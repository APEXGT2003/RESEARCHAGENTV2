import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from vector_store import create_vector_embeddings, has_changes
from web_crawler import crawler
import os
from dotenv import load_dotenv


load_dotenv('api_keys/.env')
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
api_key = os.environ["GROQ_API_KEY"]


processed_file = "Data/processed"
vectordb_path = "Data/vectordb"
cache_file = "cache/file_hashes.json"


st.title("Research Agent")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
chat_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions with atmost precision. Answer with respect to the context only.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

prompt = st.text_input("Enter your prompt")
changes_detected = False

if prompt:
    if "vectors" not in st.session_state:
        if not os.path.exists(f"{vectordb_path}/index.faiss"):
            create_vector_embeddings(processed_file, vectordb_path, embeddings, st)
        st.session_state.vectors = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)

    prompt_embedded = embeddings.embed_query(prompt)
    docs_and_scores = st.session_state.vectors.similarity_search_with_score_by_vector(
        embedding=prompt_embedded, k=3
    )

    for i, (doc, score) in enumerate(docs_and_scores):
        source_name = doc.metadata.get("source", "Unknown file")
        st.markdown(f"**Match {i+1}** — Distance: `{score:.3f}`, File: `{source_name}`")

    top_docs = [doc for doc, _ in docs_and_scores]
    top_dists = [score for _, score in docs_and_scores]



    MAX_DISTANCE = 1.3
    if not top_docs or top_dists[0] > MAX_DISTANCE:
        st.warning("No strong match found. Crawling for more information...")
        crawler(prompt)


        changes_detected = any(
        has_changes(os.path.join(processed_file, f))
        for f in os.listdir(processed_file)
         if f.endswith(".pdf")
        )


    if changes_detected:
        st.info("New content found. Rebuilding vector store...")
        create_vector_embeddings(processed_file, vectordb_path, embeddings, st)
        st.session_state.vectors = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)

    document_chain = create_stuff_documents_chain(llm, chat_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt})
    st.write(response["answer"])
