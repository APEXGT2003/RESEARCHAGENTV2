import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from web_crawler import main
import os
import numpy as np

from dotenv import load_dotenv

import asyncio
from search import scrape_website

load_dotenv('api_keys/.env')
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key=os.environ["GROQ_API_KEY"]

THRESHOLD = 0.7 

def create_vector_embeddings():
    loader = PyPDFDirectoryLoader(processed_file)
    docs = loader.load()
    st.write(f"Loaded {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_docs = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_docs, embeddings)
    vectors.save_local(vectordb_path)
    st.session_state.vectors = vectors


class NumpyOllamaEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts):
        result = super().embed_documents(texts)
        arr = np.array(result, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def embed_query(self, text):
        result = super().embed_query(text)
        arr = np.asarray(result, dtype=np.float32).flatten()
        # Convert to a list of Python floats, not numpy.float32
        return [float(x) for x in arr]

embeddings = NumpyOllamaEmbeddings(model="nomic-embed-text")
 
# Get the project root directory (one level up from Scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_file = os.path.join(project_root, "Data", "processed")
vectordb_path = os.path.join(project_root, "Data", "vectordb")
 
# Create directories if they don't exist
os.makedirs(processed_file, exist_ok=True)
os.makedirs(vectordb_path, exist_ok=True)
#relod the session to get past process
# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Research agent")

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

chat_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions with at most precision, answer with respect to the context only.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Display chat history with colors
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"{entry['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"{entry['bot']}")

# Use chat_input at the bottom for prompt
prompt = st.chat_input("Type your message...")

if prompt:
    if "vectors" not in st.session_state:
        if not os.path.exists(f"{vectordb_path}/index.faiss"):
            create_vector_embeddings()
        st.session_state.vectors = FAISS.load_local(
            vectordb_path, embeddings, allow_dangerous_deserialization=True
        )
    context = "\n".join(
        [f"User: {entry['user']}\nAgent: {entry['bot']}" for entry in st.session_state.chat_history]
    )
    docs_and_scores = st.session_state.vectors.similarity_search_with_score(
        prompt,
        k=10
    )
    top_docs = [doc for doc, _ in docs_and_scores]
    top_dists = [score for _, score in docs_and_scores]
    top_sims = [1 - dist for dist in top_dists]
    # If no good docs, scrape and rebuild
    # if not top_docs or top_sims[0] < THRESHOLD:
    #     with st.spinner("Searching the web and updating knowledge base..."):
    #         asyncio.run(scrape_website(prompt))  # Save PDFs to Data/processed
    #         create_vector_embeddings()
    #         st.session_state.vectors = FAISS.load_local(
    #             vectordb_path, embeddings, allow_dangerous_deserialization=True
    #         )
    #     docs_and_scores = st.session_state.vectors.similarity_search_with_score(
    #         prompt,
    #         k=10
    #     )
    #     top_docs = [doc for doc, _ in docs_and_scores]
    #     top_dists = [score for _, score in docs_and_scores]
    #     top_sims = [1 - dist for dist in top_dists]
    # Generate answer
    document_chain = create_stuff_documents_chain(llm, chat_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt, 'context': context})
    # Save history
    st.session_state.chat_history.append({"user": prompt, "bot": response["answer"]})
    # Show the latest messages in chat colors
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response["answer"])