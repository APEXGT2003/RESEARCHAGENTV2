import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from web_crawler import crawler
import os


from dotenv import load_dotenv
load_dotenv('../api_keys/.env')
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key=os.environ["GROQ_API_KEY"]
embeddings = OllamaEmbeddings(model="nomic-embed-text") #pull this model in ollama local.

# Get the project root directory (one level up from Scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_file = os.path.join(project_root, "Data", "processed")
vectordb_path = os.path.join(project_root, "Data", "vectordb")

# Create directories if they don't exist
os.makedirs(processed_file, exist_ok=True)
os.makedirs(vectordb_path, exist_ok=True)
#relod the session to get past process
st.title("Research agent") 

llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")
prompt=st.text_input("Enter your prompt")
chat_prompt=ChatPromptTemplate.from_template(
"""  
Answer the questions with atmost precision ,answer with respect to the context only.
<context>
{context}
<context>
Question:{input}
"""

)

def create_vector_embeddings():
    loader = PyPDFDirectoryLoader(processed_file)
    docs = loader.load()
    st.write(f"Loaded {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_docs = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_docs, embeddings)
    vectors.save_local(vectordb_path)
    st.session_state.vectors = vectors

THRESHOLD=0.4 #if u have good suggestion value put it here.


if prompt:
    if "vectors" not in st.session_state:
            if not os.path.exists(f"{vectordb_path}/index.faiss"): #edge case->starting with processed but vectors not created.
                create_vector_embeddings()
            st.session_state.vectors = FAISS.load_local(vectordb_path, embeddings,allow_dangerous_deserialization=True) 
    prompt_embedded = embeddings.embed_query(prompt)
    docs_and_scores = st.session_state.vectors.similarity_search_with_score_by_vector(
        embedding=prompt_embedded,
        k=10  # top results
    )
    
    top_docs = [doc for doc, _ in docs_and_scores]
    top_dists = [score for _, score in docs_and_scores]
    top_sims = [1 - dist for dist in top_dists]
    if not top_docs or top_sims[0] < THRESHOLD:



        crawler(prompt)
        create_vector_embeddings()
        
        #web_crawler code here
        #llm gives query to web crawler (take the prompt and make it into a query type using llm's capability to get the best result )
        #web crawler finds relvant pages and makes it into pdf's
        #the pdfs are saved in the processed folder under Data/
        #create_vector_embeddings()#the vector embeddings are created for the llm from the pdf.
        #st.session_state.vectors = FAISS.load_local(vectordb_path, embeddings)
        pass
    document_chain=create_stuff_documents_chain(llm,chat_prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':prompt})
    st.write(response["answer"])