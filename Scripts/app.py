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
import base64

image_path = os.path.join(os.path.dirname(__file__), "images", "bg.png")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_base64 = get_base64_image(image_path)

st.markdown("""
    <style>
    /* Make entire app background transparent */
    .main, .block-container {
        background-color: rgba(0, 0, 0, 0);
    }

    /* Remove background of input area container */
    .st-emotion-cache-z5fcl4 {
        background-color: rgba(0, 0, 0, 0) !important;
    }

    /* Make text input box transparent */
    .stTextInput>div>div>input {
        background-color: rgba(0, 0, 0, 0) !important;
        color: white !important;
    }

    /* Make the container of input and submit button blend */
    .st-emotion-cache-1kyxreq {
        background-color: rgba(0, 0, 0, 0) !important;
    }

    /* Optional: remove red border if undesired */
    .stTextInput>div>div {
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 30px;
    }

    </style>
""", unsafe_allow_html=True)








load_dotenv('api_keys/.env')
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
api_key = os.environ["GROQ_API_KEY"]


processed_file = "Data/processed"
vectordb_path = "Data/vectordb"
cache_file = "cache/file_hashes.json"

st.set_page_config(
    page_title="Research Agent",
    page_icon="images/book.png",  # Put your image in the `images` folder
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
    <style>
    .fancy-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(255, 255, 255, 0.3);
        font-family: 'Segoe UI', sans-serif;
        margin-top: 20px;
    }
    </style>

    <h1 class="fancy-title">🧠 Research Agent</h1>
""", unsafe_allow_html=True)


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

for entry in st.session_state.chat_history:
    with st.chat_message("user",avatar="👨‍🦰"):
        st.markdown(entry["user"])
    with st.chat_message("assistant",avatar="📚"):
        st.markdown(entry["bot"])


prompt = st.chat_input("Ask your research question...")
changes_detected = False

if prompt:
    with st.chat_message("user",avatar="👨‍🦰"):
        st.markdown(prompt)

    with st.chat_message("assistant",avatar="📚"):
        response_placeholder = st.empty()
        response_placeholder.markdown(
            f"""
            <div style="background-color: black; padding: 10px; border-radius: 10px;">
                *Typing...*
            </div>
            """,
            unsafe_allow_html=True
        )


    if "vectors" not in st.session_state:
        if not os.path.exists(f"{vectordb_path}/index.faiss"):
            create_vector_embeddings(processed_file, vectordb_path, embeddings, st)
        st.session_state.vectors = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)

    prompt_embedded = embeddings.embed_query(prompt)
    docs_and_scores = st.session_state.vectors.similarity_search_with_score_by_vector(
        embedding=prompt_embedded, k=3
    )

    with st.sidebar:
        st.subheader("🔍 Top Document Matches")
        for i, (doc, score) in enumerate(docs_and_scores):
            source_name = doc.metadata.get("source", "Unknown file")
            st.markdown(
            f"""
            <div style="
                padding: 10px;
                margin-bottom: 10px;
                background-color: rgba(255, 255, 255, 0.05);
                border-left: 4px solid #1f77b4;
                border-radius: 6px;
                font-size: 14px;
            ">
                <strong style="color:#ffffff;">Match {i+1}</strong><br>
                <span style="color: #cccccc;">Distance:</span> <code style="color: #eeeeee;">{score:.3f}</code><br>
                <span style="color: #cccccc;">File:</span> <code style="color: #eeeeee;">{source_name}</code>
            </div>
            """,
            unsafe_allow_html=True
        )



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

    context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['bot']}" for entry in st.session_state.chat_history])

    document_chain = create_stuff_documents_chain(llm, chat_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt, 'context': context})

    response_placeholder.markdown(response["answer"])

    # Save chat
    st.session_state.chat_history.append({"user": prompt, "bot": response["answer"]})


