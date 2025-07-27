import os
import json
import hashlib
import pickle
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def has_changes(file_path, hash_file="file_hashes.json"):
    current_hash = compute_md5(file_path)

    if not os.path.exists(hash_file):
        with open(hash_file, "w") as f:
            json.dump({}, f)

    with open(hash_file, "r") as f:
        try:
            hashes = json.load(f)
        except json.JSONDecodeError:
            hashes = {}

    file_name = os.path.basename(file_path)
    old_hash = hashes.get(file_name)

    if old_hash != current_hash:
        hashes[file_name] = current_hash
        with open(hash_file, "w") as f:
            json.dump(hashes, f)
        return True
    return False

def create_vector_embeddings(processed_folder, vectordb_path, embeddings, st=None):
    all_new_docs = []
    hash_file = os.path.join(CACHE_DIR, "file_hashes.json")

    for pdf_file in os.listdir(processed_folder):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(processed_folder, pdf_file)
            pickle_cache_path = os.path.join(CACHE_DIR, f"{pdf_file}.pkl")

            changed = has_changes(file_path, hash_file)

            if changed or not os.path.exists(pickle_cache_path):
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
                split_docs = splitter.split_documents(raw_docs)

                with open(pickle_cache_path, "wb") as f:
                    pickle.dump(split_docs, f)

                if st:
                    st.info(f"Processed and cached: {pdf_file}")
                all_new_docs.extend(split_docs)

            else:
                if st:
                    st.info(f"No changes detected for: {pdf_file}")

    # Load existing vector DB if exists
    if os.path.exists(os.path.join(vectordb_path, "index.faiss")):
        vectorstore = FAISS.load_local(vectordb_path, embeddings,allow_dangerous_deserialization=True)
        if all_new_docs:
            vectorstore.add_documents(all_new_docs)
            if st:
                st.success("New documents added to vector DB.")
        else:
            if st:
                st.info("No new documents to add.")
    else:
        if all_new_docs:
            vectorstore = FAISS.from_documents(all_new_docs, embeddings)
            if st:
                st.success("New vector DB created.")
        else:
            if st:
                st.warning("No vector DB found and no new documents available.")
            return

    vectorstore.save_local(vectordb_path)

