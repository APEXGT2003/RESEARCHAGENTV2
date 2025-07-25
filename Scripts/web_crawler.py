import os
import requests
import sys
from arxiv_search import search_arxiv_with_pdf

# Get the project root directory (one level up from Scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the path to store PDFs
PROCESSED_DIR = os.path.join(project_root, "Data", "processed")

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def is_pdf_available(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

def fetch_urls(query):
    raw_urls = search_arxiv_with_pdf(query)
    if not raw_urls:
        print("No URLs found for the query.")
        return []

    urls = []
    count = 1  # Separate counter for valid PDFs only
    max_pdfs = 2  # Maximum number of PDFs to download

    for url in raw_urls:
        if count > max_pdfs:
            break
        if not url or not url.endswith(".pdf"):
            continue
        if is_pdf_available(url):
            filename = f"article_{count}.pdf"
            filepath = os.path.join(PROCESSED_DIR, filename)  # Save in Data/processed folder
            urls.append({'url': url, 'filename': filepath})
            count += 1
    return urls

def download_arxiv_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"✅ Successfully downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {save_path}: {str(e)}")
        return False

def crawler(question):

    # Ensure the processed folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    urls = fetch_urls(question)
    
    if not urls:
        print("No PDFs to download.")
        return
    
    successful_downloads = 0
    for url_info in urls:
        if download_arxiv_pdf(url_info['url'], url_info['filename']):
            successful_downloads += 1
    

