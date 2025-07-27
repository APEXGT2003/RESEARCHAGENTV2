import requests
import xml.etree.ElementTree as ET

def search_arxiv_with_pdf(question, max_results=5):
    # Convert question into a URL-encoded query
    query = "+".join(question.strip().split())
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    # Send the request to arXiv API
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch from arXiv")
        return []

    # Parse the Atom XML response
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    pdf_urls = []
    for entry in root.findall("atom:entry", ns):
        try:
            arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_urls.append(pdf_url)
        except:
            continue
            
    return pdf_urls
