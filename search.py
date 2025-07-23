from googlesearch import search
import asyncio
from playwright.async_api import async_playwright

def fetch_urls(query, num_results):
    raw_urls = list(search(query, num_results=num_results))
    urls = []

    for i, url in enumerate(raw_urls):
        filename = f"article{i+1}.pdf"  # Assign filenames like article1.pdf, article2.pdf, etc.
        urls.append({'url': url, 'filename': filename})

    return urls

async def scrape_website():
    urls = fetch_urls(question, 5)
    async with async_playwright() as p:
        # Headless must be True to use page.pdf()
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        for entry in urls:
            url = entry['url']
            filename = entry['filename']

            await page.goto(url, wait_until='domcontentloaded', timeout=10000)
            await page.wait_for_timeout(2000)

            await page.pdf(path=filename, format="A4")
            print(f'Saved {filename}')

        await browser.close()

if __name__ == "__main__":
    question = "DSA "
    asyncio.run(scrape_website())







