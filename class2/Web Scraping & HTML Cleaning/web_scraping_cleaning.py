
from PIL import Image #pip install Pillow
import pytesseract 
import arxiv
import requests
import trafilatura
import json
import os


def save_to_file(data, filename='arxiv_clean.json'):

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n Saved {len(data)} papers to {file_path}")
    except Exception as e:
        print(f"Failed to save data: {e}")


def fetch_text(url):
 
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        #use trafilatura to clean data
        extracted = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        return extracted.strip() 
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return "failed to extract content"
    

# Query the latest 200 papers in cs.CL
client = arxiv.Client()

search = arxiv.Search(
    query="cat:cs.CL",
    max_results=2,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
    #print("Fetching papers from arXiv...")

papers = []
for result in client.results(search):
        url = result.entry_id
        title = result.title.strip()
        #summary = result.summary.strip()
        authors = [author.name for author in result.authors]
        date = result.published.strftime("%A, %B %d, %Y")

        text = fetch_text(url)

        paper_data = {
            "url": url,
            "title": title,
            "abstract": text,
            "authors": authors,
            "date": date
        }

        papers.append(paper_data)

save_to_file(papers, 'arxiv_clean.json')


