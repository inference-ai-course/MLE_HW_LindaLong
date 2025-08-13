
from PIL import Image
import pytesseract 
import arxiv
import requests
import trafilatura

import arxiv
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


def fetch_text(url, fallback_summary):
 
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        extracted = trafilatura.extract(response.text)
        return extracted.strip() if extracted else fallback_summary.strip()
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return fallback_summary.strip()
    

# Query the latest 200 papers in cs.CL
client = arxiv.Client()

search = arxiv.Search(
    query="cat:cs.CL",
    max_results=200,#200 reduced to 2 for testing
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
    #print("Fetching papers from arXiv...")

papers = []
for result in client.results(search):
        url = result.entry_id
        title = result.title.strip()
        summary = result.summary.strip()
        authors = [author.name for author in result.authors]
        date = result.published.strftime("%A, %B %d, %Y")

        text = fetch_text(url, summary)

        paper_data = {
            "url": url,
            "title": title,
            "abstract": text,
            "authors": authors,
            "date": date
        }

        papers.append(paper_data)

save_to_file(papers, 'arxiv_clean.json')


