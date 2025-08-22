from PIL import Image
import arxiv
import os
import time
from pathlib import Path

    

def fetch_arxiv_papers(q:str, count:int) -> list[dict[str, str]]:

    client = arxiv.Client()

    search = arxiv.Search(
        query= q,
        max_results= count,
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

            download_pdf("./pdf", result)

            paper_data = {
                "url": url,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "date": date
            }

            papers.append(paper_data)
    print("Fetching papers from arXiv... " + str(len(papers)) + " papers found.")
    return papers


# def extract_text_from_pdf(pdf_path: str) -> str:
#     doc = fitz.open(pdf_path)
#     return "\n".join(page.get_text() for page in doc)


def download_pdf(papers_dir:Path, paper:arxiv.Result):

    #raise NotImplementedError
    papers_dir = Path(papers_dir)  # Convert to Path object    
    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)   
    # Generate safe filename
    safe_title = "".join(c for c in paper.title[:50] 
                        if c.isalnum() or c in (' ', '-', '_')).rstrip()
    pdf_filename = papers_dir / f"{safe_title}.pdf"
    print(f"Safe title: {safe_title}, PDF filename: {pdf_filename}")
    
    # Check if already exists
    if pdf_filename.exists():
        print(f"Already exists: {safe_title}")
    else:
        print(f"Downloading PDF: {safe_title}")
        paper.download_pdf(dirpath=str(papers_dir), filename=f"{safe_title}.pdf")
        time.sleep(1)  # Rate limiting


if(__name__ == "__main__"):
    query = "cat:cs.CL"
    count = 50 # Number of papers to fetch
    papers = fetch_arxiv_papers(query, count)
