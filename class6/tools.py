import arxiv
import time
from pathlib import Path
import sympy


def search_arxiv(q:str, count:int) -> list[dict[str, str]]:
    """
    Search arXiv for papers matching the query `q` and return a list of papers.
    Each paper is represented as a dictionary with keys: 'url', 'title', 'authors', 'date', 'summary'.
    Also downloads the PDFs to ./pdf/*.pdf
    """

    client = arxiv.Client()

    search = arxiv.Search(
        query= q,
        max_results= count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

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
                "authors": authors,
                "date": date,
                "summary": summary
            }

            papers.append(paper_data)

    return papers


def simulate_search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a dummy passage for the given query.
    In a real system, this might query the arXiv API and extract a summary.
    """
    # TODO: Example placeholder implementation:

    return f"Here is a simulated arXiv search result snippet for query: '{query}'"



def download_pdf(papers_dir:Path, paper:arxiv.Result):
    """
    Download the PDF of the given arxiv paper to the specified directory.
    """ 

    #raise NotImplementedError
    papers_dir = Path(papers_dir)  # Convert to Path object    
    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)   
    # Generate safe filename
    safe_title = "".join(c for c in paper.title[:50] 
                        if c.isalnum() or c in (' ', '-', '_')).rstrip()
    pdf_filename = papers_dir / f"{safe_title}.pdf"
    
    # Check if already exists
    if pdf_filename.exists():
        print(f"Already exists: {safe_title}")
    else:
        # Download PDF
        paper.download_pdf(dirpath=str(papers_dir), filename=f"{safe_title}.pdf")
        time.sleep(1)  # Rate limiting


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    """
    try:

        result = sympify(expression)  # use sympy for safe evaluation
        if not isinstance(result, sympy.Number):
            raise ValueError("Expression did not evaluate to a number.")

        if isinstance(result, sympy.Float):
            result = float(result)
        elif isinstance(result, sympy.Integer):
            result = int(result)
        elif isinstance(result, sympy.Rational):
            result = float(result)

        return f"The result of {readable(expression)} is {result}"

    except Exception as e:
        # Fallback reply
        return f"I'm sorry, I encountered an error in calculation: {e}"

# Convert special mathematical symbols to more readable words
def readable(expr_str):
    return expr_str.replace('**', ' power of ').replace('*', ' times ').replace('/', ' divided by ')

# # for debugging
# if (__name__ == "__main__"):

#     print(calculate("one plus two minus minus one"))
#     print(readable("2**3 + 4 * 5"))
#     print(readable("15 / 7"))
#     print(calculate("sqrt(16) + 2"))
#     print(calculate("2^5"))

#     query = "cat:cs.CL"
#     count = 2 # Number of papers to fetch
#     papers = search_arxiv(query, count)

CALLABLE_FUNCTIONS = {
    "search_arxiv": search_arxiv,
    "calculate": calculate
}