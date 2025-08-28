
#=================== libraries installed ========================
#downloads required for langchain needed:
# pip3 install -U langchain-community
# pip3 install pypdf
# pip3 install --upgrade langchain langchain-core langchain-community langchain-openai langchain-experimental chromadb
# pip3 install openai
# pip3 install faiss-cpu -->Mac accepted faiss-cpu only
# pip3 install sentence-transformers
# pip3 install PyMuPDF

from xml.dom.minidom import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from sentence_transformers import SentenceTransformer
import sqlite_crud

#===================set env file =====================

from dotenv import get_key, load_dotenv, find_dotenv
#load_dotenv() #--> this setting to load OPENAI_API_KEY into code
# Create a `.env` file in your project root with this format:
# OPENAI_API_KEY=sk-...your real key...

_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or placeholder. Ensure a valid key is set in your .env (loaded from: {_env_path or 'not found'}).")
os.environ["OPENAI_API_KEY"] = api_key


#================ Embeding and build Faiss Index =============================

import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_path = "./pdf"
all_chunks = []

def build_faiss_index() -> faiss.IndexFlatL2:
    texts = [c["chunk"] for c in all_chunks] # Extract texts only from chunk dicts for index purpose
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

#==================Data Extraction=========================

import fitz  # PyMuPDF

def load_pdfs_from_folder(folder_path: str) -> list[tuple[str, str, str, str, str]]:
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    pdf_texts = []
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        print(f"Reading {full_path}")

        try:
            doc = fitz.open(full_path)

            title = doc.metadata.get("title", pdf_file[:-4])  # Remove .pdf extension
            author = doc.metadata.get("author", "Unknown")
            year = doc.metadata.get("creationDate", "Unknown")
            keywords = doc.metadata.get("keywords", "None")
            print(f"Extracted metadata for {pdf_file}: Title: {title}, Author: {author}, Year: {year}, Keywords: {keywords}")

            text = extract_raw_text_from_pdf(full_path, doc)
            pdf_texts.append((title, author, year, keywords, text))
        except Exception as e:
            print(f"Failed to read {pdf_file}: {e}")
    return pdf_texts  # List of (title, author, year, keywords, full_text)


def extract_raw_text_from_pdf(pdf_path: str, doc: Document) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw content from page
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text


#=================== Chunking ========================

def chunk_text(doc_id: str, title: str, text: str, chunk_size=512, overlap=50) -> list[dict]:
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk": chunk_text,
            "title": title
        })
        i += chunk_size - overlap
        chunk_id += 1
    return chunks


#=================== FAISS search =======================


def faiss_search_index(query: str) -> list[dict[str, str]]:

    query_vector = model.encode([query])[0].astype("float32")#.reshape(1, -1)
    
    # Perform FAISS search
    k = 3 #==>top 3 query results will be returned
    distances, indices = faiss_index.search(np.array([query_vector]), k)

    # Retrieve the corresponding chunks on faiss index
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity_score = float(1 / (1 + dist))  # Convert distance to similarity score      
        results.append({
            "doc_id": all_chunks[idx]["doc_id"],
            "chunk_id": all_chunks[idx]["chunk_id"],
            "chunk": all_chunks[idx]["chunk"][:20] + "...",
            "title": all_chunks[idx]["title"],
            "score": similarity_score
        })

    #print(f"Search results: {distances}, {indices}")
    return results

#=================== keyword search =======================


def fts_search_index(query: str) -> list[dict[str, str]]:

    # Perform fts search
    k = 3 #==>top 3 query results will be returned
    results = sqlite_crud.search_docs(query, k)
    for result in results:
        result["score"] = 0.5
    #print(f"Retrieved document: {results}")
    return results

#=================== Hybrid search =======================

def hybrid_score(vec_score, key_score, alpha=0.5)-> list[dict[str, str]]:
    # Assume vec_score and key_score are normalized (0-1).
    hybrid_score = alpha * vec_score + (1 - alpha) * key_score
    #print(f"Hybrid score (vec: {vec_score}, key: {key_score}): {hybrid_score}")
    return hybrid_score


def hybrid_search_index(query, k=3, alpha=0.6):

    # 1. Compute query embedding for FAISS
    vector_results = faiss_search_index(query)

    # 2. Get top-k from FAISS and top-k from SQLite FTS/BM25
    keyword_results = fts_search_index(query)
    
    # 3. Merge scores (as above) and select final top-k documents
    combined = []
    for v_result in vector_results:
        combined.append({"result":v_result,  "v_score":v_result["score"], "k_score":0.0})

    for k_result in keyword_results:
        if k_result["doc_id"] not in [c["result"]["doc_id"] for c in combined]:
            combined.append({"result":k_result, "v_score":0.0, "k_score":k_result["score"]})
        elif k_result["chunk_id"] not in [c["result"]["chunk_id"] for c in combined if c["result"]["doc_id"]==k_result["doc_id"]]:
            combined.append({"result":k_result, "v_score":0.0, "k_score":k_result["score"]})
        else:
            for c in combined:
                if c["result"]["doc_id"] == k_result["doc_id"] and c["result"]["chunk_id"] == k_result["chunk_id"]:
                    c["k_score"] = k_result["score"]
                    break

    for combined_result in combined:
        combined_result["hybrid_score"] = hybrid_score(combined_result["v_score"], combined_result["k_score"], alpha)    
    
    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined[:k]


#=================== FastAPI Route ========================

import faiss
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


app = FastAPI()

# enable fast API to handle CORS (Cross-Origin Resource 
# Sharing) preflight requests for test purpose only
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Multi-PDF RAG Search API"}


@app.post("/search")
def search(request: QueryRequest):
    #print(f"Received query: {request.query}")
    results = faiss_search_index(request.query)
    #results = fts_search_index(request.query)
    return {"query": request.query, "results": results}



@app.post("/hybrid_search")
async def hybrid_search(request: QueryRequest, k: int = 3):
    results = hybrid_search_index(request.query, k)
    return {"query": request.query, "results": results}



def process_pretraining_data():

    """
    Extract text from a PDF file.
    """
    pdfs = load_pdfs_from_folder(pdf_path)

    #chunk pdf texts and saved into all_chunks and sqlite respectively
    for title, author, year, keywords, text in pdfs:
        
        idx = sqlite_crud.get_max_id()+1
        chunks = chunk_text(idx,title, text) 
        #save chunks to all_chunks and sqlite
        all_chunks.extend(chunks)
        sqlite_crud.insert_data(idx, title, author, year, keywords, chunks)
    
    #build index
    global faiss_index
    faiss_index = build_faiss_index()


if __name__ == "__main__":

    sqlite_crud.create_table()
    process_pretraining_data()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
