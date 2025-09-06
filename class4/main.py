from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from sentence_transformers import SentenceTransformer
from cleaner import cleanup_data

#===================set env file =====================

from dotenv import load_dotenv, find_dotenv

_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or placeholder. Ensure a valid key is set in your .env (loaded from: {_env_path or 'not found'}).")


#================ Embeding and build Faiss Index =============================

import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_path = "./pdf"
index_metadata = []


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


#==================Data Extraction=========================
import fitz  # PyMuPDF

def load_pdfs_from_folder(folder_path: str) -> list[tuple[str, str]]:
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    pdf_texts = []
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        print(f"Reading {full_path}")
        try:
            text = extract_text_from_pdf(full_path)
            pdf_texts.append((pdf_file, text))
        except Exception as e:
            print(f"Failed to read {pdf_file}: {e}")
    return pdf_texts  # List of (filename, full_text)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw content from page
        page_text = cleanup_data(page_text) # clean up the raw content
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text

#=================== Chunking ========================


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    print("Chunks created:", len(chunks), "for text[:50]:", text[:50])
    return chunks

#=================== FAISS search =======================


def search_index(query: str, metadata: list[tuple[str, str]]) -> list[dict[str, str]]:

    query_vector = model.encode([query])[0].astype("float32")#.reshape(1, -1)
    
    # Perform FAISS search
    k = 3 #==>top 3 query results will be returned
    distances, indices = faiss_index.search(np.array([query_vector]), k)
    
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        #results.append(metadata[idx])
        paper_name, chunk = metadata[idx]
        results.append({
            "paper": paper_name,
            "chunk": chunk
        })
        print (f"searched out: {paper_name}, chunk: {chunk[:50]}...")  # Print first 50 chars of chunk for brevity
        
    print(f"Search results: {distances}, {indices}")
    return {"results": results}

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
    print(f"Received query: {request.query}")
    results = search_index(request.query, index_metadata)
    return {"query": request.query, "results": results}


#============ extract clean text from stored pdf folder and build FAISS index ============
def process_pretraining_data():

    """
    Extract text from a PDF file.
    """
    pdfs = load_pdfs_from_folder(pdf_path)

    all_chunks = []
    for filename, text in pdfs:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        index_metadata.extend([(filename, chunk) for chunk in chunks])


    # Create embeddings and build FAISS index
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    global faiss_index
    faiss_index = build_faiss_index(embeddings)

    # ========= below code for class4 homework use only =========
    #save trunks in json file for debug purpose
    # import json
    # with open("all_chunks.json", "w") as f:
    #     json.dump(index_metadata, f, indent=2)
    # print("Total chunks:", len(all_chunks))

    # Save FAISS index to file for homework use
    #faiss.write_index(faiss_index, "faiss.index")

    # import pickle
    # # (Optional) Save metadata (e.g., original documents) for later retrieval
    # with open("metadata.pkl", "wb") as f:
    #     pickle.dump(all_chunks, f)

    #print("FAISS index and metadata saved.")


if __name__ == "__main__":

    process_pretraining_data()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
