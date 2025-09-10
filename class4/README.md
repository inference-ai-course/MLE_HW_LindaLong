
# Class 4: Retrieval-Augmented Generation (RAG) with FastAPI and FAISS

This folder contains scripts and resources for building a Retrieval-Augmented Generation (RAG) system using FastAPI, FAISS for vector search, and language models for answer generation.

## Contents

- **main.py**  
  Main script for loading documents, splitting text, generating embeddings, building the FAISS index, and serving retrieval endpoints.
- **cleaner.py**  
  Utilities for cleaning and preprocessing text data before indexing.
- **testRAGClient.html**  
  Simple HTML client for querying the RAG system via a web interface.
- **pdf/**  
  Directory for storing downloaded or processed PDF files.

## Features

- **Document Loading:** Supports loading and preprocessing of PDF and text files.
- **Text Splitting:** Splits documents into manageable chunks for embedding and retrieval.
- **Embedding Generation:** Uses language model embeddings (e.g., OpenAI, Sentence Transformers) for vector representation.
- **FAISS Indexing:** Builds a FAISS index for efficient similarity search over document chunks.
- **RAG API:** FastAPI endpoints for querying the index and retrieving relevant context for generation.
- **Web Client:** Simple HTML/JS client for sending search queries and displaying results.

## Usage

1. **Install requirements:**
   ```sh
   pip install langchain-community pypdf openai faiss-cpu sentence-transformers PyMuPDF
