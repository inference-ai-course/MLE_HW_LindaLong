
# Class 5: Data Cleaning, Retrieval, and SQLite CRUD Utilities

This folder contains scripts for advanced data cleaning, retrieval using FAISS, and basic CRUD (Create, Read, Update, Delete) operations with SQLite databases.

## Contents

- **cleaner.py**  
  Utilities for cleaning and preprocessing text data, including normalization, deduplication, and filtering. Designed to prepare data for downstream retrieval or machine learning tasks.

- **main.py**  
  Main script for orchestrating the data pipeline. Handles loading data, cleaning with `cleaner.py`, building or querying a FAISS index, and integrating with other components as needed.

- **sqlite_crud.py**  
  Provides functions and classes for performing CRUD operations on a SQLite database. Useful for storing, updating, and retrieving metadata, document information, or other structured data related to your project.

- **pdf/**  
  Directory for storing downloaded or processed PDF files.## Features


## Features


- **Data Cleaning:**  
  - Removes duplicates, normalizes text, and filters unwanted content.
  - Prepares data for vectorization and retrieval.

- **Retrieval Pipeline:**  
  - Integrates with FAISS for fast similarity search over cleaned data.
  - Supports querying and returning relevant document chunks.

- **SQLite CRUD:**  
  - Simple interface for creating tables, inserting, updating, deleting, and querying records in a SQLite database.
  - Can be used to persist metadata, search results, or user interactions.

## Usage

1. **Install requirements:**
   ```sh
   pip install faiss-cpu sentence-transformers sqlite3
