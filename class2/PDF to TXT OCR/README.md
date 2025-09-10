# pdf_to_txt_ocr.py

This script provides utilities for extracting text from PDF files using Optical Character Recognition (OCR), making it possible to process scanned documents or image-based PDFs for downstream analysis.

## Purpose

- **pdf_to_txt_ocr.py** is designed to convert PDF files (including those that are scanned or image-based) into plain text files using OCR technology. This is useful for preparing data for natural language processing, machine learning, or archival purposes.

## Features

- **OCR Extraction:** Uses OCR to extract text from each page of a PDF, even if the PDF contains only images.
- **Batch Processing:** Can process multiple PDF files in a directory.
- **Output:** Produces `.txt` files with the extracted text, named after the original PDF files.
- **Customizable:** Easily adaptable for different OCR engines or output formats.

## Usage

1. **Install requirements:**
   ```sh
   pip install pytesseract pdf2image pillow
