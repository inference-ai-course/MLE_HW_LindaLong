
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
#brew install poppler
import pytesseract 
import arxiv
import requests
import trafilatura

import arxiv
import json
import os

from PIL import Image




def save_image_to_file(image, filename='pdf_ocr/pdf_output.json'):


    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Perform OCR on the image
        data = pytesseract.image_to_string(image)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save data: {e}")


# Query the latest 200 papers in cs.CL
client = arxiv.Client()

search = arxiv.Search(
    query="cat:cs.CL",
    max_results=2,#200 reduced to 2 for testing
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

for result in client.results(search):
        # Download the PDF file
        pdf_path = result.download_pdf()
        print("Converting PDF to image..." + pdf_path)
        # Convert PDF to image
        images = convert_from_bytes(open(pdf_path, 'rb').read())
        for image in images:
            save_image_to_file(image)
        # Remove the PDF file after processing
        os.remove(pdf_path)



