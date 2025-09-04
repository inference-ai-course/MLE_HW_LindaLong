
from PIL import Image
from pdf2image import convert_from_path
import pytesseract 
import arxiv
import os

OUTPUT_DIR = "./pdf_ocr"

def save_images_to_txtfile(images, filename):

    data = ""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        for image in images:
            # Perform OCR to extract text from the image
            data += pytesseract.image_to_string(image) + "\n\n"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)

    except Exception as e:
        print(f"Failed to save data: {e}")


def pdf_ocr():

    # Query the latest 200 papers in cs.CL
    client = arxiv.Client()

    search = arxiv.Search(
        query="cat:cs.CL",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    for result in client.results(search):
            
            try:
                pdf_path = result.download_pdf()
                images = convert_from_path(pdf_path)
                txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(pdf_path)[0] + ".txt")
                #print("Saving output to..." + txt_path)
                if not os.path.exists(txt_path):
                    save_images_to_txtfile(images, txt_path)
                # Remove the PDF file after processing
                os.remove(pdf_path)
            except Exception as e:
                print(f"Error processing {result.title}: {e}")


if __name__ == "__main__":
    pdf_ocr()
