import os
import re
import json
import fitz  # PyMuPDF
#from sys import exception
from openai import OpenAI
#from sympy import content


# --- Extract Abstract from PDF ---
def extract_abstract(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    # Take text from the first 2-3 pages (title + abstract usually appear here)
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        text += page.get_text("text") + "\n"
        # Normalize "Abstract" formatting to handle variations like "Abstract-", etc.
        text = re.sub(r'(?i)(^|\n)Abstract[-—]', r'\1Abstract: ', text)


    try:
        # --- Extract Abstract ---
        abs_match = re.search(r'(abstract[:\s\-\.])(.*?)(introduction|keywords|background)', 
                            text, flags=re.IGNORECASE | re.DOTALL)
        if abs_match:
            abstract = abs_match.group(2).strip()
        else:
            print(f"⚠️ Abstract not found in {text[:2000]}.")
            abstract = "Abstract not found."

    except Exception as e:
        print(f"Error extracting abstract from {pdf_path}: {e}")
        abstract = "Error extracting abstract."

    return abstract



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY  or OPENAI_API_KEY.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or not set.")
client = OpenAI(api_key=OPENAI_API_KEY)


# --- Generate 5 QA pairs per paper using OpenAI ---
def generate_qa_from_abstract(title, abstract, model="gpt-4o-mini"):
    prompt = f"""
        You are a helpful assistant. Based on the following paper title and abstract,
        generate 5 question-answer pairs in strict JSONLarray format. Each Q&A should be concise,
        factual, and directly answerable from the abstract.

        Title: {title}
        Abstract: {abstract}
        please strictly write each line in the format like this:
        {{"text": "<|system|>You are a helpful academic Q&A assistant specialized in scholarly content.<|user|>[Question]<|assistant|>[Answer]"}}
    """

    # prompt = f"""
    # You are a research assistant who reads academic papers and creates quiz questions.

    # Below is the abstract of a research paper. **Read the abstract and generate 5 question-answer pairs** that a student might ask after reading this paper. 
    # - Ensure the questions cover the key points or findings of the paper.
    # - Provide detailed answers based only on the information in the abstract.
    # - Include a mix of question types (factual, conceptual, etc.), and avoid ambiguous or trivial questions.

    # Abstract:{abstract}

    #     """

    try:
        #print(f"Generating Q&A for: {title} using model {model} with prompt:\n{prompt}\n")
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates study Q&A."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
    except Exception as e:
        print(f"Error generating Q&A for {title}: {e}")
        raise Exception(f"Error generating Q&A for {title}: {e}")

    # Parse model output into JSON
    try:
        #qa_pairs = json.loads(response["choices"][0]["message"]["content"])
        message = response.choices[0].message  # ChatCompletionMessage object
        #reformat content to valid json
        content = message.content.replace("json", "").replace("`", '"').strip()

        match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0) # Extract the JSONL array string
            qa_pairs = json.loads(json_str)
        else:
            raise ValueError("No JSON array found in model output.")

    except Exception as e:
        print("error in parsing qa pairs" + str(e))
        raise Exception("Error parsing QA pairs: " + str(e))
    return qa_pairs


# --- Process All PDFs ---
def process_all_pdfs(folder="./pdf", output_folder="./qa_json"):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                abstract = extract_abstract(path)
                if abstract == "Abstract not found.":
                    print(f"⚠️ Abstract not found for {filename}, skipping.")
                    continue

                qa_pairs = generate_qa_from_abstract(base_name, abstract)
                print(f"Generated {len(qa_pairs)} Q&A pairs for {filename}")
                out_path = os.path.join(output_folder, f"{base_name}_qa.json")
                if os.path.exists(out_path):
                    print(f"⚠️ Warning: {out_path} already exists. Skipping to avoid overwrite.")
                    continue

                with open(out_path, "w", encoding="utf-8") as f:
                    for qa in qa_pairs:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                print(f"✅ Generated QA for {filename} → {out_path}")

            except Exception as e:
                print(f"❌ Failed to generate QA for {filename}: {e}")


# --- Consolidate all individual QA files into a single JSONL file ---
def save_all_qa_to_single_file(input_folder="./qa_json", output_file="./all_qa.jsonl"):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith("_qa.json"):
                path = os.path.join(input_folder, filename)
                with open(path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
    print(f"✅ All Q&A pairs consolidated into {output_file}")


if __name__ == "__main__":

    #print("Starting PDF to QA generation...")
    #process_all_pdfs("./pdf", "./qa_json")

    #save all qa to a single file
    save_all_qa_to_single_file("./qa_json", "./all_qa.jsonl")
