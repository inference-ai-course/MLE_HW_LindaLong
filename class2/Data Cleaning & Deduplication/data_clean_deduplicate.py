import json
import jsonlines
import cleaner
from typing import List, Dict, Any, Tuple


#========================== Data Loading ==========================
def load_data(files: List[str]) -> List[Dict[str, Any]]:
    raw_data = []

    for file in files:
        print("Loading data...",file)
        with open(file, 'r', encoding='utf-8') as f:
            if file.endswith('.json'):
                raw_data.extend(json.load(f))
            elif file.endswith('.jsonl'):
                for line in jsonlines.Reader(f):
                    raw_data.append(line)
            elif file.endswith('.txt'):
                text = f.read()
                text_dict_list = [{'text': text}]
                raw_data.extend(text_dict_list)

    return raw_data

#========================== Data Exporting After Cleaning ===========================
def export_to_txt(data: List[Dict[str, Any]], filename: str):

    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            for line in item:
                if isinstance(item.get(line, ''), list):
                    f.write(str(line) + ":\n")
                    for x in item.get(line, []):
                        f.write(str(x) + "\n")
                else:
                    f.write(str(line) + ":\n" + str(item.get(line, '')) + "\n")


#========================== Statistic Data Exporting After Cleaning ===========================
def export_to_md(stats: List[Dict[str, Any]], filename: str):
    with open(filename, "w") as f:
            f.write(f"# Text Cleaning Statistics\n")
            f.write(f"- Total attributes: {stats['total_attrs']}\n")
            f.write(f"- Retained attributes: {stats['retained_attrs']}\n")
            f.write(f"- Removed (non-English): {stats['removed_lang']}\n")
            f.write(f"- Removed (duplicates): {stats['removed_dupes']}\n")
            f.write(f"- Total tokens: {stats['total_tokens']}\n")
            f.write(f"- Removed (repetitive): {stats['removed_repetitive_ngrams']}\n")
            f.write(f"- Removal repetitive percentage: {stats['removed_repetitive_ngrams'] / stats['total_tokens'] * 100:.2f}%\n")


def main(files_to_clean):
    # Load data from all tasks
    try:
        raw_data = load_data(files_to_clean)
    except Exception as e:
       print(f"Error loading data: {e}")
       return       

    # Clean and deduplicate data
    cleaned_data, stats = cleaner.cleanup_data(raw_data)

    # Export results
    export_to_txt(cleaned_data, 'clean_corpus.txt')
    export_to_md(stats, 'stats.md')

    print("Processing complete. Files created: clean_corpus.txt, stats.md")

if __name__ == "__main__":

    files_to_clean = ['arxiv_clean.json', 'talks_transcripts.jsonl', 'pdf_output.txt']
    main(files_to_clean)