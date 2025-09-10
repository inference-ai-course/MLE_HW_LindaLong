
```markdown
# Class 2: Data Cleaning & Deduplication

This folder contains scripts and resources for cleaning, deduplicating, and exporting text data as part of the AI Learning project.

## Contents

- **data_clean_deduplicate.py**  
  Main script for loading, cleaning, deduplicating, and exporting text data from various formats (JSON, JSONL, TXT).
- **cleaner.py**  
  Core cleaning and deduplication logic, including language detection, duplicate removal (MinHashLSH), and text normalization.
- **clean_corpus.txt**  
  Example output: cleaned and deduplicated text corpus.
- **stats.md**  
  Example output: cleaning statistics in Markdown format.

## Features

- **Flexible Data Loading:** Supports JSON, JSONL, and TXT input formats.
- **Deduplication:** Uses MinHashLSH for efficient near-duplicate detection.
- **Language Filtering:** Removes non-English entries.
- **Text Cleaning:** Strips HTML, normalizes whitespace, and more.
- **Export:** Cleaned data and statistics are saved for downstream tasks.

## Usage

1. **Install requirements:**
   ```sh
   pip install datasketch langdetect beautifulsoup4
   ```

2. **Prepare your data:**  
   Place your `.json`, `.jsonl`, or `.txt` files in this folder.

3. **Run the main script:**
   ```sh
   python data_clean_deduplicate.py
   ```
   By default, it will process the files specified in the script.

4. **Outputs:**
   - `clean_corpus.txt`: Cleaned text data, one entry per line.
   - `stats.md`: Cleaning statistics (lines processed, duplicates removed, etc.).

## Customization

- Edit the `files_to_clean` list in `data_clean_deduplicate.py` to specify your input files.
- Adjust cleaning logic in `cleaner.py` as needed for your data.

## Example

```sh
python data_clean_deduplicate.py
```
Produces:
- `clean_corpus.txt`
- `stats.md`

---

**Author:**  
Linda

**Note:**  
For questions or issues, please open an issue in the project repository.
```
