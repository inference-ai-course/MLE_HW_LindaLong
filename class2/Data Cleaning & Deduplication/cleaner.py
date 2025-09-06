from langdetect import detect
from datasketch import MinHash, MinHashLSH
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup

# Constants
SIMILARITY_THRESHOLD = 0.7
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'phone': r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
}
NEED_TO_CLEAN_ATTRIBUTES = ['abstract', 'title', 'content','text','audio_transcription', 'segments']
TRUSTED_CLEAN_ATTRIBUTES = ['url', 'authors', 'video_title','video_id','date']


stats = {
        "total_attrs": 0,
        "retained_attrs": 0,
        "removed_lang": 0,
        "removed_dupes": 0,
        "removed_pii": 0,
        "total_tokens": 0,
        "removed_repetitive_ngrams": 0
    }

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Remove personally identifiable information
def remove_pii(text: str) -> str:
    for pattern in PII_PATTERNS.values():
        text = re.sub(pattern, '[REDACTED]', text)
    return text


# Remove repetitive n-grams
def remove_repetitive_ngrams(text: str, n: int = 3) -> str:
    words = text.split()
    seen_ngrams = set()
    cleaned_words = []
    
    for i in range(len(words) - n + 1):
        stats["total_tokens"] += 1
        ngram = ' '.join(words[i:i+n])
        if ngram in seen_ngrams:
            stats["removed_repetitive_ngrams"] += 1
            continue
        seen_ngrams.add(ngram)
        cleaned_words.append(words[i])
    
    # Add remaining words
    if len(words) >= n:
        cleaned_words.extend(words[-(n-1):])
    else:
        cleaned_words.extend(words[len(cleaned_words):])
    
    return ' '.join(cleaned_words)

# Remove HTML tags
def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

def remove_duplicates(text: str, lsh: MinHashLSH) -> str:
    if not text:
        return ''
    
    mh = MinHash(num_perm=128)
    for word in set(text.split()):
        mh.update(word.encode('utf8'))

    if any(lsh.query(mh)):
        stats["removed_dupes"] += 1
        return ''
    else:
        lsh_size = len(lsh.keys)
        lsh.insert(f"item_{lsh_size}", mh)
        return text

# =================================  Cleanup data ================================

# add trusted fields that do not need cleaning
# cleanup data for specified text fields including nested dicts/lists
# For other fields, skip cleaning and ignore them

def cleanup_data(data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:

    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=128)
    cleaned_data = []

    for item_in_loop in data:

        if not isinstance(item_in_loop, dict):
            stats["total_attrs"] += 1
            continue
            # Get key and text content to process
        for key in list(item_in_loop.keys()):
            text = item_in_loop[key]

            # Handle both string and nested dict/list in one loop
            stack = [(key, text)]
            while stack:
                current_key, current_text = stack.pop()

                if current_key in TRUSTED_CLEAN_ATTRIBUTES: 
                    #trust these fields without cleaning
                    cleaned_data.append({current_key: current_text})
                    stats["retained_attrs"] += 1
                    stats["total_attrs"] += 1
                    continue

                if current_key not in NEED_TO_CLEAN_ATTRIBUTES:
                    #skip and ignore non-text and not interested fields
                    stats["total_attrs"] += 1
                    continue

                if isinstance(current_text, str) and current_text.strip() != '':
                    cleaned_current_text = remove_duplicates(clean_text(current_text), lsh)
                    cleaned_data.append({current_key: cleaned_current_text})
                    stats["retained_attrs"] += 1
                    stats["total_attrs"] += 1
                elif isinstance(current_text, dict) and current_text:
                    print("Found nested dict with key:" + current_key[:30])
                    for sub_key, sub_text in current_text.items():
                        stack.append((sub_key, sub_text))

                elif isinstance(current_text, list) and current_text:
                    print(f"Found nested list with {len(current_text)} items for key: {current_key[:30]}")
                    for sub_item in current_text:
                        if isinstance(sub_item, dict):
                            for sub_key, sub_text in sub_item.items():
                                stack.append((sub_key, sub_text))

                else:
                    print("Ignored key:{" + current_key[:30] + "}" + " of type: " + str(type(current_text)))


    return cleaned_data, stats


def clean_text(text: str) -> str:
    """Apply all cleaning steps to the text"""
    if not text or not is_english(text):
        stats["removed_lang"] += 1
        return ''
    else:
        text = remove_repetitive_ngrams(remove_pii(remove_html(text)))

    return text


