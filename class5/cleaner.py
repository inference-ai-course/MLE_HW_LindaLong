from langdetect import detect
from datasketch import MinHash, MinHashLSH
import re
from bs4 import BeautifulSoup

# Constants
SIMILARITY_THRESHOLD = 0.7
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'phone': r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
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
        ngram = ' '.join(words[i:i+n])
        if ngram in seen_ngrams:
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
        return ''
    else:
        lsh_size = len(lsh.keys)
        lsh.insert(f"item_{lsh_size}", mh)
        return text

# # =================================  Cleanup data ================================

def cleanup_data(text: str) -> str:
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=128)
    if not text or not is_english(text):
        return ''
    else:
        text = remove_repetitive_ngrams(remove_pii(remove_html(text)))
        text = remove_duplicates(text, lsh)
    return text


# For quick testing
if (__name__ == "__main__"):
    sample_text =  "<html>Contact me at <a href='mailto:test@example.com'>test@example.com</a></html>"
    cleaned_text = cleanup_data(sample_text)
    print(cleaned_text)   # Output: "Contact me at [REDACTED]"
