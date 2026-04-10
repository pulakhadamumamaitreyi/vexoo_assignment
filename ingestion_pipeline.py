import re
from typing import List, Dict
from collections import Counter
from difflib import SequenceMatcher

# -----------------------------
# Utility Functions
# -----------------------------

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Sliding Window
# -----------------------------

def sliding_window(text: str, window_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (window_size - overlap)

    return chunks


# -----------------------------
# Knowledge Pyramid Layers
# -----------------------------

def summarize(chunk: str) -> str:
    # Placeholder summary (first 2 sentences)
    sentences = chunk.split('.')
    return '.'.join(sentences[:2])


def categorize(chunk: str) -> str:
    # Rule-based category
    if "ai" in chunk.lower():
        return "AI"
    elif "finance" in chunk.lower():
        return "Finance"
    elif "law" in chunk.lower():
        return "Legal"
    return "General"


def extract_keywords(chunk: str, top_k: int = 5) -> List[str]:
    words = re.findall(r'\w+', chunk.lower())
    freq = Counter(words)
    return [word for word, _ in freq.most_common(top_k)]


# -----------------------------
# Build Knowledge Pyramid
# -----------------------------

def build_pyramid(chunks: List[str]) -> List[Dict]:
    pyramid = []

    for chunk in chunks:
        entry = {
            "raw": chunk,
            "summary": summarize(chunk),
            "category": categorize(chunk),
            "keywords": extract_keywords(chunk)
        }
        pyramid.append(entry)

    return pyramid


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query: str, pyramid: List[Dict]) -> Dict:
    best_score = 0
    best_entry = None
    best_level = ""

    for entry in pyramid:
        for level in ["raw", "summary", "category"]:
            text = str(entry[level])
            score = similarity(query.lower(), text.lower())

            if score > best_score:
                best_score = score
                best_entry = entry
                best_level = level

        # keywords match
        keyword_text = " ".join(entry["keywords"])
        score = similarity(query.lower(), keyword_text.lower())

        if score > best_score:
            best_score = score
            best_entry = entry
            best_level = "keywords"

    return {
        "level": best_level,
        "content": best_entry[best_level] if best_level != "keywords" else best_entry["keywords"],
        "score": best_score
    }


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    document = open("sample.txt").read()
    document = clean_text(document)

    chunks = sliding_window(document)
    pyramid = build_pyramid(chunks)

    query = "What is AI?"
    result = retrieve(query, pyramid)

    print("Best Match Level:", result["level"])
    print("Content:", result["content"])
    print("Score:", result["score"])
