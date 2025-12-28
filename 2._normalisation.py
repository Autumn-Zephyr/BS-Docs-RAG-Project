import re

def normalize_single_corpus(text):
    # convert to lowercase
    text = text.lower()

    # fix hyphenated line breaks (word-\nword)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # replace all newlines with spaces
    text = re.sub(r'[\r\n]+', ' ', text)

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -------- main --------
INPUT_FILE = "Cleaned Data/extracted_text.txt"
OUTPUT_FILE = "Cleaned Data/single_corpus.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

normalized_text = normalize_single_corpus(raw_text)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(normalized_text)

print("✅ Normalized single corpus saved as:", OUTPUT_FILE)
print("📏 Character count:", len(normalized_text))
print("🔍 Preview:", normalized_text[:300])
