import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
INPUT_FILE = "Cleaned Data/single_corpus.txt"
OUTPUT_FILE = "Cleaned Data/chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SOURCE_NAME = os.path.basename(INPUT_FILE)
# --------------------------------------


def langchain_chunking(text):
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter.
    This splitter recursively tries different separators to avoid breaking sentences.
    Priority is given to sentence boundaries to ensure no sentence is broken.
    """
    # Initialize the text splitter with separators prioritizing sentence boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,  # Keep separators with the preceding chunk
        separators=[
            "\n\n",  # Double newline (paragraphs)
            "\n",    # Single newline
            ". ",    # Sentence end with space (highest priority for sentences)
            "! ",    # Exclamation with space
            "? ",    # Question mark with space
            " ",     # Space (word boundary)
            ""       # Character (last resort)
        ]
    )
    
    # Split the text
    text_chunks = text_splitter.split_text(text)
    
    # Format chunks into the desired structure
    chunks = []
    for idx, chunk_text in enumerate(text_chunks, start=1):
        chunks.append({
            "chunk_id": idx,
            "text": chunk_text.strip(),
            "source": SOURCE_NAME,
            "chunk_size": len(chunk_text.strip())
        })
        print(f"✅ Chunk {idx}: Stored successfully ({len(chunk_text.strip())} chars)")
    
    return chunks


# ---------------- RUN ----------------
if __name__ == "__main__":
    # Read the input text
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"📖 Reading text corpus from: {INPUT_FILE}")
    print(f"📏 Total text length: {len(text)} characters")
    print(f"⚙️  Chunk size: {CHUNK_SIZE} characters")
    print(f"🔄 Overlap size: {CHUNK_OVERLAP} characters")
    print("-" * 60)
    
    # Perform chunking using LangChain
    chunks = langchain_chunking(text)
    
    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print("-" * 60)
    print("✅ Chunking complete!")
    print(f"📦 Total chunks: {len(chunks)}")
    print(f"💾 Output saved to: {OUTPUT_FILE}")
