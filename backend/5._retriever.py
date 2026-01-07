import chromadb
from chromadb.config import Settings
import ollama
import os

# ---------------- CONFIG ----------------
# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "bs_rag_collection"
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model (alternatively: mxbai-embed-large)
TOP_K = 5
# --------------------------------------


def get_ollama_embedding(text):
    """
    Generate embeddings using Ollama's embedding model.
    """
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def retrieve_relevant_chunks(query, top_k=TOP_K):
    """
    Retrieve the most relevant chunks for a given query using ChromaDB.
    
    Args:
        query (str): The search query
        top_k (int): Number of top chunks to retrieve (default: 5)
    
    Returns:
        list: List of dictionaries containing ranked chunks with metadata and scores
    """
    # Connect to ChromaDB (persistent)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Load collection
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print(f"✅ ChromaDB loaded - Collection: {COLLECTION_NAME}")
    
    # Generate query embedding
    print("[INFO] Generating query embedding...")
    query_embedding = get_ollama_embedding(query)
    
    # Perform similarity search
    print(f"[INFO] Searching for top {top_k} relevant chunks...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results with ranking scores
    ranked_chunks = []
    for i in range(len(results["documents"][0])):
        chunk_data = {
            "rank": i + 1,
            "score": results["distances"][0][i],  # Lower distance = higher similarity
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i]
        }
        ranked_chunks.append(chunk_data)
    
    return ranked_chunks



user_query = input("\n🔍 Enter your query: ").strip()

if not user_query:
    print("❌ Empty query")
    exit()

# Retrieve relevant chunks
retrieved_chunks = retrieve_relevant_chunks(user_query, top_k=5)

# Print results
print("\n" + "="*80)
print("📄 RETRIEVED CHUNKS (Ranked by Relevance)")
print("="*80 + "\n")

for chunk in retrieved_chunks:
    print(f"🔹 Rank: {chunk['rank']}")
    print(f"📊 Similarity Score: {chunk['score']:.4f}")
    print(f"📂 Metadata: {chunk['metadata']}")
    print(f"📝 Text:\n{chunk['text']}\n")
    print("-" * 80 + "\n")
