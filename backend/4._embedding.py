import json
import os
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import ollama

# ---------------- CONFIG ----------------
INPUT_FILE = "Cleaned Data/chunks.json"
CHROMA_DB_PATH = "Cleaned Data/chroma_db"
COLLECTION_NAME = "bs_rag_collection"
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model
# --------------------------------------


def load_chunks(file_path):
    """
    Load chunks from JSON file.
    """
    print(f"[INFO] Loading chunks from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"[SUCCESS] Loaded {len(chunks)} chunks")
    return chunks


def initialize_chromadb(db_path, collection_name):
    """
    Initialize ChromaDB client and create/get collection.
    """
    print(f"[INFO] Initializing ChromaDB at: {db_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB client with persistent storage
    client = Client(Settings(
        persist_directory=db_path,
        anonymized_telemetry=False
    ))
    
    # Try to get existing collection or create new one
    try:
        # Delete existing collection if it exists (for fresh start)
        try:
            client.delete_collection(name=collection_name)
            print(f"[INFO] Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "BS RAG embeddings collection"}
        )
        print(f"[SUCCESS] Created new collection: {collection_name}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create collection: {e}")
        raise
    
    return client, collection


def generate_embeddings_with_ollama(texts, model_name):
    """
    Generate embeddings for a list of texts using Ollama.
    """
    print(f"[INFO] Generating embeddings using Ollama model: {model_name}")
    
    embeddings = []
    total = len(texts)
    
    for idx, text in enumerate(texts, 1):
        try:
            # Generate embedding using Ollama
            response = ollama.embeddings(
                model=model_name,
                prompt=text
            )
            
            embedding = response['embedding']
            embeddings.append(embedding)
            
            # Progress indicator
            if idx % 10 == 0 or idx == total:
                print(f"[PROGRESS] Generated embeddings: {idx}/{total}")
                
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding for chunk {idx}: {e}")
            raise
    
    print(f"[SUCCESS] Generated {len(embeddings)} embeddings")
    return embeddings


def store_embeddings_in_chromadb(collection, chunks, embeddings):
    """
    Store chunks and their embeddings in ChromaDB.
    """
    print(f"[INFO] Storing embeddings in ChromaDB collection")
    
    # Prepare data for ChromaDB
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk['chunk_id'],
            "source": chunk['source'],
            "chunk_size": chunk['chunk_size']
        }
        for chunk in chunks
    ]
    
    # Add to collection
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[SUCCESS] Stored {len(chunks)} chunks with embeddings in ChromaDB")
        
    except Exception as e:
        print(f"[ERROR] Failed to store embeddings: {e}")
        raise


def verify_storage(collection):
    """
    Verify the stored data in ChromaDB.
    """
    print(f"[INFO] Verifying stored data...")
    
    # Get collection count
    count = collection.count()
    print(f"[INFO] Total documents in collection: {count}")
    
    # Peek at first few documents
    results = collection.peek(limit=3)
    print(f"[INFO] Sample documents stored:")
    for i, (doc_id, doc) in enumerate(zip(results['ids'], results['documents']), 1):
        print(f"  {i}. ID: {doc_id}, Text preview: {doc[:100]}...")
    
    print(f"[SUCCESS] Verification complete")


def test_query(collection, query_text, n_results=3):
    """
    Test querying the ChromaDB collection with Ollama embeddings.
    """
    print(f"\n[INFO] Testing query functionality")
    print(f"[INFO] Query: '{query_text}'")
    
    # Generate embedding for query using Ollama
    query_response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query_text
    )
    query_embedding = query_response['embedding']
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"[SUCCESS] Retrieved {len(results['ids'][0])} relevant chunks:")
    
    for i, (doc_id, doc, distance) in enumerate(
        zip(results['ids'][0], results['documents'][0], results['distances'][0]), 1
    ):
        print(f"\n  Result {i}:")
        print(f"    ID: {doc_id}")
        print(f"    Distance: {distance:.4f}")
        print(f"    Text preview: {doc[:200]}...")


def main():
    """
    Main function to orchestrate the embedding pipeline.
    """
    print("\n" + "="*70)
    print("           BS RAG - EMBEDDING & STORAGE PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Step 1: Load chunks
        chunks = load_chunks(INPUT_FILE)
        
        # Step 2: Initialize ChromaDB
        client, collection = initialize_chromadb(CHROMA_DB_PATH, COLLECTION_NAME)
        
        # Step 3: Generate embeddings using Ollama
        texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings_with_ollama(texts, EMBEDDING_MODEL)
        
        # Step 4: Store embeddings in ChromaDB
        store_embeddings_in_chromadb(collection, chunks, embeddings)
        
        # Step 5: Verify storage
        verify_storage(collection)
        
        # Step 6: Test query
        test_query(
            collection,
            query_text="When is quiz 1 scheduled?",
            n_results=3
        )
        
        print("\n" + "="*70)
        print("           EMBEDDING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
