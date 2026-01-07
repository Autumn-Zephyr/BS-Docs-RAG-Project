import os
import sys
import ollama

# Add parent directory to path to import retriever module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the retrieval function from 5._retriever.py
# Using the module name with underscore prefix
retriever = __import__("5._retriever")

# ---------------- CONFIG ----------------
GENERATION_MODEL = "llama3"  # Ollama generation model
TOP_K = 5
# --------------------------------------


def generate_answer(query, retrieved_chunks):
    """
    Generate an answer using Llama 3 based on retrieved context chunks.
    
    Args:
        query (str): The user's question
        retrieved_chunks (list): List of retrieved chunk dictionaries
    
    Returns:
        str: Generated answer from Llama 3
    """
    # Combine retrieved chunks as context
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    # Create prompt for Llama 3
    prompt = f"""Based on the following context, answer the user's question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate response using Ollama Llama 3
    print("\n🤖 Generating answer using Llama 3...\n")
    response = ollama.generate(
        model=GENERATION_MODEL,
        prompt=prompt
    )
    
    return response["response"]


def rag_query_pipeline(query, top_k=TOP_K, show_chunks=False):
    """
    Complete RAG pipeline: Retrieve relevant chunks and generate answer.
    
    Args:
        query (str): The user's question
        top_k (int): Number of chunks to retrieve (default: 5)
        show_chunks (bool): Whether to display retrieved chunks (default: False)
    
    Returns:
        dict: Dictionary containing retrieved chunks and generated answer
    """
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}\n")
    
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve_relevant_chunks(query, top_k=top_k)
    
    # Step 2: Display retrieved chunks if requested
    if show_chunks:
        print("\n" + "="*80)
        print("📄 RETRIEVED CHUNKS (Ranked by Relevance)")
        print("="*80 + "\n")
        
        for chunk in retrieved_chunks:
            print(f"🔹 Rank: {chunk['rank']}")
            print(f"📊 Similarity Score: {chunk['score']:.4f}")
            print(f"📂 Metadata: {chunk['metadata']}")
            print(f"📝 Text:\n{chunk['text']}\n")
            print("-" * 80 + "\n")
    
    # Step 3: Generate answer using Llama 3
    answer = generate_answer(query, retrieved_chunks)
    
    # Display answer
    print("="*80)
    print("💡 GENERATED ANSWER")
    print("="*80)
    print(answer)
    print("\n" + "="*80)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer
    }


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    # Get user query
    user_query = input("\n🔍 Enter your query: ").strip()
    
    if not user_query:
        print("❌ Empty query")
        exit()
    
    # Ask if user wants to see retrieved chunks
    show_chunks_input = input("📋 Show retrieved chunks? (y/n): ").strip().lower()
    show_chunks = show_chunks_input == 'y'
    
    # Run the RAG pipeline
    result = rag_query_pipeline(
        query=user_query,
        top_k=5,
        show_chunks=show_chunks
    )
    
    print("\n✅ Query completed successfully!")
