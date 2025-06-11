from llm import get_llm_response

def get_rag_response(query: str, documents: list) -> str:
    """Performs a simple RAG operation using provided documents and an LLM."""
    try:
        # Simple similarity search (case-insensitive)
        relevant_docs = [doc for doc in documents if query.lower() in doc.lower()]
        
        # Combine relevant documents
        context = "\n".join(relevant_docs)
        
        if not context:
            return "No relevant context found for the query."

        # Get LLM response with context
        prompt = f"Context: {context}\n\nQuestion: {query}"
        return get_llm_response(prompt)
    except Exception as e:
        return f"Error in RAG operation: {str(e)}"

if __name__ == "__main__":
    print("--- RAG Simple Example ---")
    documents = [
        "Python is a versatile programming language.",
        "It is widely used for web development, data analysis, artificial intelligence, and scientific computing.",
        "Python's simplicity and extensive libraries make it popular for beginners and experts alike."
    ]
    rag_query = "What are the common uses of Python?"
    rag_response = get_rag_response(rag_query, documents)
    print("RAG Query:", rag_query)
    print("RAG Response:", rag_response) 