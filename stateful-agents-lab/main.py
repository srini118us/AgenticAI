from llm import get_llm_response
from rag import get_rag_response
from web_scraper import scrape_website
from validator import validate_output

def simple_agent_workflow(task_type: str, query: str = None, urls: list = None, documents: list = None) -> str:
    """Demonstrates a simple agent workflow based on task type."""
    result = ""
    is_valid = False

    print(f"\n--- Starting Simple Workflow for Task: {task_type} ---")

    if task_type == "llm_query":
        if not query:
            return "Error: Query is required for LLM task."
        print(f"Calling LLM with query: '{query}'")
        result = get_llm_response(query)
        print(f"LLM Raw Result: {result[:100]}...")
        is_valid = validate_output(result, "not_empty")
        print(f"LLM Result Valid: {is_valid}")

    elif task_type == "rag_query":
        if not query or not documents:
            return "Error: Query and documents are required for RAG task."
        print(f"Calling RAG with query: '{query}' and {len(documents)} documents.")
        result = get_rag_response(query, documents)
        print(f"RAG Raw Result: {result[:100]}...")
        is_valid = validate_output(result, "not_empty")
        print(f"RAG Result Valid: {is_valid}")

    elif task_type == "web_scrape":
        if not urls:
            return "Error: URLs are required for Web Scraper task."
        print(f"Calling Web Scraper for URL: {urls[0]}")
        # For simplicity, just scraping the first URL
        result = scrape_website(urls[0])
        print(f"Web Scraper Raw Result: {result[:100]}...")
        is_valid = validate_output(result, "not_empty")
        print(f"Web Scraper Result Valid: {is_valid}")

    else:
        return f"Error: Unknown task type '{task_type}'"

    if is_valid:
        final_output = f"Workflow Completed Successfully!\nFinal Validated Output: {result}"
    else:
        final_output = f"Workflow Failed Validation!\nRaw Output: {result}"

    print(f"--- Workflow for Task: {task_type} Finished ---\n")
    return final_output

if __name__ == "__main__":
    # Example 1: LLM Query
    print(simple_agent_workflow(task_type="llm_query", query="Explain quantum computing in simple terms."))

    # Example 2: RAG Query
    sample_documents = [
        "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement.",
        "It can solve problems that are intractable for classical computers.",
        "Early quantum computers are expected to have fewer than 100 qubits."
    ]
    print(simple_agent_workflow(task_type="rag_query", query="What are the key concepts of quantum computing?", documents=sample_documents))

    # Example 3: Web Scrape
    # Be mindful of robots.txt and website terms of service when using a real URL.
    print(simple_agent_workflow(task_type="web_scrape", urls=["http://example.com"]))
