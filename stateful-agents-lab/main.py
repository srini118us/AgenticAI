from llm import get_llm_response
from rag import get_rag_response, PDFRAG
from web_scraper import search_web
from validator import validate_output
import os

def simple_agent_workflow(task_type: str, query: str = None, urls: list = None, documents: list = None, pdf_path: str = None) -> str:
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
        if not query or (not documents and not pdf_path):
            return "Error: Query and documents/pdf_path are required for RAG task."
        print(f"Calling RAG with query: '{query}' and {'PDF at ' + pdf_path if pdf_path else str(len(documents)) + ' documents'}.")
        
        if pdf_path:
            # Directly use PDFRAG for PDF loading and querying within simple_agent_workflow
            try:
                rag_instance = PDFRAG()
                rag_instance.load_pdf(pdf_path)
                result = rag_instance.get_response(query)
            except Exception as e:
                result = f"Error in PDF RAG operation: {str(e)}"
        else:
            result = get_rag_response(query=query, documents=documents)
        
        print(f"RAG Raw Result: {result[:100]}...")
        is_valid = validate_output(result, "not_empty")
        print(f"RAG Result Valid: {is_valid}")

    elif task_type == "web_scrape":
        if not query:
            return "Error: Query is required for Web Scraper task."
        print(f"Calling Web Searcher for query: {query}")
        result = search_web(query)
        print(f"Web Searcher Raw Result: {result[:100]}...")
        is_valid = validate_output(result, "not_empty")
        print(f"Web Searcher Result Valid: {is_valid}")

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

    # Example 2: RAG Query (using PDF)
    pdf_path_rag_example = "attention.pdf"
    print(simple_agent_workflow(task_type="rag_query", query="What are the key concepts of attention mechanisms?", pdf_path=pdf_path_rag_example))

    # Example 3: Web Search
    print(simple_agent_workflow(task_type="web_scrape", query="What is latest updates in AI?"))
