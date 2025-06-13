import asyncio
import os
from dotenv import load_dotenv
from nodes.supervisor_node import SupervisorNode
from nodes.llm_node import LLMNode
from nodes.rag_node import RAGNode
from nodes.web_scraper_node import WebScraperNode
from nodes.validator_node import ValidatorNode

# Load environment variables
load_dotenv()

async def main():
    # Initialize nodes
    supervisor = SupervisorNode()
    llm_node = LLMNode()
    rag_node = RAGNode()
    web_scraper_node = WebScraperNode()
    validator_node = ValidatorNode()
    
    # Register nodes with supervisor
    supervisor.register_node(llm_node)
    supervisor.register_node(rag_node)
    supervisor.register_node(web_scraper_node)
    supervisor.register_node(validator_node)
    
    # Example 1: LLM Query
    print("\nExample 1: LLM Query")
    llm_result = await supervisor.execute({
        "task_type": "llm",
        "query": "What is the capital of France?",
        "timestamp": "2024-03-20T10:00:00Z"
    })
    print(f"LLM Result: {llm_result.result}")
    
    # Example 2: RAG Query
    print("\nExample 2: RAG Query")
    # Add some sample documents to RAG
    rag_node.add_documents([
        "Paris is the capital of France. It is known as the City of Light.",
        "France is a country in Western Europe. Its capital is Paris.",
        "The Eiffel Tower is located in Paris, France."
    ])
    rag_result = await supervisor.execute({
        "task_type": "rag",
        "query": "What is the capital of France and what is it known for?",
        "timestamp": "2024-03-20T10:01:00Z"
    })
    print(f"RAG Result: {rag_result.result}")
    
    # Example 3: Web Scraping
    print("\nExample 3: Web Scraping")
    web_result = await supervisor.execute({
        "task_type": "web_search",
        "urls": ["https://en.wikipedia.org/wiki/Paris"],
        "timestamp": "2024-03-20T10:02:00Z"
    })
    print(f"Web Scraping Result: {web_result.result[0]['status']}")
    
    # Print execution history
    print("\nExecution History:")
    for entry in supervisor.get_execution_history():
        print(f"Node: {entry['node_id']}")
        print(f"Status: {entry['state']['status']}")
        print(f"Timestamp: {entry['timestamp']}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main()) 