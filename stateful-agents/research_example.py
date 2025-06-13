import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from nodes.supervisor_node import SupervisorNode
from nodes.llm_node import LLMNode
from nodes.rag_node import RAGNode
from nodes.web_scraper_node import WebScraperNode
from nodes.validator_node import ValidatorNode

# Load environment variables
load_dotenv()

async def research_quantum_computing():
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
    
    print("\n=== Quantum Computing Research Assistant ===")
    print("Starting research process...\n")
    
    # Step 1: Web Scraping
    print("Step 1: Gathering recent articles...")
    web_result = await supervisor.execute({
        "task_type": "web_search",
        "urls": [
            "https://en.wikipedia.org/wiki/Quantum_computing",
            "https://www.ibm.com/quantum-computing/learn/what-is-quantum-computing/",
            "https://www.nature.com/subjects/quantum-computing"
        ],
        "timestamp": datetime.utcnow().isoformat()
    })
    
    if web_result.status == "failed":
        print(f"Web scraping failed: {web_result.error}")
        return
    
    # Extract successful scrapes
    successful_scrapes = [r for r in web_result.result if r["status"] == "success"]
    print(f"Successfully gathered {len(successful_scrapes)} articles")
    
    # Step 2: Add scraped content to RAG
    print("\nStep 2: Processing articles with RAG...")
    for scrape in successful_scrapes:
        rag_node.add_documents([scrape["content"]])
    
    # Query RAG for key information
    rag_result = await supervisor.execute({
        "task_type": "rag",
        "query": """What are the key recent developments in quantum computing? 
        Include information about:
        1. Major breakthroughs
        2. Current challenges
        3. Potential applications
        4. Leading companies and research institutions""",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    if rag_result.status == "failed":
        print(f"RAG processing failed: {rag_result.error}")
        return
    
    print("RAG analysis completed")
    
    # Step 3: Generate comprehensive summary using LLM
    print("\nStep 3: Generating comprehensive summary...")
    llm_result = await supervisor.execute({
        "task_type": "llm",
        "query": f"""Based on the following research findings about quantum computing, 
        create a comprehensive summary that:
        1. Synthesizes the key information
        2. Highlights the most significant developments
        3. Provides a clear explanation of current state and future prospects
        4. Is accessible to a general audience
        
        Research findings:
        {rag_result.result}""",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    if llm_result.status == "failed":
        print(f"LLM processing failed: {llm_result.error}")
        return
    
    # Print final result
    print("\n=== Final Research Summary ===")
    print(llm_result.result)
    
    # Print execution history
    print("\n=== Execution History ===")
    for entry in supervisor.get_execution_history():
        print(f"\nNode: {entry['node_id']}")
        print(f"Status: {entry['state']['status']}")
        print(f"Timestamp: {entry['timestamp']}")
        if entry['state'].get('error'):
            print(f"Error: {entry['state']['error']}")

if __name__ == "__main__":
    asyncio.run(research_quantum_computing()) 