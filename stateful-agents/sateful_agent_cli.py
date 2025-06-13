"""
Stateful Agent System - Terminal Tutorial
Run this script in IPython or Jupyter console for an interactive tutorial.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from nodes.supervisor_node import SupervisorNode
from nodes.llm_node import LLMNode
from nodes.rag_node import RAGNode
from nodes.web_scraper_node import WebScraperNode
from nodes.validator_node import ValidatorNode
from nodes.base_node import BaseNode, NodeState

def print_section(title):
    """Print a section header"""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def print_step(step_num, title):
    """Print a step header"""
    print(f"\nStep {step_num}: {title}")
    print("-" * 40)

async def tutorial_base_node():
    """Tutorial for understanding the Base Node"""
    print_section("Understanding the Base Node")
    
    print("The Base Node provides:")
    print("1. State management")
    print("2. Error handling")
    print("3. Common interface for all nodes")
    
    print_step(1, "Creating a Test Node")
    
    class TestNode(BaseNode):
        async def execute(self, input_data):
            try:
                self.update_state(status="running")
                result = f"Processed input: {input_data.get('test_input', 'no input')}"
                self.update_state(status="completed", result=result)
                return self.state
            except Exception as e:
                self.update_state(status="failed", error=str(e))
                return self.state
    
    print_step(2, "Testing Successful Execution")
    node = TestNode(node_id="test_node")
    result = await node.execute({"test_input": "Hello, World!"})
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")
    print(f"State: {result.dict()}")
    
    print_step(3, "Testing Error Handling")
    result = await node.execute({})
    print(f"Status: {result.status}")
    print(f"Error: {result.error}")

async def tutorial_llm_node():
    """Tutorial for the LLM Node"""
    print_section("Working with the LLM Node")
    
    print("The LLM Node handles interactions with language models.")
    
    print_step(1, "Creating and Testing LLM Node")
    llm_node = LLMNode()
    
    print("Testing with a simple query...")
    result = await llm_node.execute({
        "query": "Explain quantum computing in one sentence."
    })
    
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")
    print(f"Metadata: {result.metadata}")

async def tutorial_rag_node():
    """Tutorial for the RAG Node"""
    print_section("Working with the RAG Node")
    
    print("The RAG Node combines document retrieval with language model generation.")
    
    print_step(1, "Creating RAG Node")
    rag_node = RAGNode()
    
    print_step(2, "Adding Sample Documents")
    sample_docs = [
        "Python is a high-level programming language known for its simplicity.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python is widely used in data science, web development, and AI."
    ]
    rag_node.add_documents(sample_docs)
    print("Documents added successfully!")
    
    print_step(3, "Testing RAG with a Query")
    result = await rag_node.execute({
        "query": "What is Python and when was it created?"
    })
    
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")
    print(f"Metadata: {result.metadata}")

async def tutorial_web_scraper():
    """Tutorial for the Web Scraper Node"""
    print_section("Working with the Web Scraper Node")
    
    print("The Web Scraper Node fetches and processes content from web pages.")
    
    print_step(1, "Creating Web Scraper Node")
    scraper = WebScraperNode()
    
    print_step(2, "Testing with a Simple URL")
    result = await scraper.execute({
        "urls": ["https://en.wikipedia.org/wiki/Python_(programming_language)"]
    })
    
    print(f"Status: {result.status}")
    if result.status == "completed":
        print(f"Number of successful scrapes: {result.metadata['successful_scrapes']}")
        if result.result and len(result.result) > 0:
            print(f"\nFirst 200 characters of scraped content:")
            print(f"{result.result[0]['content'][:200]}...")
    else:
        print(f"Error: {result.error}")

async def tutorial_validator():
    """Tutorial for the Validator Node"""
    print_section("Working with the Validator Node")
    
    print("The Validator Node checks the quality and relevance of outputs.")
    
    print_step(1, "Creating Validator Node")
    validator = ValidatorNode()
    
    print_step(2, "Testing with Sample Output")
    result = await validator.execute({
        "task_type": "llm",
        "input": "Python is a programming language. It was created in 1991."
    })
    
    print(f"Status: {result.status}")
    print(f"Validation Result: {result.result}")
    print(f"Metadata: {result.metadata}")

async def tutorial_supervisor():
    """Tutorial for the Supervisor Node"""
    print_section("Understanding the Supervisor")
    
    print("The Supervisor Node coordinates between all other nodes.")
    
    print_step(1, "Initializing All Nodes")
    supervisor = SupervisorNode()
    llm_node = LLMNode()
    rag_node = RAGNode()
    web_scraper_node = WebScraperNode()
    validator_node = ValidatorNode()
    
    print_step(2, "Registering Nodes with Supervisor")
    supervisor.register_node(llm_node)
    supervisor.register_node(rag_node)
    supervisor.register_node(web_scraper_node)
    supervisor.register_node(validator_node)
    print("Nodes registered successfully!")
    
    print_step(3, "Testing with a Simple Task")
    result = await supervisor.execute({
        "task_type": "llm",
        "query": "What is artificial intelligence?",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    print(f"\nFinal Status: {result.status}")
    print(f"Final Result: {result.result}")
    
    print_step(4, "Execution History")
    for entry in supervisor.get_execution_history():
        print(f"\nNode: {entry['node_id']}")
        print(f"Status: {entry['state']['status']}")
        print(f"Timestamp: {entry['timestamp']}")

async def run_complete_example():
    """Run a complete example combining all nodes"""
    print_section("Complete Example: Research Assistant")
    
    print("This example combines all nodes to create a simple research assistant.")
    print("Note: This example will take a few minutes to complete.")
    
    # Initialize all components
    supervisor = SupervisorNode()
    llm_node = LLMNode()
    rag_node = RAGNode()
    web_scraper_node = WebScraperNode()
    validator_node = ValidatorNode()
    
    # Register nodes
    supervisor.register_node(llm_node)
    supervisor.register_node(rag_node)
    supervisor.register_node(web_scraper_node)
    supervisor.register_node(validator_node)
    
    topic = "Artificial Intelligence"
    print(f"\nResearching: {topic}")
    
    try:
        print_step(1, "Web Scraping")
        print("Scraping Wikipedia page (this may take a minute)...")
        web_result = await asyncio.wait_for(
            supervisor.execute({
                "task_type": "web_search",
                "urls": [f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"],
                "timestamp": datetime.utcnow().isoformat()
            }),
            timeout=60  # 60 second timeout
        )
        
        if web_result.status == "failed":
            print(f"Web scraping failed: {web_result.error}")
            return
        
        print(f"Successfully scraped {len(web_result.result)} pages")
        
        print_step(2, "RAG Processing")
        print("Processing content (this may take a minute)...")
        successful_scrapes = [r for r in web_result.result if r["status"] == "success"]
        for i, scrape in enumerate(successful_scrapes, 1):
            print(f"Processing document {i}/{len(successful_scrapes)}...")
            # Limit content size to first 5000 characters
            content = scrape["content"][:5000]
            rag_node.add_documents([content])
        
        rag_result = await asyncio.wait_for(
            supervisor.execute({
                "task_type": "rag",
                "query": f"What are the key points about {topic}? Include main concepts and important details.",
                "timestamp": datetime.utcnow().isoformat()
            }),
            timeout=120  # 2 minute timeout
        )
        
        if rag_result.status == "failed":
            print(f"RAG processing failed: {rag_result.error}")
            return
        
        print_step(3, "Generating Summary")
        print("Generating summary with LLM (this may take a minute)...")
        llm_result = await asyncio.wait_for(
            supervisor.execute({
                "task_type": "llm",
                "query": f"Create a clear and concise summary about {topic} based on this information:\n{rag_result.result}",
                "timestamp": datetime.utcnow().isoformat()
            }),
            timeout=120  # 2 minute timeout
        )
        
        if llm_result.status == "failed":
            print(f"LLM processing failed: {llm_result.error}")
            return
        
        print("\n=== Final Summary ===")
        print(llm_result.result)
        
        print("\n=== Execution History ===")
        for entry in supervisor.get_execution_history():
            print(f"\nNode: {entry['node_id']}")
            print(f"Status: {entry['state']['status']}")
            print(f"Timestamp: {entry['timestamp']}")
            
    except asyncio.TimeoutError:
        print("\nError: Operation timed out. The process took too long.")
        print("This might be due to slow internet or API response times.")
        print("Please try again or try a different tutorial.")
    except Exception as e:
        print(f"\nError: An unexpected error occurred: {str(e)}")
        print("Please try again or try a different tutorial.")

async def main():
    """Main tutorial function"""
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables!")
        return
    
    print("Welcome to the Stateful Agent System Tutorial!")
    print("This tutorial will guide you through understanding each component of the system.")
    
    while True:
        print("\nAvailable Tutorials:")
        print("1. Base Node Tutorial")
        print("2. LLM Node Tutorial")
        print("3. RAG Node Tutorial")
        print("4. Web Scraper Tutorial")
        print("5. Validator Tutorial")
        print("6. Supervisor Tutorial")
        print("7. Complete Example")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == "1":
            await tutorial_base_node()
        elif choice == "2":
            await tutorial_llm_node()
        elif choice == "3":
            await tutorial_rag_node()
        elif choice == "4":
            await tutorial_web_scraper()
        elif choice == "5":
            await tutorial_validator()
        elif choice == "6":
            await tutorial_supervisor()
        elif choice == "7":
            await run_complete_example()
        elif choice == "8":
            print("\nThank you for using the tutorial!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Run the tutorial
    asyncio.run(main()) 