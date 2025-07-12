# Test the new agent structure
try:
    from agents import (
        BaseAgent, PDFProcessorAgent, EmbeddingAgent, 
        SearchAgent, SummaryAgent, EmailAgent, SystemOutputAgent
    )
    print("✅ All agents imported successfully!")
    print(f"Available agents: {len([BaseAgent, PDFProcessorAgent, EmbeddingAgent, SearchAgent, SummaryAgent, EmailAgent, SystemOutputAgent])}")
    
    # Test agent initialization
    config = {"test": True}
    pdf_agent = PDFProcessorAgent(config)
    print(f"✅ PDFProcessorAgent created: {pdf_agent.name}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")