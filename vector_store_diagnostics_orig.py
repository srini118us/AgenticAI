# vector_store_diagnostic.py
import logging
import traceback

def diagnose_vector_store_issue(workflow_instance, result_state):
    """
    Comprehensive diagnostic for vector store issues
    """
    print("🔍 VECTOR STORE DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Check result state
    print(f"📊 Workflow Status: {result_state.get('workflow_status', 'Unknown')}")
    print(f"📊 Vector Store Ready: {result_state.get('vector_store_ready', False)}")
    print(f"📊 Current Agent: {result_state.get('current_agent', 'Unknown')}")
    print(f"📊 Error Message: {result_state.get('error_message', 'None')}")
    
    # Check processing times
    processing_times = result_state.get('processing_times', {})
    print(f"\n⏱️ Processing Times:")
    for step, time_taken in processing_times.items():
        print(f"  - {step}: {time_taken:.2f}s")
    
    # Check agent messages
    messages = result_state.get('agent_messages', [])
    print(f"\n📝 Agent Messages ({len(messages)} total):")
    for msg in messages[-10:]:  # Show last 10 messages
        status_emoji = {"processing": "🔄", "completed": "✅", "error": "❌", "skipped": "⏭️"}.get(msg.get('status', ''), "ℹ️")
        print(f"  {status_emoji} [{msg.get('agent_name', 'Unknown')}] {msg.get('message', '')}")
    
    # Check workflow instance
    print(f"\n🔧 Workflow Instance Check:")
    print(f"  - Has vector_store_manager: {hasattr(workflow_instance, 'vector_store_manager')}")
    print(f"  - Has search_agent: {hasattr(workflow_instance, 'search_agent')}")
    
    if hasattr(workflow_instance, 'vector_store_manager'):
        vsm = workflow_instance.vector_store_manager
        print(f"  - VectorStoreManager type: {type(vsm).__name__}")
        print(f"  - VectorStoreManager store_type: {getattr(vsm, 'store_type', 'Unknown')}")
    
    if hasattr(workflow_instance, 'search_agent') and workflow_instance.search_agent:
        sa = workflow_instance.search_agent
        print(f"  - SearchAgent type: {type(sa).__name__}")
        print(f"  - SearchAgent has vector_store: {hasattr(sa, 'vector_store')}")
        if hasattr(sa, 'vector_store') and sa.vector_store:
            print(f"  - SearchAgent vector_store type: {type(sa.vector_store).__name__}")
        else:
            print(f"  - SearchAgent vector_store: {getattr(sa, 'vector_store', 'Missing')}")
    
    # Check documents and embeddings
    docs = result_state.get('processed_documents', [])
    embeds = result_state.get('embeddings', [])
    print(f"\n📄 Data Check:")
    print(f"  - Documents count: {len(docs)}")
    print(f"  - Embeddings count: {len(embeds)}")
    
    if docs:
        first_doc = docs[0]
        print(f"  - First document type: {type(first_doc).__name__}")
        if hasattr(first_doc, 'page_content'):
            content_length = len(first_doc.page_content) if first_doc.page_content else 0
            print(f"  - First document content length: {content_length}")
        elif isinstance(first_doc, dict):
            print(f"  - First document keys: {list(first_doc.keys())}")
    
    # Try to use the vector store status checker if available
    if hasattr(workflow_instance, 'check_vector_store_status'):
        try:
            status = workflow_instance.check_vector_store_status()
            print(f"\n🎯 Vector Store Status:")
            for key, value in status.items():
                print(f"  - {key}: {value}")
        except Exception as e:
            print(f"\n❌ Error checking vector store status: {e}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not result_state.get('vector_store_ready', False):
        print("  1. Vector store is not ready - check the vector_storage_node for errors")
        
        if result_state.get('error_message'):
            print(f"  2. Error found: {result_state.get('error_message')}")
        
        if not hasattr(workflow_instance, 'search_agent') or not workflow_instance.search_agent:
            print("  3. SearchAgent is missing - check SearchAgent initialization")
        
        if len(docs) == 0:
            print("  4. No documents found - check PDF processing")
        
        if len(embeds) == 0:
            print("  5. No embeddings found - check embedding creation")
    
    print("=" * 50)

# Quick fix function
def quick_fix_vector_store(workflow_instance):
    """
    Attempt to quickly fix vector store issues
    """
    print("🔧 ATTEMPTING QUICK FIX...")
    
    try:
        # Check if we have the necessary components
        if not hasattr(workflow_instance, 'vector_store_manager'):
            print("❌ Missing vector_store_manager - cannot fix")
            return False
        
        # Try to get processed documents from somewhere
        if hasattr(workflow_instance, '_last_documents'):
            documents = workflow_instance._last_documents
        else:
            print("❌ No documents available for vector store creation")
            return False
        
        # Try to create vector store manually
        print("🔄 Creating vector store manually...")
        vector_store = workflow_instance.vector_store_manager.create_vector_store(documents, [])
        
        if vector_store is None:
            print("❌ Vector store creation returned None")
            return False
        
        # Try to create search agent
        print("🔄 Creating search agent...")
        from agentsold import SearchAgent
        workflow_instance.search_agent = SearchAgent({'vector_store': vector_store})
        
        if hasattr(workflow_instance.search_agent, 'vector_store') and workflow_instance.search_agent.vector_store:
            print("✅ Quick fix successful!")
            return True
        else:
            print("❌ SearchAgent still missing vector_store")
            return False
            
    except Exception as e:
        print(f"❌ Quick fix failed: {e}")
        traceback.print_exc()
        return False

# Usage example:
if __name__ == "__main__":
    # Example usage in your Streamlit app:
    """
    # After getting the result from workflow
    result = workflow.run_workflow(...)
    
    # If vector store is not ready, diagnose
    if not result.get('vector_store_ready', False):
        diagnose_vector_store_issue(workflow, result)
        
        # Try quick fix
        if quick_fix_vector_store(workflow):
            print("✅ Quick fix worked! Try searching now.")
        else:
            print("❌ Quick fix failed. Check the diagnostic output above.")
    """
    pass