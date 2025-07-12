# sap_rag_diagnostics.py
import os
import sys
import traceback

def test_all_imports():
    """Test all critical imports for SAP RAG system"""
    
    print("=" * 60)
    print("SAP RAG SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Environment Check
    print("\n🔧 ENVIRONMENT CHECK:")
    print(f"Python version: {sys.version}")
    print(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
    if 'OPENAI_API_KEY' in os.environ:
        key_length = len(os.environ['OPENAI_API_KEY'])
        print(f"OPENAI_API_KEY length: {key_length} chars")
        print(f"OPENAI_API_KEY starts with: {os.environ['OPENAI_API_KEY'][:7]}...")
    
    # 2. Core Package Imports
    print("\n📦 CORE PACKAGE IMPORTS:")
    
    packages_to_test = [
        ("streamlit", "st"),
        ("pandas", "pd"), 
        ("numpy", "np"),
        ("PyPDF2", "PyPDF2"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("openai", "openai"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
        ("faiss", "faiss"),
        ("sklearn", "sklearn")
    ]
    
    for package_name, import_name in packages_to_test:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
    
    # 3. LangChain/LangGraph Specific Imports
    print("\n🔗 LANGCHAIN/LANGGRAPH IMPORTS:")
    
    langchain_imports = [
        ("langgraph.graph", "StateGraph"),
        ("langgraph.graph", "END"), 
        ("langchain_openai", "OpenAIEmbeddings"),
        ("langchain_openai", "ChatOpenAI"),
        ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),
        ("langchain.schema", "Document"),
        ("langchain.callbacks", "get_openai_callback")
    ]
    
    for module_name, class_name in langchain_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
        except AttributeError as e:
            print(f"  ⚠️  {module_name}.{class_name}: Module imported but class not found")
    
    # 4. Test OpenAI Connection
    print("\n🤖 OPENAI CONNECTION TEST:")
    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        
        # Test embeddings
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            test_embedding = embeddings.embed_query("test")
            print(f"  ✅ OpenAI Embeddings: Working (returned {len(test_embedding)} dimensions)")
        except Exception as e:
            print(f"  ❌ OpenAI Embeddings: {e}")
        
        # Test chat
        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
            response = llm.predict("Say 'OpenAI connection test successful'")
            print(f"  ✅ OpenAI Chat: Working (response: {response[:50]}...)")
        except Exception as e:
            print(f"  ❌ OpenAI Chat: {e}")
            
    except ImportError as e:
        print(f"  ❌ Cannot test OpenAI - import failed: {e}")
    
    # 5. Config Import Test
    print("\n⚙️ CONFIG IMPORT TEST:")
    try:
        from configorig import Config
        print(f"  ✅ Config imported successfully")
        
        config_attrs = [attr for attr in dir(Config) if not attr.startswith('_')]
        print(f"  📋 Config attributes: {config_attrs}")
        
        # Test specific config values
        important_configs = [
            'OPENAI_API_KEY', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 
            'EMBEDDING_MODEL', 'LLM_MODEL', 'CHROMA_PATH'
        ]
        
        for attr in important_configs:
            if hasattr(Config, attr):
                value = getattr(Config, attr)
                if 'API_KEY' in attr or 'PASSWORD' in attr:
                    print(f"  🔑 {attr}: {'*' * min(len(str(value)), 8)} (hidden)")
                else:
                    print(f"  📝 {attr}: {value}")
            else:
                print(f"  ⚠️  {attr}: NOT FOUND")
                
    except ImportError as e:
        print(f"  ❌ Config import failed: {e}")
    
    # 6. Test Document Processing
    print("\n📄 DOCUMENT PROCESSING TEST:")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        
        # Test text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        test_text = "This is a test document. " * 100
        chunks = splitter.split_text(test_text)
        print(f"  ✅ Text Splitter: Created {len(chunks)} chunks")
        
        # Test document creation
        docs = [Document(page_content=chunk, metadata={"test": True}) for chunk in chunks]
        print(f"  ✅ Document Creation: Created {len(docs)} documents")
        
    except Exception as e:
        print(f"  ❌ Document processing test failed: {e}")
    
    # 7. Vector Store Test
    print("\n💾 VECTOR STORE TEST:")
    
    # Test ChromaDB
    try:
        import chromadb
        client = chromadb.Client()
        print(f"  ✅ ChromaDB: Available")
    except Exception as e:
        print(f"  ❌ ChromaDB: {e}")
    
    # Test FAISS
    try:
        import faiss
        index = faiss.IndexFlatL2(1536)
        print(f"  ✅ FAISS: Available")
    except Exception as e:
        print(f"  ❌ FAISS: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

def test_workflow_creation():
    """Test workflow creation specifically"""
    print("\n🔄 WORKFLOW CREATION TEST:")
    
    try:
        from workfloworig import SAPRAGWorkflow
        
        config = {
            'vector_store_type': 'chroma',
            'embedding_type': 'openai',
            'chunk_size': 1000,
            'top_k': 10,
            'temperature': 0.1
        }
        
        workflow = SAPRAGWorkflow(config)
        print("  ✅ SAPRAGWorkflow created successfully")
        
        # Check if agents were initialized properly
        if hasattr(workflow, 'pdf_processor'):
            print("  ✅ PDF Processor initialized")
        if hasattr(workflow, 'embedding_agent'):
            print("  ✅ Embedding Agent initialized")
        if hasattr(workflow, 'summary_agent'):
            print("  ✅ Summary Agent initialized")
            
        # Test if LLM is dummy or real
        try:
            llm = workflow.summary_agent._get_llm()
            test_response = llm.predict("test")
            if "test summary generated by the dummy LLM" in test_response:
                print("  ⚠️  WARNING: Using DUMMY LLM - OpenAI connection failed!")
            else:
                print("  ✅ Real LLM working")
        except Exception as e:
            print(f"  ❌ LLM test failed: {e}")
            
        # Test embeddings
        try:
            embeddings = workflow.embedding_agent._get_embeddings()
            if embeddings.__class__.__name__ == "DummyEmbeddings":
                print("  ⚠️  WARNING: Using DUMMY EMBEDDINGS - OpenAI connection failed!")
            else:
                print("  ✅ Real embeddings working")
        except Exception as e:
            print(f"  ❌ Embeddings test failed: {e}")
            
    except Exception as e:
        print(f"  ❌ Workflow creation failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_all_imports()
    test_workflow_creation()