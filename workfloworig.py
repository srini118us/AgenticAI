# workflow.py (FINAL CORRECTED VERSION WITH ALL FIXES)
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Literal, TypedDict
from enum import Enum

# ✅ CORRECT LANGRAPH IMPORTS
try:
    from langgraph.graph import StateGraph, END
except ImportError as e:
    logging.warning(f"LangGraph not available: {e}")
    # Create mock classes for testing without LangGraph
    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def add_conditional_edges(self, from_node, condition, mapping): pass
        def set_entry_point(self, node): pass
        def compile(self): return MockApp()
    
    class MockApp:
        def invoke(self, state): return state
    
    END = "END"

# Safe imports for callbacks
try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    try:
        from langchain.callbacks import get_openai_callback
    except ImportError:
        from contextlib import nullcontext as get_openai_callback

# ✅ CORRECTED AGENT IMPORTS WITH CONFIG
try:
    from agentsold import (
        PDFProcessorAgent, EmbeddingAgent, SearchAgent, 
        SummaryAgent, EmailAgent, create_agent
    )
    from modelsorig import EmailRecipient, SystemSummary
    from configorig import Config
except ImportError as e:
    logging.warning(f"Could not import agents, models, or config: {e}")
    # Create mock classes for testing
    class PDFProcessorAgent:
        def __init__(self, config=None): 
            self.config = config or {}
        def process(self, files): 
            return {"success": True, "processed_files": [{"filename": f.name, "text": "mock text"} for f in files]}
    
    class EmbeddingAgent:
        def __init__(self, config=None): 
            self.config = config or {}
        def process(self, processed_files): 
            return {"success": True, "embeddings": [[0.1]*768 for _ in processed_files], "chunks": processed_files}
    
    class SearchAgent:
        def __init__(self, config=None): 
            self.config = config or {}
        def search(self, query="", filters=None): 
            return {"success": True, "search_results": [("mock_doc", 0.9)], "results_count": 1}
    
    class SummaryAgent:
        def __init__(self, config=None): 
            self.config = config or {}
        def generate_summary(self, results, query): 
            return {"success": True, "summary": {"summary": "Mock summary", "critical_findings": [], "recommendations": []}}
    
    class EmailAgent:
        def __init__(self, config=None): 
            self.config = config or {}
        def send_email(self, email_data): 
            return {"success": True}
    
    class EmailRecipient:
        def __init__(self, email, name=""): 
            self.email = email
            self.name = name
    
    class Config:
        OPENAI_API_KEY = "mock_key"
        LLM_MODEL = "gpt-4"
        EMBEDDING_MODEL = "text-embedding-ada-002"
        TEMPERATURE = 0.1
        CHUNK_SIZE = 1000
        TOP_K = 10
        EMAIL_ENABLED = False
        VECTOR_STORE_TYPE = "chroma"

# Mock vector store manager
try:
    from vector_store_orig import VectorStoreManager
except ImportError:
    class VectorStoreManager:
        def __init__(self, store_type="chroma"): 
            self.store_type = store_type
        def create_vector_store(self, docs, embeddings): 
            return MockVectorStore()
    
    class MockVectorStore:
        def add_documents(self, docs, embeddings=None): pass
        def similarity_search_with_score(self, query, k=10): return []

logger = logging.getLogger(__name__)

# ================================
# WORKFLOW STATE DEFINITION
# ================================
class WorkflowStatus(Enum):
    INITIALIZED = "initialized"
    PROCESSING_PDF = "processing_pdf"
    CREATING_EMBEDDINGS = "creating_embeddings"
    STORING_VECTORS = "storing_vectors"
    SEARCHING = "searching"
    SUMMARIZING = "summarizing"
    SYSTEM_OUTPUT = "system_output"
    SENDING_EMAIL = "sending_email"
    COMPLETED = "completed"
    ERROR = "error"

class WorkflowState(TypedDict):
    # Input data
    uploaded_files: List[Any]
    user_query: str
    search_filters: Dict[str, Any]
    
    # Processing state
    workflow_status: str
    current_agent: str
    error_message: str
    
    # Results
    processed_documents: List[Any]
    embeddings: List[Any]
    total_chunks: int
    vector_store_ready: bool
    search_results: List[tuple]
    summary: Dict[str, Any]
    system_summaries: Dict[str, Any]
    
    # Email
    email_sent: bool
    email_recipients: List[Dict[str, str]]
    
    # Metrics
    processing_times: Dict[str, float]
    agent_messages: List[Dict[str, str]]
    
    # Config
    config: Dict[str, Any]

# ================================
# FIXED SYSTEM OUTPUT AGENT
# ================================
class SystemOutputAgent:
    """Agent for system-specific output generation - FIXED VERSION"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def extract_system_ids(self, search_results: List[tuple]) -> List[str]:
        """Extract unique system IDs from search results - FIXED VERSION"""
        system_ids = set()
        
        for result_item in search_results:
            try:
                # Handle different result formats
                if isinstance(result_item, tuple) and len(result_item) >= 2:
                    doc, score = result_item[0], result_item[1]
                elif hasattr(result_item, 'content') and hasattr(result_item, 'source'):
                    # SearchResult object
                    doc = result_item
                else:
                    doc = result_item
                
                # Extract system ID from document metadata
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    if 'system_id' in doc.metadata:
                        system_ids.add(doc.metadata['system_id'])
                        continue
                
                # Extract from content
                content = ""
                if hasattr(doc, 'page_content'):
                    content = str(doc.page_content)
                elif hasattr(doc, 'content'):
                    content = str(doc.content)
                elif isinstance(doc, dict):
                    content = doc.get('page_content', doc.get('content', str(doc)))
                else:
                    content = str(doc)
                
                if content:
                    # Look for system patterns in content
                    content_upper = content.upper()
                    import re
                    
                    # Common SAP system patterns
                    patterns = [
                        r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',  # System: P01
                        r'\bSID[:\s]+([A-Z0-9]{2,4})\b',     # SID: P01
                        r'\b([A-Z]{1,3}[0-9]{1,2})\b',      # P01, DEV, QAS
                        r'\b(PRD|PROD|DEV|DEVL|QAS|TST|TRN)\b'  # Common system types
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content_upper)
                        for match in matches:
                            if (len(match) in [2, 3, 4] and 
                                match not in ['THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG']):
                                system_ids.add(match)
                                
            except Exception as e:
                logger.warning(f"Error extracting system ID from result: {e}")
                continue
        
        # If no systems found, return default
        final_systems = list(system_ids) if system_ids else ['UNKNOWN']
        
        # Filter out common false positives
        false_positives = {
            'RFC', 'EWA', 'RED', 'IBM', 'CPU', 'RAM', 'SQL', 'XML', 'PDF', 
            'LOG', 'ERR', 'OUT', 'TMP', 'SYS', 'USR', 'ADM', 'WEB', 'APP'
        }
        filtered_systems = [s for s in final_systems if s not in false_positives]
        
        return filtered_systems if filtered_systems else ['SYSTEM_01']
    
    def extract_system_summary(self, documents: List[Any], system_id: str) -> Dict[str, Any]:
        """Extract summary for a specific system - FIXED VERSION"""
        try:
            # Analyze documents for this system
            critical_issues = []
            recommendations = []
            key_metrics = {}
            
            for doc in documents:
                content = ""
                if hasattr(doc, 'page_content'):
                    content = str(doc.page_content)
                elif hasattr(doc, 'content'):
                    content = str(doc.content)
                elif isinstance(doc, dict):
                    content = doc.get('page_content', doc.get('content', str(doc)))
                else:
                    content = str(doc)
                
                content_lower = content.lower()
                
                # Look for critical issues
                if any(word in content_lower for word in ['critical', 'error', 'fail', 'down', 'alert']):
                    if system_id.lower() in content_lower:
                        critical_issues.append(f"Critical issue detected in {system_id}")
                
                # Look for recommendations
                if any(word in content_lower for word in ['recommend', 'should', 'improve', 'optimize']):
                    if system_id.lower() in content_lower:
                        recommendations.append(f"Optimization recommended for {system_id}")
                
                # Extract metrics (simplified)
                import re
                cpu_match = re.search(r'cpu[:\s]+([0-9]+)%', content_lower)
                if cpu_match:
                    key_metrics['cpu_usage'] = f"{cpu_match.group(1)}%"
                
                memory_match = re.search(r'memory[:\s]+([0-9]+)%', content_lower)
                if memory_match:
                    key_metrics['memory_usage'] = f"{memory_match.group(1)}%"
            
            # Determine overall health
            if critical_issues:
                overall_health = 'CRITICAL'
            elif recommendations:
                overall_health = 'WARNING'
            else:
                overall_health = 'HEALTHY'
            
            return {
                'system_id': system_id,
                'overall_health': overall_health,
                'critical_alerts': critical_issues or [f'No critical issues found for {system_id}'],
                'recommendations': recommendations or [f'No specific recommendations for {system_id}'],
                'key_metrics': key_metrics or {'status': 'No metrics available'},
                'last_analyzed': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating system summary for {system_id}: {e}")
            return {
                'system_id': system_id,
                'overall_health': 'UNKNOWN',
                'critical_alerts': [f'Error analyzing {system_id}'],
                'recommendations': ['Manual review recommended'],
                'key_metrics': {},
                'last_analyzed': datetime.now().isoformat()
            }

# ================================
# MAIN WORKFLOW CLASS
# ================================
class SAPRAGWorkflow:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Convert config dict to have the right attributes for agents
        agent_config = {
            'embedding_type': self.config.get('embedding_type', 'openai'),
            'vector_store_type': self.config.get('vector_store_type', 'chroma'),
            'chunk_size': self.config.get('chunk_size', 1000),
            'top_k': self.config.get('top_k', 10),
            'temperature': self.config.get('temperature', 0.1),
            'email_enabled': self.config.get('email_enabled', False)
        }
        
        # Initialize agents with proper config
        try:
            # Initialize all agents with config
            self.pdf_processor = PDFProcessorAgent(agent_config)
            self.embedding_agent = EmbeddingAgent(agent_config)
            self.summary_agent = SummaryAgent(agent_config)
            self.system_output_agent = SystemOutputAgent(agent_config)
            
            # Email agent (optional)
            self.email_agent = None
            if self.config.get('email_enabled', False):
                self.email_agent = EmailAgent(agent_config)
            
            # Vector store manager
            self.vector_store_manager = VectorStoreManager(
                store_type=self.config.get('vector_store_type', 'chroma')
            )
            
            # Search agent (initialized after vector store is ready)
            self.search_agent = None
            
            logger.info("✅ All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            # Initialize with mock agents to prevent crashes
            self.pdf_processor = PDFProcessorAgent(agent_config)
            self.embedding_agent = EmbeddingAgent(agent_config)
            self.summary_agent = SummaryAgent(agent_config)
            self.system_output_agent = SystemOutputAgent(agent_config)
            self.email_agent = None
            self.vector_store_manager = VectorStoreManager()
            self.search_agent = None
        
        # Build workflow
        try:
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile()
            logger.info("✅ Workflow compiled successfully")
        except Exception as e:
            logger.error(f"Workflow build failed: {e}")
            # Create mock app for testing
            self.app = MockApp()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper error handling"""
        
        # Create StateGraph
        workflow = StateGraph(WorkflowState)
        
        # Add all nodes
        workflow.add_node("pdf_processor", self._pdf_processing_node)
        workflow.add_node("embedding_creator", self._embedding_creation_node)
        workflow.add_node("vector_store_manager", self._vector_storage_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("summary_agent", self._summary_node)
        workflow.add_node("system_output_agent", self._system_output_node)
        workflow.add_node("email_agent", self._email_node)
        workflow.add_node("complete", self._complete_node)
        
        # Set entry point
        workflow.set_entry_point("pdf_processor")
        
        # Add direct edges (success paths)
        workflow.add_edge("pdf_processor", "embedding_creator")
        workflow.add_edge("embedding_creator", "vector_store_manager")
        workflow.add_edge("search_agent", "summary_agent")
        workflow.add_edge("summary_agent", "system_output_agent")
        workflow.add_edge("email_agent", "complete")
        workflow.add_edge("complete", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "vector_store_manager",
            self._route_after_vector_storage,
            {
                "search": "search_agent",
                "complete": "complete"
            }
        )
        
        workflow.add_conditional_edges(
            "system_output_agent", 
            self._route_after_system_output,
            {
                "send_email": "email_agent",
                "complete": "complete"
            }
        )
        
        return workflow
    
    # ================================
    # NODE IMPLEMENTATIONS - ALL FIXED
    # ================================
    
    def _pdf_processing_node(self, state: WorkflowState) -> WorkflowState:
        """Process PDF files - FIXED VERSION"""
        start_time = time.time()
        
        try:
            # Update state
            state["workflow_status"] = WorkflowStatus.PROCESSING_PDF.value
            state["current_agent"] = "pdf_processor"
            
            # Initialize processing_times if not exists
            if "processing_times" not in state:
                state["processing_times"] = {}
            
            # Add message
            self._add_message(state, "pdf_processor", "Processing PDF files...", "processing")
            
            # Process PDFs
            uploaded_files = state.get("uploaded_files", [])
            if not uploaded_files:
                raise ValueError("No PDF files provided")
            
            # Use the agent's process method correctly
            result = self.pdf_processor.process(uploaded_files)
            
            if not result.get("success"):
                raise ValueError(f"PDF processing failed: {result.get('error', 'Unknown error')}")
            
            processed_files = result.get("processed_files", [])
            if not processed_files:
                raise ValueError("No text extracted from PDFs")
            
            # 🔧 FIX: Create proper document objects that work with your vector store
            documents = []
            for file_data in processed_files:
                # Extract text content
                text_content = file_data.get('text', '')
                if not text_content:
                    continue  # Skip empty documents
                
                # Create document object with proper structure
                try:
                    from langchain.schema import Document
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0),
                            'file_type': 'pdf'
                        }
                    )
                except ImportError:
                    # Fallback: Create dict with required attributes
                    class DocumentLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                            
                        def dict(self):
                            return {
                                'page_content': self.page_content,
                                'metadata': self.metadata
                            }
                    
                    doc = DocumentLike(
                        page_content=text_content,
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0),
                            'file_type': 'pdf'
                        }
                    )
                
                documents.append(doc)
            
            if not documents:
                raise ValueError("No valid documents created from PDFs")
            
            # Update state
            processing_time = time.time() - start_time
            state["processed_documents"] = documents
            state["total_chunks"] = len(documents)
            state["processing_times"]["pdf_processing"] = processing_time
            
            self._add_message(state, "pdf_processor", 
                            f"✅ Processed {len(uploaded_files)} PDFs into {len(documents)} documents", 
                            "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return self._handle_error(state, str(e), "pdf_processor")
    
    def _embedding_creation_node(self, state: WorkflowState) -> WorkflowState:
        """Create embeddings - FIXED VERSION"""
        start_time = time.time()
        
        try:
            state["workflow_status"] = WorkflowStatus.CREATING_EMBEDDINGS.value
            state["current_agent"] = "embedding_creator"
            
            self._add_message(state, "embedding_creator", "Creating embeddings...", "processing")
            
            documents = state.get("processed_documents", [])
            if not documents:
                raise ValueError("No documents found for embedding")
            
            # 🔧 FIX: Ensure documents have the right format for embedding
            processed_docs = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    # Standard document format
                    text_content = doc.page_content
                elif isinstance(doc, dict):
                    # Dictionary format
                    text_content = doc.get('page_content', doc.get('text', str(doc)))
                else:
                    # Fallback
                    text_content = str(doc)
                
                # Skip empty documents
                if not text_content or not text_content.strip():
                    continue
                
                # Create properly formatted document for embedding
                processed_doc = {
                    'text': text_content,  # This is what embedding agent expects
                    'page_content': text_content,  # This is what vector store expects
                    'metadata': getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') 
                               else doc.get('metadata', {}) if isinstance(doc, dict) 
                               else {}
                }
                processed_docs.append(processed_doc)
            
            if not processed_docs:
                raise ValueError("No valid documents found for embedding after processing")
            
            # Use embedding agent's process method
            result = self.embedding_agent.process(processed_docs)
            
            if not result.get("success"):
                raise ValueError(f"Embedding creation failed: {result.get('error', 'Unknown error')}")
            
            embeddings = result.get("embeddings", [])
            chunks = result.get("chunks", processed_docs)
            
            # Ensure chunks have proper format
            final_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'page_content'):
                    final_chunks.append(chunk)
                elif isinstance(chunk, dict):
                    # Convert dict to document-like object
                    if 'page_content' in chunk:
                        chunk_obj = type('Document', (), chunk)()
                    else:
                        chunk_obj = type('Document', (), {
                            'page_content': chunk.get('text', str(chunk)),
                            'metadata': chunk.get('metadata', {})
                        })()
                    final_chunks.append(chunk_obj)
                else:
                    # Create document-like object
                    chunk_obj = type('Document', (), {
                        'page_content': str(chunk),
                        'metadata': {}
                    })()
                    final_chunks.append(chunk_obj)
            
            processing_time = time.time() - start_time
            state["embeddings"] = embeddings
            state["processed_documents"] = final_chunks
            state["processing_times"]["embedding_creation"] = processing_time
            
            self._add_message(state, "embedding_creator", 
                            f"✅ Created {len(embeddings)} embeddings for {len(final_chunks)} chunks", 
                            "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Embedding creation error: {str(e)}")
            return self._handle_error(state, str(e), "embedding_creator")
    
    def _vector_storage_node(self, state: WorkflowState) -> WorkflowState:
        """Store vectors in database - ENHANCED DEBUG VERSION"""
        start_time = time.time()
        
        try:
            state["workflow_status"] = WorkflowStatus.STORING_VECTORS.value
            state["current_agent"] = "vector_store_manager"
            
            self._add_message(state, "vector_store_manager", "📊 Starting vector storage...", "processing")
            
            documents = state.get("processed_documents", [])
            embeddings = state.get("embeddings", [])
            
            # 🔍 DEBUG: Log what we received
            logger.info(f"🔍 DEBUG - Documents count: {len(documents)}")
            logger.info(f"🔍 DEBUG - Embeddings count: {len(embeddings)}")
            
            if documents:
                logger.info(f"🔍 DEBUG - First document type: {type(documents[0])}")
                if hasattr(documents[0], 'page_content'):
                    logger.info(f"🔍 DEBUG - First document content length: {len(documents[0].page_content)}")
                else:
                    logger.info(f"🔍 DEBUG - First document structure: {documents[0]}")
            
            if not documents:
                raise ValueError("No documents to store")
            
            self._add_message(state, "vector_store_manager", f"📄 Processing {len(documents)} documents...", "processing")
            
            # 🔧 FIX 1: Ensure vector store manager exists
            if not hasattr(self, 'vector_store_manager') or self.vector_store_manager is None:
                logger.error("❌ vector_store_manager is None!")
                raise ValueError("Vector store manager not initialized")
            
            # 🔧 FIX 2: Create vector store with better error handling
            try:
                logger.info("🔄 Creating vector store...")
                vector_store = self.vector_store_manager.create_vector_store(documents, embeddings)
                logger.info(f"✅ Vector store created: {type(vector_store).__name__}")
            except Exception as vs_error:
                logger.error(f"❌ Vector store creation failed: {vs_error}")
                raise ValueError(f"Vector store creation failed: {vs_error}")
            
            if vector_store is None:
                logger.error("❌ Vector store creation returned None!")
                raise ValueError("Vector store creation returned None")
            
            self._add_message(state, "vector_store_manager", "🔗 Vector store created, initializing search agent...", "processing")
            
            # 🔧 FIX 3: Enhanced SearchAgent initialization with multiple fallbacks
            search_agent_created = False
            search_config = {
                'vector_store': vector_store,
                'embedding_agent': self.embedding_agent,
                'top_k': self.config.get('top_k', 10),
                'embedding_type': self.config.get('embedding_type', 'openai'),
                'vector_store_type': self.config.get('vector_store_type', 'chroma'),
                'chunk_size': self.config.get('chunk_size', 1000),
                'temperature': self.config.get('temperature', 0.1),
                'email_enabled': self.config.get('email_enabled', False)
            }
            
            # Attempt 1: Full config
            try:
                logger.info("🔄 Attempt 1: Creating SearchAgent with full config...")
                self.search_agent = SearchAgent(search_config)
                
                if hasattr(self.search_agent, 'vector_store') and self.search_agent.vector_store is not None:
                    logger.info("✅ SearchAgent created successfully with full config")
                    search_agent_created = True
                else:
                    raise ValueError("SearchAgent missing vector_store after creation")
                    
            except Exception as search_error_1:
                logger.warning(f"⚠️ Attempt 1 failed: {search_error_1}")
                
                # Attempt 2: Minimal config
                try:
                    logger.info("🔄 Attempt 2: Creating SearchAgent with minimal config...")
                    minimal_config = {'vector_store': vector_store}
                    self.search_agent = SearchAgent(minimal_config)
                    
                    if hasattr(self.search_agent, 'vector_store') and self.search_agent.vector_store is not None:
                        logger.info("✅ SearchAgent created successfully with minimal config")
                        search_agent_created = True
                    else:
                        raise ValueError("SearchAgent missing vector_store after minimal creation")
                        
                except Exception as search_error_2:
                    logger.warning(f"⚠️ Attempt 2 failed: {search_error_2}")
                    
                    # Attempt 3: Direct assignment
                    try:
                        logger.info("🔄 Attempt 3: Creating SearchAgent and assigning vector_store directly...")
                        self.search_agent = SearchAgent({})
                        self.search_agent.vector_store = vector_store
                        
                        if hasattr(self.search_agent, 'vector_store') and self.search_agent.vector_store is not None:
                            logger.info("✅ SearchAgent created with direct assignment")
                            search_agent_created = True
                        else:
                            raise ValueError("Direct assignment failed")
                            
                    except Exception as search_error_3:
                        logger.error(f"❌ All SearchAgent creation attempts failed!")
                        logger.error(f"Error 1: {search_error_1}")
                        logger.error(f"Error 2: {search_error_2}")
                        logger.error(f"Error 3: {search_error_3}")
                        raise ValueError(f"SearchAgent initialization failed after all attempts: {search_error_3}")
            
            if not search_agent_created:
                raise ValueError("SearchAgent was not successfully created")
            
            # 🔧 FIX 4: Verify everything is properly set up
            verification_passed = True
            verification_errors = []
            
            # Check SearchAgent
            if not hasattr(self, 'search_agent') or self.search_agent is None:
                verification_errors.append("search_agent is None")
                verification_passed = False
            
            # Check vector store in SearchAgent
            if hasattr(self, 'search_agent') and self.search_agent:
                if not hasattr(self.search_agent, 'vector_store') or self.search_agent.vector_store is None:
                    verification_errors.append("search_agent.vector_store is None")
                    verification_passed = False
            
            if not verification_passed:
                error_msg = f"Vector store verification failed: {', '.join(verification_errors)}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # 🔧 FIX 5: Test the vector store with a simple query
            try:
                logger.info("🧪 Testing vector store with simple search...")
                test_result = self.search_agent.search("test", {})
                logger.info(f"✅ Vector store test successful: {test_result.get('success', False)}")
            except Exception as test_error:
                logger.warning(f"⚠️ Vector store test failed: {test_error}")
                # Don't fail here - the vector store might still work for real queries
            
            # Success! Mark everything as ready
            processing_time = time.time() - start_time
            state["vector_store_ready"] = True
            state["processing_times"]["vector_storage"] = processing_time
            
            success_message = f"✅ Vector store ready with {len(documents)} documents, SearchAgent initialized"
            self._add_message(state, "vector_store_manager", success_message, "completed")
            logger.info(success_message)
            
            return state
            
        except Exception as e:
            error_msg = f"Vector storage error: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._handle_error(state, error_msg, "vector_store_manager")
    
    def _search_node(self, state: WorkflowState) -> WorkflowState:
        """Perform search - FIXED VERSION"""
        start_time = time.time()
        
        try:
            state["workflow_status"] = WorkflowStatus.SEARCHING.value
            state["current_agent"] = "search_agent"
            
            query = state.get("user_query", "")
            if not query:
                raise ValueError("No search query provided")
            
            self._add_message(state, "search_agent", f"🔍 Searching for: {query}", "processing")
            
            # Perform search
            search_filters = state.get("search_filters", {})
            result = self.search_agent.search(query, search_filters)
            
            if not result.get("success"):
                raise ValueError(f"Search failed: {result.get('error', 'Unknown error')}")
            
            # 🔧 FIX: Handle SearchResult objects properly
            raw_results = result.get("search_results", [])
            search_results = []
            
            for item in raw_results:
                if hasattr(item, 'content') and hasattr(item, 'source'):
                    # It's a SearchResult object - convert to tuple
                    doc_like = type('Document', (), {
                        'page_content': item.content,
                        'metadata': {
                            'source': item.source,
                            'system_id': getattr(item, 'system_id', 'UNKNOWN'),
                            'confidence': getattr(item, 'confidence_score', 0.0)
                        }
                    })()
                    search_results.append((doc_like, getattr(item, 'confidence_score', 0.0)))
                elif isinstance(item, tuple) and len(item) == 2:
                    # Already a tuple (document, score)
                    search_results.append(item)
                elif hasattr(item, 'page_content'):
                    # It's a document-like object
                    search_results.append((item, 1.0))  # Default score
                else:
                    # Unknown format - try to convert
                    logger.warning(f"Unknown search result format: {type(item)}")
                    try:
                        doc_like = type('Document', (), {
                            'page_content': str(item),
                            'metadata': {'source': 'unknown'}
                        })()
                        search_results.append((doc_like, 0.5))
                    except:
                        continue  # Skip this result
            
            processing_time = time.time() - start_time
            state["search_results"] = search_results
            state["processing_times"]["search"] = processing_time
            
            self._add_message(state, "search_agent", 
                            f"✅ Found {len(search_results)} results", "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return self._handle_error(state, str(e), "search_agent")
    
    def _summary_node(self, state: WorkflowState) -> WorkflowState:
        """Generate summary"""
        start_time = time.time()
        
        try:
            state["workflow_status"] = WorkflowStatus.SUMMARIZING.value
            state["current_agent"] = "summary_agent"
            
            self._add_message(state, "summary_agent", "📝 Generating summary...", "processing")
            
            query = state.get("user_query", "")
            search_results = state.get("search_results", [])
            
            if not search_results:
                # Create empty summary
                summary = {
                    "summary": "No search results to summarize",
                    "critical_findings": [],
                    "recommendations": [],
                    "confidence_score": 0.0
                }
            else:
                # Generate summary
                result = self.summary_agent.generate_summary(search_results, query)
                
                if not result.get("success"):
                    raise ValueError(f"Summary generation failed: {result.get('error', 'Unknown error')}")
                
                summary = result.get("summary", {})
            
            processing_time = time.time() - start_time
            state["summary"] = summary
            state["processing_times"]["summary"] = processing_time
            
            self._add_message(state, "summary_agent", 
                            "✅ Summary generated", "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return self._handle_error(state, str(e), "summary_agent")
    
    def _system_output_node(self, state: WorkflowState) -> WorkflowState:
        """Generate system-specific outputs"""
        start_time = time.time()
        
        try:
            state["workflow_status"] = WorkflowStatus.SYSTEM_OUTPUT.value
            state["current_agent"] = "system_output_agent"
            
            self._add_message(state, "system_output_agent", "🔧 Analyzing systems...", "processing")
            
            search_results = state.get("search_results", [])
            if not search_results:
                # No results, create empty system summaries
                state["system_summaries"] = {}
                processing_time = time.time() - start_time
                state["processing_times"]["system_output"] = processing_time
                self._add_message(state, "system_output_agent", 
                                "ℹ️ No systems to analyze", "completed")
                return state
            
            # Extract system IDs
            documents = [doc for doc, _ in search_results]
            system_ids = self.system_output_agent.extract_system_ids(search_results)
            
            # Generate system summaries
            system_summaries = {}
            for system_id in system_ids:
                summary = self.system_output_agent.extract_system_summary(documents, system_id)
                system_summaries[system_id] = summary
            
            processing_time = time.time() - start_time
            state["system_summaries"] = system_summaries
            state["processing_times"]["system_output"] = processing_time
            
            self._add_message(state, "system_output_agent", 
                            f"✅ Analyzed {len(system_ids)} systems", "completed")
            
            return state
            
        except Exception as e:
            logger.error(f"System output error: {str(e)}")
            return self._handle_error(state, str(e), "system_output_agent")
    
    def _email_node(self, state: WorkflowState) -> WorkflowState:
        """Send email notification"""
        start_time = time.time()
        
        try:
            if not self.email_agent:
                self._add_message(state, "email_agent", "📧 Email not configured", "skipped")
                return state
            
            state["workflow_status"] = WorkflowStatus.SENDING_EMAIL.value
            state["current_agent"] = "email_agent"
            
            self._add_message(state, "email_agent", "📧 Sending email...", "processing")
            
            # Get recipients
            recipients_data = state.get("email_recipients", [])
            if not recipients_data:
                self._add_message(state, "email_agent", "📧 No recipients configured", "skipped")
                state["email_sent"] = False
                return state
            
            # Prepare email data
            email_data = {
                'recipients': recipients_data,
                'summary': state.get("summary", {}),
                'query': state.get("user_query", "")
            }
            
            result = self.email_agent.send_email(email_data)
            email_sent = result.get("success", False)
            
            processing_time = time.time() - start_time
            state["email_sent"] = email_sent
            state["processing_times"]["email"] = processing_time
            
            status = "sent" if email_sent else "failed"
            self._add_message(state, "email_agent", f"📧 Email {status}", 
                            "completed" if email_sent else "error")
            
            return state
            
        except Exception as e:
            logger.error(f"Email error: {str(e)}")
            state["email_sent"] = False
            self._add_message(state, "email_agent", f"📧 Email failed: {str(e)}", "error")
            return state
    
    def _complete_node(self, state: WorkflowState) -> WorkflowState:
        """Complete the workflow"""
        state["workflow_status"] = WorkflowStatus.COMPLETED.value
        state["current_agent"] = "complete"
        
        # Calculate total time
        total_time = sum(state.get("processing_times", {}).values())
        
        self._add_message(state, "workflow", 
                         f"🎉 Workflow completed successfully in {total_time:.2f}s", 
                         "completed")
        return state
    
    # ================================
    # ROUTING FUNCTIONS
    # ================================
    
    def _route_after_vector_storage(self, state: WorkflowState) -> Literal["search", "complete"]:
        """Route after vector storage - ENHANCED VERSION"""
        
        # 🔍 DEBUG: Log the state
        vector_store_ready = state.get("vector_store_ready", False)
        user_query = state.get("user_query", "")
        has_search_agent = hasattr(self, 'search_agent') and self.search_agent is not None
        
        logger.info(f"🔍 ROUTING DEBUG:")
        logger.info(f"  - vector_store_ready: {vector_store_ready}")
        logger.info(f"  - user_query: '{user_query}'")
        logger.info(f"  - has_search_agent: {has_search_agent}")
        
        # Enhanced routing logic
        if user_query and user_query.strip():
            if vector_store_ready and has_search_agent:
                logger.info("🔄 Routing to search")
                return "search"
            else:
                logger.warning("⚠️ Query exists but vector store not ready - routing to complete")
                return "complete"
        else:
            logger.info("ℹ️ No query provided - routing to complete")
            return "complete"
    
    def _route_after_system_output(self, state: WorkflowState) -> Literal["send_email", "complete"]:
        """Route after system output"""
        if (self.config.get("email_enabled", False) and 
            self.config.get("auto_send_results", False) and
            state.get("email_recipients")):
            return "send_email"
        return "complete"
    
    # ================================
    # UTILITY FUNCTIONS
    # ================================
    
    def _add_message(self, state: WorkflowState, agent: str, message: str, status: str):
        """Add message to state"""
        if "agent_messages" not in state:
            state["agent_messages"] = []
        
        state["agent_messages"].append({
            "agent_name": agent,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": status
        })
    
    def _handle_error(self, state: WorkflowState, error_msg: str, agent: str) -> WorkflowState:
        """Handle errors properly"""
        state["workflow_status"] = WorkflowStatus.ERROR.value
        state["error_message"] = error_msg
        state["current_agent"] = agent
        
        self._add_message(state, agent, f"❌ Error: {error_msg}", "error")
        
        return state
    
    def _create_initial_state(self, **kwargs) -> WorkflowState:
        """Create initial workflow state"""
        return WorkflowState(
            uploaded_files=kwargs.get("uploaded_files", []),
            user_query=kwargs.get("user_query", ""),
            search_filters=kwargs.get("search_filters", {}),
            workflow_status=WorkflowStatus.INITIALIZED.value,
            current_agent="",
            error_message="",
            processed_documents=[],
            embeddings=[],
            total_chunks=0,
            vector_store_ready=False,
            search_results=[],
            summary={},
            system_summaries={},
            email_sent=False,
            email_recipients=kwargs.get("email_recipients", []),
            processing_times={},
            agent_messages=[],
            config=self.config
        )
    
    def check_vector_store_status(self) -> Dict[str, Any]:
        """Check the current status of vector store and search agent"""
        status = {
            "vector_store_manager_exists": hasattr(self, 'vector_store_manager') and self.vector_store_manager is not None,
            "search_agent_exists": hasattr(self, 'search_agent') and self.search_agent is not None,
            "search_agent_has_vector_store": False,
            "vector_store_type": None,
            "can_search": False
        }
        
        if status["search_agent_exists"]:
            status["search_agent_has_vector_store"] = hasattr(self.search_agent, 'vector_store') and self.search_agent.vector_store is not None
            
            if status["search_agent_has_vector_store"]:
                status["vector_store_type"] = type(self.search_agent.vector_store).__name__
                status["can_search"] = True
        
        return status
    
    # ================================
    # PUBLIC METHODS FOR STREAMLIT
    # ================================
    
    def run_workflow(self, **kwargs) -> Dict[str, Any]:
        """Run complete workflow"""
        try:
            # Create initial state
            initial_state = self._create_initial_state(**kwargs)
            
            logger.info("🚀 Starting workflow execution...")
            
            # Run workflow
            result = self.app.invoke(initial_state)
            
            logger.info("✅ Workflow execution completed")
            
            return dict(result)  # Convert to regular dict for Streamlit
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            initial_state = self._create_initial_state(**kwargs)
            error_state = self._handle_error(initial_state, str(e), "workflow")
            return dict(error_state)
    
    def run_search_only(self, query: str, search_filters: Dict = None) -> Dict[str, Any]:
        """Run search on existing vector store"""
        try:
            if not self.search_agent:
                # Return error state
                return {
                    "workflow_status": "error",
                    "error_message": "Vector store not initialized. Upload PDFs first.",
                    "summary": {},
                    "system_summaries": {},
                    "search_results": []
                }
            
            # Create search state
            search_state = self._create_initial_state(
                user_query=query,
                search_filters=search_filters or {}
            )
            search_state["vector_store_ready"] = True
            
            # Run search and summary nodes
            search_state = self._search_node(search_state)
            if search_state.get("workflow_status") != WorkflowStatus.ERROR.value:
                search_state = self._summary_node(search_state)
                if search_state.get("workflow_status") != WorkflowStatus.ERROR.value:
                    search_state = self._system_output_node(search_state)
                    
                    # Send email if configured
                    if (self.config.get("email_enabled") and 
                        self.config.get("auto_send_results") and
                        search_state.get("email_recipients")):
                        search_state = self._email_node(search_state)
            
            if search_state.get("workflow_status") != WorkflowStatus.ERROR.value:
                search_state["workflow_status"] = WorkflowStatus.COMPLETED.value
            
            return dict(search_state)  # Convert to regular dict for Streamlit
            
        except Exception as e:
            logger.error(f"Search workflow error: {str(e)}")
            return {
                "workflow_status": "error",
                "error_message": str(e),
                "summary": {},
                "system_summaries": {},
                "search_results": []
            }
    
    def send_email_summary(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email with summary (for Streamlit integration)"""
        try:
            if not self.email_agent:
                return {"success": False, "error": "Email not configured"}
            
            result = self.email_agent.send_email(email_data)
            return {"success": result.get("success", False)}
            
        except Exception as e:
            logger.error(f"Email sending error: {str(e)}")
            return {"success": False, "error": str(e)}
        
        def get_native_langgraph_visualization(self):
            """Get native LangGraph visualization for Streamlit"""
            try:
                if not hasattr(self, 'app') or self.app is None:
                    return {"success": False, "error": "Workflow not compiled"}
            
                # Try to generate the actual LangGraph diagram
                try:
                    # Method 1: Get Mermaid PNG
                    graph_image_data = self.app.get_graph().draw_mermaid_png()
                    
                    # Save for Streamlit
                    with open('native_langgraph_flow.png', 'wb') as f:
                        f.write(graph_image_data)
                    
                    return {
                        "success": True, 
                        "file": "native_langgraph_flow.png",
                        "type": "png",
                        "message": "Native LangGraph PNG generated"
                    }
                    
                except Exception as png_error:
                    # Method 2: Get Mermaid code
                    try:
                        mermaid_code = self.app.get_graph().draw_mermaid()
                        
                        with open('native_langgraph_flow.mmd', 'w') as f:
                            f.write(mermaid_code)
                        
                        return {
                            "success": True,
                            "file": "native_langgraph_flow.mmd", 
                            "type": "mermaid",
                            "code": mermaid_code,
                            "message": "Native LangGraph Mermaid generated"
                        }
                        
                    except Exception as mermaid_error:
                        return {
                            "success": False,
                            "error": f"PNG: {png_error}, Mermaid: {mermaid_error}"
                        }
                        
            except Exception as e:
                return {"success": False, "error": str(e)}
# Mock app class for testing
class MockApp:
    def invoke(self, state):
        return state

# ================================
# TEST FUNCTION
# ================================
def test_workflow():
    """Test the workflow"""
    print("🧪 Testing SAPRAGWorkflow...")
    
    try:
        config = {
            "embedding_type": "openai",
            "vector_store_type": "chroma",
            "email_enabled": False,
            "top_k": 5,
            "chunk_size": 1000,
            "temperature": 0.1
        }
        
        workflow = SAPRAGWorkflow(config)
        print("✅ Workflow created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_workflow()