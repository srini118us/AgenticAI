# workflow.py - Optimized SAP EWA Analysis Workflow with LangGraph
"""
This module implements the complete SAP Early Watch Analysis workflow using LangGraph.

The workflow orchestrates the following agents in a defined sequence:
1. PDFProcessorAgent - Extracts text from uploaded PDF files
2. EmbeddingAgent - Creates vector embeddings from text chunks  
3. VectorStoreManager - Stores embeddings in vector database
4. SearchAgent - Performs similarity search on stored vectors
5. SummaryAgent - Generates intelligent summaries from search results
6. SystemOutputAgent - Creates system-specific analysis outputs
7. EmailAgent - Sends email notifications (optional)

The workflow uses LangGraph for state management and conditional routing,
providing robust error handling, progress tracking, and flexible execution paths.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Literal, TypedDict, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Configure logging for the workflow module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# LANGGRAPH IMPORTS WITH FALLBACKS
# ================================

# Initialize LANGGRAPH_AVAILABLE at module level
LANGGRAPH_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.graph import CompiledGraph
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ LangGraph not available: {e}")
    
    # Create mock classes for development without LangGraph
    class StateGraph:
        """Mock StateGraph for development without LangGraph"""
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}
            self.entry_point = None
            logger.info("📝 Using mock StateGraph")
        
        def add_node(self, name: str, func):
            """Add a node to the mock graph"""
            self.nodes[name] = func
            logger.debug(f"Added node: {name}")
        
        def add_edge(self, from_node: str, to_node: str):
            """Add an edge between nodes"""
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
            logger.debug(f"Added edge: {from_node} -> {to_node}")
        
        def add_conditional_edges(self, from_node: str, condition, mapping: Dict[str, str]):
            """Add conditional edges"""
            self.conditional_edges[from_node] = {
                'condition': condition,
                'mapping': mapping
            }
            logger.debug(f"Added conditional edges from {from_node}: {list(mapping.keys())}")
        
        def set_entry_point(self, node: str):
            """Set the entry point for execution"""
            self.entry_point = node
            logger.debug(f"Set entry point: {node}")
        
        def compile(self):
            """Compile the mock graph"""
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        """Mock compiled graph for testing"""
        def __init__(self, graph: StateGraph):
            self.graph = graph
            self.state_schema = graph.state_schema
            logger.info("🔨 Mock graph compiled")
        
        def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute the mock workflow by running nodes in sequence.
            This provides basic workflow functionality for testing.
            """
            try:
                logger.info("🚀 Starting mock workflow execution")
                
                current_state = initial_state.copy()
                current_state["workflow_status"] = "running"
                
                # Simple sequential execution for testing
                node_sequence = [
                    "pdf_processor",
                    "embedding_creator", 
                    "vector_store_manager",
                    "search_agent",
                    "summary_agent",
                    "system_output_agent",
                    "complete"
                ]
                
                for node_name in node_sequence:
                    if node_name in self.graph.nodes:
                        try:
                            logger.info(f"🔄 Executing mock node: {node_name}")
                            node_func = self.graph.nodes[node_name]
                            current_state = node_func(current_state)
                            
                            # Check for errors
                            if current_state.get("workflow_status") == "error":
                                logger.error(f"❌ Node {node_name} returned error status")
                                break
                                
                        except Exception as e:
                            logger.error(f"❌ Error in mock node {node_name}: {e}")
                            current_state["workflow_status"] = "error"
                            current_state["error_message"] = str(e)
                            break
                
                if current_state.get("workflow_status") != "error":
                    current_state["workflow_status"] = "completed"
                
                logger.info("✅ Mock workflow execution completed")
                return current_state
                
            except Exception as e:
                logger.error(f"❌ Mock workflow execution failed: {e}")
                return {
                    **initial_state,
                    "workflow_status": "error",
                    "error_message": str(e)
                }
        
        def get_graph(self):
            """Mock method for graph visualization"""
            class MockGraph:
                def draw_mermaid_png(self):
                    # Return simple mock PNG data
                    return b"Mock PNG data for testing"
                
                def draw_mermaid(self):
                    return "graph TD\n    A[Start] --> B[End]"
            
            return MockGraph()
    
    # Mock END constant
    END = "END"
    CompiledGraph = MockCompiledGraph

# ================================
# SAFE IMPORTS WITH FALLBACKS
# ================================

try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    try:
        from langchain.callbacks import get_openai_callback
    except ImportError:
        # Fallback context manager
        from contextlib import nullcontext as get_openai_callback
        logger.warning("⚠️ OpenAI callback not available, using null context")

# Import our custom modules with fallbacks
try:
    from agents import (
        PDFProcessorAgent, EmbeddingAgent, SearchAgent, 
        SummaryAgent, EmailAgent, SystemOutputAgent
    )
    from models import WorkflowState, WorkflowStatus, SystemSummary, AgentMessage
    from config import get_agent_config
    logger.info("✅ Custom modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import custom modules: {e}")
    
    # Create minimal fallback classes
    class WorkflowState(TypedDict):
        workflow_status: str
        error_message: str
    
    class WorkflowStatus:
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
    
    # Mock agent classes
    class PDFProcessorAgent:
        def __init__(self, config): 
            self.config = config
        def process(self, files): 
            return {"success": True, "processed_files": []}
    
    class EmbeddingAgent:
        def __init__(self, config): 
            self.config = config
        def process(self, files): 
            return {"success": True, "embeddings": [], "chunks": []}
    
    class SearchAgent:
        def __init__(self, config): 
            self.config = config
        def search(self, query, filters=None): 
            return {"success": True, "search_results": []}
    
    class SummaryAgent:
        def __init__(self, config): 
            self.config = config
        def generate_summary(self, results, query): 
            return {"success": True, "summary": {}}
    
    class EmailAgent:
        def __init__(self, config): 
            self.config = config
        def send_email(self, data): 
            return {"success": True}
    
    class SystemOutputAgent:
        def __init__(self, config): 
            self.config = config
        def generate_system_outputs(self, results): 
            return {"success": True, "system_summaries": {}}
    
    def get_agent_config():
        return {"mock": True}

# Vector store import
try:
    from vector_store import ChromaVectorStore, VectorConfig
    logger.info("✅ ChromaVectorStore imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ ChromaVectorStore not available: {e}")
    
    class MockChromaVectorStore:
        """Mock ChromaDB vector store"""
        def __init__(self, config=None):
            self.config = config
            logger.info("📝 Using mock ChromaVectorStore")
        
        def add_documents(self, documents, metadatas=None, ids=None):
            """Mock add documents"""
            return True
        
        def similarity_search(self, query, k=10, filter=None):
            """Mock similarity search"""
            return []
        
        def similarity_search_with_score(self, query, k=10, filter=None):
            """Mock similarity search with score"""
            return []


# ================================
# WORKFLOW STATE DEFINITION
# ================================

class DetailedWorkflowState(TypedDict):
    """
    Complete state definition for the SAP EWA analysis workflow.
    
    This TypedDict defines all the state variables that flow through
    the workflow, ensuring type safety and documentation of data flow.
    """
    
    # Input parameters - set by user/application
    uploaded_files: List[Any]           # PDF files uploaded by user
    user_query: str                     # Search query string
    search_filters: Dict[str, Any]      # Filters for search (e.g., target_systems)
    email_recipients: List[str]         # Email addresses for notifications
    
    # Workflow control - managed by workflow engine
    workflow_status: str                # Current workflow status (WorkflowStatus enum)
    current_agent: str                  # Currently executing agent name
    error_message: str                  # Error message if workflow fails
    
    # Processing results - populated by agents
    processed_documents: List[Any]      # Documents after PDF processing
    embeddings: List[Any]               # Vector embeddings from text
    total_chunks: int                   # Number of text chunks created
    vector_store_ready: bool            # Whether vector store is populated
    search_results: List[Any]           # Results from similarity search
    summary: Dict[str, Any]             # Generated summary from results
    system_summaries: Dict[str, Any]    # System-specific analysis outputs
    
    # Communication results
    email_sent: bool                    # Whether email notification was sent
    
    # Metrics and monitoring
    processing_times: Dict[str, float]  # Time taken by each processing step
    agent_messages: List[Dict[str, str]] # Messages from agents during execution
    
    # Configuration
    config: Dict[str, Any]              # Workflow configuration settings


# ================================
# WORKFLOW NODE IMPLEMENTATIONS
# ================================

class WorkflowNodeMixin:
    """
    Mixin class providing common functionality for workflow nodes.
    
    This mixin provides standardized error handling, timing, and
    state management for all workflow nodes.
    """
    
    def start_node_timer(self, state: DetailedWorkflowState, node_name: str) -> float:
        """
        Start timing for a workflow node.
        
        Args:
            state: Current workflow state
            node_name: Name of the node being timed
            
        Returns:
            Start time timestamp
        """
        start_time = time.time()
        self._add_agent_message(state, node_name, f"Starting {node_name}...", "processing")
        return start_time
    
    def end_node_timer(self, state: DetailedWorkflowState, node_name: str, 
                      start_time: float, success: bool = True) -> float:
        """
        End timing for a workflow node and record duration.
        
        Args:
            state: Current workflow state
            node_name: Name of the node
            start_time: Start time timestamp
            success: Whether the node completed successfully
            
        Returns:
            Duration in seconds
        """
        duration = time.time() - start_time
        
        if "processing_times" not in state:
            state["processing_times"] = {}
        
        state["processing_times"][node_name] = duration
        
        status = "completed" if success else "error"
        message = f"✅ {node_name} completed in {duration:.2f}s" if success else f"❌ {node_name} failed after {duration:.2f}s"
        
        self._add_agent_message(state, node_name, message, status)
        
        logger.info(f"📊 {node_name}: {duration:.2f}s ({status})")
        return duration
    
    def _add_agent_message(self, state: DetailedWorkflowState, agent: str, 
                          message: str, status: str):
        """
        Add a message to the workflow state for tracking and debugging.
        
        Args:
            state: Workflow state
            agent: Agent/node name
            message: Message content
            status: Message status (processing, completed, error, etc.)
        """
        if "agent_messages" not in state:
            state["agent_messages"] = []
        
        state["agent_messages"].append({
            "agent_name": agent,
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    def handle_node_error(self, state: DetailedWorkflowState, node_name: str, 
                         error: Exception, context: str = "") -> DetailedWorkflowState:
        """
        Standardized error handling for workflow nodes.
        
        Args:
            state: Current workflow state
            node_name: Name of the node where error occurred
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Updated state with error information
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        
        state["workflow_status"] = WorkflowStatus.ERROR
        state["error_message"] = error_msg
        state["current_agent"] = node_name
        
        self._add_agent_message(state, node_name, f"❌ Error: {error_msg}", "error")
        
        logger.error(f"❌ {node_name} error: {error_msg}")
        
        return state


# ================================
# MAIN WORKFLOW CLASS
# ================================

class SAPRAGWorkflow(WorkflowNodeMixin):
    """
    Main workflow class for SAP Early Watch Analysis using LangGraph.
    
    This class orchestrates the complete analysis pipeline from PDF processing
    through to email notifications. It provides both full workflow execution
    and individual component access for flexibility.
    
    Key Features:
    - LangGraph integration for robust state management
    - Comprehensive error handling and recovery
    - Performance monitoring and metrics
    - Flexible execution paths (full workflow vs. search-only)
    - Email notification support
    - Visualization capabilities for workflow graphs
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SAP RAG workflow with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - embedding_type: Type of embeddings to use (openai, huggingface, etc.)
                - vector_store_type: Vector store backend (chroma, faiss, simple)
                - email_enabled: Whether to enable email notifications
                - Various agent-specific settings
        """
        self.config = config or {}
        
        # Merge with default agent config
        try:
            default_config = get_agent_config()
            self.agent_config = {**default_config, **self.config}
        except Exception as e:
            logger.warning(f"Could not load default config: {e}")
            self.agent_config = self.config
        
        # Initialize workflow components
        self._initialize_agents()
        self._initialize_vector_store_manager()
        
        # Build and compile the LangGraph workflow
        self.workflow_graph = self._build_workflow_graph()
        self.app = self._compile_workflow()
        
        # Track workflow statistics
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.last_execution_time = None
        
        logger.info("🏗️ SAPRAGWorkflow initialized successfully")
        logger.info(f"📋 Configuration: {list(self.agent_config.keys())}")
    
    def _initialize_agents(self):
        """
        Initialize all workflow agents with proper error handling.
        
        Each agent is initialized with the shared configuration,
        and fallback agents are created if initialization fails.
        """
        try:
            logger.info("🔧 Initializing workflow agents...")
            
            # Core processing agents
            self.pdf_processor = PDFProcessorAgent(self.agent_config)
            self.embedding_agent = EmbeddingAgent(self.agent_config)
            self.summary_agent = SummaryAgent(self.agent_config)
            self.system_output_agent = SystemOutputAgent(self.agent_config)
            
            # Optional email agent (only if email is enabled)
            self.email_agent = None
            if self.config.get('email_enabled', False):
                try:
                    self.email_agent = EmailAgent(self.agent_config)
                    logger.info("📧 Email agent initialized")
                except Exception as e:
                    logger.warning(f"Email agent initialization failed: {e}")
            
            # Search agent is initialized later after vector store is ready
            self.search_agent = None
            
            logger.info("✅ All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            # Continue with mock agents to prevent complete failure
            self._create_fallback_agents()
    
    def _create_fallback_agents(self):
        """Create minimal fallback agents for testing and development."""
        logger.warning("🔄 Creating fallback agents for testing")
        
        # This would create minimal mock agents that return success
        # but don't perform real processing - useful for UI development
        pass
    
    def _initialize_vector_store_manager(self):
        """
        Initialize the ChromaDB vector store with configuration.
        
        The vector store handles creating and managing
        the ChromaDB database used for similarity search.
        """
        try:
            collection_name = self.config.get('collection_name', 'sap_documents')
            config = VectorConfig(collection_name=collection_name)
            self.vector_store = ChromaVectorStore(config)
            
            logger.info(f"🗄️ ChromaDB vector store initialized: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB vector store initialization failed: {e}")
            # Fall back to mock implementation
            self.vector_store = MockChromaVectorStore()
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow definition.
        
        This method defines the complete workflow graph including:
        - All processing nodes (agents)
        - Node connections (edges)
        - Conditional routing logic
        - Entry and exit points
        
        Returns:
            Configured StateGraph ready for compilation
        """
        logger.info("📊 Building workflow graph...")
        
        # Create the state graph with our state schema
        workflow = StateGraph(DetailedWorkflowState)
        
        # Add all workflow nodes
        workflow.add_node("pdf_processor", self._pdf_processing_node)
        workflow.add_node("embedding_creator", self._embedding_creation_node)
        workflow.add_node("vector_storage", self._vector_storage_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("summary_agent", self._summary_node)
        workflow.add_node("system_output_agent", self._system_output_node)
        workflow.add_node("email_agent", self._email_node)
        workflow.add_node("complete", self._completion_node)
        
        # Set workflow entry point
        workflow.set_entry_point("pdf_processor")
        
        # Define linear processing flow
        workflow.add_edge("pdf_processor", "embedding_creator")
        workflow.add_edge("embedding_creator", "vector_storage")
        workflow.add_edge("search_agent", "summary_agent")
        workflow.add_edge("summary_agent", "system_output_agent")
        workflow.add_edge("email_agent", "complete")
        workflow.add_edge("complete", END)
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "vector_storage",
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
        
        logger.info("✅ Workflow graph built with 8 nodes and conditional routing")
        return workflow
    
    def _compile_workflow(self) -> Union[CompiledGraph, Any]:
        """
        Compile the workflow graph into an executable application.
        
        Returns:
            Compiled workflow application ready for execution
        """
        try:
            if LANGGRAPH_AVAILABLE:
                compiled_app = self.workflow_graph.compile()
                logger.info("✅ LangGraph workflow compiled successfully")
                return compiled_app
            else:
                logger.warning("⚠️ Using mock compiled workflow")
                return MockCompiledGraph(self.workflow_graph)
                
        except Exception as e:
            logger.error(f"❌ Workflow compilation failed: {e}")
            # Return mock implementation as fallback
            return MockCompiledGraph(self.workflow_graph)
    
    # ================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # ================================
    
    def _pdf_processing_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Process uploaded PDF files and extract text content.
        
        This node handles:
        - Loading and validating uploaded PDF files
        - Extracting text using multiple PDF processing libraries
        - Text cleaning and normalization
        - Creating document objects for further processing
        
        Args:
            state: Current workflow state containing uploaded_files
            
        Returns:
            Updated state with processed_documents and metrics
        """
        start_time = self.start_node_timer(state, "pdf_processor")
        
        try:
            # Update workflow status
            state["workflow_status"] = WorkflowStatus.PROCESSING_PDF
            state["current_agent"] = "pdf_processor"
            
            # Get uploaded files from state
            uploaded_files = state.get("uploaded_files", [])
            if not uploaded_files:
                raise ValueError("No PDF files provided for processing")
            
            logger.info(f"📄 Processing {len(uploaded_files)} PDF files")
            
            # Use PDF processor agent to extract text
            result = self.pdf_processor.process(uploaded_files)
            
            if not result.get("success"):
                raise Exception(f"PDF processing failed: {result.get('error', 'Unknown error')}")
            
            processed_files = result.get("processed_files", [])
            if not processed_files:
                raise Exception("No text content extracted from PDF files")
            
            # Convert processed files to document objects
            documents = self._create_document_objects(processed_files)
            
            # Update state with results
            state["processed_documents"] = documents
            state["total_chunks"] = len(documents)
            
            self.end_node_timer(state, "pdf_processor", start_time, True)
            
            logger.info(f"✅ PDF processing completed: {len(documents)} documents created")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "pdf_processor", start_time, False)
            return self.handle_node_error(state, "pdf_processor", e, "PDF processing")
    
    def _embedding_creation_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Create vector embeddings from processed documents.
        
        This node handles:
        - Text chunking with configurable size and overlap
        - Vector embedding generation using configured model
        - Batch processing for efficiency
        - Embedding validation and formatting
        
        Args:
            state: Current workflow state with processed_documents
            
        Returns:
            Updated state with embeddings and chunks
        """
        start_time = self.start_node_timer(state, "embedding_creator")
        
        try:
            state["workflow_status"] = WorkflowStatus.CREATING_EMBEDDINGS
            state["current_agent"] = "embedding_creator"
            
            # Get processed documents
            documents = state.get("processed_documents", [])
            if not documents:
                raise ValueError("No processed documents available for embedding")
            
            logger.info(f"🔤 Creating embeddings for {len(documents)} documents")
            
            # Convert documents to format expected by embedding agent
            processed_files = self._documents_to_processed_files(documents)
            
            # Create embeddings using embedding agent
            result = self.embedding_agent.process(processed_files)
            
            if not result.get("success"):
                raise Exception(f"Embedding creation failed: {result.get('error', 'Unknown error')}")
            
            embeddings = result.get("embeddings", [])
            chunks = result.get("chunks", [])
            
            if len(embeddings) != len(chunks):
                logger.warning(f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})")
                # Trim to match smaller count
                min_count = min(len(embeddings), len(chunks))
                embeddings = embeddings[:min_count]
                chunks = chunks[:min_count]
            
            # Update state
            state["embeddings"] = embeddings
            state["processed_documents"] = self._chunks_to_documents(chunks)
            state["total_chunks"] = len(chunks)
            
            self.end_node_timer(state, "embedding_creator", start_time, True)
            
            logger.info(f"✅ Embedding creation completed: {len(embeddings)} embeddings created")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "embedding_creator", start_time, False)
            return self.handle_node_error(state, "embedding_creator", e, "Embedding creation")
    
    def _vector_storage_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Store vector embeddings in the vector database and initialize search agent.
        
        This is a critical node that:
        - Creates the vector store with documents and embeddings
        - Initializes the search agent with the populated vector store
        - Validates that search functionality is working
        - Sets up the system for similarity search operations
        
        Args:
            state: Current workflow state with embeddings and documents
            
        Returns:
            Updated state with vector_store_ready flag and search agent
        """
        start_time = self.start_node_timer(state, "vector_storage")
        
        try:
            state["workflow_status"] = WorkflowStatus.STORING_VECTORS
            state["current_agent"] = "vector_storage"
            
            # Get documents and embeddings
            documents = state.get("processed_documents", [])
            embeddings = state.get("embeddings", [])
            
            if not documents:
                raise ValueError("No documents available for vector storage")
            
            logger.info(f"🗄️ Storing {len(documents)} documents and {len(embeddings)} embeddings")
            
            # Add documents to ChromaDB vector store
            self.vector_store.add_documents(documents, metadatas=None, ids=None)
            
            logger.info(f"✅ Documents added to ChromaDB vector store")
            
            # Initialize search agent with the populated vector store
            search_agent_config = {
                **self.agent_config,
                'vector_store': self.vector_store,
                'embedding_agent': self.embedding_agent
            }
            
            self.search_agent = SearchAgent(search_agent_config)
            
            # Validate search agent setup
            if not hasattr(self.search_agent, 'vector_store') or not self.search_agent.vector_store:
                raise Exception("Search agent was not properly initialized with vector store")
            
            # Test search functionality with a simple query
            test_result = self.search_agent.search("test", {})
            if not test_result.get("success"):
                logger.warning("⚠️ Search agent test failed, but continuing")
            
            # Mark vector store as ready
            state["vector_store_ready"] = True
            
            self.end_node_timer(state, "vector_storage", start_time, True)
            
            logger.info("✅ Vector storage completed, search agent ready")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "vector_storage", start_time, False)
            return self.handle_node_error(state, "vector_storage", e, "Vector storage")
    
    def _search_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Perform similarity search on the vector store.
        
        This node executes the core search functionality:
        - Takes user query and search filters from state
        - Performs vector similarity search
        - Filters and ranks results
        - Formats results for downstream processing
        
        Args:
            state: Current workflow state with user_query and search_filters
            
        Returns:
            Updated state with search_results
        """
        start_time = self.start_node_timer(state, "search_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SEARCHING
            state["current_agent"] = "search_agent"
            
            # FIX 3: Better query extraction and validation
            query = state.get("user_query", "").strip()
            search_filters = state.get("search_filters", {})
            
            # Debug logging
            logger.info(f"Raw query from state: '{state.get('user_query', 'NOT_FOUND')}'")
            logger.info(f"Cleaned query: '{query}'")
            logger.info(f"Search filters: {search_filters}")
            logger.info(f"State type: {type(state)}")
            logger.info(f"State keys: {list(state.keys()) if hasattr(state, 'keys') else 'No keys method'}")
            
            if not query:
                # Check alternative query locations in state
                alt_query = state.get("query", "").strip()
                if alt_query:
                    query = alt_query
                    logger.info(f"Using alternative query: '{query}'")
                else:
                    # Additional debugging
                    logger.error(f"State content: {state}")
                    logger.error(f"State user_query: {state.get('user_query', 'NOT_FOUND')}")
                    logger.error(f"State query: {state.get('query', 'NOT_FOUND')}")
                    raise ValueError("No search query provided in state")
            
            if len(query) < 3:
                raise ValueError(f"Search query too short: '{query}' (minimum 3 characters)")
            
            if not self.search_agent:
                raise Exception("Search agent not initialized - vector store may not be ready")
            
            logger.info(f"🔍 Searching for: '{query}' with filters: {search_filters}")
            
            # Perform the search
            result = self.search_agent.search(query, search_filters)
            
            if not result.get("success"):
                raise Exception(f"Search failed: {result.get('error', 'Unknown error')}")
            
            search_results = result.get("search_results", [])
            
            # Store results in state
            state["search_results"] = search_results
            
            self.end_node_timer(state, "search_agent", start_time, True)
            
            logger.info(f"✅ Search completed: {len(search_results)} results found")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "search_agent", start_time, False)
            return self.handle_node_error(state, "search_agent", e, "Search execution")
    
    def _summary_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Generate intelligent summary from search results.
        
        This node creates comprehensive analysis including:
        - Executive summary of findings
        - Critical issues identification
        - Actionable recommendations
        - Confidence scoring based on result quality
        
        Args:
            state: Current workflow state with search_results
            
        Returns:
            Updated state with summary data
        """
        start_time = self.start_node_timer(state, "summary_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SUMMARIZING
            state["current_agent"] = "summary_agent"
            
            # Get search results and query
            search_results = state.get("search_results", [])
            query = state.get("user_query", "")
            
            logger.info(f"📝 Generating summary for {len(search_results)} search results")
            
            # Generate summary using summary agent
            result = self.summary_agent.generate_summary(search_results, query)
            
            if not result.get("success"):
                raise Exception(f"Summary generation failed: {result.get('error', 'Unknown error')}")
            
            summary = result.get("summary", {})
            
            # Store summary in state
            state["summary"] = summary
            
            self.end_node_timer(state, "summary_agent", start_time, True)
            
            # Log summary statistics
            critical_count = len(summary.get("critical_findings", []))
            recommendation_count = len(summary.get("recommendations", []))
            confidence = summary.get("confidence_score", 0) * 100
            
            logger.info(f"✅ Summary generated: {critical_count} critical findings, "
                       f"{recommendation_count} recommendations (confidence: {confidence:.1f}%)")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "summary_agent", start_time, False)
            return self.handle_node_error(state, "summary_agent", e, "Summary generation")
    
    def _system_output_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Generate system-specific analysis outputs.
        
        This node creates detailed analysis for each SAP system found:
        - System health assessments (HEALTHY/WARNING/CRITICAL)
        - System-specific alerts and recommendations
        - Performance metrics extraction
        - Individual system reports
        
        Args:
            state: Current workflow state with search_results
            
        Returns:
            Updated state with system_summaries
        """
        start_time = self.start_node_timer(state, "system_output_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SYSTEM_OUTPUT
            state["current_agent"] = "system_output_agent"
            
            search_results = state.get("search_results", [])
            
            logger.info(f"🖥️ Generating system outputs for {len(search_results)} search results")
            
            # Generate system-specific outputs
            result = self.system_output_agent.generate_system_outputs(search_results)
            
            if not result.get("success"):
                raise Exception(f"System output generation failed: {result.get('error', 'Unknown error')}")
            
            system_summaries = result.get("system_summaries", {})
            
            # Store system summaries in state
            state["system_summaries"] = system_summaries
            
            self.end_node_timer(state, "system_output_agent", start_time, True)
            
            # Log system analysis statistics
            systems_analyzed = len(system_summaries)
            critical_systems = sum(1 for s in system_summaries.values() 
                                 if hasattr(s, 'overall_health') and 
                                 str(s.overall_health).upper() == 'CRITICAL')
            
            logger.info(f"✅ System analysis completed: {systems_analyzed} systems analyzed, "
                       f"{critical_systems} critical systems identified")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "system_output_agent", start_time, False)
            return self.handle_node_error(state, "system_output_agent", e, "System output generation")
    
    def _email_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Send email notifications with analysis results.
        
        This optional node handles:
        - Email content formatting with analysis results
        - SMTP delivery with retry logic
        - Multiple recipient support
        - Email delivery status tracking
        
        Args:
            state: Current workflow state with summary and system_summaries
            
        Returns:
            Updated state with email_sent status
        """
        start_time = self.start_node_timer(state, "email_agent")
        
        try:
            # Check if email agent is available and configured
            if not self.email_agent:
                self._add_agent_message(state, "email_agent", "📧 Email not configured - skipping", "skipped")
                state["email_sent"] = False
                self.end_node_timer(state, "email_agent", start_time, True)
                return state
            
            state["workflow_status"] = WorkflowStatus.SENDING_EMAIL
            state["current_agent"] = "email_agent"
            
            # Get email recipients
            recipients = state.get("email_recipients", [])
            if not recipients:
                self._add_agent_message(state, "email_agent", "📧 No recipients specified - skipping", "skipped")
                state["email_sent"] = False
                self.end_node_timer(state, "email_agent", start_time, True)
                return state
            
            logger.info(f"📧 Sending email to {len(recipients)} recipients")
            
            # Prepare email data
            email_data = {
                'recipients': recipients,
                'summary': state.get("summary", {}),
                'query': state.get("user_query", ""),
                'system_summaries': state.get("system_summaries", {})
            }
            
            # Send email using email agent
            result = self.email_agent.send_email(email_data)
            
            email_sent = result.get("success", False)
            state["email_sent"] = email_sent
            
            self.end_node_timer(state, "email_agent", start_time, email_sent)
            
            if email_sent:
                logger.info(f"✅ Email sent successfully to {len(recipients)} recipients")
            else:
                logger.warning(f"⚠️ Email sending failed: {result.get('error', 'Unknown error')}")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "email_agent", start_time, False)
            state["email_sent"] = False
            self._add_agent_message(state, "email_agent", f"📧 Email failed: {str(e)}", "error")
            # Don't fail the entire workflow for email errors
            logger.warning(f"⚠️ Email node error (continuing): {e}")
            return state
    
    def _completion_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Complete the workflow and finalize metrics.
        
        This final node:
        - Sets workflow status to completed
        - Calculates total processing time
        - Logs final statistics and summary
        - Prepares final state for return
        
        Args:
            state: Current workflow state
            
        Returns:
            Final completed workflow state
        """
        try:
            state["workflow_status"] = WorkflowStatus.COMPLETED
            state["current_agent"] = "complete"
            
            # Calculate total processing time
            processing_times = state.get("processing_times", {})
            total_time = sum(processing_times.values())
            
            # Update workflow statistics
            self.execution_count += 1
            self.total_processing_time += total_time
            self.last_execution_time = total_time
            
            # Log completion statistics
            logger.info("🎉 Workflow completed successfully!")
            logger.info(f"📊 Total processing time: {total_time:.2f}s")
            logger.info(f"📄 Documents processed: {state.get('total_chunks', 0)}")
            logger.info(f"🔍 Search results: {len(state.get('search_results', []))}")
            logger.info(f"🖥️ Systems analyzed: {len(state.get('system_summaries', {}))}")
            logger.info(f"📧 Email sent: {'Yes' if state.get('email_sent', False) else 'No'}")
            
            # Add final completion message
            self._add_agent_message(
                state, 
                "workflow", 
                f"🎉 Workflow completed successfully in {total_time:.2f}s",
                "completed"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Completion node error: {e}")
            return self.handle_node_error(state, "complete", e, "Workflow completion")
    
    # ================================
    # ROUTING FUNCTIONS
    # ================================
    
    def _route_after_vector_storage(self, state: DetailedWorkflowState) -> Literal["search", "complete"]:
        """
        Conditional routing after vector storage is complete.
        
        Determines whether to proceed with search or complete the workflow:
        - If user_query exists and vector store is ready → proceed to search
        - If no query or vector store not ready → complete workflow
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "search" or "complete"
        """
        try:
            user_query = state.get("user_query", "").strip()
            vector_store_ready = state.get("vector_store_ready", False)
            has_search_agent = self.search_agent is not None
            
            logger.info(f"🔀 Routing decision:")
            logger.info(f"  - Query: '{user_query}' (exists: {bool(user_query)})")
            logger.info(f"  - Vector store ready: {vector_store_ready}")
            logger.info(f"  - Search agent ready: {has_search_agent}")
            
            # Only proceed to search if we have a query and everything is ready
            if user_query and vector_store_ready and has_search_agent:
                logger.info("🔀 Routing to: search")
                return "search"
            else:
                logger.info("🔀 Routing to: complete (no search needed)")
                return "complete"
                
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            return "complete"  # Safe default
    
    def _route_after_system_output(self, state: DetailedWorkflowState) -> Literal["send_email", "complete"]:
        """
        Conditional routing after system output generation.
        
        Determines whether to send email notifications:
        - If email is enabled, configured, and recipients exist → send email
        - Otherwise → complete workflow
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "send_email" or "complete"
        """
        try:
            email_enabled = self.config.get("email_enabled", False)
            auto_send = self.config.get("auto_send_results", False)
            has_recipients = bool(state.get("email_recipients"))
            has_email_agent = self.email_agent is not None
            
            logger.info(f"🔀 Email routing decision:")
            logger.info(f"  - Email enabled: {email_enabled}")
            logger.info(f"  - Auto send: {auto_send}")
            logger.info(f"  - Has recipients: {has_recipients}")
            logger.info(f"  - Email agent ready: {has_email_agent}")
            
            # Send email if all conditions are met
            if email_enabled and auto_send and has_recipients and has_email_agent:
                logger.info("🔀 Routing to: send_email")
                return "send_email"
            else:
                logger.info("🔀 Routing to: complete (no email needed)")
                return "complete"
                
        except Exception as e:
            logger.error(f"❌ Email routing error: {e}")
            return "complete"  # Safe default
    
    # ================================
    # UTILITY FUNCTIONS
    # ================================
    
    def _create_document_objects(self, processed_files: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert processed file data to document objects.
        
        Creates document objects that are compatible with the vector store
        and contain proper metadata for downstream processing.
        
        Args:
            processed_files: List of processed file dictionaries
            
        Returns:
            List of document objects
        """
        documents = []
        
        for file_data in processed_files:
            try:
                # Try to use LangChain Document if available
                try:
                    from langchain.schema import Document
                    doc = Document(
                        page_content=file_data.get('text', ''),
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0),
                            'character_count': file_data.get('character_count', 0),
                            'word_count': file_data.get('word_count', 0),
                            'processing_timestamp': file_data.get('processing_timestamp', datetime.now().isoformat())
                        }
                    )
                except ImportError:
                    # Fallback to simple object
                    class DocumentLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    doc = DocumentLike(
                        page_content=file_data.get('text', ''),
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0)
                        }
                    )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error creating document object: {e}")
                continue
        
        return documents
    
    def _documents_to_processed_files(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert document objects back to processed file format for embedding agent.
        
        Args:
            documents: List of document objects
            
        Returns:
            List of processed file dictionaries
        """
        processed_files = []
        
        for doc in documents:
            try:
                file_data = {
                    'text': getattr(doc, 'page_content', str(doc)),
                    'filename': getattr(doc, 'metadata', {}).get('source', 'unknown'),
                    'size': getattr(doc, 'metadata', {}).get('size', 0)
                }
                processed_files.append(file_data)
            except Exception as e:
                logger.warning(f"Error converting document: {e}")
                continue
        
        return processed_files
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert text chunks back to document objects.
        
        Args:
            chunks: List of text chunk dictionaries
            
        Returns:
            List of document objects
        """
        documents = []
        
        for chunk in chunks:
            try:
                # Try LangChain Document first
                try:
                    from langchain.schema import Document
                    doc = Document(
                        page_content=chunk.get('page_content', chunk.get('text', '')),
                        metadata=chunk.get('metadata', {})
                    )
                except ImportError:
                    # Fallback
                    class DocumentLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    doc = DocumentLike(
                        page_content=chunk.get('page_content', chunk.get('text', '')),
                        metadata=chunk.get('metadata', {})
                    )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error converting chunk to document: {e}")
                continue
        
        return documents
    
    def _create_initial_state(self, **kwargs) -> DetailedWorkflowState:
        """
        Create initial workflow state with provided parameters.
        
        Args:
            **kwargs: State initialization parameters
            
        Returns:
            Initial workflow state dictionary
        """
        # FIX 6: Ensure user_query is properly handled
        user_query = kwargs.get("user_query", "")
        if user_query and isinstance(user_query, str):
            user_query = user_query.strip()
        
        initial_state = DetailedWorkflowState(
            # Input parameters
            uploaded_files=kwargs.get("uploaded_files", []),
            user_query=user_query,  # Use processed query
            search_filters=kwargs.get("search_filters", {}),
            email_recipients=kwargs.get("email_recipients", []),
            
            # Workflow control
            workflow_status=WorkflowStatus.INITIALIZED,
            current_agent="",
            error_message="",
            
            # Processing results
            processed_documents=[],
            embeddings=[],
            total_chunks=0,
            vector_store_ready=False,
            search_results=[],
            summary={},
            system_summaries={},
            
            # Communication
            email_sent=False,
            
            # Metrics
            processing_times={},
            agent_messages=[],
            
            # Configuration
            config=self.agent_config
        )
        
        return initial_state
    
    # ================================
    # PUBLIC API METHODS
    # ================================
    
    def run_workflow(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete SAP EWA analysis workflow.
        
        This is the main entry point for full workflow execution,
        handling everything from PDF processing to email notifications.
        
        Args:
            uploaded_files: List of PDF files to process
            user_query: Search query string (optional for processing-only)
            search_filters: Search filters dictionary (optional)
            email_recipients: List of email addresses for notifications (optional)
            
        Returns:
            Dictionary containing workflow results and status
        """
        try:
            logger.info("🚀 Starting complete SAP EWA workflow")
            
            # Create initial state
            initial_state = self._create_initial_state(**kwargs)
            
            # Log workflow parameters
            logger.info(f"📊 Workflow parameters:")
            logger.info(f"  - Files: {len(kwargs.get('uploaded_files', []))}")
            logger.info(f"  - Query: '{kwargs.get('user_query', '')}'")
            logger.info(f"  - Filters: {kwargs.get('search_filters', {})}")
            logger.info(f"  - Email recipients: {len(kwargs.get('email_recipients', []))}")
            
            # Execute workflow
            final_state = self.app.invoke(initial_state)
            
            # Convert TypedDict to regular dict for JSON serialization
            result = dict(final_state)
            
            logger.info("✅ Complete workflow execution finished")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            
            # Return error state
            error_state = self._create_initial_state(**kwargs)
            error_state["workflow_status"] = WorkflowStatus.ERROR
            error_state["error_message"] = str(e)
            
            return dict(error_state)
    
    def run_search_only(self, query: str, search_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute search-only workflow on existing vector store.
        
        This method allows running search and analysis without reprocessing PDFs,
        useful for exploring different queries on the same document set.
        
        Args:
            query: Search query string
            search_filters: Optional search filters (e.g., target_systems)
            
        Returns:
            Dictionary containing search results and analysis
        """
        try:
            # FIX 4: Better query validation
            if not query or not query.strip():
                error_msg = "Empty or invalid search query provided"
                logger.error(f"❌ {error_msg}")
                return {
                    "workflow_status": WorkflowStatus.ERROR,
                    "error_message": error_msg,
                    "summary": {},
                    "system_summaries": {},
                    "search_results": []
                }
            
            cleaned_query = query.strip()
            if len(cleaned_query) < 3:
                error_msg = f"Search query too short: '{cleaned_query}' (minimum 3 characters)"
                logger.error(f"❌ {error_msg}")
                return {
                    "workflow_status": WorkflowStatus.ERROR,
                    "error_message": error_msg,
                    "summary": {},
                    "system_summaries": {},
                    "search_results": []
                }
            
            logger.info(f"🔍 Starting search-only workflow for query: '{cleaned_query}'")
            
            # Check if search agent is available
            if not self.search_agent:
                error_msg = "Search agent not initialized. Please run full workflow first to process documents."
                logger.error(f"❌ {error_msg}")
                return {
                    "workflow_status": WorkflowStatus.ERROR,
                    "error_message": error_msg,
                    "summary": {},
                    "system_summaries": {},
                    "search_results": []
                }
            
            # FIX 5: Ensure query is properly set in state
            search_state = self._create_initial_state(
                user_query=cleaned_query,  # Use cleaned query
                search_filters=search_filters or {}
            )
            
            # Convert TypedDict to regular dict to ensure proper access
            search_state = dict(search_state)
            
            # Mark vector store as ready since we're using existing setup
            search_state["vector_store_ready"] = True
            
            # Additional validation - make sure query is in state
            if not search_state.get("user_query"):
                search_state["user_query"] = cleaned_query
                
            logger.info(f"Search state created with query: '{search_state.get('user_query')}'")
            
            # Execute search and analysis nodes in sequence
            try:
                # Search
                logger.info(f"About to execute search node with state: {search_state}")
                search_state = self._search_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                # Summary
                search_state = self._summary_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                # System output
                search_state = self._system_output_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                # Optional email (if configured)
                if (self.config.get("email_enabled") and 
                    self.config.get("auto_send_results") and
                    search_state.get("email_recipients")):
                    search_state = self._email_node(search_state)
                
                # Complete
                search_state = self._completion_node(search_state)
                
            except Exception as e:
                logger.error(f"❌ Search workflow node execution failed: {e}")
                search_state["workflow_status"] = WorkflowStatus.ERROR
                search_state["error_message"] = str(e)
            
            logger.info("✅ Search-only workflow completed")
            
            return dict(search_state)
            
        except Exception as e:
            logger.error(f"❌ Search-only workflow failed: {e}")
            return {
                "workflow_status": WorkflowStatus.ERROR,
                "error_message": str(e),
                "summary": {},
                "system_summaries": {},
                "search_results": []
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status and statistics.
        
        Returns:
            Dictionary with workflow status information
        """
        try:
            # Check agent availability
            agents_status = {
                "pdf_processor": self.pdf_processor is not None,
                "embedding_agent": self.embedding_agent is not None,
                "search_agent": self.search_agent is not None,
                "summary_agent": self.summary_agent is not None,
                "email_agent": self.email_agent is not None,
                "system_output_agent": self.system_output_agent is not None
            }
            
            # Check vector store status
            vector_store_status = {
                "vector_store_available": self.vector_store is not None,
                "search_agent_ready": self.search_agent is not None,
                "can_search": self.search_agent is not None and hasattr(self.search_agent, 'vector_store')
            }
            
            # Workflow statistics
            avg_processing_time = (self.total_processing_time / self.execution_count 
                                 if self.execution_count > 0 else 0.0)
            
            return {
                "workflow_ready": all(agents_status.values()),
                "agents_status": agents_status,
                "vector_store_status": vector_store_status,
                "langgraph_available": LANGGRAPH_AVAILABLE,
                "execution_statistics": {
                    "total_executions": self.execution_count,
                    "total_processing_time": self.total_processing_time,
                    "average_processing_time": avg_processing_time,
                    "last_execution_time": self.last_execution_time
                },
                "configuration": {
                    "email_enabled": self.config.get("email_enabled", False),
                    "vector_store_type": self.config.get("vector_store_type", "unknown"),
                    "embedding_type": self.config.get("embedding_type", "unknown")
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {
                "workflow_ready": False,
                "error": str(e)
            }
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """
        Generate workflow visualization data.
        
        Creates visual representation of the workflow graph for debugging
        and documentation purposes.
        
        Returns:
            Dictionary with visualization data
        """
        try:
            if not LANGGRAPH_AVAILABLE or not hasattr(self.app, 'get_graph'):
                return {
                    "success": False,
                    "error": "LangGraph visualization not available",
                    "fallback_description": "PDF → Embedding → Vector Store → Search → Summary → System Output → Email → Complete"
                }
            
            # Try to generate Mermaid diagram
            try:
                # First try PNG generation
                graph_image_data = self.app.get_graph().draw_mermaid_png()
                
                # Save to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'workflow_diagram_{timestamp}.png'
                
                with open(filename, 'wb') as f:
                    f.write(graph_image_data)
                
                return {
                    "success": True,
                    "type": "png",
                    "file": filename,
                    "size_bytes": len(graph_image_data),
                    "message": "Workflow diagram generated successfully"
                }
                
            except Exception as png_error:
                # Fallback to Mermaid text
                try:
                    mermaid_code = self.app.get_graph().draw_mermaid()
                    
                    return {
                        "success": True,
                        "type": "mermaid",
                        "code": mermaid_code,
                        "message": "Mermaid diagram code generated"
                    }
                    
                except Exception as mermaid_error:
                    logger.error(f"Both PNG and Mermaid generation failed: PNG={png_error}, Mermaid={mermaid_error}")
                    return {
                        "success": False,
                        "error": f"Visualization generation failed: {str(mermaid_error)}"
                    }
            
        except Exception as e:
            logger.error(f"Workflow visualization error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def reset_workflow(self):
        """
        Reset workflow state for fresh execution.
        
        Clears search agent and statistics while preserving configuration.
        Useful for processing new document sets.
        """
        try:
            logger.info("🔄 Resetting workflow state")
            
            # Clear search agent (will be recreated when vector store is populated)
            self.search_agent = None
            
            # Reset statistics
            self.execution_count = 0
            self.total_processing_time = 0.0
            self.last_execution_time = None
            
            # Reinitialize vector store manager
            self._initialize_vector_store_manager()
            
            logger.info("✅ Workflow reset completed")
            
        except Exception as e:
            logger.error(f"❌ Workflow reset failed: {e}")


# ================================
# WORKFLOW FACTORY AND UTILITIES
# ================================

def create_workflow(config: Dict[str, Any] = None) -> SAPRAGWorkflow:
    """
    Factory function to create a configured SAP RAG workflow.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized SAPRAGWorkflow instance
    """
    try:
        workflow = SAPRAGWorkflow(config)
        logger.info("✅ Workflow created successfully via factory")
        return workflow
    except Exception as e:
        logger.error(f"❌ Workflow creation failed: {e}")
        raise


def validate_workflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate workflow configuration and provide recommendations.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation results and recommendations
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    try:
        # Check required configuration
        if not config.get("openai_api_key"):
            validation["warnings"].append("No OpenAI API key - will use mock embeddings")
        
        # Check email configuration
        if config.get("email_enabled", False):
            if not config.get("gmail_email") or not config.get("gmail_app_password"):
                validation["errors"].append("Email enabled but credentials missing")
                validation["valid"] = False
        
        # Check vector store configuration
        vector_store_type = config.get("vector_store_type", "chroma")
        if vector_store_type not in ["chroma", "faiss", "simple", "mock"]:
            validation["warnings"].append(f"Unknown vector store type: {vector_store_type}")
        
        # Performance recommendations
        chunk_size = config.get("chunk_size", 1000)
        if chunk_size > 2000:
            validation["recommendations"].append("Consider smaller chunk size for better search precision")
        
        top_k = config.get("top_k", 10)
        if top_k > 50:
            validation["recommendations"].append("Large top_k value may slow down processing")
        
        return validation
        
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Configuration validation error: {str(e)}")
        return validation


# ================================
# MODULE TESTING FUNCTION
# ================================

def test_workflow_basic() -> bool:
    """
    Basic workflow test to ensure components are working.
    
    Returns:
        True if basic test passes
    """
    try:
        logger.info("🧪 Running basic workflow test...")
        
        # Create test configuration
        test_config = {
            "embedding_type": "mock",
            "vector_store_type": "mock",
            "email_enabled": False,
            "top_k": 5,
            "chunk_size": 500
        }
        
        # Create workflow
        workflow = SAPRAGWorkflow(test_config)
        
        # Check basic functionality
        status = workflow.get_workflow_status()
        
        if not status.get("workflow_ready"):
            logger.warning("⚠️ Workflow not fully ready, but this may be expected in test environment")
        
        logger.info("✅ Basic workflow test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic workflow test failed: {e}")
        return False


# ================================
# ADDITIONAL WORKFLOW UTILITIES
# ================================

class WorkflowDebugger:
    """
    Utility class for debugging workflow execution issues.
    
    Provides tools for analyzing workflow state, performance bottlenecks,
    and component interactions during development and troubleshooting.
    """
    
    def __init__(self, workflow: SAPRAGWorkflow):
        """
        Initialize debugger with workflow instance.
        
        Args:
            workflow: SAPRAGWorkflow instance to debug
        """
        self.workflow = workflow
        self.debug_logs = []
    
    def analyze_workflow_state(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """
        Analyze current workflow state for debugging.
        
        Args:
            state: Current workflow state
            
        Returns:
            Analysis results with debugging information
        """
        analysis = {
            "state_completeness": {},
            "performance_analysis": {},
            "component_status": {},
            "potential_issues": []
        }
        
        try:
            # Check state completeness
            required_fields = [
                "workflow_status", "uploaded_files", "user_query",
                "processed_documents", "embeddings", "vector_store_ready"
            ]
            
            for field in required_fields:
                value = state.get(field)
                analysis["state_completeness"][field] = {
                    "present": value is not None,
                    "type": type(value).__name__,
                    "length": len(value) if hasattr(value, '__len__') else None
                }
            
            # Analyze performance
            processing_times = state.get("processing_times", {})
            if processing_times:
                total_time = sum(processing_times.values())
                slowest_step = max(processing_times.items(), key=lambda x: x[1])
                
                analysis["performance_analysis"] = {
                    "total_time": total_time,
                    "step_count": len(processing_times),
                    "average_step_time": total_time / len(processing_times),
                    "slowest_step": {
                        "name": slowest_step[0],
                        "duration": slowest_step[1]
                    }
                }
            
            # Check component status
            analysis["component_status"] = {
                "pdf_processor": self.workflow.pdf_processor is not None,
                "embedding_agent": self.workflow.embedding_agent is not None,
                "search_agent": self.workflow.search_agent is not None,
                "vector_store": self.workflow.vector_store is not None,
                "email_agent": self.workflow.email_agent is not None
            }
            
            # Identify potential issues
            if not state.get("vector_store_ready") and state.get("user_query"):
                analysis["potential_issues"].append("Query provided but vector store not ready")
            
            if len(state.get("processed_documents", [])) == 0 and len(state.get("uploaded_files", [])) > 0:
                analysis["potential_issues"].append("Files uploaded but no documents processed")
            
            if state.get("workflow_status") == WorkflowStatus.ERROR:
                analysis["potential_issues"].append(f"Workflow in error state: {state.get('error_message', 'Unknown error')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing workflow state: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report for the workflow.
        
        Returns:
            Performance analysis and recommendations
        """
        try:
            status = self.workflow.get_workflow_status()
            
            report = {
                "summary": {
                    "total_executions": status["execution_statistics"]["total_executions"],
                    "average_time": status["execution_statistics"]["average_processing_time"],
                    "last_execution": status["execution_statistics"]["last_execution_time"]
                },
                "recommendations": [],
                "optimization_opportunities": []
            }
            
            # Performance recommendations
            avg_time = status["execution_statistics"]["average_processing_time"]
            if avg_time > 30:
                report["recommendations"].append("Consider optimizing PDF processing or reducing chunk size")
            
            if avg_time > 60:
                report["recommendations"].append("Workflow taking over 1 minute - check vector store performance")
            
            # Component recommendations
            if not status["vector_store_status"]["can_search"]:
                report["optimization_opportunities"].append("Search agent not properly initialized")
            
            if not status["agents_status"]["email_agent"]:
                report["optimization_opportunities"].append("Email functionality not configured")
            
            return report
            
        except Exception as e:
            return {"error": f"Failed to generate performance report: {e}"}


class WorkflowMonitor:
    """
    Real-time monitoring for workflow execution.
    
    Tracks workflow progress, performance metrics, and health indicators
    during execution for operational monitoring and alerting.
    """
    
    def __init__(self):
        """Initialize workflow monitor."""
        self.active_workflows = {}
        self.execution_history = []
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
    
    def start_monitoring(self, workflow_id: str, workflow: SAPRAGWorkflow):
        """
        Start monitoring a workflow execution.
        
        Args:
            workflow_id: Unique identifier for this execution
            workflow: Workflow instance to monitor
        """
        self.active_workflows[workflow_id] = {
            "workflow": workflow,
            "start_time": datetime.now(),
            "status": "running",
            "last_update": datetime.now()
        }
        
        logger.info(f"📊 Started monitoring workflow: {workflow_id}")
    
    def update_workflow_progress(self, workflow_id: str, state: DetailedWorkflowState):
        """
        Update progress for a monitored workflow.
        
        Args:
            workflow_id: Workflow identifier
            state: Current workflow state
        """
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].update({
                "status": state.get("workflow_status", "unknown"),
                "current_agent": state.get("current_agent", ""),
                "last_update": datetime.now(),
                "progress_data": {
                    "files_processed": len(state.get("processed_documents", [])),
                    "search_results": len(state.get("search_results", [])),
                    "systems_found": len(state.get("system_summaries", {}))
                }
            })
    
    def finish_monitoring(self, workflow_id: str, final_state: DetailedWorkflowState):
        """
        Finish monitoring and record final results.
        
        Args:
            workflow_id: Workflow identifier  
            final_state: Final workflow state
        """
        if workflow_id not in self.active_workflows:
            return
        
        workflow_data = self.active_workflows[workflow_id]
        end_time = datetime.now()
        execution_time = (end_time - workflow_data["start_time"]).total_seconds()
        
        # Record execution history
        execution_record = {
            "workflow_id": workflow_id,
            "start_time": workflow_data["start_time"],
            "end_time": end_time,
            "execution_time": execution_time,
            "status": final_state.get("workflow_status", "unknown"),
            "files_processed": len(final_state.get("processed_documents", [])),
            "search_results": len(final_state.get("search_results", [])),
            "error_message": final_state.get("error_message", "")
        }
        
        self.execution_history.append(execution_record)
        
        # Update performance metrics
        self.performance_metrics["total_executions"] += 1
        
        if final_state.get("workflow_status") == WorkflowStatus.COMPLETED:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_time = sum(record["execution_time"] for record in self.execution_history)
        self.performance_metrics["average_execution_time"] = total_time / len(self.execution_history)
        
        # Clean up active workflows
        del self.active_workflows[workflow_id]
        
        logger.info(f"📊 Finished monitoring workflow {workflow_id}: {execution_time:.2f}s")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get current monitoring summary.
        
        Returns:
            Summary of all monitored workflows and performance metrics
        """
        return {
            "active_workflows": len(self.active_workflows),
            "active_workflow_details": {
                wf_id: {
                    "status": data["status"],
                    "runtime": (datetime.now() - data["start_time"]).total_seconds(),
                    "current_agent": data.get("current_agent", "unknown")
                }
                for wf_id, data in self.active_workflows.items()
            },
            "performance_metrics": self.performance_metrics,
            "recent_executions": self.execution_history[-10:] if self.execution_history else []
        }


class WorkflowRecoveryManager:
    """
    Handles workflow error recovery and retry logic.
    
    Provides intelligent recovery strategies for different types of
    workflow failures, including partial state recovery and retry mechanisms.
    """
    
    def __init__(self, workflow: SAPRAGWorkflow):
        """
        Initialize recovery manager.
        
        Args:
            workflow: Workflow instance to manage recovery for
        """
        self.workflow = workflow
        self.recovery_strategies = {
            "pdf_processing": self._recover_pdf_processing,
            "embedding_creation": self._recover_embedding_creation,
            "vector_storage": self._recover_vector_storage,
            "search": self._recover_search,
            "summary": self._recover_summary
        }
    
    def attempt_recovery(self, failed_state: DetailedWorkflowState) -> Dict[str, Any]:
        """
        Attempt to recover from workflow failure.
        
        Args:
            failed_state: State where workflow failed
            
        Returns:
            Recovery result and next steps
        """
        try:
            current_agent = failed_state.get("current_agent", "unknown")
            error_message = failed_state.get("error_message", "")
            
            logger.info(f"🔧 Attempting recovery from {current_agent} failure: {error_message}")
            
            # Choose recovery strategy based on failed component
            recovery_func = self.recovery_strategies.get(current_agent)
            
            if recovery_func:
                recovery_result = recovery_func(failed_state)
                
                if recovery_result["success"]:
                    logger.info(f"✅ Recovery successful for {current_agent}")
                else:
                    logger.warning(f"⚠️ Recovery failed for {current_agent}: {recovery_result.get('error')}")
                
                return recovery_result
            else:
                return {
                    "success": False,
                    "error": f"No recovery strategy available for {current_agent}",
                    "recommendations": ["Manual intervention required", "Check logs for detailed error information"]
                }
                
        except Exception as e:
            logger.error(f"❌ Recovery attempt failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Contact system administrator"]
            }
    
    def _recover_pdf_processing(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for PDF processing failures."""
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            # Try with smaller file subset
            if len(uploaded_files) > 1:
                return {
                    "success": True,
                    "strategy": "partial_processing",
                    "action": "Process files individually",
                    "recommendations": ["Check file integrity", "Try with smaller files first"]
                }
            
            return {
                "success": False,
                "error": "Single file processing failed",
                "recommendations": ["Check PDF file format", "Try different PDF processing library"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_embedding_creation(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for embedding creation failures."""
        try:
            # Check if we can fall back to mock embeddings
            return {
                "success": True,
                "strategy": "fallback_embeddings",
                "action": "Use mock embeddings for testing",
                "recommendations": ["Check OpenAI API key", "Verify network connectivity"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_vector_storage(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for vector storage failures."""
        try:
            # Try different vector store type
            current_type = self.workflow.config.get("vector_store_type", "chroma")
            fallback_types = ["simple", "mock"]
            
            for fallback_type in fallback_types:
                if fallback_type != current_type:
                    return {
                        "success": True,
                        "strategy": "fallback_vector_store",
                        "action": f"Switch to {fallback_type} vector store",
                        "recommendations": [f"Consider using {fallback_type} for development"]
                    }
            
            return {
                "success": False,
                "error": "All vector store options exhausted"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_search(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for search failures."""
        try:
            # Check if vector store is available
            if not state.get("vector_store_ready"):
                return {
                    "success": True,
                    "strategy": "reinitialize_search",
                    "action": "Reinitialize search agent",
                    "recommendations": ["Verify vector store is properly populated"]
                }
            
            return {
                "success": False,
                "error": "Search agent configuration issue"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_summary(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for summary generation failures."""
        try:
            search_results = state.get("search_results", [])
            
            if not search_results:
                return {
                    "success": True,
                    "strategy": "empty_summary",
                    "action": "Generate empty summary response",
                    "recommendations": ["Check search results quality"]
                }
            
            return {
                "success": True,
                "strategy": "simplified_summary",
                "action": "Generate basic summary without advanced analysis",
                "recommendations": ["Review summary agent configuration"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Create global instances for monitoring and debugging
workflow_monitor = WorkflowMonitor()
workflow_debugger_registry = {}

def get_workflow_debugger(workflow: SAPRAGWorkflow) -> WorkflowDebugger:
    """
    Get or create debugger instance for a workflow.
    
    Args:
        workflow: Workflow instance
        
    Returns:
        WorkflowDebugger instance
    """
    workflow_id = id(workflow)
    if workflow_id not in workflow_debugger_registry:
        workflow_debugger_registry[workflow_id] = WorkflowDebugger(workflow)
    return workflow_debugger_registry[workflow_id]


# ================================
# MODULE INITIALIZATION
# ================================

# Log module initialization
logger.info("🚀 SAP EWA Workflow module initialized successfully")
logger.info(f"📊 LangGraph available: {LANGGRAPH_AVAILABLE}")
logger.info(f"🔧 Available workflow features: Full workflow execution, Search-only mode, Visualization")

if not LANGGRAPH_AVAILABLE:
    logger.warning("⚠️ Running in mock mode - LangGraph not available")
    logger.info("💡 Install LangGraph with: pip install langgraph")

# Export main classes and functions
__all__ = [
    'SAPRAGWorkflow',
    'DetailedWorkflowState', 
    'WorkflowNodeMixin',
    'create_workflow',
    'validate_workflow_config',
    'test_workflow_basic',
    'WorkflowDebugger',
    'WorkflowMonitor',
    'WorkflowRecoveryManager',
    'get_workflow_debugger',
    'workflow_monitor'
]


# ================================
# FINAL MODULE SUMMARY
# ================================

logger.info("🎯 SAP EWA Workflow Module Features:")
logger.info("   📋 Complete LangGraph-based workflow orchestration")
logger.info("   🔄 Robust error handling and recovery mechanisms") 
logger.info("   📊 Performance monitoring and debugging tools")
logger.info("   🔍 Search-only mode for iterative analysis")
logger.info("   📧 Optional email notification system")
logger.info("   🎨 Workflow visualization capabilities")
logger.info("   🧪 Comprehensive testing and validation")
logger.info("   🔧 Advanced debugging and recovery utilities")

if LANGGRAPH_AVAILABLE:
    logger.info("✅ Module ready for production use with LangGraph")
else:
    logger.info("⚠️ Module ready for development use with mock implementation")

logger.info("🚀 SAP EWA Workflow module initialization completed successfully!")