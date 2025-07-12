# workflow/core_workflow.py - Main Workflow Class
"""
Core workflow implementation for SAP Early Watch Analysis using LangGraph.

This module contains the main SAPRAGWorkflow class and state definitions.
It orchestrates the complete analysis pipeline from PDF processing through
to email notifications using LangGraph state management.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Literal, TypedDict, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# LangGraph imports with fallbacks
LANGGRAPH_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.graph import CompiledGraph
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ LangGraph not available: {e}")
    
    # Mock classes for development
    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}
            self.entry_point = None
        
        def add_node(self, name: str, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node: str, to_node: str):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
        
        def add_conditional_edges(self, from_node: str, condition, mapping: Dict[str, str]):
            self.conditional_edges[from_node] = {'condition': condition, 'mapping': mapping}
        
        def set_entry_point(self, node: str):
            self.entry_point = node
        
        def compile(self):
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph: StateGraph):
            self.graph = graph
            self.state_schema = graph.state_schema
        
        def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            current_state = initial_state.copy()
            current_state["workflow_status"] = "completed"
            return current_state
        
        def get_graph(self):
            class MockGraph:
                def draw_mermaid_png(self):
                    return b"Mock PNG data"
                def draw_mermaid(self):
                    return "graph TD\n    A[Start] --> B[End]"
            return MockGraph()
    
    END = "END"
    CompiledGraph = MockCompiledGraph

# Safe imports with fallbacks
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
    
    # Fallback classes
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
    
    def get_agent_config():
        return {"mock": True}

# Vector store import
try:
    from vector_store import ChromaVectorStore, VectorConfig
    logger.info("✅ ChromaVectorStore imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ ChromaVectorStore not available: {e}")
    
    class MockChromaVectorStore:
        def __init__(self, config=None):
            self.config = config
        
        def add_documents(self, documents, metadatas=None, ids=None):
            return True
        
        def similarity_search_with_score(self, query, k=10, filter=None):
            return []


# ================================
# WORKFLOW STATE DEFINITION
# ================================

class DetailedWorkflowState(TypedDict):
    """Complete state definition for the SAP EWA analysis workflow."""
    
    # Input parameters
    uploaded_files: List[Any]
    user_query: str
    search_filters: Dict[str, Any]
    email_recipients: List[str]
    
    # Workflow control
    workflow_status: str
    current_agent: str
    error_message: str
    
    # Processing results
    processed_documents: List[Any]
    embeddings: List[Any]
    total_chunks: int
    vector_store_ready: bool
    search_results: List[Any]
    summary: Dict[str, Any]
    system_summaries: Dict[str, Any]
    
    # Communication
    email_sent: bool
    
    # Metrics and monitoring
    processing_times: Dict[str, float]
    agent_messages: List[Dict[str, str]]
    
    # Configuration
    config: Dict[str, Any]


# ================================
# WORKFLOW NODE MIXIN
# ================================

class WorkflowNodeMixin:
    """Mixin providing common functionality for workflow nodes."""
    
    def start_node_timer(self, state: DetailedWorkflowState, node_name: str) -> float:
        """Start timing for a workflow node."""
        start_time = time.time()
        self._add_agent_message(state, node_name, f"Starting {node_name}...", "processing")
        return start_time
    
    def end_node_timer(self, state: DetailedWorkflowState, node_name: str, 
                      start_time: float, success: bool = True) -> float:
        """End timing and record duration."""
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
        """Add a message to workflow state."""
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
        """Standardized error handling for workflow nodes."""
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
    through to email notifications with robust state management and error handling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the SAP RAG workflow with configuration."""
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
    
    def _initialize_agents(self):
        """Initialize all workflow agents with error handling."""
        try:
            logger.info("🔧 Initializing workflow agents...")
            
            # Core processing agents
            self.pdf_processor = PDFProcessorAgent(self.agent_config)
            self.embedding_agent = EmbeddingAgent(self.agent_config)
            self.summary_agent = SummaryAgent(self.agent_config)
            self.system_output_agent = SystemOutputAgent(self.agent_config)
            
            # Optional email agent
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
            self._create_fallback_agents()
    
    def _create_fallback_agents(self):
        """Create minimal fallback agents for testing."""
        logger.warning("🔄 Creating fallback agents for testing")
    
    def _initialize_vector_store_manager(self):
        """Initialize the ChromaDB vector store."""
        try:
            collection_name = self.config.get('collection_name', 'sap_documents')
            config = VectorConfig(collection_name=collection_name)
            self.vector_store = ChromaVectorStore(config)
            
            logger.info(f"🗄️ ChromaDB vector store initialized: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB vector store initialization failed: {e}")
            self.vector_store = MockChromaVectorStore()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow definition."""
        logger.info("📊 Building workflow graph...")
        
        # Import workflow nodes
        from .workflow_nodes import WorkflowNodes
        
        # Create the state graph
        workflow = StateGraph(DetailedWorkflowState)
        nodes = WorkflowNodes(self)
        
        # Add all workflow nodes
        workflow.add_node("pdf_processor", nodes.pdf_processing_node)
        workflow.add_node("embedding_creator", nodes.embedding_creation_node)
        workflow.add_node("vector_storage", nodes.vector_storage_node)
        workflow.add_node("search_agent", nodes.search_node)
        workflow.add_node("summary_agent", nodes.summary_node)
        workflow.add_node("system_output_agent", nodes.system_output_node)
        workflow.add_node("email_agent", nodes.email_node)
        workflow.add_node("complete", nodes.completion_node)
        
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
            nodes.route_after_vector_storage,
            {"search": "search_agent", "complete": "complete"}
        )
        
        workflow.add_conditional_edges(
            "system_output_agent",
            nodes.route_after_system_output,
            {"send_email": "email_agent", "complete": "complete"}
        )
        
        logger.info("✅ Workflow graph built with 8 nodes and conditional routing")
        return workflow
    
    def _compile_workflow(self) -> Union[CompiledGraph, Any]:
        """Compile the workflow graph into an executable application."""
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
            return MockCompiledGraph(self.workflow_graph)
    
    def _create_initial_state(self, **kwargs) -> DetailedWorkflowState:
        """Create initial workflow state with provided parameters."""
        user_query = kwargs.get("user_query", "")
        if user_query and isinstance(user_query, str):
            user_query = user_query.strip()
        
        initial_state = DetailedWorkflowState(
            # Input parameters
            uploaded_files=kwargs.get("uploaded_files", []),
            user_query=user_query,
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
        """Execute the complete SAP EWA analysis workflow."""
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
        """Execute search-only workflow on existing vector store."""
        try:
            # Fixed query validation
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
            
            # Create search state
            search_state = self._create_initial_state(
                user_query=cleaned_query,
                search_filters=search_filters or {}
            )
            
            # Convert TypedDict to regular dict
            search_state = dict(search_state)
            search_state["vector_store_ready"] = True
            
            # Import and execute search nodes
            from .workflow_nodes import WorkflowNodes
            nodes = WorkflowNodes(self)
            
            try:
                # Execute search and analysis nodes in sequence
                search_state = nodes.search_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                search_state = nodes.summary_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                search_state = nodes.system_output_node(search_state)
                if search_state["workflow_status"] == WorkflowStatus.ERROR:
                    return dict(search_state)
                
                # Optional email
                if (self.config.get("email_enabled") and 
                    self.config.get("auto_send_results") and
                    search_state.get("email_recipients")):
                    search_state = nodes.email_node(search_state)
                
                # Complete
                search_state = nodes.completion_node(search_state)
                
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
        """Get current workflow status and statistics."""
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
            return {"workflow_ready": False, "error": str(e)}
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Generate workflow visualization data."""
        try:
            if not LANGGRAPH_AVAILABLE or not hasattr(self.app, 'get_graph'):
                return {
                    "success": False,
                    "error": "LangGraph visualization not available",
                    "fallback_description": "PDF → Embedding → Vector Store → Search → Summary → System Output → Email → Complete"
                }
            
            try:
                # Try PNG generation
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
            return {"success": False, "error": str(e)}
    
    def reset_workflow(self):
        """Reset workflow state for fresh execution."""
        try:
            logger.info("🔄 Resetting workflow state")
            
            # Clear search agent
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