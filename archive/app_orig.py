"""
SAP Early Watch Analyzer - Main Streamlit Application
====================================================

This is the main Streamlit application for the SAP Early Watch Analyzer (EWA).
It provides a comprehensive web interface for analyzing SAP system health reports
using AI-powered document processing and intelligent search capabilities.

Key Features:
- Document upload and processing (PDF files)
- Vector-based similarity search using ChromaDB
- LangGraph workflow orchestration
- System-specific health analysis
- Email notification system (Gmail/Outlook support)
- Real-time processing status
- Interactive search interface
- Comprehensive error handling and recovery

Architecture:
- Frontend: Streamlit web interface
- Backend: LangGraph workflow with specialized agents
- Vector Store: ChromaDB for document embeddings
- AI Models: OpenAI GPT for analysis and embeddings
- Email: SMTP integration for notifications (Gmail/Outlook)

Author: Your Team
Date: 2025
Version: 1.0 (Production Ready)
"""

import streamlit as st
import os
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import tempfile
from pathlib import Path

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sap_ewa_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# SAFE IMPORTS WITH FALLBACKS
# ================================

try:
    from workflow import SAPRAGWorkflow, create_workflow, validate_workflow_config
    from config import get_config, get_agent_config, is_email_enabled
    from models import WorkflowStatus, HealthStatus, EmailRecipient
    from vector_store import create_vector_store
    MODULES_AVAILABLE = True
    logger.info("✅ All custom modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import custom modules: {e}")
    MODULES_AVAILABLE = False
    
    # Create minimal fallback classes for development
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
    
    class HealthStatus:
        HEALTHY = "HEALTHY"
        WARNING = "WARNING"
        CRITICAL = "CRITICAL"
        UNKNOWN = "UNKNOWN"

# Email functionality imports
try:
    import smtplib
    import ssl
    from email.message import EmailMessage
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.warning("⚠️ Email functionality not available")

# Environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    logger.warning("⚠️ dotenv not available - using system environment variables")

# ================================
# APPLICATION CONSTANTS
# ================================

APP_TITLE = "SAP Early Watch Analyzer"
APP_ICON = "📊"
APP_VERSION = "1.0"
APP_DESCRIPTION = "AI-Powered SAP System Health Analysis with LangGraph"

# File upload constraints
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = ['.pdf']
MAX_FILES_PER_UPLOAD = 10

# UI Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 10
DEFAULT_TEMPERATURE = 0.1

# Status messages
STATUS_MESSAGES = {
    WorkflowStatus.INITIALIZED: "🔄 Workflow initialized",
    WorkflowStatus.PROCESSING_PDF: "📄 Processing PDF files",
    WorkflowStatus.CREATING_EMBEDDINGS: "🔤 Creating embeddings",
    WorkflowStatus.STORING_VECTORS: "🗄️ Storing in vector database",
    WorkflowStatus.SEARCHING: "🔍 Searching documents",
    WorkflowStatus.SUMMARIZING: "📝 Generating summary",
    WorkflowStatus.SYSTEM_OUTPUT: "🖥️ Analyzing systems",
    WorkflowStatus.SENDING_EMAIL: "📧 Sending notifications",
    WorkflowStatus.COMPLETED: "✅ Analysis completed",
    WorkflowStatus.ERROR: "❌ Error occurred"
}

# Quick search templates
QUICK_SEARCH_TEMPLATES = [
    {
        "label": "🚨 Critical Alerts",
        "query": "Show me all critical alerts and errors in the systems",
        "description": "Find urgent issues requiring immediate attention"
    },
    {
        "label": "📈 Performance Issues", 
        "query": "Analyze performance bottlenecks and slow queries",
        "description": "Identify system performance problems"
    },
    {
        "label": "💡 Recommendations",
        "query": "What are the top recommendations for optimization",
        "description": "Get actionable improvement suggestions"
    },
    {
        "label": "🔧 System Health",
        "query": "Overall system health status and metrics",
        "description": "Get comprehensive health overview"
    },
    {
        "label": "💾 Memory Issues",
        "query": "Find memory usage problems and memory leaks",
        "description": "Analyze memory-related issues"
    },
    {
        "label": "🗄️ Database Problems",
        "query": "Show database performance issues and errors",
        "description": "Focus on database-specific problems"
    }
]


# ================================
# SESSION STATE MANAGEMENT
# ================================

class SessionStateManager:
    """
    Centralized session state management for the Streamlit application.
    
    This class provides a clean interface for managing all session state
    variables with proper initialization, validation, and type safety.
    """
    
    # Default values for all session state variables
    DEFAULT_VALUES = {
        # Workflow state
        'workflow': None,
        'workflow_initialized': False,
        'workflow_status': WorkflowStatus.INITIALIZED,
        'processing_status': 'ready',
        'current_agent': '',
        'error_message': '',
        
        # Document processing
        'uploaded_files': [],
        'processed_documents': [],
        'vector_store_ready': False,
        'total_chunks': 0,
        
        # Search and analysis
        'selected_systems': [],
        'search_query': '',
        'last_search_query': '',
        'search_results': None,
        'search_filters': {},
        
        # Results and summaries
        'summary': {},
        'system_summaries': {},
        'critical_findings': [],
        'recommendations': [],
        
        # Email configuration
        'email_enabled': False,
        'email_recipients': [],
        'email_sent': False,
        'email_config_checked': False,
        
        # UI state
        'current_tab': 'upload',
        'show_advanced_options': False,
        'debug_mode': False,
        
        # Configuration
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
        'top_k': DEFAULT_TOP_K,
        'temperature': DEFAULT_TEMPERATURE,
        
        # Performance metrics
        'processing_times': {},
        'agent_messages': [],
        'execution_count': 0,
        'last_execution_time': None,
        
        # Timestamps
        'session_start_time': datetime.now(),
        'last_activity_time': datetime.now()
    }
    
    @classmethod
    def initialize(cls):
        """
        Initialize all session state variables with default values.
        
        This method ensures that all required session state variables exist
        with appropriate default values, preventing KeyError exceptions.
        """
        logger.info("🔄 Initializing session state")
        
        initialized_count = 0
        for key, default_value in cls.DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                initialized_count += 1
        
        # Update activity time
        st.session_state.last_activity_time = datetime.now()
        
        if initialized_count > 0:
            logger.info(f"✅ Initialized {initialized_count} session state variables")
        
        # Validate critical state
        cls.validate_state()
    
    @classmethod
    def validate_state(cls):
        """
        Validate session state for consistency and fix any issues.
        
        Performs sanity checks on session state variables and corrects
        any inconsistencies or invalid values.
        """
        try:
            # Ensure workflow status is valid
            if st.session_state.workflow_status not in [
                WorkflowStatus.INITIALIZED, WorkflowStatus.PROCESSING_PDF,
                WorkflowStatus.CREATING_EMBEDDINGS, WorkflowStatus.STORING_VECTORS,
                WorkflowStatus.SEARCHING, WorkflowStatus.SUMMARIZING,
                WorkflowStatus.SYSTEM_OUTPUT, WorkflowStatus.SENDING_EMAIL,
                WorkflowStatus.COMPLETED, WorkflowStatus.ERROR
            ]:
                st.session_state.workflow_status = WorkflowStatus.INITIALIZED
                logger.warning("⚠️ Fixed invalid workflow status")
            
            # Ensure lists are actually lists
            list_fields = ['uploaded_files', 'selected_systems', 'email_recipients', 
                          'critical_findings', 'recommendations', 'agent_messages']
            for field in list_fields:
                if not isinstance(st.session_state.get(field), list):
                    st.session_state[field] = []
                    logger.warning(f"⚠️ Fixed invalid list field: {field}")
            
            # Ensure dictionaries are actually dictionaries
            dict_fields = ['search_filters', 'summary', 'system_summaries', 'processing_times']
            for field in dict_fields:
                if not isinstance(st.session_state.get(field), dict):
                    st.session_state[field] = {}
                    logger.warning(f"⚠️ Fixed invalid dict field: {field}")
            
            # Ensure numeric fields are valid
            numeric_fields = {
                'chunk_size': (100, 4000, DEFAULT_CHUNK_SIZE),
                'chunk_overlap': (0, 1000, DEFAULT_CHUNK_OVERLAP),
                'top_k': (1, 100, DEFAULT_TOP_K),
                'temperature': (0.0, 2.0, DEFAULT_TEMPERATURE)
            }
            
            for field, (min_val, max_val, default) in numeric_fields.items():
                value = st.session_state.get(field, default)
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    st.session_state[field] = default
                    logger.warning(f"⚠️ Fixed invalid numeric field: {field}")
            
            logger.debug("✅ Session state validation completed")
            
        except Exception as e:
            logger.error(f"❌ Session state validation failed: {e}")
    
    @classmethod
    def reset(cls, preserve_config: bool = True):
        """
        Reset session state to initial values.
        
        Args:
            preserve_config: Whether to preserve user configuration settings
        """
        logger.info("🔄 Resetting session state")
        
        # Preserve certain values if requested
        preserved_values = {}
        if preserve_config:
            config_keys = ['chunk_size', 'chunk_overlap', 'top_k', 'temperature', 
                          'debug_mode', 'show_advanced_options']
            preserved_values = {k: st.session_state.get(k) for k in config_keys 
                              if k in st.session_state}
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize with defaults
        cls.initialize()
        
        # Restore preserved values
        for key, value in preserved_values.items():
            st.session_state[key] = value
        
        logger.info("✅ Session state reset completed")
    
    @classmethod
    def get_status_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current session state for debugging and monitoring.
        
        Returns:
            Dictionary containing key session state information
        """
        return {
            "workflow_initialized": st.session_state.get('workflow_initialized', False),
            "workflow_status": st.session_state.get('workflow_status', 'unknown'),
            "vector_store_ready": st.session_state.get('vector_store_ready', False),
            "documents_count": len(st.session_state.get('processed_documents', [])),
            "selected_systems_count": len(st.session_state.get('selected_systems', [])),
            "has_search_results": st.session_state.get('search_results') is not None,
            "email_enabled": st.session_state.get('email_enabled', False),
            "session_duration": (datetime.now() - st.session_state.get('session_start_time', datetime.now())).total_seconds(),
            "execution_count": st.session_state.get('execution_count', 0)
        }


# ================================
# WORKFLOW MANAGER
# ================================

class WorkflowManager:
    """
    Manages the SAP RAG workflow lifecycle and integration with Streamlit.
    
    This class handles workflow initialization, execution, monitoring,
    and error recovery. It provides a clean interface between the Streamlit
    UI and the underlying LangGraph workflow.
    """
    
    def __init__(self):
        """Initialize the workflow manager."""
        self.workflow = None
        self.config = None
        self.last_error = None
        
    def initialize_workflow(self) -> tuple[bool, str]:
        """
        Initialize the SAP RAG workflow with current configuration.
        
        Creates and configures the workflow instance based on session state
        and environment variables. Handles fallbacks for missing dependencies.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info("🔧 Initializing SAP RAG workflow")
            
            if not MODULES_AVAILABLE:
                return False, "Required modules not available. Check imports."
            
            # Load configuration
            try:
                self.config = self._build_workflow_config()
                logger.info(f"📋 Configuration loaded: {list(self.config.keys())}")
            except Exception as config_error:
                logger.error(f"❌ Configuration loading failed: {config_error}")
                return False, f"Configuration error: {str(config_error)}"
            
            # Validate configuration
            validation_result = validate_workflow_config(self.config)
            if not validation_result.get('valid', True):
                errors = validation_result.get('errors', [])
                return False, f"Configuration validation failed: {'; '.join(errors)}"
            
            # Create workflow
            try:
                self.workflow = create_workflow(self.config)
                st.session_state.workflow = self.workflow
                st.session_state.workflow_initialized = True
                
                logger.info("✅ Workflow initialized successfully")
                return True, "Workflow initialized successfully"
                
            except Exception as workflow_error:
                logger.error(f"❌ Workflow creation failed: {workflow_error}")
                return False, f"Workflow creation failed: {str(workflow_error)}"
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ Workflow initialization failed: {e}")
            return False, f"Initialization failed: {str(e)}"
    
    def _build_workflow_config(self) -> Dict[str, Any]:
        """
        Build workflow configuration from session state and environment.
        
        Returns:
            Configuration dictionary for the workflow
        """
        config = {
            # Core workflow settings
            "embedding_type": "openai",
            "vector_store_type": "chroma",
            "llm_provider": "openai",
            "llm_model": "gpt-4-turbo-preview",
            
            # Processing parameters
            "chunk_size": st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
            "chunk_overlap": st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
            "top_k": st.session_state.get('top_k', DEFAULT_TOP_K),
            "temperature": st.session_state.get('temperature', DEFAULT_TEMPERATURE),
            
            # Email configuration
            "email_enabled": self._check_email_configuration(),
            "auto_send_results": False,  # Manual send only in Streamlit
            
            # Performance settings
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "batch_size": 10,
            "timeout": 300,
            "retry_attempts": 3,
            
            # Logging and debugging
            "debug_mode": st.session_state.get('debug_mode', False),
            "verbose_logging": True
        }
        
        # Add API keys from environment
        if ENV_AVAILABLE:
            config.update({
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "gmail_email": os.getenv("GMAIL_EMAIL"),
                "gmail_app_password": os.getenv("GMAIL_APP_PASSWORD"),
                # Support for Outlook email
                "outlook_email": os.getenv("OUTLOOK_EMAIL"),
                "outlook_password": os.getenv("OUTLOOK_PASSWORD"),
                "email_provider": os.getenv("EMAIL_PROVIDER", "gmail")  # gmail or outlook
            })
        
        # Add advanced options if enabled
        if st.session_state.get('show_advanced_options', False):
            config.update({
                "similarity_threshold": 0.7,
                "max_results": 100,
                "enable_system_detection": True,
                "enable_performance_metrics": True
            })
        
        return config
    
    def _check_email_configuration(self) -> bool:
        """
        Check if email is properly configured for both Gmail and Outlook.
        
        Returns:
            True if email can be sent, False otherwise
        """
        try:
            if not EMAIL_AVAILABLE:
                return False
            
            if not ENV_AVAILABLE:
                return False
            
            email_provider = os.getenv("EMAIL_PROVIDER", "gmail").lower()
            email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
            
            if not email_enabled:
                return False
            
            # Check Gmail configuration
            if email_provider == "gmail":
                gmail_email = os.getenv("GMAIL_EMAIL")
                gmail_password = os.getenv("GMAIL_APP_PASSWORD")
                is_configured = gmail_email and gmail_password
            
            # Check Outlook configuration
            elif email_provider == "outlook":
                outlook_email = os.getenv("OUTLOOK_EMAIL")
                outlook_password = os.getenv("OUTLOOK_PASSWORD")
                is_configured = outlook_email and outlook_password
            
            else:
                logger.warning(f"⚠️ Unknown email provider: {email_provider}")
                return False
            
            st.session_state.email_enabled = is_configured
            return is_configured
            
        except Exception as e:
            logger.warning(f"⚠️ Email configuration check failed: {e}")
            return False
    
    def process_documents(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        Process uploaded documents using the workflow.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Processing result dictionary
        """
        try:
            if not self.workflow:
                return {"success": False, "error": "Workflow not initialized"}
            
            if not uploaded_files:
                return {"success": False, "error": "No files provided"}
            
            logger.info(f"📄 Processing {len(uploaded_files)} documents")
            
            # Validate files
            validation_result = self._validate_uploaded_files(uploaded_files)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Prepare workflow parameters
            workflow_params = {
                "uploaded_files": uploaded_files,
                "user_query": "",  # Empty for processing-only mode
                "search_filters": {},
                "email_recipients": []
            }
            
            # Update status
            st.session_state.workflow_status = WorkflowStatus.PROCESSING_PDF
            st.session_state.processing_status = 'processing'
            
            # Execute workflow
            result = self.workflow.run_workflow(**workflow_params)
            
            # Process results
            if result.get("workflow_status") == WorkflowStatus.COMPLETED:
                st.session_state.vector_store_ready = True
                st.session_state.processed_documents = result.get("processed_documents", [])
                st.session_state.total_chunks = result.get("total_chunks", 0)
                st.session_state.processing_times = result.get("processing_times", {})
                st.session_state.processing_status = 'completed'
                st.session_state.execution_count = st.session_state.get('execution_count', 0) + 1
                
                logger.info("✅ Document processing completed successfully")
                
                return {
                    "success": True,
                    "message": "Documents processed successfully",
                    "total_chunks": result.get("total_chunks", 0),
                    "processing_times": result.get("processing_times", {}),
                    "workflow_result": result
                }
                
            else:
                error_msg = result.get("error_message", "Unknown processing error")
                st.session_state.workflow_status = WorkflowStatus.ERROR
                st.session_state.error_message = error_msg
                st.session_state.processing_status = 'error'
                
                logger.error(f"❌ Document processing failed: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "workflow_result": result
                }
            
        except Exception as e:
            error_msg = f"Document processing exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            st.session_state.workflow_status = WorkflowStatus.ERROR
            st.session_state.error_message = error_msg
            st.session_state.processing_status = 'error'
            
            return {"success": False, "error": error_msg}
    
    def execute_search(self, query: str, search_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute search using the workflow.
        
        Args:
            query: Search query string
            search_filters: Optional search filters
            
        Returns:
            Search result dictionary
        """
        try:
            if not self.workflow:
                return {"success": False, "error": "Workflow not initialized"}
            
            if not st.session_state.get('vector_store_ready', False):
                return {"success": False, "error": "Vector store not ready. Process documents first."}
            
            # FIX 1: Better query validation
            if not query or not query.strip():
                return {"success": False, "error": "Empty search query provided"}
            
            # Clean and validate the query
            cleaned_query = query.strip()
            if len(cleaned_query) < 3:
                return {"success": False, "error": "Search query too short (minimum 3 characters)"}
            
            logger.info(f"🔍 Executing search: '{cleaned_query}'")
            
            # Prepare search filters
            filters = search_filters or {}
            if st.session_state.get('selected_systems'):
                filters['target_systems'] = st.session_state.selected_systems
            
            # Update status
            st.session_state.workflow_status = WorkflowStatus.SEARCHING
            st.session_state.search_query = cleaned_query
            
            # FIX 2: Pass cleaned query to workflow
            result = self.workflow.run_search_only(query=cleaned_query, search_filters=filters)
            
            # Process results
            if result.get("workflow_status") == WorkflowStatus.COMPLETED:
                # Store results in session state
                summary = result.get("summary", {})
                system_summaries = result.get("system_summaries", {})
                
                search_results = {
                    "query": cleaned_query,  # Use cleaned query
                    "timestamp": datetime.now().isoformat(),
                    "selected_systems": st.session_state.get('selected_systems', []),
                    "summary": summary.get("summary", "Search completed"),
                    "critical_findings": summary.get("critical_findings", []),
                    "recommendations": summary.get("recommendations", []),
                    "performance_insights": summary.get("performance_insights", []),
                    "confidence_score": summary.get("confidence_score", 0.0),
                    "system_summaries": system_summaries,
                    "search_results_count": len(result.get("search_results", [])),
                    "processing_time": result.get("processing_time", 0.0)
                }
                
                # Update session state with results
                st.session_state.search_results = search_results
                st.session_state.summary = summary
                st.session_state.system_summaries = system_summaries
                st.session_state.critical_findings = summary.get("critical_findings", [])
                st.session_state.recommendations = summary.get("recommendations", [])
                st.session_state.last_search_query = cleaned_query
                st.session_state.last_execution_time = result.get("processing_time", 0.0)
                
                logger.info(f"✅ Search completed: {search_results['search_results_count']} results")
                
                return {
                    "success": True,
                    "search_results": search_results,
                    "workflow_result": result
                }
                
            else:
                error_msg = result.get("error_message", "Unknown search error")
                st.session_state.workflow_status = WorkflowStatus.ERROR
                st.session_state.error_message = error_msg
                
                logger.error(f"❌ Search failed: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "workflow_result": result
                }
            
        except Exception as e:
            error_msg = f"Search execution exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            st.session_state.workflow_status = WorkflowStatus.ERROR
            st.session_state.error_message = error_msg
            
            return {"success": False, "error": error_msg}
    
    def _validate_uploaded_files(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        Validate uploaded files for size and type constraints.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Validation result dictionary
        """
        try:
            if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
                return {
                    "valid": False,
                    "error": f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files allowed."
                }
            
            for file in uploaded_files:
                # Check file size
                file_size = len(file.getvalue())
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    return {
                        "valid": False,
                        "error": f"File {file.name} exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
                    }
                
                # Check file type
                file_extension = Path(file.name).suffix.lower()
                if file_extension not in SUPPORTED_FILE_TYPES:
                    return {
                        "valid": False,
                        "error": f"File {file.name} has unsupported type. Supported: {SUPPORTED_FILE_TYPES}"
                    }
                
                # Check if file is empty
                if file_size == 0:
                    return {
                        "valid": False,
                        "error": f"File {file.name} is empty"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"File validation error: {str(e)}"
            }


# ================================
# STREAMLIT PAGE CONFIGURATION
# ================================

def configure_page():
    """
    Configure Streamlit page settings and apply custom CSS styling.
    
    Sets up the page layout, theme, and custom CSS for a professional
    appearance with proper branding and responsive design.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/sap-ewa-analyzer',
            'Report a bug': 'https://github.com/your-repo/sap-ewa-analyzer/issues',
            'About': f"{APP_TITLE} v{APP_VERSION} - AI-Powered SAP Analysis Tool"
        }
    )
    
    # Apply custom CSS styling
    st.markdown("""
    <style>
        /* Main application styling */
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Status boxes */
        .status-box {
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 5px solid;
        }
        
        .success-box {
            background: #d4edda;
            border-left-color: #28a745;
            border: 1px solid #c3e6cb;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left-color: #ffc107;
            border: 1px solid #ffeeba;
        }
        
        .error-box {
            background: #f8d7da;
            border-left-color: #dc3545;
            border: 1px solid #f5c6cb;
        }
        
        .info-box {
            background: #d1ecf1;
            border-left-color: #17a2b8;
            border: 1px solid #bee5eb;
        }
        
        .processing-box {
            background: #e2e3e5;
            border-left-color: #6c757d;
            border: 1px solid #d6d8db;
        }
        
        /* Cards and containers */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 0.5rem 0;
        }
        
        .system-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin: 0.5rem 0;
        }
        
        /* Progress and status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-critical { background-color: #dc3545; }
        .status-unknown { background-color: #6c757d; }
        
        /* Button improvements */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .quick-search-button {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-search-button:hover {
            background: #e9ecef;
            border-color: #1f77b4;
            transform: translateY(-1px);
        }
        
        /* Sidebar styling */
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        
        /* Footer */
        .app-footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 3rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 { font-size: 1.8rem; }
            .main-header p { font-size: 1rem; }
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)






# ================================
# EMAIL MANAGER
# ================================

class EmailManager:
    """
    Manages email notifications with support for both Gmail and Outlook.
    
    This class handles email configuration, formatting, and sending
    with automatic provider detection and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the email manager."""
        self.provider = None
        self.config = {}
        self._detect_email_provider()
    
    def _detect_email_provider(self):
        """
        Detect and configure email provider from environment variables.
        """
        try:
            if not ENV_AVAILABLE or not EMAIL_AVAILABLE:
                return
            
            email_provider = os.getenv("EMAIL_PROVIDER", "gmail").lower()
            
            if email_provider == "gmail":
                self.provider = "gmail"
                self.config = {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "email": os.getenv("GMAIL_EMAIL"),
                    "password": os.getenv("GMAIL_APP_PASSWORD"),
                    "use_tls": True
                }
            
            elif email_provider == "outlook":
                self.provider = "outlook"
                self.config = {
                    "smtp_server": "smtp-mail.outlook.com",
                    "smtp_port": 587,
                    "email": os.getenv("OUTLOOK_EMAIL"),
                    "password": os.getenv("OUTLOOK_PASSWORD"),
                    "use_tls": True
                }
            
            # Validate configuration
            if self.config.get("email") and self.config.get("password"):
                st.session_state.email_enabled = True
                logger.info(f"✅ Email configured for {self.provider}")
            else:
                st.session_state.email_enabled = False
                logger.warning(f"⚠️ Email credentials missing for {self.provider}")
                
        except Exception as e:
            logger.error(f"❌ Email provider detection failed: {e}")
            st.session_state.email_enabled = False
    
    def send_email(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send email using the configured provider.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body content
            
        Returns:
            Result dictionary with success status and message
        """
        try:
            if not self.provider or not self.config.get("email"):
                return {"success": False, "error": "Email not configured"}
            
            logger.info(f"📧 Sending email via {self.provider} to {to_email}")
            
            # Create email message
            msg = EmailMessage()
            msg['From'] = self.config["email"]
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.set_content(body)
            
            # Send email using SMTP
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                if self.config["use_tls"]:
                    server.starttls(context=context)
                
                server.login(self.config["email"], self.config["password"])
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully via {self.provider}")
            return {"success": True, "message": f"Email sent successfully to {to_email}"}
            
        except Exception as e:
            error_msg = f"Email sending failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def format_analysis_email(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Format analysis results into email content.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dictionary with formatted subject and body
        """
        try:
            # Extract data
            query = results.get('query', 'SAP Analysis')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            systems = ', '.join(results.get('selected_systems', []))
            critical_count = len(results.get('critical_findings', []))
            
            # Format subject
            if critical_count > 0:
                urgency = "CRITICAL" if critical_count >= 3 else "ALERT"
                subject = f"[{urgency}] SAP EWA Analysis - {query}"
            else:
                subject = f"SAP EWA Analysis Results - {query}"
            
            # Format body
            body = f"""SAP Early Watch Analyzer - Analysis Results
{'=' * 50}

Query: {query}
Analysis Time: {timestamp}
Systems Analyzed: {systems}
Results Found: {results.get('search_results_count', 0)}

EXECUTIVE SUMMARY:
{results.get('summary', 'Analysis completed successfully')}

CRITICAL FINDINGS ({critical_count}):
"""
            
            # Add critical findings
            for i, finding in enumerate(results.get('critical_findings', []), 1):
                body += f"{i}. {finding}\n"
            
            if not results.get('critical_findings'):
                body += "✅ No critical issues found\n"
            
            # Add recommendations
            recommendations = results.get('recommendations', [])
            body += f"\nRECOMMENDATIONS ({len(recommendations)}):\n"
            
            for i, rec in enumerate(recommendations, 1):
                body += f"{i}. {rec}\n"
            
            if not recommendations:
                body += "ℹ️ No specific recommendations at this time\n"
            
            # Add system details if available
            system_summaries = results.get('system_summaries', {})
            if system_summaries:
                body += "\nSYSTEM DETAILS:\n"
                for sys_id, sys_data in system_summaries.items():
                    health = sys_data.get('overall_health', 'UNKNOWN')
                    alerts = len(sys_data.get('critical_alerts', []))
                    body += f"• {sys_id}: {health} ({alerts} alerts)\n"
            
            # Footer
            body += f"""
---
Generated by SAP EWA Analyzer v{APP_VERSION}
Report generated at: {timestamp}

This is an automated analysis report. For questions or concerns,
please contact your SAP BASIS team or system administrator.
"""
            
            return {"subject": subject, "body": body}
            
        except Exception as e:
            logger.error(f"❌ Email formatting failed: {e}")
            return {
                "subject": f"SAP Analysis Results - {results.get('query', 'Report')}",
                "body": f"SAP analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nPlease check the system for detailed results."
            }


# ================================
# UI COMPONENTS
# ================================

def create_main_header():
    """
    Create and display the main application header.
    
    Displays the application title, description, and branding
    with a professional gradient background.
    """
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
        <small>Version {APP_VERSION} | Powered by LangGraph & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)


def create_file_upload_section() -> List[Any]:
    """
    Create the file upload section with validation and preview.
    
    Returns:
        List of uploaded file objects
    """
    st.header("📁 Document Upload")
    
    # Initialize workflow if needed
    if not st.session_state.workflow_initialized:
        with st.spinner("Initializing workflow..."):
            workflow_manager = WorkflowManager()
            success, message = workflow_manager.initialize_workflow()
            
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                st.stop()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload SAP EWA PDF Reports",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"Upload up to {MAX_FILES_PER_UPLOAD} PDF files, max {MAX_FILE_SIZE_MB}MB each"
    )
    
    # Display file information
    if uploaded_files:
        total_size = sum(len(file.getvalue()) for file in uploaded_files)
        total_size_mb = total_size / 1024 / 1024
        
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ {len(uploaded_files)} files uploaded successfully</strong><br>
            Total size: {total_size_mb:.1f} MB
        </div>
        """, unsafe_allow_html=True)
        
        # File details in expander
        with st.expander("📋 File Details"):
            for i, file in enumerate(uploaded_files, 1):
                size_mb = len(file.getvalue()) / 1024 / 1024
                st.write(f"{i}. **{file.name}** ({size_mb:.1f} MB)")
    
    return uploaded_files


def create_processing_section(uploaded_files: List[Any]):
    """
    Create the document processing section with controls and status.
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    st.header("⚙️ Document Processing")
    
    if not uploaded_files:
        st.info("📄 Upload PDF files first to begin processing")
        return
    
    # Processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_button = st.button(
            "🚀 Process Documents", 
            type="primary", 
            use_container_width=True,
            help="Extract text and create vector embeddings"
        )
    
    with col2:
        # Status indicator
        if st.session_state.vector_store_ready:
            st.success("✅ Documents Ready")
        elif st.session_state.processing_status == 'processing':
            st.info("⏳ Processing...")
        elif st.session_state.processing_status == 'error':
            st.error("❌ Processing Failed")
        else:
            st.info("⏳ Not Processed")
    
    with col3:
        clear_button = st.button(
            "🗑️ Clear All", 
            use_container_width=True,
            help="Clear all processed data and start over"
        )
    
    # Handle button clicks
    if process_button:
        _handle_document_processing(uploaded_files)
    
    if clear_button:
        _handle_clear_data()
    
    # Show processing status and metrics
    if st.session_state.processing_status == 'completed' and st.session_state.vector_store_ready:
        _display_processing_metrics()


def _handle_document_processing(uploaded_files: List[Any]):
    """
    Handle the document processing workflow execution.
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    try:
        workflow_manager = WorkflowManager()
        
        # Create progress indicators
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing documents..."):
                # Start processing
                status_text.text("🔄 Initializing workflow...")
                progress_bar.progress(10)
                
                # Initialize the workflow first
                success, message = workflow_manager.initialize_workflow()
                if not success:
                    progress_bar.progress(0)
                    status_text.text(f"❌ Error: {message}")
                    st.error(f"❌ Workflow initialization failed: {message}")
                    return
                
                progress_bar.progress(30)
                status_text.text("📄 Processing documents...")
                
                result = workflow_manager.process_documents(uploaded_files)
                
                if result["success"]:
                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")
                    
                    # Show success message with metrics
                    processing_time = sum(result.get("processing_times", {}).values())
                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} files "
                        f"in {processing_time:.2f} seconds"
                    )
                    
                    # Rerun to update UI
                    time.sleep(1)
                    st.rerun()
                    
                else:
                    progress_bar.progress(0)
                    error_msg = result.get("error", "Unknown error")
                    status_text.text(f"❌ Error: {error_msg}")
                    st.error(f"❌ Processing failed: {error_msg}")
                    
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())


def _handle_clear_data():
    """Handle clearing all processed data."""
    try:
        # Reset session state
        SessionStateManager.reset(preserve_config=True)
        st.success("🗑️ All data cleared successfully")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing data: {str(e)}")


def _display_processing_metrics():
    """Display processing metrics and statistics."""
    processing_times = st.session_state.get('processing_times', {})
    total_chunks = st.session_state.get('total_chunks', 0)
    
    if processing_times:
        with st.expander("📊 Processing Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Documents", total_chunks)
                st.metric("Processing Time", f"{sum(processing_times.values()):.2f}s")
            
            with col2:
                if processing_times:
                    slowest_step = max(processing_times.items(), key=lambda x: x[1])
                    st.metric("Slowest Step", slowest_step[0])
                    st.metric("Duration", f"{slowest_step[1]:.2f}s")
            
            # Processing timeline
            st.subheader("Processing Timeline")
            for step, duration in processing_times.items():
                st.write(f"• **{step}**: {duration:.2f}s")


def create_system_selection_section():
    """
    Create the system selection interface.
    
    Allows users to specify which SAP systems to analyze.
    """
    if not st.session_state.vector_store_ready:
        return
    
    st.header("🎯 System Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        systems_input = st.text_input(
            "SAP System IDs (comma-separated)",
            value=", ".join(st.session_state.selected_systems),
            placeholder="P01, Q01, DEV, QAS, PRD",
            help="Enter the SAP system IDs you want to analyze"
        )
    
    with col2:
        if st.button("🔍 Auto-Detect", use_container_width=True):
            _handle_system_detection()
    
    # Update selected systems
    if systems_input:
        systems = [s.strip().upper() for s in systems_input.split(',') if s.strip()]
        st.session_state.selected_systems = systems
        
        if systems:
            st.markdown(f"""
            <div class="success-box">
                <strong>🖥️ Selected Systems ({len(systems)}):</strong> {', '.join(systems)}
            </div>
            """, unsafe_allow_html=True)


def _handle_system_detection():
    """Handle automatic system detection from processed documents."""
    try:
        with st.spinner("Detecting SAP systems..."):
            # This would use the search agent to detect systems
            # For now, using mock detection
            detected_systems = ["PRD", "QAS", "DEV"]  # Mock systems
            
            st.session_state.selected_systems = detected_systems
            st.success(f"✅ Detected systems: {', '.join(detected_systems)}")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ System detection failed: {str(e)}")


def create_search_section():
    """
    Create the search and analysis interface.
    
    Provides quick search templates and custom search functionality.
    """
    if not st.session_state.vector_store_ready:
        return
    
    if not st.session_state.selected_systems:
        st.warning("⚠️ Please select SAP systems first")
        return
    
    st.header("🔍 Search & Analysis")
    
    # Quick search buttons
    st.subheader("🚀 Quick Searches")
    
    # Create grid of quick search buttons
    cols = st.columns(3)
    for i, template in enumerate(QUICK_SEARCH_TEMPLATES):
        col_idx = i % 3
        
        with cols[col_idx]:
            if st.button(
                template["label"], 
                use_container_width=True,
                help=template["description"]
            ):
                _handle_search_execution(template["query"])
    
    # Custom search
    st.subheader("✏️ Custom Search")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question",
            value=st.session_state.last_search_query,
            placeholder="What are the memory usage issues in the production system?",
            help="Ask specific questions about your SAP systems"
        )
    
    with col2:
        search_button = st.button(
            "🔍 Search", 
            type="primary", 
            use_container_width=True
        )
    
    # Handle search execution
    if search_button and query.strip():
        _handle_search_execution(query)
    elif search_button:
        st.error("❌ Please enter a search query")


def _handle_search_execution(query: str):
    """
    Handle search execution with progress tracking.
    
    Args:
        query: Search query string
    """
    try:
        # FIX 7: Better query validation before execution
        if not query or not query.strip():
            st.error("❌ Please enter a valid search query")
            return
            
        cleaned_query = query.strip()
        if len(cleaned_query) < 3:
            st.error("❌ Search query must be at least 3 characters long")
            return
        
        workflow_manager = WorkflowManager()
        
        with st.spinner(f"🔍 Searching: '{cleaned_query}'..."):
            # Prepare search filters
            search_filters = {
                "target_systems": st.session_state.selected_systems
            }
            
            # Execute search with cleaned query
            result = workflow_manager.execute_search(cleaned_query, search_filters)
            
            if result["success"]:
                search_results = result["search_results"]
                st.success(
                    f"✅ Search completed! Found {search_results['search_results_count']} results"
                )
                st.rerun()
                
            else:
                st.error(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"❌ Search error: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())


def create_results_section():
    """
    Create the results display section with tabs and analysis.
    
    Shows search results, critical findings, recommendations, and system details.
    """
    if not st.session_state.get('search_results'):
        return
    
    st.header("📊 Analysis Results")
    
    results = st.session_state.search_results
    
    # Results header with query info
    st.markdown(f"""
    <div class="info-box">
        <h4>🔍 Query: "{results['query']}"</h4>
        <strong>Systems:</strong> {', '.join(results['selected_systems'])} • 
        <strong>Results:</strong> {results.get('search_results_count', 0)} • 
        <strong>Confidence:</strong> {results.get('confidence_score', 0) * 100:.1f}% •
        <strong>Time:</strong> {datetime.fromisoformat(results['timestamp']).strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", 
        "🚨 Critical Issues", 
        "💡 Recommendations", 
        "🖥️ System Details",
        "📧 Actions"
    ])
    
    with tab1:
        _display_summary_tab(results)
    
    with tab2:
        _display_critical_issues_tab(results)
    
    with tab3:
        _display_recommendations_tab(results)
    
    with tab4:
        _display_system_details_tab(results)
    
    with tab5:
        _display_actions_tab(results)


def _display_summary_tab(results: Dict[str, Any]):
    """Display the summary tab content."""
    summary = results.get('summary', 'No summary available')
    confidence = results.get('confidence_score', 0) * 100
    
    # Confidence indicator
    if confidence >= 80:
        confidence_color = "success"
        confidence_icon = "🟢"
    elif confidence >= 60:
        confidence_color = "warning"
        confidence_icon = "🟡"
    else:
        confidence_color = "error"
        confidence_icon = "🔴"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(summary)
    
    with col2:
        st.metric(
            "Confidence Score",
            f"{confidence:.1f}%",
            help="AI confidence in the analysis results"
        )
        st.write(f"{confidence_icon} {confidence_color.title()} Confidence")
    
    # Performance insights if available
    insights = results.get('performance_insights', [])
    if insights:
        st.subheader("⚡ Performance Insights")
        for insight in insights:
            st.write(f"• {insight}")


def _display_critical_issues_tab(results: Dict[str, Any]):
    """Display the critical issues tab content."""
    critical_findings = results.get('critical_findings', [])
    
    if critical_findings:
        st.error(f"🚨 Found {len(critical_findings)} critical issues requiring attention")
        
        for i, finding in enumerate(critical_findings, 1):
            st.markdown(f"""
            <div class="error-box">
                <strong>🚨 Critical Alert #{i}:</strong><br>
                {finding}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical issues found in the analyzed systems")
        st.balloons()


def _display_recommendations_tab(results: Dict[str, Any]):
    """Display the recommendations tab content."""
    recommendations = results.get('recommendations', [])
    
    if recommendations:
        st.info(f"💡 Found {len(recommendations)} optimization recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="warning-box">
                <strong>💡 Recommendation #{i}:</strong><br>
                {rec}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No specific recommendations available at this time")


def _display_system_details_tab(results: Dict[str, Any]):
    """Display the system details tab content."""
    system_summaries = results.get('system_summaries', {})
    
    if system_summaries:
        for sys_id, sys_data in system_summaries.items():
            # Health status indicator
            health = sys_data.get('overall_health', 'UNKNOWN')
            health_icons = {
                'HEALTHY': '🟢',
                'WARNING': '🟡', 
                'CRITICAL': '🔴',
                'UNKNOWN': '⚪'
            }
            health_icon = health_icons.get(health, '⚪')
            
            with st.expander(f"{health_icon} System {sys_id} - {health}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**System Health**")
                    st.write(f"Status: {health_icon} {health}")
                    st.write(f"Last Analyzed: {sys_data.get('last_analyzed', 'Unknown')}")
                    
                    # Key metrics
                    metrics = sys_data.get('key_metrics', {})
                    if metrics:
                        st.markdown("**Key Metrics**")
                        for metric, value in metrics.items():
                            st.write(f"• {metric}: {value}")
                
                with col2:
                    # Critical alerts
                    alerts = sys_data.get('critical_alerts', [])
                    st.markdown(f"**Critical Alerts ({len(alerts)})**")
                    
                    if alerts:
                        for alert in alerts[:3]:  # Show max 3
                            st.write(f"• {alert}")
                        if len(alerts) > 3:
                            st.write(f"• ... and {len(alerts) - 3} more")
                    else:
                        st.write("• No critical alerts")
    else:
        st.info("ℹ️ No system-specific details available")


def _display_actions_tab(results: Dict[str, Any]):
    """Display the actions tab with email and export options."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📧 Email Report")
        
        # Check email configuration
        email_manager = EmailManager()
        
        if st.session_state.email_enabled:
            st.success(f"✅ Email ready ({email_manager.provider})")
            
            with st.form("email_form"):
                to_email = st.text_input(
                    "Send to:",
                    placeholder="recipient@company.com",
                    help="Enter recipient's email address"
                )
                
                include_details = st.checkbox(
                    "Include detailed system analysis", 
                    value=True
                )
                
                send_button = st.form_submit_button(
                    "📧 Send Email",
                    type="primary",
                    use_container_width=True
                )
                
                if send_button and to_email:
                    _handle_email_sending(email_manager, to_email, results)
                elif send_button:
                    st.error("❌ Please enter an email address")
        else:
            st.error("❌ Email not configured")
            st.info("Configure email in your .env file")
    
    with col2:
        st.subheader("📄 Export Options")
        
        # Export to text file
        if st.button("📄 Download Report", use_container_width=True):
            _handle_report_export(results)
        
        # New search
        if st.button("🔍 New Search", use_container_width=True):
            st.session_state.search_results = None
            st.session_state.last_search_query = ''
            st.rerun()
        
        # Refresh current search
        if st.button("🔄 Refresh Results", use_container_width=True):
            _handle_search_execution(results['query'])


def _handle_email_sending(email_manager: EmailManager, to_email: str, results: Dict[str, Any]):
    """
    Handle email sending with results.
    
    Args:
        email_manager: EmailManager instance
        to_email: Recipient email address
        results: Analysis results
    """
    try:
        with st.spinner("Sending email..."):
            # Format email content
            email_content = email_manager.format_analysis_email(results)
            
            # Send email
            result = email_manager.send_email(
                to_email=to_email,
                subject=email_content["subject"],
                body=email_content["body"]
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"❌ Email failed: {result['error']}")
                
    except Exception as e:
        st.error(f"❌ Email sending error: {str(e)}")


def _handle_report_export(results: Dict[str, Any]):
    """
    Handle report export to text file.
    
    Args:
        results: Analysis results to export
    """
    try:
        # Format export content
        export_content = f"""SAP EWA Analysis Report
{'=' * 50}

Query: {results['query']}
Analysis Time: {datetime.fromisoformat(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
Systems Analyzed: {', '.join(results['selected_systems'])}
Results Found: {results.get('search_results_count', 0)}
Confidence Score: {results.get('confidence_score', 0) * 100:.1f}%

EXECUTIVE SUMMARY:
{results.get('summary', 'No summary available')}

CRITICAL FINDINGS ({len(results.get('critical_findings', []))}):
"""
        
        # Add critical findings
        for i, finding in enumerate(results.get('critical_findings', []), 1):
            export_content += f"{i}. {finding}\n"
        
        if not results.get('critical_findings'):
            export_content += "✅ No critical issues found\n"
        
        # Add recommendations
        recommendations = results.get('recommendations', [])
        export_content += f"\nRECOMMENDATIONS ({len(recommendations)}):\n"
        
        for i, rec in enumerate(recommendations, 1):
            export_content += f"{i}. {rec}\n"
        
        if not recommendations:
            export_content += "ℹ️ No specific recommendations available\n"
        
        # Add system details
        system_summaries = results.get('system_summaries', {})
        if system_summaries:
            export_content += "\nSYSTEM DETAILS:\n"
            for sys_id, sys_data in system_summaries.items():
                health = sys_data.get('overall_health', 'UNKNOWN')
                alerts = len(sys_data.get('critical_alerts', []))
                export_content += f"• {sys_id}: {health} ({alerts} critical alerts)\n"
        
        # Footer
        export_content += f"""
---
Generated by SAP EWA Analyzer v{APP_VERSION}
Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Create download button
        filename = f"sap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        st.download_button(
            label="📥 Download Report",
            data=export_content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")


def create_sidebar():
    """
    Create the enhanced sidebar with status, controls, and workflow visualization.
    
    Provides system status, workflow controls, email configuration,
    and LangGraph workflow visualization capabilities.
    """
    with st.sidebar:
        st.header("📊 System Status")
        
        # System status indicators
        _display_system_status()
        
        st.markdown("---")
        
        # Email configuration status
        _display_email_status()
        
        st.markdown("---")
        
        # LangGraph workflow section
        _display_workflow_section()
        
        st.markdown("---")
        
        # Quick actions
        _display_quick_actions()


def _display_system_status():
    """Display current system status in sidebar."""
    # Processing status
    if st.session_state.vector_store_ready:
        st.success("✅ Documents Processed")
        st.write(f"📄 Chunks: {st.session_state.get('total_chunks', 0)}")
    else:
        st.info("⏳ Upload & Process Documents")
    
    # Selected systems
    selected_systems = st.session_state.get('selected_systems', [])
    if selected_systems:
        st.write(f"🖥️ **Systems ({len(selected_systems)}):**")
        for system in selected_systems:
            st.write(f"• {system}")
    
    # Latest search results
    search_results = st.session_state.get('search_results')
    if search_results:
        st.write("🔍 **Latest Search:**")
        st.write(f"• Query: {search_results['query'][:25]}...")
        st.write(f"• Results: {search_results.get('search_results_count', 0)}")
        st.write(f"• Critical: {len(search_results.get('critical_findings', []))}")
        st.write(f"• Recommendations: {len(search_results.get('recommendations', []))}")


def _display_email_status():
    """Display email configuration status."""
    st.header("📧 Email Status")
    
    email_manager = EmailManager()
    
    if st.session_state.email_enabled and email_manager.provider:
        st.success(f"✅ {email_manager.provider.title()} Ready")
        st.write(f"Provider: {email_manager.provider}")
        st.write(f"Account: {email_manager.config.get('email', 'Not configured')}")
    else:
        st.warning("⚠️ Email Not Configured")
        
        with st.expander("📧 Email Setup"):
            st.write("Add to your .env file:")
            
            # Show configuration for both providers
            provider_choice = st.selectbox(
                "Email Provider",
                ["gmail", "outlook"],
                help="Choose your email provider"
            )
            
            if provider_choice == "gmail":
                st.code("""
EMAIL_ENABLED=true
EMAIL_PROVIDER=gmail
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your-app-password
                """)
                st.info("📝 Use App Password, not regular password for Gmail")
            
            else:  # outlook
                st.code("""
EMAIL_ENABLED=true
EMAIL_PROVIDER=outlook
OUTLOOK_EMAIL=your-email@outlook.com
OUTLOOK_PASSWORD=your-password
                """)
                st.info("📝 Use your regular Outlook/Hotmail password")


def _display_workflow_section():
    """Display LangGraph workflow status and visualization."""
    st.header("🔄 LangGraph Workflow")
    
    workflow = st.session_state.get('workflow')
    
    if workflow:
        # Workflow debug information
        workflow_debug = {
            "workflow_exists": workflow is not None,
            "workflow_type": type(workflow).__name__,
            "has_app": hasattr(workflow, 'app'),
            "app_exists": getattr(workflow, 'app', None) is not None,
            "app_type": type(getattr(workflow, 'app', None)).__name__,
            "has_viz_method": hasattr(workflow, 'get_workflow_visualization')
        }
        
        # Show workflow status
        st.subheader("🔍 Status")
        for key, value in workflow_debug.items():
            color = "🟢" if value else "🔴"
            st.write(f"{color} **{key.replace('_', ' ').title()}**: {value}")
        
        # Workflow visualization
        if workflow_debug["app_exists"] and workflow_debug["app_type"] != "MockCompiledGraph":
            st.subheader("🎨 Visualization")
            
            if st.button("📊 Generate Workflow Diagram", use_container_width=True):
                _handle_workflow_visualization(workflow)
        
        else:
            st.warning("⚠️ Using Mock Workflow")
            st.info("💡 LangGraph not fully available")
        
        # Debug mode toggle
        debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            with st.expander("🐛 Debug Info"):
                st.json(workflow_debug)
    
    else:
        st.warning("⚠️ No workflow initialized")
        st.info("💡 Upload documents to initialize")


def _handle_workflow_visualization(workflow):
    """
    Handle workflow visualization generation.
    
    Args:
        workflow: Workflow instance
    """
    try:
        with st.spinner("Generating workflow diagram..."):
            # Get workflow visualization
            viz_result = workflow.get_workflow_visualization()
            
            if viz_result.get('success'):
                st.success("✅ Diagram generated successfully!")
                
                file_path = viz_result.get('file')
                viz_type = viz_result.get('type')
                
                if viz_type == 'png' and file_path:
                    # Display PNG image
                    if os.path.exists(file_path):
                        st.image(file_path, caption="LangGraph Workflow", use_column_width=True)
                        
                        # Download button
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="📥 Download PNG",
                                data=f.read(),
                                file_name="workflow_diagram.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    else:
                        st.error(f"❌ File not found: {file_path}")
                
                elif viz_type == 'mermaid':
                    # Display Mermaid code
                    mermaid_code = viz_result.get('code', '')
                    st.code(mermaid_code, language='mermaid')
                    st.info("💡 Copy to https://mermaid.live/ to view")
                    
                    st.download_button(
                        label="📥 Download Mermaid",
                        data=mermaid_code,
                        file_name="workflow_diagram.mmd",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            else:
                st.error(f"❌ Generation failed: {viz_result.get('error')}")
                
    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())


def _display_quick_actions():
    """Display quick action buttons in sidebar."""
    st.header("⚡ Quick Actions")
    
    # System health check
    if st.button("💊 System Health", use_container_width=True):
        _show_system_health_modal()
    
    # Configuration panel
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.show_advanced_options = not st.session_state.get('show_advanced_options', False)
        st.rerun()
    
    # Restart application
    if st.button("🔄 Restart App", use_container_width=True):
        SessionStateManager.reset(preserve_config=False)
        st.rerun()


def _show_system_health_modal():
    """Display system health information in modal."""
    with st.expander("💊 System Health Check", expanded=True):
        # Session state health
        health_checks = {
            "Session State": "✅ Active" if st.session_state else "❌ Missing",
            "Workflow": "✅ Ready" if st.session_state.get('workflow') else "❌ Not initialized",
            "Vector Store": "✅ Ready" if st.session_state.get('vector_store_ready') else "❌ Not ready",
            "Email Config": "✅ Configured" if st.session_state.get('email_enabled') else "⚠️ Not configured",
            "Selected Systems": f"✅ {len(st.session_state.get('selected_systems', []))} systems" if st.session_state.get('selected_systems') else "⚠️ None selected"
        }
        
        for check, status in health_checks.items():
            st.write(f"**{check}:** {status}")
        
        # Session metrics
        st.write("**Session Duration:**", 
                f"{(datetime.now() - st.session_state.get('session_start_time', datetime.now())).total_seconds():.0f}s")
        
        # Memory usage if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.write(f"**Memory Usage:** {memory.percent}%")
        except ImportError:
            st.write("**Memory Usage:** Not available")


def create_footer():
    """Create the application footer with branding and version info."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**{APP_ICON} {APP_TITLE}**")
        st.caption("AI-Powered SAP Analysis Tool")
    
    with col2:
        st.markdown("**🔧 Powered by:**")
        st.caption("LangGraph • ChromaDB • OpenAI • Streamlit")
    
    with col3:
        st.markdown("**📅 Status:**")
        current_time = datetime.now().strftime('%H:%M:%S')
        st.caption(f"Version {APP_VERSION} • {current_time}")


def create_debug_section():
    """Create debug section when debug mode is enabled."""
    if not st.session_state.get('debug_mode'):
        return
    
    st.markdown("---")
    st.subheader("🐛 Debug Information")
    
    # Session state debug
    with st.expander("📊 Session State"):
        # Filter and truncate session state for display
        debug_state = {}
        for key, value in st.session_state.items():
            if key.startswith('_'):  # Skip private keys
                continue
            
            # Truncate long values
            if isinstance(value, str) and len(value) > 100:
                debug_state[key] = value[:100] + "..."
            elif isinstance(value, list) and len(value) > 5:
                debug_state[key] = f"List with {len(value)} items"
            elif isinstance(value, dict) and len(value) > 10:
                debug_state[key] = f"Dict with {len(value)} keys"
            else:
                debug_state[key] = value
        
        st.json(debug_state)
    
    # System information
    with st.expander("💻 System Information"):
        import sys
        import platform
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        
        # Check key packages
        try:
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            key_packages = ['streamlit', 'langchain', 'langgraph', 'openai', 'chromadb']
            
            st.write("**Key Packages:**")
            for pkg in key_packages:
                status = "✅ Installed" if pkg in installed_packages else "❌ Missing"
                st.write(f"• {pkg}: {status}")
                
        except Exception as e:
            st.write(f"**Package check failed:** {e}")
    
    # Performance metrics
    with st.expander("⚡ Performance Metrics"):
        status_summary = SessionStateManager.get_status_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Session Duration", f"{status_summary['session_duration']:.0f}s")
            st.metric("Executions", status_summary['execution_count'])
        
        with col2:
            st.metric("Documents", status_summary['documents_count'])
            st.metric("Systems", status_summary['selected_systems_count'])


# ================================
# ERROR HANDLING & RECOVERY
# ================================

def handle_global_error(error: Exception, context: str = "Application"):
    """
    Global error handler with recovery options.
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
    """
    logger.error(f"❌ Global error in {context}: {str(error)}")
    
    st.error(f"❌ Error in {context}: {str(error)}")
    
    # Show debug information if debug mode is enabled
    if st.session_state.get('debug_mode'):
        st.code(traceback.format_exc())
    
    # Recovery options
    with st.expander("🔧 Recovery Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart App", key=f"error_restart_{context.lower().replace(' ', '_')}"):
                SessionStateManager.reset(preserve_config=False)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Data", key=f"error_clear_{context.lower().replace(' ', '_')}"):
                SessionStateManager.reset(preserve_config=True)
                st.rerun()
        
        with col3:
            if st.button("🐛 Enable Debug", key=f"error_debug_{context.lower().replace(' ', '_')}"):
                st.session_state.debug_mode = True
                st.rerun()


# ================================
# MAIN APPLICATION
# ================================

def main():
    """
    Main application function that orchestrates the entire UI.
    
    This function sets up the page configuration, initializes session state,
    and renders all UI components in the correct order.
    """
    try:
        # Configure page and apply styling
        configure_page()
        
        # Initialize session state
        SessionStateManager.initialize()
        
        # Create sidebar
        create_sidebar()
        
        # Main content area
        create_main_header()
        
        # File upload section
        uploaded_files = create_file_upload_section()
        
        # Document processing section
        create_processing_section(uploaded_files)
        
        # System selection (shown after processing)
        if st.session_state.vector_store_ready:
            create_system_selection_section()
            
            # Search section (shown after system selection)
            if st.session_state.selected_systems:
                create_search_section()
        
        # Results section (shown after search)
        create_results_section()
        
        # Footer
        create_footer()
        
        # Debug section (only in debug mode)
        if st.session_state.get('debug_mode'):
            create_debug_section()
            add_debug_section_to_app()
        
        # Advanced options (if enabled)
        if st.session_state.get('show_advanced_options'):
            _show_advanced_options()
        
    except Exception as e:
        handle_global_error(e, "Main Application")


def _show_advanced_options():
    """Display advanced configuration options."""
    st.markdown("---")
    st.subheader("⚙️ Advanced Configuration")
    
    with st.expander("🔧 Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                step=100,
                help="Size of text chunks for processing"
            )
            
            st.session_state.top_k = st.slider(
                "Search Results",
                min_value=1,
                max_value=50,
                value=st.session_state.get('top_k', DEFAULT_TOP_K),
                help="Number of results to retrieve"
            )
        
        with col2:
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                step=50,
                help="Overlap between text chunks"
            )
            
            st.session_state.temperature = st.slider(
                "AI Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get('temperature', DEFAULT_TEMPERATURE),
                step=0.1,
                help="AI creativity level (0=focused, 2=creative)"
            )


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    try:
        # Set up logging
        logger.info(f"🚀 Starting {APP_TITLE} v{APP_VERSION}")
        
        # Run main application
        main()
        
        # Log successful completion
        logger.info("✅ Application running successfully")
        
    except Exception as e:
        # Handle any uncaught exceptions
        logger.critical(f"💥 Critical application error: {str(e)}")
        handle_global_error(e, "Application Startup")
        st.stop()


# ================================
# MODULE EXPORTS
# ================================

__all__ = [
    'main',
    'SessionStateManager',
    'WorkflowManager', 
    'EmailManager',
    'configure_page',
    'handle_global_error'
]

# ================================
# MODULE INITIALIZATION LOG
# ================================

logger.info("🏗️ SAP EWA Analyzer Streamlit App initialized")
logger.info(f"📋 App version: {APP_VERSION}")
logger.info(f"🔧 Modules available: {MODULES_AVAILABLE}")
logger.info(f"📧 Email available: {EMAIL_AVAILABLE}")
logger.info(f"⚙️ Environment loading: {ENV_AVAILABLE}")

if not MODULES_AVAILABLE:
    logger.warning("⚠️ Running with limited functionality - check module imports")

logger.info("✅ SAP EWA Analyzer app.py module ready for production")

# 6. Additional debugging helper function
def debug_search_state(workflow_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug helper to analyze search state issues.
    
    Args:
        workflow_state: Current workflow state
        
    Returns:
        Debug information about search state
    """
    debug_info = {
        "user_query_present": "user_query" in workflow_state,
        "user_query_value": workflow_state.get("user_query", "NOT_FOUND"),
        "user_query_type": type(workflow_state.get("user_query", None)).__name__,
        "user_query_length": len(workflow_state.get("user_query", "")),
        "search_filters_present": "search_filters" in workflow_state,
        "search_filters_value": workflow_state.get("search_filters", {}),
        "vector_store_ready": workflow_state.get("vector_store_ready", False),
        "workflow_status": workflow_state.get("workflow_status", "unknown"),
        "all_state_keys": list(workflow_state.keys())
    }
    
    return debug_info


# Search Query Diagnostic Functions
def diagnose_search_issue():
    """
    Diagnostic function to help identify search query issues.
    Add this to your Streamlit app for debugging.
    """
    st.subheader("🔍 Search Query Diagnostics")
    
    # Check session state
    with st.expander("📊 Session State Diagnostics"):
        st.write("**Search-related session state:**")
        search_keys = [
            'search_query', 'last_search_query', 'search_results', 
            'vector_store_ready', 'workflow_initialized', 'selected_systems'
        ]
        
        for key in search_keys:
            value = st.session_state.get(key, "NOT_FOUND")
            st.write(f"• **{key}**: `{value}` (type: {type(value).__name__})")
    
    # Test query input
    with st.expander("🧪 Test Query Processing"):
        test_query = st.text_input("Test Query:", placeholder="Enter a test query")
        
        if st.button("Test Query Processing"):
            if test_query:
                st.write("**Query Processing Results:**")
                st.write(f"• Original: `{test_query}`")
                st.write(f"• Stripped: `{test_query.strip()}`")
                st.write(f"• Length: {len(test_query.strip())}")
                st.write(f"• Valid: {bool(test_query.strip() and len(test_query.strip()) >= 3)}")
                
                # Test workflow state creation
                try:
                    if hasattr(st.session_state, 'workflow') and st.session_state.workflow:
                        workflow = st.session_state.workflow
                        test_state = workflow._create_initial_state(user_query=test_query.strip())
                        st.write("**Test State Created:**")
                        st.write(f"• user_query in state: `{test_state.get('user_query')}`")
                        st.write(f"• State keys: {list(test_state.keys())}")
                    else:
                        st.warning("Workflow not initialized")
                except Exception as e:
                    st.error(f"State creation failed: {e}")
            else:
                st.warning("Enter a test query")
    
    # Check workflow status
    with st.expander("🔧 Workflow Status"):
        if hasattr(st.session_state, 'workflow') and st.session_state.workflow:
            try:
                status = st.session_state.workflow.get_workflow_status()
                st.json(status)
            except Exception as e:
                st.error(f"Error getting workflow status: {e}")
        else:
            st.warning("Workflow not available in session state")
    
    # Manual search test
    with st.expander("🚀 Manual Search Test"):
        manual_query = st.text_input("Manual Test Query:", key="manual_test")
        
        if st.button("Execute Manual Search") and manual_query:
            try:
                if not st.session_state.get('vector_store_ready'):
                    st.error("Vector store not ready")
                    return
                
                # Create workflow manager
                workflow_manager = WorkflowManager()
                
                # Test search execution
                st.write("**Executing search...**")
                
                # Show what we're passing
                st.write(f"Query: `{manual_query}`")
                st.write(f"Selected systems: {st.session_state.get('selected_systems', [])}")
                
                # Execute
                result = workflow_manager.execute_search(manual_query.strip())
                
                if result["success"]:
                    st.success("✅ Manual search succeeded!")
                    st.json(result)
                else:
                    st.error(f"❌ Manual search failed: {result.get('error')}")
                    
            except Exception as e:
                st.error(f"Manual search error: {e}")
                st.code(traceback.format_exc())


def test_quick_search_templates():
    """Test the quick search template functionality."""
    st.subheader("🚀 Quick Search Template Test")
    
    for i, template in enumerate(QUICK_SEARCH_TEMPLATES):
        with st.expander(f"Template {i+1}: {template['label']}"):
            st.write(f"**Query:** `{template['query']}`")
            st.write(f"**Description:** {template['description']}")
            
            if st.button(f"Test {template['label']}", key=f"test_template_{i}"):
                # Test the template query
                query = template['query']
                st.write(f"Testing query: `{query}`")
                
                if not query or not query.strip():
                    st.error("❌ Template query is empty!")
                elif len(query.strip()) < 3:
                    st.error("❌ Template query too short!")
                else:
                    st.success(f"✅ Template query valid: `{query.strip()}`")
                    
                    # Try to execute if vector store is ready
                    if st.session_state.get('vector_store_ready'):
                        _handle_search_execution(query)
                    else:
                        st.info("Vector store not ready - query validation only")


def enhanced_handle_search_execution(query: str):
    """
    Enhanced search execution with detailed error reporting.
    """
    try:
        # Step 1: Validate inputs
        st.write("**Step 1: Input Validation**")
        
        if not query:
            st.error("❌ Query is None or empty")
            return
            
        cleaned_query = query.strip()
        st.write(f"✓ Original query: `{query}`")
        st.write(f"✓ Cleaned query: `{cleaned_query}`")
        
        if not cleaned_query:
            st.error("❌ Query is empty after cleaning")
            return
            
        if len(cleaned_query) < 3:
            st.error(f"❌ Query too short: {len(cleaned_query)} chars (minimum 3)")
            return
            
        st.success("✓ Query validation passed")
        
        # Step 2: Check prerequisites
        st.write("**Step 2: Prerequisites Check**")
        
        if not st.session_state.get('vector_store_ready', False):
            st.error("❌ Vector store not ready")
            return
            
        if not st.session_state.get('workflow_initialized', False):
            st.error("❌ Workflow not initialized")
            return
            
        st.success("✓ Prerequisites check passed")
        
        # Step 3: Execute search
        st.write("**Step 3: Search Execution**")
        
        workflow_manager = WorkflowManager()
        
        search_filters = {
            "target_systems": st.session_state.get('selected_systems', [])
        }
        
        st.write(f"✓ Search filters: {search_filters}")
        
        with st.spinner(f"🔍 Searching: '{cleaned_query}'..."):
            result = workflow_manager.execute_search(cleaned_query, search_filters)
            
            if result["success"]:
                st.success("✅ Search completed successfully!")
                search_results = result["search_results"]
                st.write(f"Found {search_results['search_results_count']} results")
                st.rerun()
            else:
                st.error(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                
                # Show detailed error info
                if 'workflow_result' in result:
                    workflow_result = result['workflow_result']
                    st.write("**Workflow Error Details:**")
                    st.write(f"• Status: {workflow_result.get('workflow_status', 'unknown')}")
                    st.write(f"• Error: {workflow_result.get('error_message', 'no details')}")
                    st.write(f"• Current Agent: {workflow_result.get('current_agent', 'unknown')}")
                
    except Exception as e:
        st.error(f"❌ Search execution exception: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())


def add_debug_section_to_app():
    """
    Add debug section to main app when debug mode is enabled.
    """
    if st.session_state.get('debug_mode'):
        st.markdown("---")
        st.header("🐛 Search Debug Section")
        
        tab1, tab2, tab3 = st.tabs(["Diagnostics", "Template Test", "Manual Test"])
        
        with tab1:
            diagnose_search_issue()
        
        with tab2:
            test_quick_search_templates()
        
        with tab3:
            st.subheader("Enhanced Manual Search")
            debug_query = st.text_input("Debug Search Query:")
            if st.button("Execute Debug Search") and debug_query:
                enhanced_handle_search_execution(debug_query)


# ================================
# MAIN APPLICATION
# ================================


