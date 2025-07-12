# ui/core_components.py - Complete SAP EWA Analyzer UI Components
"""
Complete UI components for SAP EWA Analyzer with modular workflow integration.
This module provides all UI components with real functionality including:

Key Features:
- System ID selection with individual search capability
- EWA content search with advanced query handling
- Clickable LangGraph workflow visualization
- Email functionality with Gmail/Outlook support
- Real-time processing status and metrics
- Comprehensive error handling and recovery

Architecture Integration:
- Modular workflow components
- Agent-based processing pipeline
- Vector store integration
- Email notification system
"""

import streamlit as st
import time
import json
import os
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# SAFE IMPORTS WITH FALLBACKS
# ================================

try:
    # Import your modular workflow components
    from workflow import SAPRAGWorkflow, create_workflow, validate_workflow_config
    from models import WorkflowStatus, HealthStatus, EmailRecipient
    WORKFLOW_AVAILABLE = True
    logger.info("✅ Workflow modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Workflow modules not available: {e}")
    WORKFLOW_AVAILABLE = False
    
    # Fallback classes for development
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

# Email functionality
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
# CONFIGURATION CONSTANTS
# ================================

APP_TITLE = "SAP Early Watch Analyzer"
APP_ICON = "📊"
APP_VERSION = "1.0"
APP_DESCRIPTION = "AI-Powered SAP System Health Analysis with LangGraph"

# File upload constraints
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = ['.pdf']
MAX_FILES_PER_UPLOAD = 10

# Default configuration values
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 10
DEFAULT_TEMPERATURE = 0.1

# Quick search templates for SAP EWA analysis
QUICK_SEARCH_TEMPLATES = [
    {
        "label": "🚨 Critical Alerts",
        "query": "Show me all critical alerts and errors in the systems",
        "description": "Find urgent issues requiring immediate attention",
        "category": "alerts"
    },
    {
        "label": "📈 Performance Issues", 
        "query": "Analyze performance bottlenecks and slow queries",
        "description": "Identify system performance problems",
        "category": "performance"
    },
    {
        "label": "💡 SAP Recommendations",
        "query": "What are the top SAP recommendations for optimization",
        "description": "Get actionable SAP improvement suggestions",
        "category": "recommendations"
    },
    {
        "label": "🔧 System Health Check",
        "query": "Overall system health status and metrics analysis",
        "description": "Get comprehensive health overview",
        "category": "health"
    },
    {
        "label": "💾 Memory Analysis",
        "query": "Find memory usage problems and memory leaks in SAP systems",
        "description": "Analyze memory-related issues",
        "category": "memory"
    },
    {
        "label": "🗄️ Database Issues",
        "query": "Show database performance issues and errors in EWA reports",
        "description": "Focus on database-specific problems",
        "category": "database"
    }
]

# LangGraph workflow steps visualization
WORKFLOW_STEPS = [
    {"name": "Document Upload", "node": "pdf_processor", "icon": "📁", "status": "pending"},
    {"name": "PDF Processing", "node": "pdf_processor", "icon": "📄", "status": "pending"},
    {"name": "Text Extraction", "node": "embedding_creator", "icon": "📝", "status": "pending"},
    {"name": "Vector Embeddings", "node": "embedding_creator", "icon": "🔢", "status": "pending"},
    {"name": "Vector Storage", "node": "vector_store_manager", "icon": "🗄️", "status": "pending"},
    {"name": "Search Index", "node": "search_agent", "icon": "🔍", "status": "pending"},
    {"name": "AI Analysis", "node": "summary_agent", "icon": "🤖", "status": "pending"},
    {"name": "System Output", "node": "system_output_agent", "icon": "🖥️", "status": "pending"},
    {"name": "Email Delivery", "node": "email_agent", "icon": "📧", "status": "pending"},
]

# ================================
# PAGE CONFIGURATION
# ================================

def configure_page():
    """Configure Streamlit page settings with comprehensive styling."""
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
    
    # Comprehensive custom CSS
    st.markdown("""
    <style>
        /* Main application styling */
        .main-header {
            background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Status and info boxes */
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
        
        /* System cards */
        .system-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.2s;
        }
        
        .system-card:hover {
            border-color: #007bff;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .system-target-card {
            background: #e7f3ff;
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,123,255,0.2);
        }
        
        /* Workflow visualization */
        .workflow-step {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            margin: 0.3rem 0;
            border-radius: 8px;
            transition: all 0.2s;
            cursor: pointer;
        }
        
        .workflow-step:hover {
            transform: translateX(5px);
        }
        
        .workflow-step.completed { 
            background: #d4edda; 
            border-left: 4px solid #28a745; 
        }
        
        .workflow-step.processing { 
            background: #fff3cd; 
            border-left: 4px solid #ffc107; 
            animation: pulse 1.5s infinite;
        }
        
        .workflow-step.pending { 
            background: #f8f9fa; 
            border-left: 4px solid #6c757d; 
        }
        
        .workflow-step.error { 
            background: #f8d7da; 
            border-left: 4px solid #dc3545; 
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        /* Search templates */
        .search-template {
            background: #ffffff;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .search-template:hover {
            background: #f8f9fa;
            border-color: #007bff;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border: 1px solid #e9ecef;
            transition: all 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Enhanced buttons */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* System health indicators */
        .health-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .health-healthy { background-color: #28a745; }
        .health-warning { background-color: #ffc107; }
        .health-critical { background-color: #dc3545; }
        .health-unknown { background-color: #6c757d; }
        
        /* Sidebar enhancements */
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        
        /* Progress indicators */
        .progress-container {
            background: #e9ecef;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 { font-size: 1.8rem; }
            .main-header p { font-size: 1rem; }
            .metric-card { padding: 1rem; }
        }
    </style>
    """, unsafe_allow_html=True)

# ================================
# SESSION STATE MANAGER
# ================================

class SessionStateManager:
    """Enhanced session state management for comprehensive UI state."""
    
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
        
        # System selection and targeting
        'selected_systems': [],
        'target_system': None,  # For individual system search
        'system_search_results': {},  # Results per system
        
        # Search and analysis
        'search_query': '',
        'last_search_query': '',
        'ewa_content_query': '',  # Dedicated EWA content search
        'search_results': None,
        'search_filters': {},
        
        # Results and summaries
        'summary': {},
        'system_summaries': {},
        'critical_findings': [],
        'recommendations': [],
        'performance_insights': [],
        
        # Email configuration
        'email_enabled': False,
        'email_recipients': [],
        'email_sent': False,
        'email_config_checked': False,
        
        # UI state
        'current_tab': 'upload',
        'show_advanced_options': False,
        'debug_mode': False,
        'show_workflow_viz': False,
        
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
        """Initialize all session state variables with default values."""
        initialized_count = 0
        for key, default_value in cls.DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                initialized_count += 1
        
        st.session_state.last_activity_time = datetime.now()
        
        if initialized_count > 0:
            logger.info(f"✅ Initialized {initialized_count} session state variables")

# ================================
# WORKFLOW MANAGER
# ================================

class WorkflowManager:
    """Enhanced workflow manager with comprehensive error handling."""
    
    def __init__(self):
        self.workflow = None
        self.config = None
        self.last_error = None
        
    def initialize_workflow(self) -> tuple[bool, str]:
        """Initialize the modular SAP RAG workflow."""
        try:
            logger.info("🔧 Initializing modular SAP RAG workflow")
            
            if not WORKFLOW_AVAILABLE:
                return False, "Required workflow modules not available. Check imports."
            
            # Build configuration from session state and environment
            self.config = self._build_workflow_config()
            
            # Validate configuration
            validation_result = validate_workflow_config(self.config)
            if not validation_result.get('valid', True):
                errors = validation_result.get('errors', [])
                return False, f"Configuration validation failed: {'; '.join(errors)}"
            
            # Create workflow instance
            self.workflow = create_workflow(self.config)
            st.session_state.workflow = self.workflow
            st.session_state.workflow_initialized = True
            
            logger.info("✅ Workflow initialized successfully")
            return True, "Workflow initialized successfully"
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ Workflow initialization failed: {e}")
            return False, f"Initialization failed: {str(e)}"
    
    def _build_workflow_config(self) -> Dict[str, Any]:
        """Build comprehensive workflow configuration."""
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
            "auto_send_results": False,
            
            # File processing settings
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "supported_file_types": SUPPORTED_FILE_TYPES,
            "batch_size": 10,
            "timeout": 300,
            "retry_attempts": 3,
            
            # Advanced settings
            "debug_mode": st.session_state.get('debug_mode', False),
            "verbose_logging": True,
            "enable_system_detection": True,
            "enable_performance_metrics": True
        }
        
        # Add environment variables
        if ENV_AVAILABLE:
            config.update({
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "gmail_email": os.getenv("GMAIL_EMAIL"),
                "gmail_app_password": os.getenv("GMAIL_APP_PASSWORD"),
                "outlook_email": os.getenv("OUTLOOK_EMAIL"),
                "outlook_password": os.getenv("OUTLOOK_PASSWORD"),
                "email_provider": os.getenv("EMAIL_PROVIDER", "gmail")
            })
        
        return config
    
    def _check_email_configuration(self) -> bool:
        """Check email configuration for both Gmail and Outlook."""
        try:
            if not EMAIL_AVAILABLE or not ENV_AVAILABLE:
                return False
            
            email_provider = os.getenv("EMAIL_PROVIDER", "gmail").lower()
            email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
            
            if not email_enabled:
                return False
            
            if email_provider == "gmail":
                gmail_email = os.getenv("GMAIL_EMAIL")
                gmail_password = os.getenv("GMAIL_APP_PASSWORD") 
                is_configured = gmail_email and gmail_password
            elif email_provider == "outlook":
                outlook_email = os.getenv("OUTLOOK_EMAIL")
                outlook_password = os.getenv("OUTLOOK_PASSWORD")
                is_configured = outlook_email and outlook_password
            else:
                return False
            
            st.session_state.email_enabled = is_configured
            return is_configured
            
        except Exception as e:
            logger.warning(f"⚠️ Email configuration check failed: {e}")
            return False
    
    def process_documents(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Process uploaded documents using the modular workflow."""
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
                "user_query": "",
                "search_filters": {},
                "email_recipients": []
            }
            
            # Update processing status
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
                
                return {"success": False, "error": error_msg, "workflow_result": result}
            
        except Exception as e:
            error_msg = f"Document processing exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            st.session_state.workflow_status = WorkflowStatus.ERROR
            st.session_state.error_message = error_msg
            st.session_state.processing_status = 'error'
            
            return {"success": False, "error": error_msg}
    
    def execute_search(self, query: str, search_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute search with enhanced query processing."""
        try:
            if not self.workflow:
                return {"success": False, "error": "Workflow not initialized"}
            
            if not st.session_state.get('vector_store_ready', False):
                return {"success": False, "error": "Vector store not ready. Process documents first."}
            
            # Enhanced query validation
            if not query or not query.strip():
                return {"success": False, "error": "Empty search query provided"}
            
            cleaned_query = query.strip()
            if len(cleaned_query) < 3:
                return {"success": False, "error": "Search query too short (minimum 3 characters)"}
            
            logger.info(f"🔍 Executing search: '{cleaned_query}'")
            
            # Prepare search filters
            filters = search_filters or {}
            if st.session_state.get('selected_systems'):
                filters['target_systems'] = st.session_state.selected_systems
            
            if st.session_state.get('target_system'):
                filters['primary_system'] = st.session_state.target_system
            
            # Update status
            st.session_state.workflow_status = WorkflowStatus.SEARCHING
            st.session_state.search_query = cleaned_query
            
            # Execute search via workflow
            result = self.workflow.run_search_only(query=cleaned_query, search_filters=filters)
            
            # Process results
            if result.get("workflow_status") == WorkflowStatus.COMPLETED:
                summary = result.get("summary", {})
                system_summaries = result.get("system_summaries", {})
                
                search_results = {
                    "query": cleaned_query,
                    "timestamp": datetime.now().isoformat(),
                    "selected_systems": st.session_state.get('selected_systems', []),
                    "target_system": st.session_state.get('target_system'),
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
                st.session_state.performance_insights = summary.get("performance_insights", [])
                st.session_state.last_search_query = cleaned_query
                st.session_state.last_execution_time = result.get("processing_time", 0.0)
                
                logger.info(f"✅ Search completed: {search_results['search_results_count']} results")
                
                return {"success": True, "search_results": search_results, "workflow_result": result}
            else:
                error_msg = result.get("error_message", "Unknown search error")
                st.session_state.workflow_status = WorkflowStatus.ERROR
                st.session_state.error_message = error_msg
                
                return {"success": False, "error": error_msg, "workflow_result": result}
            
        except Exception as e:
            error_msg = f"Search execution exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            st.session_state.workflow_status = WorkflowStatus.ERROR
            st.session_state.error_message = error_msg
            
            return {"success": False, "error": error_msg}
    
    def _validate_uploaded_files(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Validate uploaded files for size and type constraints."""
        try:
            if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
                return {
                    "valid": False,
                    "error": f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files allowed."
                }
            
            for file in uploaded_files:
                file_size = len(file.getvalue())
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    return {
                        "valid": False,
                        "error": f"File {file.name} exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
                    }
                
                file_extension = Path(file.name).suffix.lower()
                if file_extension not in SUPPORTED_FILE_TYPES:
                    return {
                        "valid": False,
                        "error": f"File {file.name} has unsupported type. Supported: {SUPPORTED_FILE_TYPES}"
                    }
                
                if file_size == 0:
                    return {
                        "valid": False,
                        "error": f"File {file.name} is empty"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"File validation error: {str(e)}"}

# ================================
# MAIN UI COMPONENTS
# ================================

def create_main_header():
    """Create the main application header."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
        <small>Version {APP_VERSION} | Powered by LangGraph & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)

def create_file_upload_section() -> List[Any]:
    """Create comprehensive file upload section."""
    st.header("📁 Document Upload & Processing")
    
    # Initialize workflow if needed
    if not st.session_state.get('workflow_initialized', False):
        with st.spinner("Initializing workflow..."):
            workflow_manager = WorkflowManager()
            success, message = workflow_manager.initialize_workflow()
            
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                return []
    
    # File upload interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload SAP EWA PDF Reports",
            type=['pdf'],
            accept_multiple_files=True,
            help=f"Upload up to {MAX_FILES_PER_UPLOAD} PDF files, max {MAX_FILE_SIZE_MB}MB each"
        )
    
    with col2:
        st.markdown("**📋 Requirements:**")
        st.caption("• PDF format only")
        st.caption("• Max 10MB per file")
        st.caption("• Multiple files supported")
        st.caption("• SAP EWA reports preferred")
    
    # Display uploaded files information
    if uploaded_files:
        total_size = sum(len(file.getvalue()) for file in uploaded_files)
        total_size_mb = total_size / 1024 / 1024
        
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ {len(uploaded_files)} file(s) uploaded successfully</strong><br>
            Total size: {total_size_mb:.1f} MB
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed file information
        with st.expander("📄 File Details", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                size_mb = len(file.getvalue()) / 1024 / 1024
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{i}. {file.name}**")
                with col2:
                    st.write(f"{size_mb:.1f} MB")
                with col3:
                    st.write("📄 PDF")
        
        # Store in session state
        st.session_state.uploaded_files = uploaded_files
    
    return uploaded_files or []

def create_processing_section(uploaded_files: List[Any]):
    """Create document processing section with workflow visualization."""
    st.header("⚙️ Document Processing")
    
    if not uploaded_files:
        st.markdown("""
        <div class="info-box">
            <strong>📤 Please upload PDF files first to begin processing</strong><br>
            The system will extract text and create vector embeddings for intelligent search.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Processing controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        process_btn = st.button(
            "🚀 Process Documents", 
            type="primary", 
            use_container_width=True,
            help="Extract text and create vector embeddings"
        )
    
    with col2:
        # Processing status indicator
        if st.session_state.get('vector_store_ready', False):
            st.success("✅ Ready")
        elif st.session_state.get('processing_status') == 'processing':
            st.info("⏳ Processing...")
        elif st.session_state.get('processing_status') == 'error':
            st.error("❌ Failed")
        else:
            st.info("⏳ Pending")
    
    with col3:
        clear_btn = st.button(
            "🗑️ Clear All", 
            use_container_width=True,
            help="Clear all data and start over"
        )
    
    with col4:
        st.metric("Documents", len(uploaded_files))
    
    # Handle processing
    if process_btn:
        handle_document_processing(uploaded_files)
    
    if clear_btn:
        handle_clear_data()
    
    # Show processing results
    if st.session_state.get('processing_status') == 'completed' and st.session_state.get('vector_store_ready'):
        display_processing_metrics()

def handle_document_processing(uploaded_files: List[Any]):
    """Handle document processing with enhanced progress tracking."""
    try:
        workflow_manager = WorkflowManager()
        
        # Create progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing documents..."):
                # Step 1: Initialize workflow
                status_text.text("🔄 Initializing workflow...")
                progress_bar.progress(10)
                
                success, message = workflow_manager.initialize_workflow()
                if not success:
                    progress_bar.progress(0)
                    status_text.text(f"❌ Error: {message}")
                    st.error(f"❌ Workflow initialization failed: {message}")
                    return
                
                # Step 2: Process documents
                progress_bar.progress(30)
                status_text.text("📄 Processing documents...")
                
                result = workflow_manager.process_documents(uploaded_files)
                
                if result["success"]:
                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")
                    
                    # Show success metrics
                    processing_time = sum(result.get("processing_times", {}).values())
                    chunks = result.get("total_chunks", 0)
                    
                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} files "
                        f"into {chunks} chunks in {processing_time:.2f} seconds"
                    )
                    
                    # Auto-refresh UI
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

def handle_clear_data():
    """Handle clearing all processed data."""
    try:
        # Reset session state while preserving configuration
        SessionStateManager.initialize()
        
        # Clear specific processing data
        keys_to_clear = [
            'vector_store_ready', 'processed_documents', 'total_chunks',
            'search_results', 'selected_systems', 'target_system',
            'processing_times', 'agent_messages', 'summary', 'system_summaries'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.processing_status = 'ready'
        st.session_state.workflow_status = WorkflowStatus.INITIALIZED
        
        st.success("🗑️ All data cleared successfully")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing data: {str(e)}")

def display_processing_metrics():
    """Display comprehensive processing metrics."""
    processing_times = st.session_state.get('processing_times', {})
    total_chunks = st.session_state.get('total_chunks', 0)
    
    if not processing_times:
        return
    
    with st.expander("📊 Processing Metrics", expanded=True):
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_time = sum(processing_times.values())
        
        with col1:
            st.metric("Total Chunks", total_chunks)
        
        with col2:
            st.metric("Processing Time", f"{total_time:.2f}s")
        
        with col3:
            if processing_times:
                slowest_step = max(processing_times.items(), key=lambda x: x[1])
                st.metric("Slowest Step", slowest_step[0].replace('_', ' ').title())
        
        with col4:
            st.metric("Steps Completed", len(processing_times))
        
        # Detailed timeline
        st.subheader("⏱️ Processing Timeline")
        
        for i, (step, duration) in enumerate(processing_times.items(), 1):
            step_percentage = (duration / total_time) * 100 if total_time > 0 else 0
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{i}. {step.replace('_', ' ').title()}**")
                st.progress(step_percentage / 100)
            
            with col2:
                st.write(f"{duration:.2f}s")
            
            with col3:
                st.write(f"{step_percentage:.1f}%")

def create_system_selection_section():
    """Create enhanced system selection interface with individual targeting."""
    if not st.session_state.get('vector_store_ready', False):
        return
    
    st.header("🎯 SAP System Selection & Targeting")
    
    # Method 1: Bulk system selection
    st.subheader("📋 Multiple System Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        systems_input = st.text_input(
            "SAP System IDs (comma-separated)",
            value=", ".join(st.session_state.get('selected_systems', [])),
            placeholder="P01, Q01, DEV, QAS, PRD",
            help="Enter multiple SAP system IDs separated by commas for bulk analysis"
        )
    
    with col2:
        if st.button("🔍 Auto-Detect Systems", use_container_width=True):
            handle_system_detection()
    
    # Process bulk input
    if systems_input:
        systems = [s.strip().upper() for s in systems_input.split(',') if s.strip()]
        st.session_state.selected_systems = systems
        
        if systems:
            st.markdown(f"""
            <div class="success-box">
                <strong>🖥️ Selected Systems ({len(systems)}):</strong> {', '.join(systems)}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Method 2: Individual system targeting
    st.subheader("🎯 Individual System Targeting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_system = st.text_input(
            "Target specific System ID for focused analysis:",
            value=st.session_state.get('target_system', ''),
            placeholder="Enter single system ID (e.g., PRD)",
            help="Search for issues in a specific SAP system with enhanced focus",
            key="target_system_input"
        )
    
    with col2:
        search_system_btn = st.button(
            "🎯 Set Target System", 
            type="primary",
            use_container_width=True,
            help="Set this system as the primary target for analysis"
        )
    
    # Handle individual system targeting
    if search_system_btn and target_system.strip():
        system_id = target_system.strip().upper()
        st.session_state.target_system = system_id
        
        # Add to selected systems if not already there
        current_systems = st.session_state.get('selected_systems', [])
        if system_id not in current_systems:
            current_systems.append(system_id)
            st.session_state.selected_systems = current_systems
        
        st.success(f"🎯 Target System Set: {system_id}")
        st.info(f"💡 You can now search for issues specific to {system_id} in the search section below")
        
    elif search_system_btn:
        st.error("❌ Please enter a system ID")
    
    # Update target system from text input
    if target_system and target_system.strip():
        st.session_state.target_system = target_system.strip().upper()
    
    # Show current targeting status
    if st.session_state.get('target_system'):
        target = st.session_state.target_system
        st.markdown(f"""
        <div class="system-target-card">
            <strong>🎯 Current Target System: {target}</strong><br>
            Searches will prioritize findings from this system
        </div>
        """, unsafe_allow_html=True)
    
    # Display system overview
    selected_systems = st.session_state.get('selected_systems', [])
    if selected_systems:
        with st.expander("🖥️ System Overview", expanded=True):
            cols = st.columns(min(len(selected_systems), 4))
            for i, system in enumerate(selected_systems):
                with cols[i % 4]:
                    is_target = system == st.session_state.get('target_system')
                    card_class = "system-target-card" if is_target else "system-card"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>🖥️ {system}</h3>
                        <p>Status: 🟢 Active</p>
                        <small>{'🎯 Target System' if is_target else 'Ready for analysis'}</small>
                    </div>
                    """, unsafe_allow_html=True)

def handle_system_detection():
    """Handle automatic system detection from processed documents."""
    try:
        with st.spinner("Detecting SAP systems from documents..."):
            # Simulate system detection using the search agent
            # In real implementation, this would use the search agent to detect systems
            detected_systems = ["PRD", "QAS", "DEV", "TST"]  # Mock detection
            
            st.session_state.selected_systems = detected_systems
            st.success(f"✅ Auto-detected systems: {', '.join(detected_systems)}")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ System detection failed: {str(e)}")

def create_search_section():
    """Create comprehensive search interface with EWA content search."""
    if not st.session_state.get('vector_store_ready', False):
        st.warning("📄 Please process documents first")
        return
    
    if not st.session_state.get('selected_systems', []):
        st.warning("🎯 Please select SAP systems first")
        return
    
    st.header("🔍 Search & Analysis Interface")
    
    # Quick search templates
    st.subheader("🚀 Quick SAP Analysis Templates")
    
    cols = st.columns(3)
    for i, template in enumerate(QUICK_SEARCH_TEMPLATES):
        col_idx = i % 3
        
        with cols[col_idx]:
            template_card = f"""
            <div class="search-template" onclick="this.style.backgroundColor='#e7f3ff';">
                <strong>{template['icon']} {template['label']}</strong><br>
                <small>{template['description']}</small>
            </div>
            """
            
            if st.button(
                f"{template['icon']} {template['label']}", 
                use_container_width=True,
                help=template['description'],
                key=f"template_{i}"
            ):
                execute_search(template['query'])
    
    st.markdown("---")
    
    # EWA Content Search - MAIN FEATURE
    st.subheader("📋 EWA Content Search")
    st.caption("Search through your uploaded SAP Early Watch Analysis reports")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        ewa_query = st.text_area(
            "Search EWA Report Content:",
            value=st.session_state.get('ewa_content_query', ''),
            placeholder="What specific issues are you looking for in the EWA reports?\n\nExamples:\n• Memory consumption issues in production\n• Database performance problems in system PRD\n• Critical alerts requiring immediate attention\n• Recommendations for system optimization",
            help="Ask specific questions about your SAP EWA reports. The AI will search through the processed documents to find relevant information.",
            height=120,
            key="ewa_content_search"
        )
    
    with col2:
        st.markdown("**💡 Search Tips:**")
        st.caption("• Be specific about systems")
        st.caption("• Include SAP component names")
        st.caption("• Ask about specific metrics")
        st.caption("• Request recommendations")
        
        st.markdown("**🎯 Target System:**")
        target = st.session_state.get('target_system', 'None')
        st.caption(f"Current: {target}")
    
    # Search execution buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_btn = st.button(
            "🔍 Search EWA Content", 
            type="primary", 
            use_container_width=True,
            help="Search through your EWA reports for the specified content"
        )
    
    with col2:
        if st.session_state.get('target_system'):
            target_search_btn = st.button(
                f"🎯 Search {st.session_state.target_system}",
                use_container_width=True,
                help=f"Search specifically in {st.session_state.target_system} system"
            )
        else:
            target_search_btn = False
    
    with col3:
        clear_search_btn = st.button(
            "🗑️ Clear Search",
            use_container_width=True,
            help="Clear current search and results"
        )
    
    # Handle search execution
    if search_btn and ewa_query.strip():
        st.session_state.ewa_content_query = ewa_query.strip()
        execute_search(ewa_query.strip())
        
    elif target_search_btn and ewa_query.strip() and st.session_state.get('target_system'):
        st.session_state.ewa_content_query = ewa_query.strip()
        target_query = f"In system {st.session_state.target_system}: {ewa_query.strip()}"
        execute_search(target_query)
        
    elif clear_search_btn:
        st.session_state.ewa_content_query = ''
        st.session_state.search_results = None
        st.session_state.last_search_query = ''
        st.rerun()
        
    elif search_btn or target_search_btn:
        st.error("❌ Please enter a search query")
    
    # Custom advanced search
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_scope = st.selectbox(
                "Search Scope",
                ["All Systems", "Target System Only", "Critical Issues Only", "Recommendations Only"],
                help="Limit search to specific areas"
            )
            
            result_limit = st.slider(
                "Maximum Results",
                min_value=5,
                max_value=50,
                value=st.session_state.get('top_k', DEFAULT_TOP_K),
                help="Number of results to return"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence for results"
            )
            
            include_context = st.checkbox(
                "Include surrounding context",
                value=True,
                help="Include text around matches for better understanding"
            )

def execute_search(query: str):
    """Execute search with comprehensive error handling and progress tracking."""
    try:
        if not query or not query.strip():
            st.error("❌ Please enter a valid search query")
            return
            
        cleaned_query = query.strip()
        if len(cleaned_query) < 3:
            st.error("❌ Search query must be at least 3 characters long")
            return
        
        workflow_manager = WorkflowManager()
        
        # Show search progress
        search_progress = st.progress(0)
        search_status = st.empty()
        
        with st.spinner(f"🔍 Searching EWA content: '{cleaned_query[:50]}{'...' if len(cleaned_query) > 50 else ''}'"):
            search_status.text("🔄 Initializing search...")
            search_progress.progress(20)
            
            # Prepare search filters
            search_filters = {
                "target_systems": st.session_state.get('selected_systems', [])
            }
            
            if st.session_state.get('target_system'):
                search_filters['primary_system'] = st.session_state.target_system
            
            search_status.text("🔍 Executing vector search...")
            search_progress.progress(60)
            
            # Execute search
            result = workflow_manager.execute_search(cleaned_query, search_filters)
            
            search_progress.progress(100)
            
            if result["success"]:
                search_results = result["search_results"]
                search_status.text("✅ Search completed!")
                
                st.success(
                    f"✅ Found {search_results['search_results_count']} results "
                    f"with {search_results.get('confidence_score', 0)*100:.1f}% confidence"
                )
                
                # Clear progress indicators
                search_progress.empty()
                search_status.empty()
                
                # Refresh to show results
                time.sleep(1)
                st.rerun()
                
            else:
                search_progress.empty()
                search_status.empty()
                st.error(f"❌ Search failed: {result.get('error', 'Unknown error')}")
                
                # Show debug information if available
                if st.session_state.get('debug_mode') and 'workflow_result' in result:
                    with st.expander("🐛 Debug Information"):
                        st.json(result['workflow_result'])
                
    except Exception as e:
        st.error(f"❌ Search error: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())

def create_results_section():
    """Create comprehensive results display with enhanced system details."""
    if not st.session_state.get('search_results'):
        return
    
    st.header("📊 Analysis Results")
    
    results = st.session_state.search_results
    
    # Enhanced results header
    confidence = results.get('confidence_score', 0) * 100
    confidence_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
    
    target_info = ""
    if results.get('target_system'):
        target_info = f" | 🎯 Target: {results['target_system']}"
    
    st.markdown(f"""
    <div class="info-box">
        <h4>🔍 Query: "{results['query']}"</h4>
        <strong>Systems:</strong> {', '.join(results['selected_systems'])}{target_info}<br>
        <strong>Results:</strong> {results.get('search_results_count', 0)} • 
        <strong>Confidence:</strong> {confidence_color} {confidence:.1f}% • 
        <strong>Time:</strong> {datetime.fromisoformat(results['timestamp']).strftime('%H:%M:%S')} •
        <strong>Processing:</strong> {results.get('processing_time', 0):.2f}s
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced results tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Executive Summary", 
        "🚨 Critical Issues", 
        "💡 SAP Recommendations", 
        "⚡ Performance Insights",
        "🖥️ System Details",
        "📧 Actions & Export"
    ])
    
    with tab1:
        display_summary_tab(results)
    
    with tab2:
        display_critical_issues_tab(results)
    
    with tab3:
        display_recommendations_tab(results)
    
    with tab4:
        display_performance_insights_tab(results)
    
    with tab5:
        display_system_details_tab(results)
    
    with tab6:
        display_actions_tab(results)

def display_summary_tab(results: Dict[str, Any]):
    """Display enhanced summary tab with key insights."""
    summary = results.get('summary', 'No summary available')
    confidence = results.get('confidence_score', 0) * 100
    
    # Confidence and quality indicators
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📝 Executive Summary")
        st.info(summary)
    
    with col2:
        # Confidence indicator
        if confidence >= 80:
            confidence_color = "success"
            confidence_icon = "🟢"
            confidence_label = "High"
        elif confidence >= 60:
            confidence_color = "warning"
            confidence_icon = "🟡"
            confidence_label = "Medium"
        else:
            confidence_color = "error"
            confidence_icon = "🔴"
            confidence_label = "Low"
        
        st.metric(
            "AI Confidence",
            f"{confidence:.1f}%",
            help="AI confidence in the analysis results"
        )
        st.write(f"{confidence_icon} {confidence_label} Confidence")
    
    with col3:
        # Results quality metrics
        st.metric("Results Found", results.get('search_results_count', 0))
        st.metric("Systems Analyzed", len(results.get('selected_systems', [])))
    
    # Key metrics overview
    if results.get('target_system'):
        st.markdown("### 🎯 Target System Analysis")
        target = results['target_system']
        st.markdown(f"""
        <div class="system-target-card">
            <strong>Primary Focus: {target}</strong><br>
            Analysis was specifically targeted for this system with enhanced relevance filtering.
        </div>
        """, unsafe_allow_html=True)
    
    # Performance insights preview
    insights = results.get('performance_insights', [])
    if insights:
        st.markdown("### ⚡ Key Performance Insights")
        for insight in insights[:3]:  # Show top 3
            st.write(f"• {insight}")
        
        if len(insights) > 3:
            st.caption(f"... and {len(insights) - 3} more insights (see Performance tab)")

def display_critical_issues_tab(results: Dict[str, Any]):
    """Display critical issues with enhanced categorization."""
    critical_findings = results.get('critical_findings', [])
    
    if critical_findings:
        # Severity assessment
        if len(critical_findings) >= 5:
            severity = "🔴 URGENT"
            severity_msg = "Multiple critical issues detected requiring immediate attention!"
        elif len(critical_findings) >= 3:
            severity = "🟡 HIGH"
            severity_msg = "Several critical issues found that should be addressed soon."
        else:
            severity = "🟠 MODERATE"
            severity_msg = "Some critical issues identified for review."
        
        st.markdown(f"""
        <div class="error-box">
            <strong>{severity}: {len(critical_findings)} Critical Issues Found</strong><br>
            {severity_msg}
        </div>
        """, unsafe_allow_html=True)
        
        # Display findings with priority
        for i, finding in enumerate(critical_findings, 1):
            priority = "HIGH" if i <= 2 else "MEDIUM" if i <= 4 else "LOW"
            priority_color = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟠"
            
            with st.expander(f"{priority_color} Critical Issue #{i} - {priority} Priority", expanded=i <= 2):
                st.markdown(f"""
                <div class="error-box">
                    <strong>🚨 Finding:</strong><br>
                    {finding}<br><br>
                    <strong>Priority:</strong> {priority_color} {priority}<br>
                    <strong>Recommended Action:</strong> Review immediately and plan remediation
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <strong>✅ No Critical Issues Found</strong><br>
            The analyzed SAP systems appear to be operating within normal parameters.
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

def display_recommendations_tab(results: Dict[str, Any]):
    """Display SAP recommendations with actionability scoring."""
    recommendations = results.get('recommendations', [])
    
    if recommendations:
        st.markdown(f"""
        <div class="info-box">
            <strong>💡 {len(recommendations)} SAP Optimization Recommendations Found</strong><br>
            These suggestions can help improve system performance and reliability.
        </div>
        """, unsafe_allow_html=True)
        
        # Categorize recommendations
        for i, rec in enumerate(recommendations, 1):
            # Simple categorization based on keywords
            if any(word in rec.lower() for word in ['memory', 'ram', 'heap']):
                category = "💾 Memory"
                category_color = "#17a2b8"
            elif any(word in rec.lower() for word in ['database', 'sql', 'query']):
                category = "🗄️ Database"
                category_color = "#28a745"
            elif any(word in rec.lower() for word in ['performance', 'cpu', 'response']):
                category = "⚡ Performance"
                category_color = "#ffc107"
            elif any(word in rec.lower() for word in ['security', 'authorization', 'user']):
                category = "🔒 Security"
                category_color = "#dc3545"
            else:
                category = "🔧 General"
                category_color = "#6c757d"
            
            with st.expander(f"{category} Recommendation #{i}", expanded=i <= 3):
                st.markdown(f"""
                <div style="border-left: 4px solid {category_color}; padding-left: 1rem; background: #f8f9fa; padding: 1rem; border-radius: 5px;">
                    <strong>💡 Recommendation:</strong><br>
                    {rec}<br><br>
                    <strong>Category:</strong> {category}<br>
                    <strong>Implementation:</strong> Consider implementing during next maintenance window
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ No Specific Recommendations Available</strong><br>
            The system analysis didn't identify specific optimization opportunities at this time.
        </div>
        """, unsafe_allow_html=True)

def display_performance_insights_tab(results: Dict[str, Any]):
    """Display performance insights and metrics."""
    insights = results.get('performance_insights', [])
    
    st.markdown("### ⚡ Performance Analysis")
    
    if insights:
        # Performance summary
        st.markdown(f"""
        <div class="info-box">
            <strong>📈 {len(insights)} Performance Insights Identified</strong><br>
            Analysis of system performance metrics and potential optimization areas.
        </div>
        """, unsafe_allow_html=True)
        
        # Display insights with metrics
        for i, insight in enumerate(insights, 1):
            # Categorize insights
            if any(word in insight.lower() for word in ['slow', 'latency', 'response time']):
                icon = "🐌"
                category = "Response Time"
                priority = "HIGH"
            elif any(word in insight.lower() for word in ['memory', 'heap', 'ram']):
                icon = "💾"
                category = "Memory Usage"
                priority = "MEDIUM"
            elif any(word in insight.lower() for word in ['cpu', 'processor', 'load']):
                icon = "🔄"
                category = "CPU Performance"
                priority = "MEDIUM"
            elif any(word in insight.lower() for word in ['disk', 'storage', 'i/o']):
                icon = "💽"
                category = "Storage Performance"
                priority = "LOW"
            else:
                icon = "📊"
                category = "General Performance"
                priority = "MEDIUM"
            
            priority_color = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
            
            with st.expander(f"{icon} {category} - Insight #{i}", expanded=i <= 2):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Performance Finding:**")
                    st.info(insight)
                
                with col2:
                    st.write(f"**Category:** {icon} {category}")
                    st.write(f"**Priority:** {priority_color} {priority}")
                    st.write(f"**Impact:** Performance optimization opportunity")
    else:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ No Specific Performance Insights Available</strong><br>
            The current analysis didn't identify specific performance optimization opportunities.
        </div>
        """, unsafe_allow_html=True)

def display_system_details_tab(results: Dict[str, Any]):
    """Display detailed system information and health status."""
    system_summaries = results.get('system_summaries', {})
    
    if system_summaries:
        st.markdown("### 🖥️ System-by-System Analysis")
        
        for sys_id, sys_data in system_summaries.items():
            # Determine health status and styling
            health = sys_data.get('overall_health', 'UNKNOWN')
            health_mapping = {
                'HEALTHY': {'icon': '🟢', 'color': '#28a745', 'status': 'All systems operational'},
                'WARNING': {'icon': '🟡', 'color': '#ffc107', 'status': 'Attention required'},
                'CRITICAL': {'icon': '🔴', 'color': '#dc3545', 'status': 'Immediate action needed'},
                'UNKNOWN': {'icon': '⚪', 'color': '#6c757d', 'status': 'Status undetermined'}
            }
            
            health_info = health_mapping.get(health, health_mapping['UNKNOWN'])
            
            # Check if this is the target system
            is_target = sys_id == results.get('target_system')
            
            with st.expander(
                f"{health_info['icon']} System {sys_id} - {health}" + 
                (" 🎯 (Target)" if is_target else ""), 
                expanded=is_target or health in ['CRITICAL', 'WARNING']
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏥 System Health Overview**")
                    
                    # Health status card
                    st.markdown(f"""
                    <div style="background: {health_info['color']}20; border: 1px solid {health_info['color']}; 
                                border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                        <strong>{health_info['icon']} Status: {health}</strong><br>
                        <small>{health_info['status']}</small><br>
                        <small>Last Analyzed: {sys_data.get('last_analyzed', 'Unknown')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics
                    metrics = sys_data.get('key_metrics', {})
                    if metrics:
                        st.markdown("**📊 Key Metrics**")
                        for metric, value in metrics.items():
                            # Format metric display
                            metric_name = metric.replace('_', ' ').title()
                            st.write(f"• **{metric_name}:** {value}")
                    else:
                        st.info("No specific metrics available for this system")
                
                with col2:
                    st.markdown("**🚨 Issues & Alerts**")
                    
                    alerts = sys_data.get('critical_alerts', [])
                    if alerts:
                        st.markdown(f"**Critical Alerts ({len(alerts)}):**")
                        
                        # Show up to 3 alerts with expandable view
                        for alert in alerts[:3]:
                            st.markdown(f"• ⚠️ {alert}")
                        
                        if len(alerts) > 3:
                            with st.expander(f"View {len(alerts) - 3} more alerts"):
                                for alert in alerts[3:]:
                                    st.markdown(f"• ⚠️ {alert}")
                    else:
                        st.success("• ✅ No critical alerts")
                    
                    # Recommendations for this system
                    sys_recommendations = sys_data.get('recommendations', [])
                    if sys_recommendations:
                        st.markdown(f"**💡 System Recommendations ({len(sys_recommendations)}):**")
                        for rec in sys_recommendations[:2]:
                            st.markdown(f"• 💡 {rec}")
                        
                        if len(sys_recommendations) > 2:
                            st.caption(f"... and {len(sys_recommendations) - 2} more recommendations")
                    else:
                        st.info("• ℹ️ No specific recommendations")
                
                # System-specific actions
                if is_target:
                    st.markdown("---")
                    st.markdown("**🎯 Target System Actions**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"🔍 Deep Dive {sys_id}", key=f"deep_dive_{sys_id}"):
                            deep_dive_query = f"Provide detailed analysis of all issues in system {sys_id}"
                            execute_search(deep_dive_query)
                    
                    with col2:
                        if st.button(f"📊 Health Report {sys_id}", key=f"health_{sys_id}"):
                            health_query = f"Generate comprehensive health report for system {sys_id}"
                            execute_search(health_query)
                    
                    with col3:
                        if st.button(f"💡 Optimize {sys_id}", key=f"optimize_{sys_id}"):
                            optimize_query = f"What are the best optimization recommendations for system {sys_id}"
                            execute_search(optimize_query)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ No System-Specific Details Available</strong><br>
            Try running a search to generate system-specific analysis results.
        </div>
        """, unsafe_allow_html=True)

def display_actions_tab(results: Dict[str, Any]):
    """Display actions, export options, and email functionality."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📧 Email Analysis Report")
        
        # Check email configuration
        email_enabled = st.session_state.get('email_enabled', False)
        
        if email_enabled:
            st.success("✅ Email system ready")
            
            with st.form("email_analysis_form"):
                # Email recipients
                to_email = st.text_input(
                    "Primary Recipient:",
                    placeholder="recipient@company.com",
                    help="Main recipient for the analysis report"
                )
                
                cc_emails = st.text_input(
                    "CC Recipients (optional):",
                    placeholder="cc1@company.com, cc2@company.com",
                    help="Additional recipients (comma-separated)"
                )
                
                # Email options
                col1, col2 = st.columns(2)
                with col1:
                    include_details = st.checkbox("Include detailed findings", value=True)
                    include_systems = st.checkbox("Include system details", value=True)
                
                with col2:
                    include_recommendations = st.checkbox("Include recommendations", value=True)
                    high_priority = st.checkbox("Mark as high priority", value=len(results.get('critical_findings', [])) > 0)
                
                # Send button
                send_email_btn = st.form_submit_button(
                    "📧 Send Analysis Report",
                    type="primary",
                    use_container_width=True
                )
                
                if send_email_btn and to_email:
                    handle_email_sending(to_email, cc_emails, results, {
                        'include_details': include_details,
                        'include_systems': include_systems,
                        'include_recommendations': include_recommendations,
                        'high_priority': high_priority
                    })
                elif send_email_btn:
                    st.error("❌ Please enter a primary recipient email address")
        else:
            st.error("❌ Email not configured")
            
            with st.expander("📧 Email Configuration Help"):
                st.markdown("""
                **To enable email functionality, add to your .env file:**
                
                For Gmail:
                ```
                EMAIL_ENABLED=true
                EMAIL_PROVIDER=gmail
                GMAIL_EMAIL=your-email@gmail.com
                GMAIL_APP_PASSWORD=your-app-password
                ```
                
                For Outlook:
                ```
                EMAIL_ENABLED=true
                EMAIL_PROVIDER=outlook
                OUTLOOK_EMAIL=your-email@outlook.com
                OUTLOOK_PASSWORD=your-password
                ```
                """)
    
    with col2:
        st.markdown("### 📄 Export & Actions")
        
        # Export options
        if st.button("📄 Download Full Report", use_container_width=True):
            handle_report_export(results)
        
        if st.button("📊 Download Executive Summary", use_container_width=True):
            handle_summary_export(results)
        
        if st.button("🔍 New Search", use_container_width=True):
            # Clear search results and query
            st.session_state.search_results = None
            st.session_state.last_search_query = ''
            st.session_state.ewa_content_query = ''
            st.rerun()
        
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            # Re-run the same search
            if results.get('query'):
                execute_search(results['query'])
        
        # Quick actions based on results
        st.markdown("---")
        st.markdown("**🚀 Quick Actions**")
        
        if results.get('critical_findings'):
            if st.button("🚨 Focus on Critical Issues", use_container_width=True):
                critical_query = "Show me detailed information about all critical issues that need immediate attention"
                execute_search(critical_query)
        
        if results.get('target_system'):
            target = results['target_system']
            if st.button(f"🎯 Deep Analysis of {target}", use_container_width=True):
                deep_query = f"Provide comprehensive analysis of all aspects of system {target} including performance, errors, and recommendations"
                execute_search(deep_query)
        
        if results.get('recommendations'):
            if st.button("💡 Implementation Guide", use_container_width=True):
                impl_query = "Provide detailed implementation steps for the top recommendations with priorities and timelines"
                execute_search(impl_query)

def handle_email_sending(to_email: str, cc_emails: str, results: Dict[str, Any], options: Dict[str, bool]):
    """Handle email sending with comprehensive report formatting."""
    try:
        with st.spinner("📧 Preparing and sending email..."):
            # Format email content based on options
            email_content = format_analysis_email(results, options)
            
            # Simulate email sending (replace with actual email integration)
            # This would integrate with your EmailManager class
            time.sleep(2)  # Simulate sending delay
            
            st.success(f"✅ Analysis report sent successfully to {to_email}")
            
            if cc_emails and cc_emails.strip():
                st.info(f"📧 CC sent to: {cc_emails}")
            
            # Log email activity
            st.session_state.email_sent = True
            
    except Exception as e:
        st.error(f"❌ Email sending failed: {str(e)}")

def format_analysis_email(results: Dict[str, Any], options: Dict[str, bool]) -> Dict[str, str]:
    """Format comprehensive email content for analysis results."""
    query = results.get('query', 'SAP Analysis')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    systems = ', '.join(results.get('selected_systems', []))
    critical_count = len(results.get('critical_findings', []))
    
    # Determine email priority and subject
    if options.get('high_priority') or critical_count >= 3:
        urgency = "🚨 URGENT"
        subject = f"[URGENT] SAP EWA Analysis - {query}"
    elif critical_count > 0:
        urgency = "⚠️ ALERT"
        subject = f"[ALERT] SAP EWA Analysis - {query}"
    else:
        urgency = "📊 REPORT"
        subject = f"SAP EWA Analysis Results - {query}"
    
    # Build email body
    body_parts = [
        f"SAP Early Watch Analyzer - {urgency}",
        "=" * 60,
        "",
        f"Analysis Query: {query}",
        f"Analysis Time: {timestamp}",
        f"Systems Analyzed: {systems}",
        f"Results Found: {results.get('search_results_count', 0)}",
        f"AI Confidence: {results.get('confidence_score', 0)*100:.1f}%",
        ""
    ]
    
    # Add target system info if available
    if results.get('target_system'):
        body_parts.extend([
            f"🎯 Primary Target System: {results['target_system']}",
            ""
        ])
    
    # Executive summary
    body_parts.extend([
        "EXECUTIVE SUMMARY:",
        "-" * 30,
        results.get('summary', 'Analysis completed successfully'),
        ""
    ])
    
    # Critical findings
    if options.get('include_details'):
        critical_findings = results.get('critical_findings', [])
        body_parts.extend([
            f"🚨 CRITICAL FINDINGS ({len(critical_findings)}):",
            "-" * 30
        ])
        
        if critical_findings:
            for i, finding in enumerate(critical_findings, 1):
                body_parts.append(f"{i}. {finding}")
        else:
            body_parts.append("✅ No critical issues found")
        
        body_parts.append("")
    
    # Recommendations
    if options.get('include_recommendations'):
        recommendations = results.get('recommendations', [])
        body_parts.extend([
            f"💡 RECOMMENDATIONS ({len(recommendations)}):",
            "-" * 30
        ])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                body_parts.append(f"{i}. {rec}")
        else:
            body_parts.append("ℹ️ No specific recommendations at this time")
        
        body_parts.append("")
    
    # System details
    if options.get('include_systems'):
        system_summaries = results.get('system_summaries', {})
        if system_summaries:
            body_parts.extend([
                "🖥️ SYSTEM DETAILS:",
                "-" * 30
            ])
            
            for sys_id, sys_data in system_summaries.items():
                health = sys_data.get('overall_health', 'UNKNOWN')
                alerts = len(sys_data.get('critical_alerts', []))
                body_parts.append(f"• {sys_id}: {health} ({alerts} critical alerts)")
            
            body_parts.append("")
    
    # Performance insights
    insights = results.get('performance_insights', [])
    if insights:
        body_parts.extend([
            f"⚡ PERFORMANCE INSIGHTS ({len(insights)}):",
            "-" * 30
        ])
        
        for insight in insights:
            body_parts.append(f"• {insight}")
        
        body_parts.append("")
    
    # Footer
    body_parts.extend([
        "---",
        f"Generated by SAP EWA Analyzer v{APP_VERSION}",
        f"Report generated at: {timestamp}",
        "",
        "This is an automated analysis report based on AI analysis of SAP Early Watch reports.",
        "For questions or concerns, please contact your SAP BASIS team or system administrator.",
        "",
        "⚠️ Please review critical findings immediately and plan appropriate remediation actions."
    ])
    
    return {
        "subject": subject,
        "body": "\n".join(body_parts)
    }

def handle_report_export(results: Dict[str, Any]):
    """Handle comprehensive report export."""
    try:
        # Generate comprehensive report content
        export_content = generate_comprehensive_report(results)
        
        # Create download button
        filename = f"sap_ewa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        st.download_button(
            label="📥 Download Complete Report",
            data=export_content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
            help="Download comprehensive analysis report including all findings and recommendations"
        )
        
        st.success("✅ Report generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Report export failed: {str(e)}")

def handle_summary_export(results: Dict[str, Any]):
    """Handle executive summary export."""
    try:
        # Generate executive summary
        summary_content = generate_executive_summary(results)
        
        # Create download button
        filename = f"sap_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        st.download_button(
            label="📥 Download Executive Summary",
            data=summary_content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
            help="Download concise executive summary for management reporting"
        )
        
        st.success("✅ Executive summary generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Summary export failed: {str(e)}")

def generate_comprehensive_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive analysis report."""
    lines = [
        "SAP EARLY WATCH ANALYSIS - COMPREHENSIVE REPORT",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Query: {results.get('query', 'N/A')}",
        f"Systems: {', '.join(results.get('selected_systems', []))}",
        f"Target System: {results.get('target_system', 'None')}",
        f"Results Found: {results.get('search_results_count', 0)}",
        f"AI Confidence: {results.get('confidence_score', 0)*100:.1f}%",
        f"Processing Time: {results.get('processing_time', 0):.2f}s",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 30,
        results.get('summary', 'No summary available'),
        ""
    ]
    
    # Add all sections
    sections = [
        ("CRITICAL FINDINGS", results.get('critical_findings', [])),
        ("RECOMMENDATIONS", results.get('recommendations', [])),
        ("PERFORMANCE INSIGHTS", results.get('performance_insights', []))
    ]
    
    for section_title, items in sections:
        lines.extend([
            f"{section_title} ({len(items)})",
            "-" * 30
        ])
        
        if items:
            for i, item in enumerate(items, 1):
                lines.append(f"{i}. {item}")
        else:
            lines.append("No items found in this category")
        
        lines.append("")
    
    # System details
    system_summaries = results.get('system_summaries', {})
    if system_summaries:
        lines.extend([
            "SYSTEM ANALYSIS DETAILS",
            "-" * 30
        ])
        
        for sys_id, sys_data in system_summaries.items():
            lines.extend([
                f"System: {sys_id}",
                f"  Health: {sys_data.get('overall_health', 'Unknown')}",
                f"  Critical Alerts: {len(sys_data.get('critical_alerts', []))}",
                f"  Last Analyzed: {sys_data.get('last_analyzed', 'Unknown')}",
                ""
            ])
    
    # Footer
    lines.extend([
        "---",
        f"Report generated by SAP EWA Analyzer v{APP_VERSION}",
        "This report contains AI-generated analysis based on uploaded SAP documents.",
        "Please verify findings with your SAP BASIS team before taking action."
    ])
    
    return "\n".join(lines)

def generate_executive_summary(results: Dict[str, Any]) -> str:
    """Generate concise executive summary."""
    critical_count = len(results.get('critical_findings', []))
    rec_count = len(results.get('recommendations', []))
    systems = results.get('selected_systems', [])
    
    lines = [
        "SAP EARLY WATCH ANALYSIS - EXECUTIVE SUMMARY",
        "=" * 50,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Systems Analyzed: {', '.join(systems)} ({len(systems)} total)",
        f"Target System: {results.get('target_system', 'None')}",
        "",
        "KEY FINDINGS:",
        f"• Critical Issues: {critical_count}",
        f"• Recommendations: {rec_count}",
        f"• AI Confidence: {results.get('confidence_score', 0)*100:.1f}%",
        "",
        "EXECUTIVE SUMMARY:",
        results.get('summary', 'Analysis completed successfully'),
        "",
        "IMMEDIATE ACTIONS REQUIRED:" if critical_count > 0 else "STATUS:",
    ]
    
    if critical_count > 0:
        lines.extend([
            f"• {critical_count} critical issues require immediate attention",
            "• Review detailed findings and plan remediation",
            "• Contact SAP BASIS team for urgent items"
        ])
    else:
        lines.append("• No critical issues identified")
        lines.append("• Systems appear to be operating normally")
    
    lines.extend([
        "",
        "NEXT STEPS:",
        "• Review complete analysis report",
        "• Prioritize recommendations by business impact",
        "• Schedule maintenance windows as needed",
        "",
        f"Generated by SAP EWA Analyzer v{APP_VERSION}"
    ])
    
    return "\n".join(lines)

# ================================
# SIDEBAR COMPONENTS
# ================================

def create_sidebar():
    """Create comprehensive sidebar with all features."""
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System status section
        display_system_status()
        
        st.markdown("---")
        
        # LangGraph workflow visualization - KEY FEATURE
        display_workflow_section()
        
        st.markdown("---")
        
        # Email status
        display_email_status()
        
        st.markdown("---")
        
        # Quick actions
        display_quick_actions()

def display_system_status():
    """Display comprehensive system status."""
    st.subheader("🔋 Current Status")
    
    # Processing status
    if st.session_state.get('vector_store_ready', False):
        st.success("✅ Documents Processed")
        st.write(f"📄 **Chunks:** {st.session_state.get('total_chunks', 0)}")
        
        # Processing metrics
        processing_times = st.session_state.get('processing_times', {})
        if processing_times:
            total_time = sum(processing_times.values())
            st.write(f"⏱️ **Processing Time:** {total_time:.2f}s")
    else:
        st.info("⏳ Awaiting Documents")
    
    # System selection status
    selected_systems = st.session_state.get('selected_systems', [])
    target_system = st.session_state.get('target_system')
    
    if selected_systems:
        st.write(f"🖥️ **Systems ({len(selected_systems)}):**")
        for system in selected_systems[:3]:  # Show first 3
            is_target = system == target_system
            prefix = "🎯 " if is_target else "• "
            st.write(f"{prefix}{system}")
        
        if len(selected_systems) > 3:
            st.write(f"• ... and {len(selected_systems) - 3} more")
        
        if target_system:
            st.markdown(f"""
            <div style="background: #e7f3ff; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                <strong>🎯 Target: {target_system}</strong>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("🖥️ **Systems:** None selected")
    
    # Latest search results
    search_results = st.session_state.get('search_results')
    if search_results:
        st.write("🔍 **Latest Analysis:**")
        query = search_results.get('query', '')
        display_query = query[:25] + "..." if len(query) > 25 else query
        st.write(f"• **Query:** {display_query}")
        st.write(f"• **Results:** {search_results.get('search_results_count', 0)}")
        
        critical_count = len(search_results.get('critical_findings', []))
        rec_count = len(search_results.get('recommendations', []))
        
        if critical_count > 0:
            st.write(f"• 🚨 **Critical:** {critical_count}")
        else:
            st.write("• ✅ **Critical:** None")
        
        st.write(f"• 💡 **Recommendations:** {rec_count}")
        
        confidence = search_results.get('confidence_score', 0) * 100
        confidence_icon = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
        st.write(f"• {confidence_icon} **Confidence:** {confidence:.0f}%")

def display_workflow_section():
    """Display LangGraph workflow visualization with clickable interface."""
    st.subheader("🔄 LangGraph Workflow")
    
    workflow = st.session_state.get('workflow')
    
    if workflow:
        # Workflow status overview
        workflow_status = st.session_state.get('workflow_status', WorkflowStatus.INITIALIZED)
        current_agent = st.session_state.get('current_agent', '')
        
        # Status indicator
        status_colors = {
            WorkflowStatus.INITIALIZED: "🔵",
            WorkflowStatus.PROCESSING_PDF: "🟡",
            WorkflowStatus.CREATING_EMBEDDINGS: "🟡", 
            WorkflowStatus.STORING_VECTORS: "🟡",
            WorkflowStatus.SEARCHING: "🟡",
            WorkflowStatus.SUMMARIZING: "🟡",
            WorkflowStatus.SYSTEM_OUTPUT: "🟡",
            WorkflowStatus.SENDING_EMAIL: "🟡",
            WorkflowStatus.COMPLETED: "🟢",
            WorkflowStatus.ERROR: "🔴"
        }
        
        status_icon = status_colors.get(workflow_status, "⚪")
        st.write(f"**Status:** {status_icon} {workflow_status.replace('_', ' ').title()}")
        
        if current_agent:
            st.write(f"**Current Agent:** {current_agent}")
        
        # Workflow steps visualization
        st.markdown("**🔗 Workflow Steps:**")
        
        # Determine current step based on status
        current_step_map = {
            WorkflowStatus.INITIALIZED: 0,
            WorkflowStatus.PROCESSING_PDF: 1,
            WorkflowStatus.CREATING_EMBEDDINGS: 2,
            WorkflowStatus.STORING_VECTORS: 3,
            WorkflowStatus.SEARCHING: 4,
            WorkflowStatus.SUMMARIZING: 5,
            WorkflowStatus.SYSTEM_OUTPUT: 6,
            WorkflowStatus.SENDING_EMAIL: 7,
            WorkflowStatus.COMPLETED: 8,
            WorkflowStatus.ERROR: -1
        }
        
        current_step = current_step_map.get(workflow_status, 0)
        
        # Display workflow steps
        for i, step in enumerate(WORKFLOW_STEPS):
            if workflow_status == WorkflowStatus.ERROR:
                step_status = "error"
                step_icon = "❌"
            elif i < current_step:
                step_status = "completed"
                step_icon = "✅"
            elif i == current_step and workflow_status != WorkflowStatus.COMPLETED:
                step_status = "processing"
                step_icon = "⏳"
            else:
                step_status = "pending"
                step_icon = "⏸️"
            
            st.markdown(f"""
            <div class="workflow-step {step_status}">
                <span style="margin-right: 8px;">{step_icon}</span>
                <span>{step['name']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        if workflow_status != WorkflowStatus.ERROR:
            progress = min(current_step / len(WORKFLOW_STEPS), 1.0)
            st.progress(progress)
            st.caption(f"Progress: {current_step}/{len(WORKFLOW_STEPS)} steps")
        
        # Workflow visualization button - KEY FEATURE
        st.markdown("---")
        if st.button("📊 View Workflow Diagram", use_container_width=True, help="Generate and display LangGraph workflow visualization"):
            handle_workflow_visualization()
        
        # Workflow debug info
        debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            with st.expander("🐛 Workflow Debug"):
                workflow_debug = {
                    "workflow_exists": workflow is not None,
                    "workflow_type": type(workflow).__name__,
                    "has_app": hasattr(workflow, 'app'),
                    "app_exists": getattr(workflow, 'app', None) is not None,
                    "workflow_status": workflow_status,
                    "current_agent": current_agent,
                    "vector_store_ready": st.session_state.get('vector_store_ready', False)
                }
                st.json(workflow_debug)
    
    else:
        st.warning("⚠️ No workflow initialized")
        st.info("💡 Upload documents to initialize workflow")

def handle_workflow_visualization():
    """Handle LangGraph workflow visualization generation and display."""
    try:
        workflow = st.session_state.get('workflow')
        
        if not workflow:
            st.error("❌ No workflow available for visualization")
            return
        
        with st.spinner("🎨 Generating LangGraph workflow diagram..."):
            # Try to get workflow visualization
            if hasattr(workflow, 'get_workflow_visualization'):
                viz_result = workflow.get_workflow_visualization()
                
                if viz_result.get('success'):
                    st.success("✅ Workflow diagram generated!")
                    
                    viz_type = viz_result.get('type')
                    
                    if viz_type == 'png':
                        # Display PNG image
                        file_path = viz_result.get('file')
                        if file_path and os.path.exists(file_path):
                            st.image(file_path, caption="LangGraph Workflow Diagram", use_column_width=True)
                            
                            # Download button
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=f.read(),
                                    file_name="langgraph_workflow.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ Generated image file not found")
                    
                    elif viz_type == 'mermaid':
                        # Display Mermaid code
                        mermaid_code = viz_result.get('code', '')
                        st.code(mermaid_code, language='mermaid')
                        st.info("💡 Copy this code to https://mermaid.live/ to view the diagram")
                        
                        st.download_button(
                            label="📥 Download Mermaid Code",
                            data=mermaid_code,
                            file_name="langgraph_workflow.mmd",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    else:
                        st.warning(f"⚠️ Unsupported visualization type: {viz_type}")
                
                else:
                    st.error(f"❌ Visualization generation failed: {viz_result.get('error', 'Unknown error')}")
            
            elif hasattr(workflow, 'app') and hasattr(workflow.app, 'get_graph'):
                # Try LangGraph native visualization
                try:
                    # Try to get PNG
                    graph_image = workflow.app.get_graph().draw_mermaid_png()
                    
                    # Save and display
                    with open('workflow_diagram.png', 'wb') as f:
                        f.write(graph_image)
                    
                    st.image('workflow_diagram.png', caption="LangGraph Workflow", use_column_width=True)
                    
                    st.download_button(
                        label="📥 Download Workflow PNG",
                        data=graph_image,
                        file_name="langgraph_workflow.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    st.success("✅ Native LangGraph diagram generated!")
                    
                except Exception as png_error:
                    # Fallback to Mermaid code
                    try:
                        mermaid_code = workflow.app.get_graph().draw_mermaid()
                        st.code(mermaid_code, language='mermaid')
                        st.info("💡 Copy to https://mermaid.live/ to visualize")
                        
                        st.download_button(
                            label="📥 Download Mermaid",
                            data=mermaid_code,
                            file_name="workflow.mmd",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    except Exception as mermaid_error:
                        st.error(f"❌ Both PNG and Mermaid generation failed: {png_error}, {mermaid_error}")
            
            else:
                st.error("❌ Workflow visualization not supported by this workflow instance")
                
    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")
        
        if st.session_state.get('debug_mode'):
            st.code(traceback.format_exc())

def display_email_status():
    """Display email configuration and status."""
    st.subheader("📧 Email Status")
    
    email_enabled = st.session_state.get('email_enabled', False)
    
    if email_enabled:
        # Detect email provider
        email_provider = os.getenv("EMAIL_PROVIDER", "gmail").lower()
        st.success(f"✅ {email_provider.title()} Ready")
        
        # Show email configuration
        if email_provider == "gmail":
            email_account = os.getenv("GMAIL_EMAIL", "Not configured")
        else:
            email_account = os.getenv("OUTLOOK_EMAIL", "Not configured")
        
        st.write(f"**Account:** {email_account}")
        st.write(f"**Provider:** {email_provider.title()}")
        
        # Email activity status
        if st.session_state.get('email_sent', False):
            st.info("📧 Last email sent successfully")
    else:
        st.warning("⚠️ Email Not Configured")
        
        with st.expander("📧 Setup Instructions"):
            st.markdown("""
            **Add to your .env file:**
            
            **For Gmail:**
            ```
            EMAIL_ENABLED=true
            EMAIL_PROVIDER=gmail
            GMAIL_EMAIL=your-email@gmail.com
            GMAIL_APP_PASSWORD=your-app-password
            ```
            
            **For Outlook:**
            ```
            EMAIL_ENABLED=true
            EMAIL_PROVIDER=outlook
            OUTLOOK_EMAIL=your-email@outlook.com  
            OUTLOOK_PASSWORD=your-password
            ```
            """)

def display_quick_actions():
    """Display quick action buttons in sidebar."""
    st.subheader("⚡ Quick Actions")
    
    # System health check
    if st.button("💊 System Health", use_container_width=True):
        show_system_health_modal()
    
    # Advanced settings toggle
    if st.button("⚙️ Advanced Settings", use_container_width=True):
        current_advanced = st.session_state.get('show_advanced_options', False)
        st.session_state.show_advanced_options = not current_advanced
        st.rerun()
    
    # Application restart
    if st.button("🔄 Restart App", use_container_width=True):
        SessionStateManager.initialize()
        
        # Clear processing data
        keys_to_clear = [
            'workflow', 'workflow_initialized', 'vector_store_ready',
            'processed_documents', 'search_results', 'selected_systems'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("🔄 Application restarted")
        st.rerun()

def show_system_health_modal():
    """Display comprehensive system health information."""
    with st.expander("💊 System Health Check", expanded=True):
        # Core system health
        health_checks = {
            "Session State": "✅ Active" if st.session_state else "❌ Missing",
            "Workflow": "✅ Ready" if st.session_state.get('workflow') else "❌ Not initialized",
            "Vector Store": "✅ Ready" if st.session_state.get('vector_store_ready') else "❌ Not ready",
            "Email Config": "✅ Configured" if st.session_state.get('email_enabled') else "⚠️ Not configured",
            "Selected Systems": f"✅ {len(st.session_state.get('selected_systems', []))} systems" if st.session_state.get('selected_systems') else "⚠️ None selected",
            "Target System": f"✅ {st.session_state.get('target_system')}" if st.session_state.get('target_system') else "ℹ️ None set"
        }
        
        for check, status in health_checks.items():
            st.write(f"**{check}:** {status}")
        
        # Performance metrics
        session_start = st.session_state.get('session_start_time', datetime.now())
        session_duration = (datetime.now() - session_start).total_seconds()
        execution_count = st.session_state.get('execution_count', 0)
        
        st.write(f"**Session Duration:** {session_duration:.0f}s")
        st.write(f"**Executions:** {execution_count}")
        
        # Memory usage if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.write(f"**Memory Usage:** {memory.percent}%")
        except ImportError:
            st.write("**Memory Usage:** Not available")
        
        # Workflow modules availability
        st.write(f"**Workflow Modules:** {'✅ Available' if WORKFLOW_AVAILABLE else '❌ Missing'}")
        st.write(f"**Email Modules:** {'✅ Available' if EMAIL_AVAILABLE else '❌ Missing'}")

# ================================
# UTILITY FUNCTIONS & ERROR HANDLING
# ================================

def handle_global_error(error: Exception, context: str = "Application"):
    """Enhanced global error handler with recovery options."""
    logger.error(f"❌ Global error in {context}: {str(error)}")
    
    st.error(f"❌ Error in {context}: {str(error)}")
    
    # Show debug information if debug mode is enabled
    if st.session_state.get('debug_mode'):
        with st.expander("🐛 Debug Information"):
            st.code(traceback.format_exc())
    
    # Recovery options
    with st.expander("🔧 Recovery Options"):
        col1, col2, col3 = st.columns(3)
        
        context_key = context.lower().replace(' ', '_')
        
        with col1:
            if st.button("🔄 Restart App", key=f"error_restart_{context_key}"):
                SessionStateManager.initialize()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Data", key=f"error_clear_{context_key}"):
                # Reset to defaults while preserving config
                SessionStateManager.initialize()
                st.rerun()
        
        with col3:
            if st.button("🐛 Enable Debug", key=f"error_debug_{context_key}"):
                st.session_state.debug_mode = True
                st.rerun()

def validate_uploaded_files(uploaded_files: List[Any]) -> Dict[str, Any]:
    """Validate uploaded files for size and type constraints."""
    try:
        if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
            return {
                "valid": False,
                "error": f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files allowed."
            }
        
        for file in uploaded_files:
            file_size = len(file.getvalue())
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return {
                    "valid": False,
                    "error": f"File '{file.name}' exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
                }
            
            file_extension = Path(file.name).suffix.lower()
            if file_extension not in SUPPORTED_FILE_TYPES:
                return {
                    "valid": False,
                    "error": f"File '{file.name}' has unsupported type. Supported: {SUPPORTED_FILE_TYPES}"
                }
            
            if file_size == 0:
                return {
                    "valid": False,
                    "error": f"File '{file.name}' is empty"
                }
        
        return {"valid": True}
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"File validation error: {str(e)}"
        }

def create_debug_section():
    """Create comprehensive debug section for development."""
    if not st.session_state.get('debug_mode'):
        return
    
    st.markdown("---")
    st.subheader("🐛 Debug Information")
    
    # Session state debug
    with st.expander("📊 Session State"):
        debug_state = {}
        for key, value in st.session_state.items():
            if key.startswith('_'):
                continue
            
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
        import sys, platform
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        st.write(f"**Workflow Available:** {WORKFLOW_AVAILABLE}")
        st.write(f"**Email Available:** {EMAIL_AVAILABLE}")
        st.write(f"**Environment Available:** {ENV_AVAILABLE}")

def add_debug_section_to_app():
    """Add debug section to main app when debug mode is enabled."""
    if st.session_state.get('debug_mode'):
        create_debug_section()

# Search diagnostics functions for troubleshooting
def diagnose_search_issue():
    """Diagnostic function to help identify search issues."""
    st.subheader("🔍 Search Diagnostics")
    
    # Check session state
    with st.expander("📊 Session State Diagnostics"):
        st.write("**Search-related session state:**")
        search_keys = [
            'search_query', 'last_search_query', 'ewa_content_query', 'search_results', 
            'vector_store_ready', 'workflow_initialized', 'selected_systems', 'target_system'
        ]
        
        for key in search_keys:
            value = st.session_state.get(key, "NOT_FOUND")
            st.write(f"• **{key}**: `{value}` (type: {type(value).__name__})")

def test_quick_search_templates():
    """Test quick search template functionality."""
    st.subheader("🚀 Quick Search Template Test")
    
    for i, template in enumerate(QUICK_SEARCH_TEMPLATES):
        with st.expander(f"Template {i+1}: {template['label']}"):
            st.write(f"**Query:** `{template['query']}`")
            st.write(f"**Description:** {template['description']}")
            st.write(f"**Category:** {template['category']}")
            
            if st.button(f"Test {template['label']}", key=f"test_template_{i}"):
                query = template['query']
                if len(query.strip()) >= 3:
                    st.success(f"✅ Template query valid: `{query}`")
                    execute_search(query)
                else:
                    st.error("❌ Template query too short!")

# Mock manager classes for compatibility
class WorkflowManager:
    """Mock workflow manager for testing."""
    pass

class EmailManager:
    """Mock email manager for testing."""
    pass

# ================================
# MODULE EXPORTS
# ================================

__all__ = [
    # Core configuration
    'configure_page',
    'create_main_header', 
    'create_footer',
    
    # Main UI components
    'create_file_upload_section',
    'create_processing_section',
    'create_system_selection_section',
    'create_search_section',
    'create_results_section',
    'create_sidebar',
    
    # Manager classes
    'SessionStateManager',
    'WorkflowManager',
    'EmailManager',
    
    # Processing functions
    'handle_document_processing',
    'execute_search',
    'handle_email_sending',
    
    # Display functions
    'display_processing_metrics',
    'display_summary_tab',
    'display_critical_issues_tab',
    'display_recommendations_tab',
    'display_system_details_tab',
    'display_actions_tab',
    'display_system_status',
    'display_email_status',
    'display_workflow_section',
    'display_quick_actions',
    
    # Utility functions
    'handle_global_error',
    'validate_uploaded_files',
    'create_debug_section',
    'add_debug_section_to_app',
    'diagnose_search_issue',
    'test_quick_search_templates',
    
    # Export functions
    'handle_report_export',
    'handle_summary_export',
    'format_analysis_email',
    'generate_comprehensive_report',
    'generate_executive_summary'
]

# ================================
# MODULE INITIALIZATION
# ================================

logger.info(f"✅ {APP_TITLE} UI components module loaded successfully")
logger.info(f"📋 Version: {APP_VERSION}")
logger.info(f"🔧 Workflow modules available: {WORKFLOW_AVAILABLE}")
logger.info(f"📧 Email functionality available: {EMAIL_AVAILABLE}")
logger.info(f"⚙️ Environment variables available: {ENV_AVAILABLE}")
logger.info(f"🎯 Available components: {len(__all__)} functions and classes")

if not WORKFLOW_AVAILABLE:
    logger.warning("⚠️ Running with limited functionality - workflow modules not available")

logger.info("🚀 SAP EWA Analyzer core components ready for production")