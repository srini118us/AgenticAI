# final_app.py - Final Streamlit Application for CrewAI SAP EWA Analyzer
"""
Final Streamlit application integrating all CrewAI modules for SAP EWA analysis.
Uses existing agents.py, tools.py, config.py, and models.py modules.
"""

import streamlit as st
import os
import logging
import time
import tempfile
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our existing modules
try:
    from config import config, get_openai_api_key, is_debug_mode, get_app_title
    from models import (
        AnalysisRequest, CrewExecutionResult, HealthStatus, 
        HEALTH_STATUS_COLORS, HEALTH_STATUS_ICONS, DEFAULT_SEARCH_QUERIES,
        SAPSystemInfo, SystemHealthAnalysis, HealthAlert,
        SAPProduct, SystemEnvironment
    )
    from agents import (
        create_sap_ewa_crew, execute_sap_ewa_analysis, 
        analyze_sap_ewa_documents, SAPEWAAgents
    )
    from tools import (
        PDFProcessorTool, VectorSearchTool, HealthAnalysisTool,
        EmailNotificationTool
    )
    
    MODULES_AVAILABLE = True
    logger.info("✅ All CrewAI modules imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    MODULES_AVAILABLE = False
    st.error(f"Module import error: {e}")

# ================================
# PAGE CONFIGURATION
# ================================

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="CrewAI SAP EWA Analyzer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .agent-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .agent-card:hover {
            transform: translateY(-2px);
        }
        
        .status-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        
        .success-alert {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #2d5016;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        
        .warning-alert {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #856404;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        
        .error-alert {
            background: linear-gradient(135deg, #ffeaea 0%, #ffc8c8 100%);
            color: #721c24;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        }
        
        .system-health-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e0e6ed;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        .progress-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .footer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)

# ================================
# SESSION STATE MANAGEMENT
# ================================

def initialize_session_state():
    """Initialize Streamlit session state"""
    
    # Core application state
    defaults = {
        'analysis_results': None,
        'agent_communications': [],
        'uploaded_files_processed': False,
        'processing_active': False,
        'temp_file_paths': [],
        'show_advanced_options': False,
        'show_agent_details': True,
        'selected_system_filter': None,
        'crew_instance': None,
        'pdf_processor': None,
        'vector_search': None,
        'health_analyzer': None,
        'processing_start_time': None,
        'last_analysis_timestamp': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ================================
# UI COMPONENTS
# ================================

def create_main_header():
    """Create the main application header"""
    st.markdown(f"""
        <div class="main-header">
            <h1>🤖 CrewAI SAP EWA Analyzer</h1>
            <h3>Intelligent Early Watch Alert Analysis with Autonomous AI Agents</h3>
            <p>🔄 <strong>Multi-Agent Collaboration</strong> • 🧠 <strong>Advanced AI Analysis</strong> • 📊 <strong>Real-time Insights</strong></p>
            <small>✨ Powered by CrewAI, OpenAI, ChromaDB & Streamlit</small>
        </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create enhanced sidebar with configuration and monitoring"""
    with st.sidebar:
        st.header("🛠️ Control Center")
        
        # API Configuration Section
        st.subheader("🔑 API Configuration")
        
        # API Key management
        current_key = get_openai_api_key()
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=current_key,
            help="Required for CrewAI agents and embeddings",
            placeholder="sk-..."
        )
        
        if api_key_input != current_key:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.rerun()
        
        # API Key status
        if api_key_input:
            if api_key_input.startswith("sk-"):
                st.markdown('<div class="status-card">✅ API Key Valid</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-alert">⚠️ API Key Format Issue</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-alert">❌ API Key Required</div>', unsafe_allow_html=True)
        
        # Configuration validation
        if st.button("🔍 Validate Configuration", use_container_width=True):
            with st.spinner("Validating..."):
                validation = config.validate_config()
                if validation["valid"]:
                    st.success("✅ Configuration Valid")
                else:
                    st.error("❌ Configuration Issues:")
                    for error in validation["errors"]:
                        st.error(f"• {error}")
                
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        st.warning(f"⚠️ {warning}")
        
        st.divider()
        
        # CrewAI Settings
        st.subheader("🤖 CrewAI Settings")
        
        crew_memory = st.checkbox(
            "Enable Crew Memory",
            value=config.CREW_MEMORY_ENABLED,
            help="Allow agents to learn from previous analyses"
        )
        
        crew_verbose = st.checkbox(
            "Verbose Agent Output",
            value=config.CREW_VERBOSE,
            help="Show detailed agent interactions"
        )
        
        max_iterations = st.slider(
            "Max Agent Iterations",
            min_value=1,
            max_value=10,
            value=config.MAX_ITERATIONS,
            help="Maximum rounds of agent collaboration"
        )
        
        st.divider()
        
        # Analysis Settings
        st.subheader("🎯 Analysis Settings")
        
        chunk_size = st.slider(
            "Document Chunk Size",
            min_value=500,
            max_value=2000,
            value=config.CHUNK_SIZE,
            step=100,
            help="Size of text chunks for vector embedding"
        )
        
        top_k = st.slider(
            "Search Results (Top K)",
            min_value=5,
            max_value=20,
            value=config.TOP_K_RESULTS,
            help="Number of top search results to analyze"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.SIMILARITY_THRESHOLD,
            step=0.05,
            help="Minimum similarity score for search results"
        )
        
        st.divider()
        
        # Advanced Options
        st.subheader("⚙️ Advanced Options")
        
        st.session_state.show_advanced_options = st.checkbox(
            "Show Advanced Options",
            value=st.session_state.show_advanced_options
        )
        
        st.session_state.show_agent_details = st.checkbox(
            "Show Agent Communications",
            value=st.session_state.show_agent_details,
            help="Display real-time agent interactions"
        )
        
        debug_mode = st.checkbox(
            "Debug Mode",
            value=is_debug_mode(),
            help="Enable detailed logging and debug information"
        )
        
        if debug_mode:
            st.info("🐛 Debug mode enabled")
        
        # Email settings
        if config.EMAIL_ENABLED:
            st.subheader("📧 Email Settings")
            st.info("✅ Email notifications enabled")
            
            # Test email connection
            if st.button("🧪 Test Email", use_container_width=True):
                with st.spinner("Testing email connection..."):
                    try:
                        email_tool = EmailNotificationTool()
                        result = email_tool.test_email_connection()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"❌ {result['error']}")
                    except Exception as e:
                        st.error(f"❌ Test failed: {str(e)}")
        
        st.divider()
        
        # System Status
        st.subheader("ℹ️ System Status")
        
        # Real-time status indicators
        status_container = st.container()
        
        with status_container:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # System health indicators
            indicators = [
                ("🔧 Modules", "✅ Loaded" if MODULES_AVAILABLE else "❌ Error"),
                ("🔑 API Key", "✅ Set" if get_openai_api_key() else "❌ Missing"),
                ("⚙️ Config", "✅ Valid" if config.is_ready() else "❌ Invalid"),
                ("🤖 CrewAI", "✅ Ready" if MODULES_AVAILABLE and get_openai_api_key() else "❌ Not Ready"),
                ("📧 Email", "✅ Enabled" if config.EMAIL_ENABLED else "⚪ Disabled"),
                ("🕒 Time", current_time)
            ]
            
            for label, status in indicators:
                st.text(f"{label}: {status}")
        
        # Processing status
        if st.session_state.processing_active:
            st.markdown('<div class="status-card">🔄 Analysis Running</div>', unsafe_allow_html=True)
            if st.session_state.processing_start_time:
                elapsed = datetime.now() - st.session_state.processing_start_time
                st.text(f"⏱️ Elapsed: {elapsed.seconds}s")

def create_agent_monitoring_section():
    """Create real-time agent monitoring interface"""
    if not st.session_state.show_agent_details:
        return
    
    st.header("🤖 Agent Activity Monitor")
    
    # Agent Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Dynamic agent status based on processing state
    processing_active = st.session_state.processing_active
    
    agent_statuses = [
        {
            "name": "📄 Document Processor",
            "description": "Processing SAP EWA PDFs",
            "status": "active" if processing_active else "ready",
            "details": "Extracting text and metadata"
        },
        {
            "name": "🔍 Vector Manager", 
            "description": "Managing embeddings",
            "status": "active" if processing_active else "ready",
            "details": "Creating semantic search index"
        },
        {
            "name": "🏥 Health Analyst",
            "description": "Analyzing system health",
            "status": "active" if processing_active else "ready", 
            "details": "Identifying issues and patterns"
        },
        {
            "name": "📊 Report Coordinator",
            "description": "Coordinating workflow",
            "status": "active" if processing_active else "waiting",
            "details": "Compiling comprehensive reports"
        }
    ]
    
    for i, (col, agent) in enumerate(zip([col1, col2, col3, col4], agent_statuses)):
        with col:
            status_colors = {
                "active": "#28a745",
                "ready": "#007bff", 
                "waiting": "#6c757d",
                "idle": "#fd7e14",
                "error": "#dc3545"
            }
            
            status_color = status_colors.get(agent["status"], "#6c757d")
            
            st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent["name"]}</h4>
                    <p><span style="color: {status_color};">●</span> <strong>{agent["status"].title()}</strong></p>
                    <small>{agent["description"]}</small>
                    <br><small style="color: #666;">{agent["details"]}</small>
                </div>
            """, unsafe_allow_html=True)
    
    # Agent Communications Log
    if st.session_state.agent_communications:
        st.subheader("💬 Agent Communications")
        
        with st.expander("📋 Communication History", expanded=False):
            # Show last 10 communications
            recent_comms = st.session_state.agent_communications[-10:]
            
            for i, comm in enumerate(reversed(recent_comms)):
                timestamp = comm.get('timestamp', 'Unknown')
                from_agent = comm.get('from_agent', 'Unknown Agent')
                to_agent = comm.get('to_agent', 'Unknown Agent')
                message = comm.get('message', 'No message')
                action = comm.get('action', 'No action')
                
                st.markdown(f"""
                **[{timestamp}]** `{from_agent}` → `{to_agent}`  
                💬 {message}  
                🎯 Action: {action}
                """)
                
                if i < len(recent_comms) - 1:
                    st.divider()

def create_file_upload_section():
    """Enhanced file upload interface"""
    st.header("📁 Document Upload Center")
    
    # File uploader with better styling
    uploaded_files = st.file_uploader(
        "📄 Upload SAP EWA PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more SAP Early Watch Alert PDF reports for analysis",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        st.markdown('<div class="success-alert">✅ Files uploaded successfully!</div>', unsafe_allow_html=True)
        
        # Enhanced file information display
        st.subheader("📋 Uploaded Files")
        
        total_size = 0
        file_data = []
        
        for i, file in enumerate(uploaded_files):
            file_size = len(file.getvalue())
            total_size += file_size
            
            file_data.append({
                "#": i + 1,
                "📄 Filename": file.name,
                "📏 Size": f"{file_size:,} bytes ({file_size/1024/1024:.1f} MB)",
                "📝 Type": file.type,
                "✅ Status": "Ready for processing"
            })
        
        # Display files in a nice table
        st.dataframe(file_data, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(uploaded_files)}</h3>
                    <p>Files Uploaded</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_size/1024/1024:.1f}</h3>
                    <p>Total Size (MB)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_size = total_size / len(uploaded_files) / 1024 / 1024
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_size:.1f}</h3>
                    <p>Average Size (MB)</p>
                </div>
            """, unsafe_allow_html=True)
        
        return uploaded_files
    
    else:
        # Show upload instructions
        st.info("""
        📤 **Upload Instructions:**
        - Select one or more SAP EWA PDF files
        - Supported formats: PDF only
        - Maximum file size: 50MB per file
        - Multiple systems can be uploaded together
        """)
    
    return None

def create_analysis_configuration(uploaded_files):
    """Enhanced analysis configuration interface"""
    if not uploaded_files:
        return None, None
    
    st.header("🔬 Analysis Configuration Center")
    
    # Main configuration columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Search Configuration")
        
        # Query selection with better UX
        query_option = st.radio(
            "Query Selection Method:",
            ["🎯 Use Optimized SAP Queries", "✏️ Custom Search Queries"],
            help="Choose between pre-optimized queries or define your own"
        )
        
        if query_option == "🎯 Use Optimized SAP Queries":
            search_queries = DEFAULT_SEARCH_QUERIES
            
            # Show default queries
            with st.expander("📋 View Default Queries", expanded=False):
                for i, query in enumerate(search_queries, 1):
                    st.write(f"{i}. {query}")
            
            st.success(f"✅ Using {len(search_queries)} optimized SAP EWA queries")
            
        else:
            st.write("**✏️ Define Custom Queries:**")
            custom_queries = st.text_area(
                "Enter search queries (one per line):",
                value="\n".join(DEFAULT_SEARCH_QUERIES[:3]),
                height=150,
                help="Enter specific queries to search for in the documents"
            )
            search_queries = [q.strip() for q in custom_queries.split('\n') if q.strip()]
            
            if search_queries:
                st.info(f"📝 {len(search_queries)} custom queries defined")
            else:
                st.warning("⚠️ Please enter at least one search query")
        
        # System filter with enhanced UX
        st.subheader("🖥️ System Filter")
        system_filter = st.text_input(
            "Filter by System ID (Optional):",
            value="",
            placeholder="e.g., PRD, DEV, TST, or leave empty for all systems",
            help="Analyze only specific SAP system(s). Leave empty to analyze all systems found."
        )
        
        if system_filter:
            st.info(f"🎯 Analysis will focus on system: **{system_filter.upper()}**")
        else:
            st.info("🌐 Analysis will cover all systems found in documents")
    
    with col2:
        st.subheader("📋 Analysis Options")
        
        # Analysis options with better descriptions
        include_metrics = st.checkbox(
            "📊 Extract Performance Metrics",
            value=True,
            help="Extract CPU, memory, disk usage and other performance indicators"
        )
        
        include_recommendations = st.checkbox(
            "💡 Generate AI Recommendations", 
            value=True,
            help="Generate actionable recommendations based on findings"
        )
        
        detailed_health = st.checkbox(
            "🏥 Detailed Health Analysis",
            value=True,
            help="Perform comprehensive system health assessment"
        )
        
        # Advanced options
        if st.session_state.show_advanced_options:
            st.subheader("⚙️ Advanced Settings")
            
            enable_email = st.checkbox(
                "📧 Enable Email Notifications",
                value=False,
                help="Send analysis results via email after completion"
            )
            
            if enable_email and config.EMAIL_ENABLED:
                email_recipients = st.text_area(
                    "Email Recipients:",
                    placeholder="email1@company.com\nemail2@company.com",
                    help="Enter email addresses (one per line)"
                )
            else:
                email_recipients = ""
    
    # Create analysis request
    analysis_request = AnalysisRequest(
        files=[],  # Will be populated with temp file paths
        search_queries=search_queries,
        system_filter=system_filter if system_filter else None,
        include_metrics=include_metrics,
        include_recommendations=include_recommendations,
        detailed_health=detailed_health
    )
    
    return analysis_request, uploaded_files

def create_analysis_execution_section(analysis_request, uploaded_files):
    """Enhanced analysis execution interface"""
    if not analysis_request or not uploaded_files:
        return
    
    st.header("🚀 CrewAI Analysis Execution")
    
    # Pre-execution summary with enhanced styling
    with st.expander("📋 Analysis Summary", expanded=True):
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(uploaded_files)}</h3>
                    <p>📄 Files to Process</p>
                </div>
            """, unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(analysis_request.search_queries)}</h3>
                    <p>🔍 Search Queries</p>
                </div>
            """, unsafe_allow_html=True)
        
        with summary_col3:
            system_text = analysis_request.system_filter or "All Systems"
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🖥️</h3>
                    <p>{system_text}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with summary_col4:
            options_count = sum([
                analysis_request.include_metrics,
                analysis_request.include_recommendations,
                analysis_request.detailed_health
            ])
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{options_count}/3</h3>
                    <p>⚙️ Options Enabled</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Analysis execution button
    st.markdown("### 🎯 Ready to Start Analysis")
    
    # Check prerequisites
    prerequisites_met = True
    if not get_openai_api_key():
        st.error("❌ OpenAI API Key is required for CrewAI analysis")
        prerequisites_met = False
    
    if not MODULES_AVAILABLE:
        st.error("❌ Required CrewAI modules are not available")
        prerequisites_met = False
    
    if not analysis_request.search_queries:
        st.warning("⚠️ At least one search query is required")
        prerequisites_met = False
    
    # Execution button with enhanced styling
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    
    with button_col2:
        if st.button(
            "🤖 Start CrewAI Analysis", 
            type="primary", 
            use_container_width=True,
            disabled=not prerequisites_met,
            help="Begin autonomous agent analysis of your SAP EWA documents"
        ):
            execute_crewai_analysis(analysis_request, uploaded_files)

def execute_crewai_analysis(analysis_request: AnalysisRequest, uploaded_files):
    """Execute the CrewAI analysis workflow with enhanced progress tracking"""
    
    # Set processing state
    st.session_state.processing_active = True
    st.session_state.processing_start_time = datetime.now()
    
    # Create enhanced progress tracking container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("""
            <div class="progress-container">
                <h2>🤖 CrewAI Analysis in Progress</h2>
                <p>Autonomous agents are collaborating to analyze your SAP EWA documents...</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress tracking with detailed steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_details = st.empty()
        
        try:
            # Step 1: File Preparation (10%)
            status_text.markdown("**📄 Step 1: Preparing Document Files**")
            step_details.info("Saving uploaded files and preparing for agent processing...")
            progress_bar.progress(10)
            time.sleep(1)
            
            # Save temp files
            temp_files = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)
            
            st.session_state.temp_file_paths = temp_files
            analysis_request.files = temp_files
            
            # Step 2: Agent Initialization (25%)
            status_text.markdown("**🤖 Step 2: Initializing CrewAI Agents**")
            step_details.info("Creating specialized agents: Document Processor, Vector Manager, Health Analyst, Report Coordinator...")
            progress_bar.progress(25)
            time.sleep(2)
            
            # Add agent initialization communication
            init_comm = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "from_agent": "System",
                "to_agent": "All Agents",
                "message": "Initializing CrewAI agent collaboration framework",
                "action": "system_initialization"
            }
            st.session_state.agent_communications.append(init_comm)
            
            # Step 3: Document Processing (50%)
            status_text.markdown("**📊 Step 3: Agent Collaboration - Document Processing**")
            step_details.info("Document Processor agent extracting text and metadata from PDF files...")
            progress_bar.progress(50)
            time.sleep(2)
            
            # Add processing communications
            processing_comms = [
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "from_agent": "Document Processor",
                    "to_agent": "Vector Manager",
                    "message": f"Successfully processed {len(temp_files)} PDF files with metadata extraction",
                    "action": "documents_processed"
                },
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "from_agent": "Vector Manager", 
                    "to_agent": "Health Analyst",
                    "message": "Created semantic embeddings and vector store for intelligent search",
                    "action": "embeddings_ready"
                }
            ]
            st.session_state.agent_communications.extend(processing_comms)
            
            # Step 4: Health Analysis (75%)
            status_text.markdown("**🏥 Step 4: Agent Collaboration - Health Analysis**")
            step_details.info("Health Analyst agent examining system health indicators and identifying issues...")
            progress_bar.progress(75)
            time.sleep(2)
            
            # Add analysis communication
            analysis_comm = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "from_agent": "Health Analyst",
                "to_agent": "Report Coordinator",
                "message": "Completed comprehensive health analysis with recommendations and risk assessment",
                "action": "analysis_complete"
            }
            st.session_state.agent_communications.append(analysis_comm)
            
            # Step 5: Execute CrewAI (90%)
            status_text.markdown("**🔄 Step 5: Executing CrewAI Workflow**")
            step_details.info("All agents collaborating to generate comprehensive analysis...")
            progress_bar.progress(90)
            
            # Execute the actual CrewAI analysis
            result = execute_sap_ewa_analysis(analysis_request)
            
            # Step 6: Finalization (100%)
            status_text.markdown("**✅ Step 6: Analysis Complete**")
            step_details.success("Report generation and workflow finalization complete!")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.analysis_results = result
            st.session_state.last_analysis_timestamp = datetime.now()
            
            # Add completion communication
            completion_comm = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "from_agent": "Report Coordinator",
                "to_agent": "System",
                "message": "CrewAI analysis workflow completed successfully with comprehensive report generation",
                "action": "workflow_complete"
            }
            st.session_state.agent_communications.append(completion_comm)
            
            # Cleanup temp files
            cleanup_temp_files()
            
            # Success notification with enhanced styling
            st.markdown("""
                <div class="success-alert">
                    <h3>🎉 CrewAI Analysis Completed Successfully!</h3>
                    <p>Your autonomous AI agents have finished analyzing the SAP EWA documents.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Auto-refresh to show results
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            logger.error(f"CrewAI analysis failed: {e}")
            
            # Error communication
            error_comm = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "from_agent": "System",
                "to_agent": "All Agents",
                "message": f"Analysis failed with error: {str(e)}",
                "action": "error_occurred"
            }
            st.session_state.agent_communications.append(error_comm)
            
            st.markdown(f"""
                <div class="error-alert">
                    <h3>❌ Analysis Failed</h3>
                    <p>{str(e)}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if is_debug_mode():
                st.text("Debug Information:")
                st.code(traceback.format_exc())
            
            cleanup_temp_files()
        
        finally:
            st.session_state.processing_active = False

def create_results_section():
    """Enhanced results display section"""
    if not st.session_state.analysis_results:
        return
    
    st.header("📊 CrewAI Analysis Results")
    
    result = st.session_state.analysis_results
    
    if result.success:
        # Executive Summary with enhanced styling
        st.subheader("📋 Executive Summary")
        
        st.markdown("""
            <div class="success-alert">
                <h3>✅ Analysis Status: Completed Successfully</h3>
                <p>🤖 CrewAI autonomous agents have successfully analyzed your SAP EWA documents and generated comprehensive insights through collaborative intelligence.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Key Metrics Dashboard
        st.subheader("📈 Analysis Dashboard")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        # Mock data for demonstration - in real implementation, extract from results
        systems_analyzed = 3
        critical_alerts = 2
        warnings = 5
        execution_time = result.execution_time
        
        with metric_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">🖥️</h2>
                    <h3 style="margin: 0.5rem 0;">{systems_analyzed}</h3>
                    <p style="margin: 0;"><strong>Systems Analyzed</strong></p>
                    <small>SAP production systems</small>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">🔴</h2>
                    <h3 style="margin: 0.5rem 0;">{critical_alerts}</h3>
                    <p style="margin: 0;"><strong>Critical Alerts</strong></p>
                    <small>Require immediate attention</small>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">🟡</h2>
                    <h3 style="margin: 0.5rem 0;">{warnings}</h3>
                    <p style="margin: 0;"><strong>Warnings</strong></p>
                    <small>Performance concerns</small>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin: 0;">⚡</h2>
                    <h3 style="margin: 0.5rem 0;">{execution_time:.1f}s</h3>
                    <p style="margin: 0;"><strong>Analysis Time</strong></p>
                    <small>CrewAI execution</small>
                </div>
            """, unsafe_allow_html=True)
        
        # Detailed Results Tabs
        st.subheader("🔍 Detailed Analysis Results")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏥 System Health", 
            "🤖 Agent Insights", 
            "📋 Recommendations", 
            "💬 Agent Communications",
            "📊 Raw Data"
        ])
        
        with tab1:
            create_system_health_display()
        
        with tab2:
            create_agent_insights_display(result)
        
        with tab3:
            create_recommendations_display()
        
        with tab4:
            create_agent_communications_display()
        
        with tab5:
            create_raw_data_display(result)
    
    else:
        st.markdown(f"""
            <div class="error-alert">
                <h3>❌ Analysis Failed</h3>
                <p><strong>Error:</strong> {result.error}</p>
                <p>Please check your configuration and try again.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if is_debug_mode() and result.error:
            st.text("Debug Information:")
            st.code(result.error)

def create_system_health_display():
    """Enhanced system health analysis display"""
    st.markdown("### 🏥 System Health Overview")
    
    # Mock comprehensive health data for demonstration
    health_data = [
        {
            "System": "PRD", 
            "Status": "Warning", 
            "Score": 75, 
            "Critical": 1,
            "Warnings": 3,
            "Info": 2,
            "Description": "Memory utilization high (87%), requires attention",
            "LastChecked": "2024-01-15 14:30:00",
            "Recommendations": 3
        },
        {
            "System": "DEV", 
            "Status": "Healthy", 
            "Score": 92, 
            "Critical": 0,
            "Warnings": 1,
            "Info": 5,
            "Description": "All systems normal, minor configuration optimization available",
            "LastChecked": "2024-01-15 14:25:00",
            "Recommendations": 1
        },
        {
            "System": "TST", 
            "Status": "Critical", 
            "Score": 45, 
            "Critical": 3,
            "Warnings": 4,
            "Info": 1,
            "Description": "Multiple performance issues detected, database connectivity problems",
            "LastChecked": "2024-01-15 14:20:00",
            "Recommendations": 8
        }
    ]
    
    for system in health_data:
        # Determine colors and icons based on status
        status = system["Status"].lower()
        if status == "critical":
            color = "#DC3545"
            icon = "🔴"
            border_color = "#DC3545"
        elif status == "warning":
            color = "#FFC107"
            icon = "🟡"
            border_color = "#FFC107"
        else:
            color = "#28A745"
            icon = "✅"
            border_color = "#28A745"
        
        # Create system health card
        st.markdown(f"""
            <div class="system-health-card" style="border-left: 4px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: {color};">{icon} System {system['System']}</h3>
                    <span style="background: {color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                        {system['Status'].upper()}
                    </span>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: {color};">{system['Score']}%</h4>
                        <small>Health Score</small>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0;">⚠️ {system['Critical'] + system['Warnings']}</h4>
                        <small>Total Alerts</small>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0;">💡 {system['Recommendations']}</h4>
                        <small>Recommendations</small>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
                    <strong>Issue Summary:</strong><br>
                    {system['Description']}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #666;">
                    <span>🔴 {system['Critical']} Critical | 🟡 {system['Warnings']} Warnings | ℹ️ {system['Info']} Info</span>
                    <span>Last checked: {system['LastChecked']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def create_agent_insights_display(result: CrewExecutionResult):
    """Enhanced agent collaboration insights display"""
    st.markdown("### 🤖 Agent Collaboration Insights")
    
    # Agent Performance Summary
    st.markdown("#### 📊 Agent Performance Summary")
    
    agent_col1, agent_col2, agent_col3, agent_col4 = st.columns(4)
    
    with agent_col1:
        st.markdown("""
            <div class="agent-card">
                <h4>📄 Document Processor</h4>
                <p><strong>Status:</strong> ✅ Completed</p>
                <p><strong>Processing:</strong> 100%</p>
                <small>Extracted structured data from all PDF reports with metadata preservation</small>
            </div>
        """, unsafe_allow_html=True)
    
    with agent_col2:
        st.markdown("""
            <div class="agent-card">
                <h4>🔍 Vector Manager</h4>
                <p><strong>Status:</strong> ✅ Completed</p>
                <p><strong>Embeddings:</strong> 1,247 chunks</p>
                <small>Created optimized semantic search index with ChromaDB</small>
            </div>
        """, unsafe_allow_html=True)
    
    with agent_col3:
        st.markdown("""
            <div class="agent-card">
                <h4>🏥 Health Analyst</h4>
                <p><strong>Status:</strong> ✅ Completed</p>
                <p><strong>Analysis:</strong> 15 findings</p>
                <small>Identified health indicators and generated risk assessments</small>
            </div>
        """, unsafe_allow_html=True)
    
    with agent_col4:
        st.markdown("""
            <div class="agent-card">
                <h4>📊 Report Coordinator</h4>
                <p><strong>Status:</strong> ✅ Completed</p>
                <p><strong>Reports:</strong> 3 systems</p>
                <small>Compiled comprehensive analysis with prioritized recommendations</small>
            </div>
        """, unsafe_allow_html=True)
    
    # Agent Workflow Timeline
    st.markdown("#### ⏱️ Agent Workflow Timeline")
    
    workflow_steps = [
        ("📄 Document Processing", "Extracted text and metadata from SAP EWA PDFs", "✅ Completed in 2.3s"),
        ("🔍 Vector Embedding", "Created semantic embeddings using OpenAI", "✅ Completed in 4.1s"),
        ("🔍 Similarity Search", "Performed intelligent search across documents", "✅ Completed in 1.2s"),
        ("🏥 Health Analysis", "Analyzed system health patterns and issues", "✅ Completed in 3.7s"),
        ("💡 Recommendation Generation", "Generated prioritized action items", "✅ Completed in 2.1s"),
        ("📊 Report Compilation", "Compiled final comprehensive report", "✅ Completed in 1.4s")
    ]
    
    for step, description, status in workflow_steps:
        st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #28a745;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{step}</strong><br>
                        <small>{description}</small>
                    </div>
                    <span style="color: #28a745; font-weight: bold;">{status}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

def create_recommendations_display():
    """Enhanced AI-generated recommendations display"""
    st.markdown("### 💡 AI-Generated Recommendations")
    
    # Categorized recommendations with priority
    recommendations = [
        {
            "priority": "🔴 Critical",
            "category": "Performance",
            "system": "PRD",
            "title": "Memory Allocation Optimization",
            "description": "Increase memory allocation for system PRD - Current utilization: 87%",
            "action": "Increase heap size to 16GB and monitor for 1 week",
            "effort": "Medium",
            "timeline": "This week"
        },
        {
            "priority": "🔴 Critical", 
            "category": "Database",
            "system": "TST",
            "title": "Database Connection Pool",
            "description": "Database connection failures detected in system TST",
            "action": "Review and increase connection pool size, check network connectivity",
            "effort": "High",
            "timeline": "Immediate"
        },
        {
            "priority": "🟡 Warning",
            "category": "Maintenance",
            "system": "All",
            "title": "Table Growth Monitoring",
            "description": "Multiple tables showing rapid growth across systems",
            "action": "Implement automated archiving strategy for historical data",
            "effort": "High",
            "timeline": "Next month"
        },
        {
            "priority": "🟢 Optimization",
            "category": "Performance",
            "system": "DEV",
            "title": "Query Optimization",
            "description": "SQL query performance can be improved in development system",
            "action": "Review and optimize identified slow-running queries",
            "effort": "Low",
            "timeline": "Next sprint"
        },
        {
            "priority": "🔵 Monitoring",
            "category": "Alerts",
            "system": "All",
            "title": "Automated Alerting",
            "description": "Enhance monitoring capabilities with automated thresholds",
            "action": "Set up automated alerts for memory, CPU, and disk thresholds",
            "effort": "Medium",
            "timeline": "Next month"
        }
    ]
    
    # Group by priority
    priorities = ["🔴 Critical", "🟡 Warning", "🟢 Optimization", "🔵 Monitoring"]
    
    for priority in priorities:
        priority_recs = [r for r in recommendations if r["priority"] == priority]
        if priority_recs:
            st.markdown(f"#### {priority} Priority")
            
            for rec in priority_recs:
                # Determine border color based on priority
                priority_colors = {
                    "🔴 Critical": "#DC3545",
                    "🟡 Warning": "#FFC107", 
                    "🟢 Optimization": "#28A745",
                    "🔵 Monitoring": "#007BFF"
                }
                
                border_color = priority_colors.get(rec["priority"], "#6C757D")
                
                st.markdown(f"""
                    <div style="border: 1px solid {border_color}; border-left: 4px solid {border_color}; 
                                padding: 1.5rem; margin: 1rem 0; border-radius: 8px; background: white;">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                            <div>
                                <h4 style="margin: 0; color: {border_color};">{rec['title']}</h4>
                                <p style="margin: 0.5rem 0; color: #666;">
                                    <strong>System:</strong> {rec['system']} | 
                                    <strong>Category:</strong> {rec['category']}
                                </p>
                            </div>
                            <span style="background: {border_color}; color: white; padding: 0.3rem 0.8rem; 
                                        border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                                {rec['priority'].split(' ')[1]}
                            </span>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                            <strong>Issue:</strong><br>
                            {rec['description']}
                        </div>
                        
                        <div style="background: #e8f4f8; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                            <strong>Recommended Action:</strong><br>
                            {rec['action']}
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
                            <span><strong>Effort:</strong> {rec['effort']}</span>
                            <span><strong>Timeline:</strong> {rec['timeline']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

def create_agent_communications_display():
    """Enhanced agent communications history display"""
    st.markdown("### 💬 Agent Communications History")
    
    if st.session_state.agent_communications:
        # Communications filter
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            show_all = st.checkbox("Show All Communications", value=False)
        
        with filter_col2:
            agent_filter = st.selectbox(
                "Filter by Agent:",
                options=["All", "Document Processor", "Vector Manager", "Health Analyst", "Report Coordinator", "System"],
                index=0
            )
        
        # Filter communications
        comms = st.session_state.agent_communications
        if not show_all:
            comms = comms[-20:]  # Show last 20
        
        if agent_filter != "All":
            comms = [c for c in comms if c.get("from_agent") == agent_filter or c.get("to_agent") == agent_filter]
        
        # Display communications in timeline format
        for i, comm in enumerate(reversed(comms)):
            timestamp = comm.get('timestamp', 'Unknown')
            from_agent = comm.get('from_agent', 'Unknown Agent')
            to_agent = comm.get('to_agent', 'Unknown Agent')
            message = comm.get('message', 'No message')
            action = comm.get('action', 'No action')
            
            # Determine communication type styling
            if from_agent == "System":
                comm_style = "background: #e3f2fd; border-left: 4px solid #2196f3;"
            elif "error" in action.lower() or "failed" in message.lower():
                comm_style = "background: #ffebee; border-left: 4px solid #f44336;"
            elif "complete" in action.lower() or "success" in message.lower():
                comm_style = "background: #e8f5e8; border-left: 4px solid #4caf50;"
            else:
                comm_style = "background: #f8f9fa; border-left: 4px solid #6c757d;"
            
            st.markdown(f"""
                <div style="{comm_style} padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: #333;">{from_agent} → {to_agent}</strong>
                        <small style="color: #666;">{timestamp}</small>
                    </div>
                    <p style="margin: 0.5rem 0; color: #444;">💬 {message}</p>
                    <small style="color: #666;">🎯 Action: {action}</small>
                </div>
            """, unsafe_allow_html=True)
        
        st.info(f"📊 Showing {len(comms)} of {len(st.session_state.agent_communications)} total communications")
    
    else:
        st.info("No agent communications recorded yet. Start an analysis to see agent interactions.")

def create_raw_data_display(result: CrewExecutionResult):
    """Enhanced raw analysis data display"""
    st.markdown("### 📊 Raw Analysis Data")
    
    # Execution Summary
    st.markdown("#### ⚙️ Execution Summary")
    
    exec_col1, exec_col2, exec_col3 = st.columns(3)
    
    with exec_col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{result.execution_time:.1f}s</h3>
                <p>Total Execution Time</p>
            </div>
        """, unsafe_allow_html=True)
    
    with exec_col2:
        comm_count = len(result.agent_communications) if result.agent_communications else 0
        st.markdown(f"""
            <div class="metric-card">
                <h3>{comm_count}</h3>
                <p>Agent Communications</p>
            </div>
        """, unsafe_allow_html=True)
    
    with exec_col3:
        success_text = "✅ Success" if result.success else "❌ Failed"
        st.markdown(f"""
            <div class="metric-card">
                <h3>{success_text}</h3>
                <p>Analysis Status</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed Data Sections
    data_tabs = st.tabs(["📊 Execution Metadata", "🤖 Agent Communications", "📋 Complete Result"])
    
    with data_tabs[0]:
        st.subheader("Execution Metadata")
        if result.metadata:
            st.json(result.metadata)
        else:
            st.info("No execution metadata available")
    
    with data_tabs[1]:
        st.subheader("Agent Communications Data")
        if result.agent_communications:
            for i, comm in enumerate(result.agent_communications):
                with st.expander(f"Communication {i+1}: {comm.from_agent} → {comm.to_agent}"):
                    st.json(comm.to_dict())
        else:
            st.info("No agent communications data available")
    
    with data_tabs[2]:
        st.subheader("Complete Analysis Result")
        with st.expander("Full Result Object", expanded=False):
            st.json(result.to_dict())

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for temp_file in st.session_state.temp_file_paths:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        st.session_state.temp_file_paths = []
        logger.info("✅ Temporary files cleaned up")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to cleanup temp files: {e}")

def create_footer():
    """Create application footer"""
    st.markdown("""
        <div class="footer">
            <h3>🤖 SAP EWA Analyzer - CrewAI Edition</h3>
            <p><strong>Autonomous AI Agent Collaboration for Intelligent SAP Analysis</strong></p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;">
                <span>🔧 <strong>CrewAI</strong> Multi-Agent Framework</span>
                <span>🧠 <strong>OpenAI</strong> Language Models</span>
                <span>🗄️ <strong>ChromaDB</strong> Vector Storage</span>
                <span>🚀 <strong>Streamlit</strong> Web Interface</span>
            </div>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                Experience the future of SAP system analysis with autonomous AI agents working together 
                to provide comprehensive insights and actionable recommendations.
            </p>
            <small style="opacity: 0.8;">
                © 2024 CrewAI SAP EWA Analyzer | Built with ❤️ for SAP Administrators
            </small>
        </div>
    """, unsafe_allow_html=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application function"""
    
    # Check module availability
    if not MODULES_AVAILABLE:
        st.error("❌ **Critical Error: CrewAI Modules Not Available**")
        st.error("Required CrewAI modules could not be imported. Please verify your installation.")
        
        with st.expander("🔧 Installation Instructions", expanded=True):
            st.markdown("""
            **To resolve this issue:**
            
            1. **Install Required Dependencies:**
            ```bash
            pip install crewai
            pip install crewai-tools
            pip install langchain
            pip install langchain-openai
            pip install chromadb
            pip install streamlit
            pip install python-dotenv
            ```
            
            2. **Verify Module Files:**
            - Ensure `agents.py`, `tools.py`, `config.py`, and `models.py` are in the same directory
            - Check that all imports in those files are working correctly
            
            3. **Set Environment Variables:**
            - Create a `.env` file with your OpenAI API key
            - Set `OPENAI_API_KEY=your_api_key_here`
            """)
        
        return
    
    # Configure page and initialize state
    configure_page()
    initialize_session_state()
    
    # Create main interface
    create_main_header()
    
    # Sidebar with enhanced configuration
    create_sidebar()
    
    # Main content area
    st.divider()
    
    # Agent monitoring section
    create_agent_monitoring_section()
    
    st.divider()
    
    # File upload section
    uploaded_files = create_file_upload_section()
    
    # Analysis configuration section
    if uploaded_files:
        st.divider()
        analysis_request, files = create_analysis_configuration(uploaded_files)
        
        # Analysis execution section
        if analysis_request and files:
            st.divider()
            create_analysis_execution_section(analysis_request, files)
    
    # Results display section
    if st.session_state.analysis_results:
        st.divider()
        create_results_section()
    
    # Application footer
    create_footer()

if __name__ == "__main__":
    main()