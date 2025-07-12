# app_ewa_enhanced.py - ENHANCED SAP EWA ANALYZER WITH IMPROVED PATTERN RECOGNITION
"""
Enhanced SAP Early Watch Analyzer based on analysis of 4 EWA sample reports:
- sample-ewa-ibp.pdf (IBP - Integrated Business Planning)
- ewa-busiinessobjects-sample-report.pdf (BusinessObjects BI Platform)
- earlywatch-alert-s4hana-security-chapter.pdf (S/4HANA Security)
- ewa-ecc_production.pdf (ECC Production)

KEY FINDINGS FROM ANALYSIS:
✅ Traffic Light System: Red/Yellow/Green ratings are consistent across all reports
✅ System ID Patterns: VMW, PR0, XXX, P01 are common system identifiers
✅ SAP Notes: Consistent format "SAP Note XXXXXX" across all reports
✅ Recommendation Structure: Clear recommendation sections with action items
✅ Critical Issues: Hardware exhaustion, security vulnerabilities, performance problems
✅ Warning Issues: Parameter deviations, outdated versions, configuration issues
"""

import streamlit as st
import os
import logging
import time
import re
import traceback
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional, Union, Literal, Set
import tempfile
from pathlib import Path

# Core libraries
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import PyPDF2
import pdfplumber

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Core settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "gmail").lower()
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL", "")
OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD", "")

# Additional settings
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE", "209715200"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")

# Configuration
CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
    "top_k": int(os.getenv("TOP_K", "10")),
    "temperature": float(os.getenv("TEMPERATURE", "0.1")),
    "collection_name": "sap_documents",
    "persist_directory": CHROMA_PATH,
    "timeout": int(os.getenv("TIMEOUT_SECONDS", "300")),
    "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "3")),
    "vector_store_type": VECTOR_STORE_TYPE,
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "debug": DEBUG,
    "email_enabled": EMAIL_ENABLED,
    "embedding_type": "openai"
}

# ===============================
# ENHANCED EWA PATTERN RECOGNITION
# ===============================

class EWAPatternRecognizer:
    """Enhanced pattern recognition based on analysis of 4 EWA sample reports"""
    
    def __init__(self):
        # Traffic Light Patterns (consistent across all reports)
        self.traffic_light_patterns = {
            'red': [
                r'red.*rating|rating.*red',
                r'critical.*red',
                r'🔴',
                r'red.*circle',
                r'severe.*problems.*may.*cause.*lose.*business',
                r'critical.*security.*issues?',
                r'hardware.*resources.*exhausted',
                r'cpu.*utilization.*9[0-9]%',
                r'memory.*utilization.*9[0-9]%',
                r'disk.*space.*exhausted',
                r'connection.*failed',
                r'service.*failed',
                r'security.*vulnerabilities',
                r'user.*system.*active.*valid'
            ],
            'yellow': [
                r'yellow.*rating|rating.*yellow',
                r'warning.*yellow',
                r'🟡',
                r'yellow.*circle',
                r'orange.*rating|rating.*orange',
                r'🟠',
                r'parameters.*deviate.*recommend',
                r'outdated.*version',
                r'performance.*degradation',
                r'configuration.*suboptimal',
                r'attention.*required',
                r'purge.*scheduling.*not.*conform',
                r'excel.*add.*in.*deviate'
            ],
            'green': [
                r'green.*rating|rating.*green',
                r'healthy.*green',
                r'✅',
                r'green.*circle',
                r'checkmark',
                r'healthy.*status'
            ]
        }
        
        # System ID Patterns (from analysis)
        self.system_id_patterns = [
            r'\bSYSTEM\s+([A-Z0-9]{2,6})\b',
            r'\bSID\s*[:=]?\s*([A-Z0-9]{2,6})\b',
            r'\[([A-Z0-9]{2,6})\]',
            r'System ID:\s*([A-Z0-9]{2,6})',
            r'System:\s*([A-Z0-9]{2,6})',
            r'\b([A-Z]{1,3}[0-9]{1,2})\b',  # P01, VMW, PR0, XXX
            r'\b(PRD|PROD|DEV|QAS|TST|TRN)\b'
        ]
        
        # SAP Note Patterns (consistent format)
        self.sap_note_patterns = [
            r'sap.*note.*([0-9]+)',
            r'sap note ([0-9]+)',
            r'note ([0-9]+)',
            r'refer.*to.*sap.*note.*([0-9]+)'
        ]
        
        # Recommendation Patterns
        self.recommendation_patterns = [
            r'recommendation[s]?.*[.!]',
            r'action.*required',
            r'immediate.*action',
            r'consider.*upgrading',
            r'review.*configuration',
            r'activate.*audit.*trail',
            r'update.*to.*version',
            r'increase.*memory',
            r'optimize.*configuration'
        ]
        
        # Product-specific patterns
        self.product_patterns = {
            'ibp': [
                r'sap.*ibp',
                r'integrated.*business.*planning',
                r'excel.*add.*in.*sap.*ibp',
                r'ibp.*parameters',
                r'ibp.*configuration'
            ],
            'businessobjects': [
                r'businessobjects',
                r'bi.*platform',
                r'crystal.*reports',
                r'web.*intelligence',
                r'bo.*server',
                r'vmw'
            ],
            's4hana': [
                r'sap.*s/4hana',
                r's/4hana',
                r'sap hana',
                r'hana.*database',
                r'hana.*audit.*trail',
                r'hana.*password.*policy'
            ],
            'ecc': [
                r'sap.*ecc',
                r'ecc.*production',
                r'abap',
                r'netweaver',
                r'st03',
                r'st06'
            ]
        }
    
    def extract_traffic_lights(self, content: str) -> Dict[str, List[str]]:
        """Extract traffic light ratings from content"""
        results = {'red': [], 'yellow': [], 'green': []}
        content_lower = content.lower()
        
        for color, patterns in self.traffic_light_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content_lower)
                if matches:
                    results[color].extend(matches)
        
        return results
    
    def extract_system_ids(self, content: str) -> Set[str]:
        """Extract system IDs from content"""
        system_ids = set()
        content_upper = content.upper()
        
        # Common false positives to filter out
        false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL',
            'CPU', 'RAM', 'GB', 'MB', 'KB', 'HTTP', 'URL', 'API', 'GUI', 'UI', 'DB',
            'RFC', 'EWA', 'RED', 'IBM', 'OUT', 'TMP', 'USR', 'ADM', 'WEB', 'APP',
            'NO', 'YES', 'ON', 'OFF', 'NEW', 'OLD', 'TOP', 'MAX', 'MIN', 'AVG',
            'OVER', 'UNDER', 'HIGH', 'LOW', 'FULL', 'NULL', 'TRUE', 'FALSE',
            'GET', 'SET', 'PUT', 'POST', 'RUN', 'END', 'START', 'STOP',
            'ID', 'IS', 'TO', 'BY', 'ARE', 'CAN', 'MUST', 'THAT', 'HAVE', 'FROM'
        }
        
        for pattern in self.system_id_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if (len(match) >= 2 and 
                    match not in false_positives and
                    not match.isdigit()):
                    system_ids.add(match)
        
        return system_ids
    
    def extract_sap_notes(self, content: str) -> List[str]:
        """Extract SAP Note numbers from content"""
        sap_notes = []
        content_lower = content.lower()
        
        for pattern in self.sap_note_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match.isdigit():
                    sap_notes.append(match)
        
        return list(set(sap_notes))  # Remove duplicates
    
    def extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from content"""
        recommendations = []
        content_lower = content.lower()
        
        for pattern in self.recommendation_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                recommendations.extend(matches)
        
        return recommendations
    
    def detect_product_type(self, content: str) -> str:
        """Detect SAP product type from content"""
        content_lower = content.lower()
        
        for product, patterns in self.product_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return product
        
        return 'unknown'
    
    def extract_enhanced_findings(self, content: str, system_id: str = "") -> Dict[str, Any]:
        """Extract comprehensive findings using enhanced patterns"""
        findings = {
            'traffic_lights': self.extract_traffic_lights(content),
            'system_ids': list(self.extract_system_ids(content)),
            'sap_notes': self.extract_sap_notes(content),
            'recommendations': self.extract_recommendations(content),
            'product_type': self.detect_product_type(content),
            'critical_issues': [],
            'warning_issues': [],
            'healthy_components': []
        }
        
        # Extract critical issues (red traffic lights)
        for red_match in findings['traffic_lights']['red']:
            findings['critical_issues'].append(f"🔴 {red_match}")
        
        # Extract warning issues (yellow traffic lights)
        for yellow_match in findings['traffic_lights']['yellow']:
            findings['warning_issues'].append(f"🟡 {yellow_match}")
        
        # Extract healthy components (green traffic lights)
        for green_match in findings['traffic_lights']['green']:
            findings['healthy_components'].append(f"✅ {green_match}")
        
        return findings

# ===============================
# ENHANCED SUMMARY AGENT
# ===============================

class EnhancedSummaryAgent:
    """Enhanced summary agent with improved EWA pattern recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.pattern_recognizer = EWAPatternRecognizer()
    
    def generate_enhanced_summary(self, search_results: List[tuple], query: str, user_system_id: str = "") -> Dict[str, Any]:
        """Generate enhanced summary with improved EWA pattern recognition"""
        try:
            documents = [doc for doc, _ in search_results]
            
            if not documents:
                return {
                    "success": True,
                    "summary": {
                        "summary": "No relevant documents found for analysis.",
                        "critical_findings": [],
                        "recommendations": [],
                        "confidence_score": 0.0
                    }
                }
            
            # Build context for analysis
            max_context_chars = 12000
            context_parts = []
            total_chars = 0
            
            # Check if user provided a specific system
            single_system_mode = (user_system_id and 
                                 user_system_id.strip() and 
                                 ',' not in user_system_id and 
                                 ' ' not in user_system_id.strip())
            
            if single_system_mode:
                system_id = user_system_id.strip().upper()
                context_parts.append(f"=== SAP EWA ANALYSIS FOR SYSTEM {system_id} ===")
            else:
                context_parts.append("=== SAP EARLY WATCH ANALYZER REPORT - MULTI-SYSTEM ===")
            
            context_parts.append("")
            
            # Extract content from documents
            all_content = ""
            for doc in documents[:5]:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                else:
                    content = str(doc)
                    source = 'Unknown'
                
                doc_content = f"Document {len(context_parts)}: {source}\n{content}"
                
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "...\n[Content truncated]"
                
                if total_chars + len(doc_content) > max_context_chars:
                    break
                
                context_parts.append(doc_content)
                all_content += content + "\n"
                total_chars += len(doc_content)
            
            context = "\n\n".join(context_parts)
            
            # Use enhanced pattern recognition
            enhanced_findings = self.pattern_recognizer.extract_enhanced_findings(all_content, user_system_id)
            
            # Generate analysis
            if OPENAI_API_KEY:
                if single_system_mode:
                    analysis = self._generate_enhanced_openai_analysis(query, context, enhanced_findings, user_system_id.strip())
                else:
                    analysis = self._generate_enhanced_openai_analysis(query, context, enhanced_findings, "")
            else:
                if single_system_mode:
                    analysis = self._generate_enhanced_fallback_analysis(query, enhanced_findings, user_system_id.strip())
                else:
                    analysis = self._generate_enhanced_fallback_analysis(query, enhanced_findings, "")
            
            # Build summary with enhanced findings
            summary = {
                "summary": analysis,
                "critical_findings": enhanced_findings['critical_issues'],
                "recommendations": enhanced_findings['recommendations'],
                "sap_notes": enhanced_findings['sap_notes'],
                "system_ids": enhanced_findings['system_ids'],
                "product_type": enhanced_findings['product_type'],
                "traffic_lights": enhanced_findings['traffic_lights'],
                "confidence_score": 0.9 if OPENAI_API_KEY else 0.6,
                "query": query,
                "results_analyzed": len(documents),
                "context_truncated": total_chars >= max_context_chars
            }
            
            return {"success": True, "summary": summary}
            
        except Exception as e:
            logger.error(f"Enhanced summary generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_enhanced_openai_analysis(self, query: str, context: str, enhanced_findings: Dict, system_id: str = "") -> str:
        """Generate enhanced OpenAI analysis with improved prompts"""
        try:
            from langchain_community.chat_models import ChatOpenAI
            
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.1,
                model="gpt-4o-mini",
                max_tokens=2000
            )
            
            # Enhanced prompt based on EWA analysis
            if system_id:
                prompt = f"""
You are analyzing a SAP Early Watch Alert (EWA) report for system {system_id}.

ENHANCED EWA PATTERN RECOGNITION:
Based on analysis of 4 EWA sample reports, look for these consistent patterns:

🚦 TRAFFIC LIGHT RATINGS (Consistent across all SAP products):
- RED ratings = CRITICAL ISSUES (hardware exhaustion, security vulnerabilities, performance problems)
- YELLOW/ORANGE ratings = WARNING ISSUES (parameter deviations, outdated versions, configuration issues)  
- GREEN ratings = HEALTHY STATUS (optimal performance, proper configuration)

📋 SAP NOTES: Always in format "SAP Note XXXXXX" - these are specific action items
🆔 SYSTEM IDs: Look for patterns like VMW, PR0, XXX, P01, etc.
📊 PRODUCT TYPES: IBP, BusinessObjects, S/4HANA, ECC have different focus areas

EWA Report Content:
{context}

Enhanced Findings Detected:
- Traffic Lights: {enhanced_findings['traffic_lights']}
- System IDs: {enhanced_findings['system_ids']}
- SAP Notes: {enhanced_findings['sap_notes']}
- Product Type: {enhanced_findings['product_type']}

User Query: {query}

Analyze the content and provide:

## System {system_id} - Enhanced EWA Analysis

### 🔴 CRITICAL ISSUES (Red Traffic Lights)
List each critical finding with:
- What component/area has the red rating
- What the specific issue is
- Why it's marked as critical
- Immediate action required

### 🟡 WARNING ISSUES (Yellow Traffic Lights)
List each warning with:
- What component/area has the yellow rating
- What the specific issue is
- What needs attention
- Recommended action

### ✅ HEALTHY COMPONENTS (Green Traffic Lights)
List what's working well.

### 📋 SAP RECOMMENDATIONS
- List all SAP Notes found: {enhanced_findings['sap_notes']}
- Specific action items from recommendations
- Priority order for addressing issues

### 📊 System Health Assessment
- Overall risk level based on traffic light distribution
- Product-specific considerations for {enhanced_findings['product_type']}
- Immediate vs. long-term actions

IMPORTANT: Focus on actual traffic light ratings and SAP Notes found in the content.
                """
            else:
                prompt = f"""
You are analyzing SAP Early Watch Alert (EWA) reports for multiple systems.

ENHANCED EWA PATTERN RECOGNITION:
Based on analysis of 4 EWA sample reports, look for these consistent patterns:

🚦 TRAFFIC LIGHT RATINGS (Consistent across all SAP products):
- RED ratings = CRITICAL ISSUES (hardware exhaustion, security vulnerabilities, performance problems)
- YELLOW/ORANGE ratings = WARNING ISSUES (parameter deviations, outdated versions, configuration issues)  
- GREEN ratings = HEALTHY STATUS (optimal performance, proper configuration)

📋 SAP NOTES: Always in format "SAP Note XXXXXX" - these are specific action items
🆔 SYSTEM IDs: Look for patterns like VMW, PR0, XXX, P01, etc.
📊 PRODUCT TYPES: IBP, BusinessObjects, S/4HANA, ECC have different focus areas

EWA Report Content:
{context}

Enhanced Findings Detected:
- Traffic Lights: {enhanced_findings['traffic_lights']}
- System IDs: {enhanced_findings['system_ids']}
- SAP Notes: {enhanced_findings['sap_notes']}
- Product Type: {enhanced_findings['product_type']}

User Query: {query}

## Multi-System Enhanced EWA Analysis

### 🖥️ System-by-System Traffic Light Summary
For each system found ({enhanced_findings['system_ids']}), analyze:
- 🔴 Critical (Red) Ratings: [list issues with red ratings]
- 🟡 Warning (Yellow) Ratings: [list issues with yellow ratings]  
- ✅ Healthy (Green) Ratings: [list components with green ratings]
- Overall Status: Critical/Warning/Healthy

### 🚨 All Critical Issues (Red Traffic Lights Across Systems)
List every component/issue that has a RED rating, organized by system.

### ⚠️ All Warning Issues (Yellow Traffic Lights Across Systems) 
List every component/issue that has a YELLOW rating, organized by system.

### 📋 SAP RECOMMENDATIONS & NOTES
- All SAP Notes found: {enhanced_findings['sap_notes']}
- Specific action items from recommendations
- Priority order for addressing issues

### 📊 Cross-System Risk Assessment
- Which systems have the most red ratings?
- What are the common issues across systems?
- Product-specific considerations for {enhanced_findings['product_type']}
- Priority order for addressing issues

IMPORTANT: Focus on actual traffic light ratings and SAP Notes found in the content.
                """
            
            return llm.predict(prompt)
            
        except Exception as e:
            logger.error(f"Enhanced OpenAI analysis failed: {e}")
            return self._generate_enhanced_fallback_analysis(query, enhanced_findings, system_id)

    def _generate_enhanced_fallback_analysis(self, query: str, enhanced_findings: Dict, system_id: str = "") -> str:
        """Generate enhanced fallback analysis"""
        if system_id:
            return f"""
## System {system_id} - Enhanced SAP EWA Analysis

### Analysis Summary
Enhanced SAP Early Watch analysis completed for system: **{system_id}**

**Query:** {query}

### 🚦 Traffic Light Analysis
- Red Ratings: {len(enhanced_findings['traffic_lights']['red'])} critical issues
- Yellow Ratings: {len(enhanced_findings['traffic_lights']['yellow'])} warning issues  
- Green Ratings: {len(enhanced_findings['traffic_lights']['green'])} healthy components

### 📋 SAP Notes Found
{enhanced_findings['sap_notes']}

### 🆔 Systems Detected
{enhanced_findings['system_ids']}

### 📊 Product Type
{enhanced_findings['product_type'].upper()}

### Next Steps
- Review all traffic light ratings for system {system_id}
- Address critical issues immediately
- Follow SAP Note recommendations
- Regular monitoring recommended

*Note: Configure OpenAI API key for detailed AI-powered insights.*
            """
        else:
            return f"""
## Enhanced SAP EWA Analysis - Multi-System Report

### Analysis Summary
Enhanced SAP Early Watch analysis completed for multiple systems.

**Query:** {query}

### 🚦 Traffic Light Analysis
- Red Ratings: {len(enhanced_findings['traffic_lights']['red'])} critical issues
- Yellow Ratings: {len(enhanced_findings['traffic_lights']['yellow'])} warning issues  
- Green Ratings: {len(enhanced_findings['traffic_lights']['green'])} healthy components

### 📋 SAP Notes Found
{enhanced_findings['sap_notes']}

### 🆔 Systems Detected
{enhanced_findings['system_ids']}

### 📊 Product Type
{enhanced_findings['product_type'].upper()}

### Next Steps
- Review all traffic light ratings across systems
- Address critical issues immediately
- Follow SAP Note recommendations
- Regular monitoring recommended

*Note: Configure OpenAI API key for detailed AI-powered insights.*
            """

# ===============================
# MAIN STREAMLIT APPLICATION
# ===============================

def main():
    """Enhanced SAP EWA Analyzer with improved pattern recognition"""
    
    st.set_page_config(
        page_title="Enhanced SAP EWA Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .traffic-light-red { color: #dc3545; font-weight: bold; }
        .traffic-light-yellow { color: #ffc107; font-weight: bold; }
        .traffic-light-green { color: #28a745; font-weight: bold; }
        .system-id-box {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #1976d2;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced SAP Early Watch Analyzer</h1>
        <p>Advanced Pattern Recognition Based on 4 EWA Sample Reports | Traffic Light Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "enhanced_workflow" not in st.session_state:
        st.session_state.enhanced_workflow = None
        st.session_state.pattern_recognizer = EWAPatternRecognizer()
        st.session_state.enhanced_summary_agent = EnhancedSummaryAgent(CONFIG)
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    # File upload section
    st.header("📁 Upload SAP EWA Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload SAP Early Watch reports (S/4HANA, IBP, BusinessObjects, ECC, etc.)"
    )
    
    if uploaded_files:
        st.write(f"📄 {len(uploaded_files)} files selected:")
        total_size = 0
        for file in uploaded_files:
            file_size = len(file.getvalue()) / (1024 * 1024)  # MB
            total_size += file_size
            st.write(f"• {file.name} ({file_size:.1f} MB)")
        st.write(f"**Total size: {total_size:.1f} MB**")
        
        # Process button
        if st.button("🚀 Process Documents with Enhanced Analysis", type="primary"):
            with st.spinner("Processing SAP EWA documents with enhanced pattern recognition..."):
                try:
                    # Extract content from all files
                    all_content = ""
                    for file in uploaded_files:
                        try:
                            file.seek(0)
                            pdf_reader = PyPDF2.PdfReader(file)
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    all_content += page_text + "\n"
                        except Exception as e:
                            st.warning(f"Warning: Could not extract text from {file.name}: {e}")
                    
                    if all_content.strip():
                        # Use enhanced pattern recognition
                        enhanced_findings = st.session_state.pattern_recognizer.extract_enhanced_findings(all_content)
                        
                        # Store results
                        st.session_state.documents_processed = True
                        st.session_state.analysis_results = {
                            'enhanced_findings': enhanced_findings,
                            'content': all_content,
                            'files_processed': len(uploaded_files),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success(f"✅ Successfully processed {len(uploaded_files)} SAP EWA files with enhanced pattern recognition!")
                        
                        # Show enhanced findings
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔴 Critical Issues", len(enhanced_findings['traffic_lights']['red']))
                        with col2:
                            st.metric("🟡 Warning Issues", len(enhanced_findings['traffic_lights']['yellow']))
                        with col3:
                            st.metric("✅ Healthy Components", len(enhanced_findings['traffic_lights']['green']))
                        
                        st.rerun()
                    else:
                        st.error("❌ No text content could be extracted from the uploaded files")
                        
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Document processing error: {e}")
    
    # Analysis section
    if st.session_state.documents_processed and st.session_state.analysis_results:
        st.markdown("---")
        st.header("🔍 Enhanced EWA Analysis")
        
        results = st.session_state.analysis_results
        enhanced_findings = results['enhanced_findings']
        
        # System ID Input
        st.markdown('<div class="system-id-box">', unsafe_allow_html=True)
        st.write("**Enter System ID to analyze (optional):**")
        selected_system = st.text_input(
            "System ID",
            placeholder="Enter system ID (e.g., VMW, PR0, XXX, P01)",
            help="Enter a specific SAP system ID to focus the analysis, or leave empty to analyze all content",
            key="system_id_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "What would you like to know about your SAP EWA documents?",
            height=100,
            placeholder="Example: Show me all critical issues and SAP recommendations"
        )
        
        # Analysis button
        if st.button("🔍 Analyze with Enhanced Patterns", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("Analyzing SAP EWA documents with enhanced pattern recognition..."):
                    try:
                        # Create mock search results for analysis
                        mock_documents = [Document(page_content=results['content'], metadata={'source': 'EWA Report'})]
                        mock_search_results = [(doc, 0.9) for doc in mock_documents]
                        
                        # Generate enhanced summary
                        summary_result = st.session_state.enhanced_summary_agent.generate_enhanced_summary(
                            mock_search_results, 
                            query, 
                            selected_system
                        )
                        
                        if summary_result.get("success"):
                            summary = summary_result.get("summary", {})
                            
                            # Display enhanced results
                            st.markdown("---")
                            st.header("📊 Enhanced EWA Analysis Results")
                            
                            # Show traffic light summary
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🔴 Critical", len(enhanced_findings['traffic_lights']['red']))
                            with col2:
                                st.metric("🟡 Warning", len(enhanced_findings['traffic_lights']['yellow']))
                            with col3:
                                st.metric("✅ Healthy", len(enhanced_findings['traffic_lights']['green']))
                            with col4:
                                st.metric("📋 SAP Notes", len(enhanced_findings['sap_notes']))
                            
                            # Display analysis
                            st.subheader("🎯 AI Analysis")
                            st.markdown(summary.get("summary", "No analysis available"))
                            
                            # Display critical findings
                            if enhanced_findings['critical_issues']:
                                st.subheader("🔴 Critical Issues")
                                for issue in enhanced_findings['critical_issues']:
                                    st.error(issue)
                            
                            # Display warning findings
                            if enhanced_findings['warning_issues']:
                                st.subheader("🟡 Warning Issues")
                                for issue in enhanced_findings['warning_issues']:
                                    st.warning(issue)
                            
                            # Display SAP Notes
                            if enhanced_findings['sap_notes']:
                                st.subheader("📋 SAP Notes Found")
                                for note in enhanced_findings['sap_notes']:
                                    st.success(f"📝 SAP Note {note}")
                            
                            # Display system information
                            if enhanced_findings['system_ids']:
                                st.subheader("🆔 Systems Detected")
                                st.info(f"Systems found: {', '.join(enhanced_findings['system_ids'])}")
                            
                            # Display product type
                            if enhanced_findings['product_type'] != 'unknown':
                                st.subheader("📊 Product Type")
                                st.info(f"Detected product: {enhanced_findings['product_type'].upper()}")
                            
                        else:
                            st.error(f"❌ Analysis failed: {summary_result.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {e}")

if __name__ == "__main__":
    main() 