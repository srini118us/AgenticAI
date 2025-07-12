# app.py - SAP EWA RAG - ALL ISSUES FIXED
"""
Complete SAP Early Watch Analyzer with ALL BUGS FIXED
✅ Issue 1: User System ID Input (FIXED)
✅ Issue 2: Real Critical Issues Display (FIXED) 
✅ Issue 3: Real SAP Recommendations (FIXED)
✅ Issue 4: Email Provider Fix (FIXED)
"""

import streamlit as st
import os
import logging
import time
import smtplib
import ssl
import re
import traceback
# CORRECTED EMAIL IMPORTS
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional, Union
import tempfile
from pathlib import Path

# Core libraries
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
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

# Core settings - Enhanced to match your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "gmail").lower()
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL", "")
OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD", "")

# Additional settings from your .env
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE", "209715200"))  # 200MB default
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
SYSTEM_OUTPUTS_PATH = os.getenv("SYSTEM_OUTPUTS_PATH", "./data/system_outputs")

# Configuration - Enhanced to use your .env settings
CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),  # Convert bytes to MB
    "top_k": int(os.getenv("TOP_K", "10")),
    "temperature": float(os.getenv("TEMPERATURE", "0.1")),
    "collection_name": "sap_documents",
    "persist_directory": CHROMA_PATH,
    "timeout": int(os.getenv("TIMEOUT_SECONDS", "300")),
    "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "3")),
    "vector_store_type": VECTOR_STORE_TYPE,
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "debug": DEBUG
}

# ===============================
# STATE DEFINITION
# ===============================

class WorkflowState(TypedDict):
    """State structure from working original"""
    uploaded_files: List[Any]
    processed_files: List[Dict[str, Any]]
    documents: List[Document]
    embeddings: Optional[List[List[float]]]
    vector_store: Optional[Any]
    search_results: List[Document]
    summary: Dict[str, Any]
    system_summaries: List[Dict[str, Any]]
    query: str
    email_data: Optional[Dict[str, Any]]
    workflow_status: str
    current_agent: str
    processing_times: Dict[str, float]
    total_processing_time: float
    error: Optional[str]
    success: bool
    system_ids: List[str]
    user_system_id: str  # FIX 1: User input system ID

# ===============================
# BASE AGENT CLASS
# ===============================

class BaseAgent:
    """Base agent class from original"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.start_time = None
        self.performance_metrics = {}
    
    def log_info(self, message: str):
        self.logger.info(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        self.logger.warning(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        self.logger.error(f"[{self.name}] {message}")
    
    def start_timer(self):
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        error_message = f"{context} failed: {str(error)}"
        self.log_error(error_message)
        return {
            "success": False,
            "error": error_message,
            "processing_time": self.get_elapsed_time()
        }

# ===============================
# PDF PROCESSOR AGENT
# ===============================

class PDFProcessorAgent(BaseAgent):
    """PDF processor from original - working perfectly"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PDFProcessor", config)
        self.max_file_size = config.get('max_file_size_mb', 50) * 1024 * 1024
        self.supported_encodings = config.get('encodings', ['utf-8', 'latin-1', 'cp1252'])
        self.clean_text = config.get('clean_text', True)
    
    def process(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Main processing method"""
        self.start_timer()
        
        try:
            self.log_info(f"Starting PDF processing for {len(uploaded_files)} files")
            
            if not uploaded_files:
                return self.handle_error(ValueError("No PDF files provided"), "PDF Processing")
            
            processed_files = []
            failed_files = []
            total_size = 0
            
            for file in uploaded_files:
                try:
                    # File validation
                    if len(file.getvalue()) > self.max_file_size:
                        failed_files.append({
                            'filename': file.name,
                            'error': f'File too large (max {self.max_file_size/1024/1024}MB)'
                        })
                        continue
                    
                    # Extract text
                    text_content = self._extract_text_from_pdf(file)
                    
                    if text_content and text_content.strip():
                        # Clean text
                        if self.clean_text:
                            text_content = self._clean_extracted_text(text_content)
                        
                        # Create file data
                        file_data = {
                            'filename': file.name,
                            'text': text_content,
                            'size': len(file.getvalue()),
                            'character_count': len(text_content),
                            'word_count': len(text_content.split()),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        processed_files.append(file_data)
                        total_size += file_data['size']
                        
                        self.log_info(f"✅ Successfully processed {file.name}: "
                                    f"{file_data['character_count']} chars, {file_data['word_count']} words")
                    else:
                        failed_files.append({
                            'filename': file.name,
                            'error': 'No text content extracted'
                        })
                        
                except Exception as e:
                    self.log_error(f"Failed to process {file.name}: {str(e)}")
                    failed_files.append({
                        'filename': file.name,
                        'error': str(e)
                    })
            
            # Calculate success rate
            total_files = len(uploaded_files)
            success_count = len(processed_files)
            success_rate = (success_count / total_files) * 100 if total_files > 0 else 0
            
            # Return results
            result = {
                "success": success_count > 0,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "processing_time": self.get_elapsed_time(),
                "success_rate": success_rate,
                "total_size": total_size,
                "files_processed": success_count,
                "files_failed": len(failed_files)
            }
            
            if success_count == 0:
                result["error"] = "No files could be processed successfully"
            
            self.log_info(f"PDF processing completed: {success_count}/{total_files} files successful")
            return result
            
        except Exception as e:
            return self.handle_error(e, "PDF Processing")
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text using multiple PDF libraries"""
        text = ""
        
        # Method 1: PyPDF2
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            self.log_warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber
        try:
            file.seek(0)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            self.log_warning(f"pdfplumber extraction failed: {e}")
        
        return text
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF artifacts
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()

# ===============================
# FIXED SYSTEM SUMMARY GENERATOR
# ===============================

class SystemSummaryGenerator:
    """Generate individual summaries for each detected system - FIX 2 & 3"""
    
    def __init__(self):
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{3})\b',
            r'\bSID[:\s]+([A-Z0-9]{3})\b',
            r'\b([A-Z]{1,2}[0-9]{1,2})\b'
        ]
    
    def generate_system_summaries(self, search_results: List[Document], system_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate individual summaries for each system - FIX 2 & 3: REAL DATA"""
        system_summaries = {}
        
        for system_id in system_ids:
            # Find documents relevant to this specific system
            system_docs = self._filter_docs_for_system(search_results, system_id)
            
            # Generate REAL summary for this system - FIX 2 & 3
            system_summary = self._generate_real_summary_for_system(system_id, system_docs)
            system_summaries[system_id] = system_summary
        
        return system_summaries
    
    def _filter_docs_for_system(self, documents: List[Document], target_system: str) -> List[Document]:
        """Filter documents that are relevant to a specific system"""
        relevant_docs = []
        
        for doc in documents:
            content = doc.page_content.upper()
            
            # Check if this document mentions the target system
            if target_system.upper() in content:
                relevant_docs.append(doc)
                continue
            
            # Check for system patterns mentioning this system
            for pattern in self.system_patterns:
                import re
                matches = re.findall(pattern, content)
                if target_system.upper() in [match.upper() for match in matches]:
                    relevant_docs.append(doc)
                    break
        
        return relevant_docs
    
    def _generate_real_summary_for_system(self, system_id: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate REAL summary for a specific system - FIX 2 & 3"""
        critical_findings = []
        recommendations = []
        performance_metrics = {}
        health_status = "HEALTHY"
        
        # FIX 2 & 3: Extract REAL critical issues and recommendations from actual document content
        for doc in documents:
            content = doc.page_content.lower()
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # FIX 2: REAL Critical Issues Detection
                critical_keywords = [
                    'critical', 'error', 'fail', 'down', 'alert', 'issue', 'problem',
                    'exception', 'abort', 'crash', 'timeout', 'memory leak',
                    'database lock', 'slow response', 'high cpu', 'disk full',
                    'connection failed', 'service unavailable', 'performance degraded'
                ]
                
                if any(keyword in sentence for keyword in critical_keywords):
                    if system_id.lower() in sentence or len(documents) == 1:  # System-specific or single system
                        critical_finding = f"Critical: {sentence.capitalize()}"
                        if critical_finding not in critical_findings and len(critical_findings) < 5:
                            critical_findings.append(critical_finding)
                            health_status = "CRITICAL"
                
                # FIX 3: REAL Recommendations Detection
                recommendation_keywords = [
                    'recommend', 'should', 'improve', 'optimize', 'upgrade', 'configure',
                    'adjust', 'tune', 'modify', 'consider', 'suggest', 'advise',
                    'increase', 'decrease', 'enable', 'disable', 'patch', 'update'
                ]
                
                if any(keyword in sentence for keyword in recommendation_keywords):
                    if system_id.lower() in sentence or len(documents) == 1:  # System-specific or single system
                        recommendation = f"Recommendation: {sentence.capitalize()}"
                        if recommendation not in recommendations and len(recommendations) < 5:
                            recommendations.append(recommendation)
                
                # Extract REAL performance metrics
                import re
                cpu_match = re.search(r'cpu[:\s]+([0-9]+)%', sentence)
                if cpu_match and (system_id.lower() in sentence or len(documents) == 1):
                    performance_metrics['cpu_usage'] = f"{cpu_match.group(1)}%"
                
                memory_match = re.search(r'memory[:\s]+([0-9]+)%', sentence)
                if memory_match and (system_id.lower() in sentence or len(documents) == 1):
                    performance_metrics['memory_usage'] = f"{memory_match.group(1)}%"
                
                disk_match = re.search(r'disk[:\s]+([0-9]+)%', sentence)
                if disk_match and (system_id.lower() in sentence or len(documents) == 1):
                    performance_metrics['disk_usage'] = f"{disk_match.group(1)}%"
        
        # Determine health status based on REAL findings
        if len(critical_findings) >= 3:
            health_status = "CRITICAL"
        elif len(critical_findings) >= 1:
            health_status = "WARNING"
        elif not critical_findings and recommendations:
            health_status = "HEALTHY"
        
        # If no real data found, provide system-specific default messages
        if not critical_findings:
            critical_findings = [f"No critical issues detected for system {system_id}"]
        
        if not recommendations:
            recommendations = [f"System {system_id} appears to be operating normally", 
                             f"Continue monitoring {system_id} performance metrics"]
        
        return {
            "system_id": system_id,
            "overall_health": health_status,
            "critical_alerts": critical_findings,
            "recommendations": recommendations,
            "key_metrics": performance_metrics,
            "documents_analyzed": len(documents),
            "last_analyzed": datetime.now().isoformat()
        }

# ===============================
# FIXED EMAIL MANAGER - FIX 4
# ===============================

class EmailManager:
    """Email manager with Gmail/Outlook support - FIX 4"""
    
    def __init__(self):
        self.provider = EMAIL_PROVIDER
        self.email_enabled = EMAIL_ENABLED
    
    def is_configured(self) -> bool:
        if not self.email_enabled:
            return False
        
        # FIX 4: Corrected provider validation
        if self.provider == "gmail":
            return bool(GMAIL_EMAIL and GMAIL_APP_PASSWORD)
        elif self.provider == "outlook":
            return bool(OUTLOOK_EMAIL and OUTLOOK_PASSWORD)
        
        return False
    
    def get_status_message(self) -> str:
        """Get email configuration status message - FIX 4"""
        if not EMAIL_ENABLED:
            return "Email disabled in environment (EMAIL_ENABLED=false)"
        elif self.provider == "gmail":
            if not GMAIL_EMAIL:
                return "Gmail email address not configured (set GMAIL_EMAIL)"
            elif not GMAIL_APP_PASSWORD:
                return "Gmail app password not configured (set GMAIL_APP_PASSWORD)" 
            else:
                return f"Gmail configured: {GMAIL_EMAIL}"
        elif self.provider == "outlook":
            if not OUTLOOK_EMAIL:
                return "Outlook email address not configured (set OUTLOOK_EMAIL)"
            elif not OUTLOOK_PASSWORD:
                return "Outlook password not configured (set OUTLOOK_PASSWORD)"
            else:
                return f"Outlook configured: {OUTLOOK_EMAIL}"
        else:
            # FIX 4: Corrected error message format
            return f"Email provider '{self.provider}' is not supported (use 'gmail' or 'outlook')"
    
    def send_email(self, recipients: List[str], subject: str, body: str, cc_recipients: List[str] = None) -> Dict[str, Any]:
        """Send email - FIX 4: Fixed provider configuration"""
        try:
            if not self.is_configured():
                return {"success": False, "error": "Email not configured"}
            
            msg = MIMEMultipart()
            
            # FIX 4: Corrected provider configuration
            if self.provider == "gmail":
                msg['From'] = GMAIL_EMAIL
                sender_email = GMAIL_EMAIL
                sender_password = GMAIL_APP_PASSWORD
                smtp_server = "smtp.gmail.com"
                smtp_port = 587
            elif self.provider == "outlook":
                msg['From'] = OUTLOOK_EMAIL
                sender_email = OUTLOOK_EMAIL
                sender_password = OUTLOOK_PASSWORD
                smtp_server = "smtp-mail.outlook.com"
                smtp_port = 587
            else:
                # FIX 4: Handle unsupported provider
                return {"success": False, "error": f"Unsupported email provider: {self.provider}"}
            
            msg['To'] = ", ".join(recipients)
            if cc_recipients:
                msg['Cc'] = ", ".join(cc_recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                
                all_recipients = recipients + (cc_recipients or [])
                server.sendmail(sender_email, all_recipients, msg.as_string())
            
            return {
                "success": True,
                "message": f"Email sent to {len(all_recipients)} recipients",
                "recipients_count": len(all_recipients)
            }
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {"success": False, "error": str(e)}

# ===============================
# FIXED SYSTEM ID EXTRACTOR - FIX 1
# ===============================

class SystemIDExtractor:
    """Extract SAP System IDs with proper validation - FIX 1"""
    
    def __init__(self):
        # SAP system ID patterns
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{3})\b',
            r'\bSID[:\s]+([A-Z0-9]{3})\b',
            r'\bSAP\s+SYSTEM[:\s]+([A-Z0-9]{3})\b',
            r'\bEWA[_\s]+([A-Z0-9]{3})\b',
            r'\b([A-Z]{1,2}[0-9]{1,2})\b',
            r'\b(PRD|PROD|DEV|QAS|TST|TRN)\b'
        ]
        
        # Expanded false positives
        self.false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL',
            'CPU', 'RAM', 'GB', 'MB', 'KB', 'HTTP', 'URL', 'API', 'GUI', 'UI', 'DB',
            'RFC', 'EWA', 'RED', 'IBM', 'OUT', 'TMP', 'USR', 'ADM', 'WEB', 'APP',
            'NO', 'YES', 'ON', 'OFF', 'NEW', 'OLD', 'TOP', 'MAX', 'MIN', 'AVG',
            'SP16', 'ST03', 'ST06', 'SM50', 'SM51', 'SM21', 'SE80', 'SE11',
            'OVER', 'UNDER', 'HIGH', 'LOW', 'FULL', 'NULL', 'TRUE', 'FALSE',
            'GET', 'SET', 'PUT', 'POST', 'RUN', 'END', 'START', 'STOP'
        }
        
        # Valid system patterns
        self.valid_patterns = [
            r'^[A-Z]{1,2}[0-9]{1,2}$',    # P01, D1, Q1, etc.
            r'^(PRD|PROD|DEV|QAS|TST|TRN)$'  # Standard names
        ]
    
    def extract_from_documents(self, documents: List[Document]) -> List[str]:
        """Extract unique system IDs from documents"""
        system_ids = set()
        
        for doc in documents:
            content = doc.page_content.upper()
            
            # Extract using patterns
            for pattern in self.system_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if self._is_valid_system_id(match):
                        system_ids.add(match)
        
        # Filter and validate
        filtered_systems = []
        for system_id in system_ids:
            if self._validate_system_id(system_id):
                filtered_systems.append(system_id)
        
        return sorted(filtered_systems) if filtered_systems else []
    
    def _is_valid_system_id(self, candidate: str) -> bool:
        """Check if candidate is a valid SAP system ID"""
        if not candidate or len(candidate) < 2 or len(candidate) > 4:
            return False
        
        # Remove false positives
        if candidate in self.false_positives:
            return False
        
        # Must be exactly 3 characters for most SAP systems
        if len(candidate) == 3:
            # Check valid patterns
            for pattern in self.valid_patterns:
                if re.match(pattern, candidate):
                    return True
        
        # Allow some 2-character systems
        elif len(candidate) == 2:
            if re.match(r'^[A-Z][0-9]$', candidate):
                return True
        
        # Standard system names
        elif candidate in ['PRD', 'PROD', 'DEV', 'QAS', 'TST', 'TRN']:
            return True
        
        return False
    
    def _validate_system_id(self, system_id: str) -> bool:
        """Final validation of system ID"""
        # Must be 2-4 characters
        if len(system_id) < 2 or len(system_id) > 4:
            return False
        
        # Must contain at least one letter or number
        if not re.search('[A-Z0-9]', system_id):
            return False
        
        # Cannot be all numbers or all letters (except standard names)
        if system_id.isdigit():
            return False
        
        if system_id.isalpha() and system_id not in ['PRD', 'PROD', 'DEV', 'QAS', 'TST', 'TRN']:
            return False
        
        return True

# ===============================
# VECTOR STORE MANAGER
# ===============================

class VectorStoreManager:
    """Vector store manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vector_store = None
        self.embeddings = None
        
    def initialize_embeddings(self):
        """Initialize embeddings"""
        try:
            if OPENAI_API_KEY:
                from langchain.embeddings import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                logger.info("✅ OpenAI embeddings initialized")
            else:
                # Mock embeddings for testing
                class MockEmbeddings:
                    def embed_documents(self, texts):
                        return [[0.1] * 1536 for _ in texts]
                    def embed_query(self, text):
                        return [0.1] * 1536
                
                self.embeddings = MockEmbeddings()
                logger.info("⚠️ Using mock embeddings (no OpenAI API key)")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create vector store from documents"""
        try:
            if not self.embeddings:
                if not self.initialize_embeddings():
                    return False
            
            # Create ChromaDB vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=CONFIG["collection_name"],
                persist_directory=CONFIG["persist_directory"]
            )
            
            logger.info(f"✅ Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []
            
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

# ===============================
# WORKFLOW NODES
# ===============================

def pdf_processing_node(state: WorkflowState) -> WorkflowState:
    """PDF processing node"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "processing_pdf"
        state["current_agent"] = "pdf_processor"
        
        uploaded_files = state.get("uploaded_files", [])
        if not uploaded_files:
            raise ValueError("No PDF files provided")
        
        # Create PDF processor
        pdf_processor = PDFProcessorAgent(CONFIG)
        result = pdf_processor.process(uploaded_files)
        
        if not result.get("success"):
            raise ValueError(f"PDF processing failed: {result.get('error', 'Unknown error')}")
        
        processed_files = result.get("processed_files", [])
        if not processed_files:
            raise ValueError("No text extracted from PDFs")
        
        # Convert to documents
        documents = []
        for file_data in processed_files:
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
            documents.append(doc)
        
        # Update state
        state["processed_files"] = processed_files
        state["documents"] = documents
        state["processing_times"]["pdf_processor"] = time.time() - start_time
        state["success"] = True
        
        logger.info(f"✅ PDF processing completed: {len(documents)} documents")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["pdf_processor"] = time.time() - start_time
        logger.error(f"❌ PDF processing failed: {e}")
        return state

def embedding_creation_node(state: WorkflowState) -> WorkflowState:
    """Embedding creation node"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "creating_embeddings"
        state["current_agent"] = "embedding_creator"
        
        documents = state.get("documents", [])
        if not documents:
            raise ValueError("No documents available for embedding creation")
        
        # Embeddings are handled by vector store creation
        state["embeddings"] = []  # Placeholder
        state["processing_times"]["embedding_creator"] = time.time() - start_time
        
        logger.info("✅ Embedding creation completed")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["embedding_creator"] = time.time() - start_time
        logger.error(f"❌ Embedding creation failed: {e}")
        return state

def vector_storage_node(state: WorkflowState) -> WorkflowState:
    """Vector storage node"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "storing_vectors"
        state["current_agent"] = "vector_storage"
        
        documents = state.get("documents", [])
        if not documents:
            raise ValueError("No documents available for vector storage")
        
        # Create vector store manager
        vector_manager = VectorStoreManager(CONFIG)
        
        if not vector_manager.create_vector_store(documents):
            raise ValueError("Failed to create vector store")
        
        # Store in state
        state["vector_store"] = vector_manager
        state["processing_times"]["vector_storage"] = time.time() - start_time
        
        logger.info(f"✅ Vector store created with {len(documents)} documents")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["vector_storage"] = time.time() - start_time
        logger.error(f"❌ Vector storage failed: {e}")
        return state

def search_node(state: WorkflowState) -> WorkflowState:
    """Search node"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "searching"
        state["current_agent"] = "search"
        
        query = state.get("query", "")
        vector_manager = state.get("vector_store")
        
        if not query:
            raise ValueError("No search query provided")
        
        if not vector_manager:
            raise ValueError("Vector store not available")
        
        # Perform search
        search_results = vector_manager.similarity_search(query, k=CONFIG["top_k"])
        
        state["search_results"] = search_results
        state["processing_times"]["search"] = time.time() - start_time
        
        logger.info(f"✅ Search completed: {len(search_results)} results found")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["search"] = time.time() - start_time
        logger.error(f"❌ Search failed: {e}")
        return state

def analysis_node(state: WorkflowState) -> WorkflowState:
    """Analysis node with token management and multi-system handling"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "analyzing"
        state["current_agent"] = "analysis"
        
        query = state.get("query", "")
        search_results = state.get("search_results", [])
        user_system_id = state.get("user_system_id", "")
        system_ids = state.get("system_ids", [])
        
        if not search_results:
            state["summary"] = {
                "summary": "No relevant documents found for analysis.",
                "critical_findings": [],
                "recommendations": [],
                "confidence_score": 0.0
            }
        else:
            # FIX: Detect all systems in the documents for proper analysis
            if not system_ids:
                extractor = SystemIDExtractor()
                detected_systems = extractor.extract_from_documents(search_results)
                system_ids = detected_systems
                state["system_ids"] = system_ids
            
            # Create system-aware context
            systems_found = system_ids if system_ids else ["UNKNOWN"]
            systems_text = ", ".join(systems_found)
            
            # Token management for large documents
            max_context_chars = 12000
            
            # Prepare context with system awareness
            context_parts = []
            total_chars = 0
            
            # Group documents by system for better context
            system_docs = {}
            for doc in search_results:
                content = doc.page_content.upper()
                doc_systems = []
                
                # Check which systems this document mentions
                for sys_id in systems_found:
                    if sys_id.upper() in content:
                        doc_systems.append(sys_id)
                
                # If no specific system found, add to general
                if not doc_systems:
                    doc_systems = ["GENERAL"]
                
                for sys_id in doc_systems:
                    if sys_id not in system_docs:
                        system_docs[sys_id] = []
                    system_docs[sys_id].append(doc)
            
            # Build context with system separation
            for sys_id, docs in system_docs.items():
                if total_chars > max_context_chars:
                    break
                    
                context_parts.append(f"\n=== SYSTEM {sys_id} DOCUMENTS ===")
                
                for i, doc in enumerate(docs[:3]):  # Limit docs per system
                    doc_content = f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                    
                    # Truncate if too long
                    if len(doc_content) > 1500:
                        doc_content = doc_content[:1500] + "...\n[Content truncated]"
                    
                    if total_chars + len(doc_content) > max_context_chars:
                        break
                    
                    context_parts.append(doc_content)
                    total_chars += len(doc_content)
            
            context = "\n\n".join(context_parts)
            
            # Generate analysis with system awareness
            if OPENAI_API_KEY:
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=CONFIG["temperature"],
                    model="gpt-4o-mini",
                    max_tokens=2000
                )
                
                # FIX: Enhanced prompt for multi-system analysis
                prompt = f"""
                You are a SAP Early Watch Analysis expert. Analyze the following documents for MULTIPLE SAP SYSTEMS.
                
                Query: {query}
                
                IMPORTANT: The documents contain information about {len(systems_found)} different SAP systems: {systems_text}
                Please treat each system (GDP, P01, etc.) as SEPARATE and DISTINCT systems.
                
                Context from SAP documents:
                {context}
                
                Please provide analysis that:
                1. Clearly separates findings for each system: {systems_text}
                2. Identifies issues specific to each system
                3. Provides system-specific recommendations
                4. Does NOT combine different systems together
                
                Format your response to clearly distinguish between systems.
                
                Analysis:
                """
                
                try:
                    analysis = llm.predict(prompt)
                except Exception as llm_error:
                    if "context_length_exceeded" in str(llm_error) or "maximum context length" in str(llm_error):
                        # Fallback with shorter context
                        short_context = context[:6000] + "\n[Content truncated due to token limits]"
                        short_prompt = f"""
                        Multi-System SAP Analysis for: {systems_text}
                        Query: {query}
                        
                        Context: {short_context}
                        
                        Provide separate analysis for each system: {systems_text}
                        """
                        analysis = llm.predict(short_prompt)
                    else:
                        raise llm_error
                        
            else:
                analysis = f"Mock analysis for systems: {systems_text}. Query: '{query}'. Found {len(search_results)} relevant documents across {len(systems_found)} systems. Configure OpenAI API key for detailed AI analysis."
            
            state["summary"] = {
                "summary": analysis,
                "critical_findings": [f"Analysis completed for systems: {systems_text}"],
                "recommendations": ["Configure OpenAI API key for enhanced analysis"] if not OPENAI_API_KEY else [f"Review findings for each system: {systems_text}"],
                "confidence_score": 0.8 if OPENAI_API_KEY else 0.1,
                "query": query,
                "results_analyzed": len(search_results),
                "systems_analyzed": systems_found,
                "context_truncated": total_chars >= max_context_chars
            }
        
        state["processing_times"]["analysis"] = time.time() - start_time
        
        logger.info("✅ Analysis completed")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["analysis"] = time.time() - start_time
        logger.error(f"❌ Analysis failed: {e}")
        return state

def system_output_node(state: WorkflowState) -> WorkflowState:
    """System output node - generates individual system summaries"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "system_output"
        state["current_agent"] = "system_output_agent"
        
        search_results = state.get("search_results", [])
        system_ids = state.get("system_ids", [])
        
        # FIX 1: Use user input system ID if provided
        user_system_id = state.get("user_system_id", "")
        if user_system_id and user_system_id.strip():
            # User provided a specific system ID
            system_ids = [user_system_id.strip().upper()]
            logger.info(f"Using user-specified system ID: {user_system_id}")
        elif not system_ids and search_results:
            # Extract system IDs from search results
            extractor = SystemIDExtractor()
            system_ids = extractor.extract_from_documents(search_results)
            logger.info(f"Auto-detected system IDs: {system_ids}")
        
        if search_results and system_ids:
            # Generate individual system summaries with REAL data
            summary_generator = SystemSummaryGenerator()
            system_summaries = summary_generator.generate_system_summaries(search_results, system_ids)
            
            # Convert to list format for workflow
            system_summaries_list = []
            for system_id, summary in system_summaries.items():
                system_summaries_list.append(summary)
            
            state["system_summaries"] = system_summaries_list
        else:
            state["system_summaries"] = []
        
        state["processing_times"]["system_output_agent"] = time.time() - start_time
        
        logger.info("✅ System output completed")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["system_output_agent"] = time.time() - start_time
        logger.error(f"❌ System output failed: {e}")
        return state

def email_node(state: WorkflowState) -> WorkflowState:
    """Email node - send analysis results"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "sending_email"
        state["current_agent"] = "email_agent"
        
        email_data = state.get("email_data")
        if email_data and EMAIL_ENABLED:
            email_manager = EmailManager()
            
            if email_manager.is_configured():
                result = email_manager.send_email(
                    recipients=email_data.get("recipients", []),
                    subject=email_data.get("subject", "SAP EWA Analysis Results"),
                    body=email_data.get("body", "Analysis completed"),
                    cc_recipients=email_data.get("cc_recipients", [])
                )
                
                state["email_result"] = result
            else:
                state["email_result"] = {"success": False, "error": "Email not configured"}
        else:
            state["email_result"] = {"success": False, "error": "Email not enabled or no email data"}
        
        state["processing_times"]["email_agent"] = time.time() - start_time
        
        logger.info("✅ Email processing completed")
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["success"] = False
        state["processing_times"]["email_agent"] = time.time() - start_time
        logger.error(f"❌ Email processing failed: {e}")
        return state

def completion_node(state: WorkflowState) -> WorkflowState:
    """Completion node"""
    state["workflow_status"] = "completed"
    state["current_agent"] = "complete"
    
    # Calculate total processing time
    total_time = sum(state.get("processing_times", {}).values())
    state["total_processing_time"] = total_time
    
    logger.info(f"✅ Workflow completed in {total_time:.2f} seconds")
    return state

# ===============================
# WORKFLOW CREATION
# ===============================

def create_workflow() -> StateGraph:
    """Create LangGraph workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes in correct order from original
    workflow.add_node("pdf_processor", pdf_processing_node)
    workflow.add_node("embedding_creator", embedding_creation_node)
    workflow.add_node("vector_store_manager", vector_storage_node)
    workflow.add_node("search_agent", search_node)
    workflow.add_node("summary_agent", analysis_node)
    workflow.add_node("system_output_agent", system_output_node)
    workflow.add_node("email_agent", email_node)
    workflow.add_node("complete", completion_node)
    
    # Set entry point
    workflow.set_entry_point("pdf_processor")
    
    # Add edges exactly like original workflow
    workflow.add_edge("pdf_processor", "embedding_creator")
    workflow.add_edge("embedding_creator", "vector_store_manager")
    
    # Conditional edges for search flow
    workflow.add_conditional_edges(
        "vector_store_manager",
        lambda state: "search" if state.get("query") else "complete",
        {
            "search": "search_agent",
            "complete": "complete"
        }
    )
    
    workflow.add_edge("search_agent", "summary_agent") 
    workflow.add_edge("summary_agent", "system_output_agent")
    
    # Conditional edges for email
    workflow.add_conditional_edges(
        "system_output_agent",
        lambda state: "send_email" if EMAIL_ENABLED and state.get("email_data") else "complete",
        {
            "send_email": "email_agent", 
            "complete": "complete"
        }
    )
    
    workflow.add_edge("email_agent", "complete")
    workflow.add_edge("complete", END)
    
    return workflow.compile()

# ===============================
# UTILITY FUNCTIONS
# ===============================

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def format_analysis_email(query: str, analysis: str, search_results: List[Document]) -> str:
    """Format analysis for email"""
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2>🔍 SAP Early Watch Analysis Report</h2>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h3>📋 Query</h3>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px;">
            <strong>{query}</strong>
        </div>
        
        <h3>📊 AI Analysis</h3>
        <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
            {analysis.replace(chr(10), '<br>')}
        </div>
        
        <h3>📚 Source Documents ({len(search_results)} found)</h3>
    """
    
    for i, doc in enumerate(search_results):
        source = doc.metadata.get('source', 'Unknown')
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        html_content += f"""
        <div style="margin: 10px 0; padding: 10px; border-left: 3px solid #1f77b4;">
            <strong>Document {i+1}: {source}</strong><br>
            {content_preview.replace(chr(10), '<br>')}
        </div>
        """
    
    html_content += """
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; text-align: center;">
            <p><em>SAP Early Watch Analyzer | Powered by LangGraph & OpenAI</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ===============================
# STREAMLIT UI
# ===============================

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="SAP EWA Analyzer",
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
        .status-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        .success-box { background: #d4edda; border-color: #c3e6cb; }
        .warning-box { background: #fff3cd; border-color: #ffeaa7; }
        .error-box { background: #f8d7da; border-color: #f5c6cb; }
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
        <h1>🔍 SAP Early Watch Analyzer</h1>
        <p>AI-Powered Document Analysis with LangGraph | All Issues Fixed</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_workflow()
        logger.info("Workflow initialized")
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = None
    
    if "system_ids" not in st.session_state:
        st.session_state.system_ids = []
    
    if "selected_system_id" not in st.session_state:
        st.session_state.selected_system_id = ""
    
    # Email manager
    email_manager = EmailManager()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System status
        st.subheader("🔋 Current Status")
        
        if OPENAI_API_KEY:
            st.markdown('<div class="status-box success-box">✅ OpenAI API Configured</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ OpenAI API Not Set (Mock Mode)</div>', unsafe_allow_html=True)
        
        # Email status - FIX 4: Enhanced status display
        st.subheader("📧 Email Status")
        if email_manager.is_configured():
            st.markdown(f'<div class="status-box success-box">✅ {email_manager.provider.title()} Ready<br><small>{email_manager.get_status_message()}</small></div>', unsafe_allow_html=True)
        else:
            status_msg = email_manager.get_status_message()
            if "disabled" in status_msg:
                st.markdown(f'<div class="status-box warning-box">ℹ️ Email Disabled<br><small>{status_msg}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box error-box">❌ Email Not Configured<br><small>{status_msg}</small></div>', unsafe_allow_html=True)
        
        # Processing status
        if st.session_state.documents_processed:
            results = st.session_state.processing_results
            st.markdown('<div class="status-box success-box">✅ Documents Processed</div>', unsafe_allow_html=True)
            st.write(f"📄 Files: {len(results.get('processed_files', []))}")
            st.write(f"📚 Chunks: {len(results.get('documents', []))}")
            if st.session_state.system_ids:
                st.write(f"🖥️ Systems Found: {len(st.session_state.system_ids)}")
        else:
            st.markdown('<div class="status-box warning-box">⚠️ No Documents Processed</div>', unsafe_allow_html=True)
    
    # Main content area
    # File upload section
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload SAP Early Watch reports or related documents"
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
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Create initial state
                    initial_state = WorkflowState(
                        uploaded_files=uploaded_files,
                        processed_files=[],
                        documents=[],
                        embeddings=None,
                        vector_store=None,
                        search_results=[],
                        summary={},
                        system_summaries=[],
                        query="",
                        email_data=None,
                        workflow_status="initialized",
                        current_agent="",
                        processing_times={},
                        total_processing_time=0.0,
                        error=None,
                        success=False,
                        system_ids=[],
                        user_system_id=""  # FIX 1: Add user system ID
                    )
                    
                    # Run workflow
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: PDF Processing
                    status_text.text("Step 1/3: Processing PDF files...")
                    progress_bar.progress(25)
                    
                    result_state = pdf_processing_node(initial_state)
                    
                    if result_state.get("error"):
                        st.error(f"❌ {result_state['error']}")
                        st.stop()
                    
                    progress_bar.progress(50)
                    
                    # Step 2: Embedding Creation
                    status_text.text("Step 2/3: Creating embeddings...")
                    result_state = embedding_creation_node(result_state)
                    
                    progress_bar.progress(75)
                    
                    # Step 3: Vector Storage
                    status_text.text("Step 3/3: Creating vector database...")
                    result_state = vector_storage_node(result_state)
                    
                    if result_state.get("error"):
                        st.error(f"❌ {result_state['error']}")
                        st.stop()
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Extract system IDs from processed documents
                    extractor = SystemIDExtractor()
                    system_ids = extractor.extract_from_documents(result_state["documents"])
                    result_state["system_ids"] = system_ids
                    
                    # Store results
                    st.session_state.vector_store = result_state["vector_store"]
                    st.session_state.documents_processed = True
                    st.session_state.processing_results = result_state
                    st.session_state.system_ids = system_ids
                    
                    # Show success
                    st.success(f"✅ Successfully processed {len(result_state['documents'])} document chunks!")
                    
                    # Show processing stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", len(result_state['processed_files']))
                    with col2:
                        st.metric("Document Chunks", len(result_state['documents']))
                    with col3:
                        st.metric("Valid Systems Found", len(system_ids))
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Document processing error: {e}")
    
    # FIX 1: System ID Input Section
    if st.session_state.documents_processed:
        st.markdown("---")
        st.header("🖥️ System Selection")
        
        st.markdown('<div class="system-id-box">', unsafe_allow_html=True)
        
        # Show auto-detected systems if any
        if st.session_state.system_ids:
            st.write("**Auto-detected SAP Systems:**")
            cols = st.columns(min(4, len(st.session_state.system_ids)))
            for i, system_id in enumerate(st.session_state.system_ids):
                with cols[i % len(cols)]:
                    st.write(f"🖥️ **{system_id}**")
        
        # FIX 1: User System ID Input
        st.write("**Enter System ID to analyze (or leave empty for all systems):**")
        selected_system = st.text_input(
            "System ID",
            value=st.session_state.selected_system_id,
            placeholder="Enter system ID (e.g., P01, PRD, DEV) or leave empty for all systems",
            help="Enter a specific SAP system ID to focus the analysis, or leave empty to search all systems",
            key="system_id_input"
        )
        
        st.session_state.selected_system_id = selected_system
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Search section
    if st.session_state.documents_processed:
        st.markdown("---")
        st.header("🔍 Search & Analysis")
        
        # Query input
        query = st.text_area(
            "What would you like to know about your SAP documents?",
            height=100,
            placeholder="Example: What are the critical performance issues mentioned in the Early Watch report?"
        )
        
        # Add system context to query if system ID selected
        if st.session_state.selected_system_id and st.session_state.selected_system_id.strip():
            enhanced_query = f"For system {st.session_state.selected_system_id}: {query}" if query else f"Show me information about system {st.session_state.selected_system_id}"
        else:
            enhanced_query = query
        
        # Quick query examples
        if st.session_state.selected_system_id:
            st.write(f"**Quick Examples for System {st.session_state.selected_system_id}:**")
            quick_queries = [
                f"What are the performance issues for system {st.session_state.selected_system_id}?",
                f"Show critical findings for {st.session_state.selected_system_id}",
                f"Database performance problems in {st.session_state.selected_system_id}",
                f"High-priority action items for {st.session_state.selected_system_id}"
            ]
        else:
            st.write("**Quick Examples:**")
            quick_queries = [
                "What are the main performance issues?",
                "Show me critical findings and recommendations",
                "What database performance problems were identified?", 
                "List all high-priority action items"
            ]
        
        # Quick query buttons
        cols = st.columns(2)
        for i, quick_query in enumerate(quick_queries):
            with cols[i % 2]:
                if st.button(f"📋 {quick_query}", key=f"quick_{i}"):
                    query = quick_query
                    enhanced_query = quick_query
                    st.rerun()
        
        # Search button
        if st.button("🔍 Search Documents", type="primary", disabled=not enhanced_query.strip()):
            if enhanced_query.strip():
                with st.spinner("Searching and analyzing documents..."):
                    try:
                        # Create search state
                        search_state = WorkflowState(
                            uploaded_files=[],
                            processed_files=st.session_state.processing_results["processed_files"],
                            documents=st.session_state.processing_results["documents"],
                            embeddings=None,
                            vector_store=st.session_state.vector_store,
                            search_results=[],
                            summary={},
                            system_summaries=[],
                            query=enhanced_query,
                            email_data=None,
                            workflow_status="searching",
                            current_agent="search",
                            processing_times={},
                            total_processing_time=0.0,
                            error=None,
                            success=False,
                            system_ids=st.session_state.system_ids,
                            user_system_id=st.session_state.selected_system_id  # FIX 1: Pass user system ID
                        )
                        
                        # Run search and analysis workflow
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Search
                        status_text.text("Step 1/4: Searching relevant documents...")
                        progress_bar.progress(25)
                        
                        search_state = search_node(search_state)
                        
                        if search_state.get("error"):
                            st.error(f"❌ Search failed: {search_state['error']}")
                            st.stop()
                        
                        progress_bar.progress(50)
                        
                        # Step 2: Analysis
                        status_text.text("Step 2/4: Generating AI analysis...")
                        
                        search_state = analysis_node(search_state)
                        
                        if search_state.get("error"):
                            st.error(f"❌ Analysis failed: {search_state['error']}")
                            st.stop()
                        
                        progress_bar.progress(75)
                        
                        # Step 3: System-specific analysis
                        status_text.text("Step 3/4: Generating system summaries...")
                        
                        search_state = system_output_node(search_state)
                        
                        progress_bar.progress(90)
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        
                        # Display results
                        st.markdown("---")
                        st.header("📊 Analysis Results")
                        
                        summary = search_state.get("summary", {})
                        search_results = search_state.get("search_results", [])
                        system_summaries = search_state.get("system_summaries", [])
                        
                        # Show query context with multiple systems
                        detected_systems = summary.get("systems_analyzed", st.session_state.system_ids)
                        if detected_systems and len(detected_systems) > 1:
                            st.info(f"🖥️ Analysis covers {len(detected_systems)} systems: **{', '.join(detected_systems)}**")
                        elif st.session_state.selected_system_id:
                            st.info(f"🖥️ Analysis focused on system: **{st.session_state.selected_system_id}**")
                        
                        # Main analysis
                        st.subheader("🎯 Overall AI Analysis")
                        st.write(summary.get("summary", "No analysis available"))
                        
                        # Show context truncation warning if applicable
                        if summary.get("context_truncated"):
                            st.warning("⚠️ Large document content was truncated to fit context limits. Analysis is based on available content.")
                        
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Documents Found", len(search_results))
                        with col_b:
                            st.metric("Systems Analyzed", len(system_summaries))
                        with col_c:
                            st.metric("Analysis Time", f"{search_state.get('processing_times', {}).get('analysis', 0):.1f}s")
                        
                        # System-specific summaries
                        if system_summaries:
                            st.markdown("---")
                            st.subheader("🖥️ System-Specific Analysis")
                            
                            for system_summary in system_summaries:
                                system_id = system_summary.get("system_id", "Unknown")
                                health_status = system_summary.get("overall_health", "UNKNOWN")
                                
                                # Color coding based on health status
                                if health_status == "CRITICAL":
                                    status_color = "🔴"
                                    status_style = "background: #f8d7da; border-color: #f5c6cb;"
                                elif health_status == "WARNING":
                                    status_color = "🟡"
                                    status_style = "background: #fff3cd; border-color: #ffeaa7;"
                                else:
                                    status_color = "🟢"
                                    status_style = "background: #d4edda; border-color: #c3e6cb;"
                                
                                st.markdown(f"""
                                <div style="{status_style} padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                                    <h4>{status_color} System {system_id} - {health_status}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Critical alerts
                                critical_alerts = system_summary.get("critical_alerts", [])
                                if critical_alerts:
                                    st.write("**🚨 Critical Alerts:**")
                                    for alert in critical_alerts:
                                        st.write(f"• {alert}")
                                
                                # Recommendations
                                recommendations = system_summary.get("recommendations", [])
                                if recommendations:
                                    st.write("**💡 Recommendations:**")
                                    for rec in recommendations:
                                        st.write(f"• {rec}")
                                
                                # Key metrics
                                key_metrics = system_summary.get("key_metrics", {})
                                if key_metrics:
                                    st.write("**📊 Key Metrics:**")
                                    metric_cols = st.columns(len(key_metrics))
                                    for i, (metric_name, metric_value) in enumerate(key_metrics.items()):
                                        with metric_cols[i]:
                                            st.metric(metric_name.replace('_', ' ').title(), metric_value)
                                
                                st.markdown("---")
                        
                        # Email section
                        if EMAIL_ENABLED and email_manager.is_configured():
                            st.markdown("---")
                            st.subheader("📧 Email Results")
                            
                            email_recipients = st.text_input(
                                "Email Recipients (comma-separated)",
                                placeholder="email1@example.com, email2@example.com",
                                help="Enter email addresses separated by commas"
                            )
                            
                            email_subject = st.text_input(
                                "Email Subject",
                                value=f"SAP EWA Analysis Results - {datetime.now().strftime('%Y-%m-%d')}",
                                help="Subject line for the email"
                            )
                            
                            if st.button("📧 Send Analysis Email", type="secondary"):
                                if email_recipients.strip():
                                    # Validate emails
                                    recipient_list = [email.strip() for email in email_recipients.split(",")]
                                    valid_emails = [email for email in recipient_list if validate_email(email)]
                                    
                                    if valid_emails:
                                        # Format email content
                                        email_body = format_analysis_email(
                                            enhanced_query,
                                            summary.get("summary", "No analysis available"),
                                            search_results
                                        )
                                        
                                        # Send email
                                        email_result = email_manager.send_email(
                                            recipients=valid_emails,
                                            subject=email_subject,
                                            body=email_body
                                        )
                                        
                                        if email_result.get("success"):
                                            st.success(f"✅ Email sent successfully to {len(valid_emails)} recipients!")
                                        else:
                                            st.error(f"❌ Email failed: {email_result.get('error', 'Unknown error')}")
                                    else:
                                        st.error("❌ Please enter valid email addresses")
                                else:
                                    st.error("❌ Please enter email recipients")
                        
                    except Exception as e:
                        st.error(f"❌ Search and analysis failed: {str(e)}")
                        logger.error(f"Search and analysis error: {e}")
    
    # Email configuration section
    if EMAIL_ENABLED:
        st.markdown("---")
        st.header("📧 Email Configuration")
        
        st.write("**Current Email Status:**")
        st.write(f"• Provider: {email_manager.provider.title()}")
        st.write(f"• Status: {email_manager.get_status_message()}")
        
        if not email_manager.is_configured():
            st.warning("⚠️ Email is enabled but not properly configured. Please check your environment variables.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px;">
        <p><strong>SAP Early Watch Analyzer</strong> | Powered by LangGraph & OpenAI</p>
        <p><small>All Issues Fixed ✅ | User System ID Input ✅ | Real Critical Issues ✅ | Real SAP Recommendations ✅ | Email Provider Fix ✅</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()