# app.py - SAP EWA RAG - FINAL CLEAN VERSION
"""
Complete SAP Early Watch Analyzer - FINAL VERSION
✅ User provides valid system IDs manually
✅ No automatic system detection 
✅ No display of invalid system IDs
✅ System Analysis section commented out (shows only AI Analysis)
✅ All issues fixed and optimized
"""

import streamlit as st
import os
import logging
import time
import smtplib
import ssl
import re
import traceback
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
    "debug": DEBUG
}

# ===============================
# STATE DEFINITION
# ===============================

class WorkflowState(TypedDict):
    """State structure"""
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
    user_system_id: str

# ===============================
# BASE AGENT CLASS
# ===============================

class BaseAgent:
    """Base agent class"""
    
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
    """PDF processor"""
    
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
# SYSTEM SUMMARY GENERATOR
# ===============================

class SystemSummaryGenerator:
    """Generate summaries for user-specified systems only"""
    
    def generate_system_summary(self, search_results: List[Document], system_id: str) -> Dict[str, Any]:
        """Generate summary for a specific user-provided system"""
        critical_findings = []
        recommendations = []
        performance_metrics = {}
        health_status = "HEALTHY"
        
        # Filter documents for this specific system
        system_specific_docs = []
        for doc in search_results:
            content = doc.page_content.lower()
            # Check if document mentions this system
            if (system_id.lower() in content or 
                f"system {system_id.lower()}" in content or
                f"sid {system_id.lower()}" in content):
                system_specific_docs.append(doc)
        
        # If no system-specific docs found, use all docs
        if not system_specific_docs:
            system_specific_docs = search_results
        
        # Extract issues and recommendations
        for doc in system_specific_docs:
            content = doc.page_content.lower()
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Only process sentences that mention this specific system or use all content
                if not (system_id.lower() in sentence or len(system_specific_docs) == len(search_results)):
                    continue
                
                # Critical Issues Detection
                critical_keywords = [
                    'critical', 'error', 'fail', 'down', 'alert', 'issue', 'problem',
                    'exception', 'abort', 'crash', 'timeout', 'memory leak',
                    'database lock', 'slow response', 'high cpu', 'disk full',
                    'connection failed', 'service unavailable', 'performance degraded',
                    'deadlock', 'memory shortage', 'space critical', 'tablespace full'
                ]
                
                if any(keyword in sentence for keyword in critical_keywords):
                    critical_finding = f"[{system_id}] {sentence.capitalize()}"
                    if critical_finding not in critical_findings and len(critical_findings) < 5:
                        critical_findings.append(critical_finding)
                        health_status = "CRITICAL"
                
                # Recommendations Detection
                recommendation_keywords = [
                    'recommend', 'should', 'improve', 'optimize', 'upgrade', 'configure',
                    'adjust', 'tune', 'modify', 'consider', 'suggest', 'advise',
                    'increase', 'decrease', 'enable', 'disable', 'patch', 'update',
                    'archive', 'reorganize', 'index', 'parameter'
                ]
                
                if any(keyword in sentence for keyword in recommendation_keywords):
                    recommendation = f"[{system_id}] {sentence.capitalize()}"
                    if recommendation not in recommendations and len(recommendations) < 5:
                        recommendations.append(recommendation)
                
                # Extract performance metrics
                if system_id.lower() in sentence:
                    cpu_match = re.search(r'cpu[:\s]+([0-9]+)%', sentence)
                    if cpu_match:
                        performance_metrics['cpu_usage'] = f"{cpu_match.group(1)}%"
                    
                    memory_match = re.search(r'memory[:\s]+([0-9]+)%', sentence)
                    if memory_match:
                        performance_metrics['memory_usage'] = f"{memory_match.group(1)}%"
                    
                    disk_match = re.search(r'disk[:\s]+([0-9]+)%', sentence)
                    if disk_match:
                        performance_metrics['disk_usage'] = f"{disk_match.group(1)}%"
        
        # Determine health status
        if len(critical_findings) >= 3:
            health_status = "CRITICAL"
        elif len(critical_findings) >= 1:
            health_status = "WARNING"
        elif not critical_findings and recommendations:
            health_status = "HEALTHY"
        
        # Default messages if no specific data found
        if not critical_findings:
            critical_findings = [f"System {system_id}: No critical issues detected in current analysis"]
        
        if not recommendations:
            recommendations = [f"System {system_id}: Regular monitoring recommended", 
                             f"System {system_id}: Review system configuration periodically"]
        
        return {
            "system_id": system_id,
            "overall_health": health_status,
            "critical_alerts": critical_findings,
            "recommendations": recommendations,
            "key_metrics": performance_metrics,
            "documents_analyzed": len(system_specific_docs),
            "last_analyzed": datetime.now().isoformat()
        }

# ===============================
# EMAIL MANAGER
# ===============================

class EmailManager:
    """Email manager with Gmail/Outlook support"""
    
    def __init__(self):
        self.provider = EMAIL_PROVIDER
        self.email_enabled = EMAIL_ENABLED
    
    def is_configured(self) -> bool:
        if not self.email_enabled:
            return False
        
        if self.provider == "gmail":
            return bool(GMAIL_EMAIL and GMAIL_APP_PASSWORD)
        elif self.provider == "outlook":
            return bool(OUTLOOK_EMAIL and OUTLOOK_PASSWORD)
        
        return False
    
    def get_status_message(self) -> str:
        """Get email configuration status message"""
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
            return f"Email provider '{self.provider}' is not supported (use 'gmail' or 'outlook')"
    
    def send_email(self, recipients: List[str], subject: str, body: str, cc_recipients: List[str] = None) -> Dict[str, Any]:
        """Send email"""
        try:
            if not self.is_configured():
                return {"success": False, "error": f"Email not configured: {self.get_status_message()}"}
            
            logger.info(f"Attempting to send email via {self.provider} to {recipients}")
            
            msg = MIMEMultipart()
            
            if self.provider == "gmail":
                if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
                    return {"success": False, "error": "Gmail credentials missing in .env file"}
                msg['From'] = GMAIL_EMAIL
                sender_email = GMAIL_EMAIL
                sender_password = GMAIL_APP_PASSWORD
                smtp_server = "smtp.gmail.com"
                smtp_port = 587
            elif self.provider == "outlook":
                if not OUTLOOK_EMAIL or not OUTLOOK_PASSWORD:
                    return {"success": False, "error": "Outlook credentials missing in .env file"}
                msg['From'] = OUTLOOK_EMAIL
                sender_email = OUTLOOK_EMAIL
                sender_password = OUTLOOK_PASSWORD
                smtp_server = "smtp-mail.outlook.com"
                smtp_port = 587
            else:
                return {"success": False, "error": f"Unsupported email provider: {self.provider}"}
            
            msg['To'] = ", ".join(recipients)
            if cc_recipients:
                msg['Cc'] = ", ".join(cc_recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                
                all_recipients = recipients + (cc_recipients or [])
                server.sendmail(sender_email, all_recipients, msg.as_string())
            
            logger.info("Email sent successfully!")
            return {
                "success": True,
                "message": f"Email sent successfully to {len(recipients)} recipients",
                "recipients_count": len(recipients)
            }
            
        except Exception as e:
            error_msg = f"Email sending failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

# ===============================
# VECTOR STORE MANAGER
# ===============================

class VectorStoreManager:
    """Vector store manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vector_store = None
        self.embeddings = None
        self.store_type = self.config.get('vector_store_type', 'chroma')
        
    def initialize_embeddings(self):
        """Initialize embeddings"""
        try:
            if OPENAI_API_KEY:
                try:
                    from langchain.embeddings import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(
                        openai_api_key=OPENAI_API_KEY,
                        model=EMBEDDING_MODEL
                    )
                    logger.info("✅ OpenAI embeddings initialized")
                    return True
                except Exception as openai_error:
                    logger.warning(f"OpenAI embeddings failed: {openai_error}")
            
            # Fallback to HuggingFace
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("✅ HuggingFace embeddings initialized as fallback")
                return True
            except Exception as hf_error:
                logger.warning(f"HuggingFace embeddings failed: {hf_error}")
            
            # Mock embeddings as final fallback
            logger.warning("Using mock embeddings")
            self.embeddings = self._create_mock_embeddings()
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create vector store"""
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            if not documents:
                logger.error("No documents provided")
                return False
            
            if not self.embeddings:
                if not self.initialize_embeddings():
                    logger.error("Failed to initialize embeddings")
                    return False
            
            os.makedirs(CONFIG["persist_directory"], exist_ok=True)
            
            # Try ChromaDB first
            if self.store_type.lower() == 'chroma':
                success = self._create_chroma_store(documents)
                if success:
                    return True
                logger.warning("ChromaDB failed, trying fallback...")
            
            # Fallback to in-memory store
            success = self._create_memory_store(documents)
            return success
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            return self._create_fallback_store(documents)
    
    def _create_chroma_store(self, documents: List[Document]) -> bool:
        """Create ChromaDB vector store"""
        try:
            from langchain.vectorstores import Chroma
            import chromadb
            import uuid
            
            collection_name = f"sap_docs_{uuid.uuid4().hex[:8]}"
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=CONFIG["persist_directory"]
            )
            
            logger.info(f"✅ ChromaDB vector store created")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB creation failed: {e}")
            return False
    
    def _create_memory_store(self, documents: List[Document]) -> bool:
        """Create FAISS in-memory store"""
        try:
            from langchain.vectorstores import FAISS
            
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info(f"✅ FAISS vector store created")
            return True
            
        except Exception as e:
            logger.error(f"FAISS creation failed: {e}")
            return False
    
    def _create_fallback_store(self, documents: List[Document]) -> bool:
        """Create fallback store"""
        try:
            class SimpleFallbackStore:
                def __init__(self, documents):
                    self.documents = documents
                
                def similarity_search(self, query: str, k: int = 5) -> List[Document]:
                    query_lower = query.lower()
                    scored_docs = []
                    
                    for doc in self.documents:
                        content_lower = doc.page_content.lower()
                        score = 0
                        for word in query_lower.split():
                            if len(word) > 2:
                                score += content_lower.count(word)
                        
                        if score > 0:
                            scored_docs.append((doc, score))
                    
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    return [doc for doc, score in scored_docs[:k]]
            
            self.vector_store = SimpleFallbackStore(documents)
            logger.info("✅ Fallback vector store created")
            return True
            
        except Exception as e:
            logger.error(f"Fallback store creation failed: {e}")
            return False
    
    def _create_mock_embeddings(self):
        """Create mock embeddings"""
        class MockEmbeddings:
            def embed_documents(self, texts):
                import random
                return [[random.random() for _ in range(384)] for _ in texts]
            
            def embed_query(self, text):
                import random
                return [random.random() for _ in range(384)]
        
        return MockEmbeddings()
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []
            
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} documents for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
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

def vector_storage_node(state: WorkflowState) -> WorkflowState:
    """Vector storage node"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "storing_vectors"
        state["current_agent"] = "vector_storage"
        
        documents = state.get("documents", [])
        if not documents:
            raise ValueError("No documents available for vector storage")
        
        vector_manager = VectorStoreManager(CONFIG)
        success = vector_manager.create_vector_store(documents)
        
        if not success:
            raise ValueError("Failed to create vector store")
        
        state["vector_store"] = vector_manager
        state["processing_times"]["vector_storage"] = time.time() - start_time
        
        logger.info(f"✅ Vector store created successfully")
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
    """Analysis node with user-specified system focus"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "analyzing"
        state["current_agent"] = "analysis"
        
        query = state.get("query", "")
        search_results = state.get("search_results", [])
        user_system_id = state.get("user_system_id", "")
        
        if not search_results:
            state["summary"] = {
                "summary": "No relevant documents found for analysis.",
                "critical_findings": [],
                "recommendations": [],
                "confidence_score": 0.0
            }
        else:
            # Build context for analysis
            max_context_chars = 12000
            context_parts = []
            total_chars = 0
            
            if user_system_id and user_system_id.strip():
                context_parts.append(f"=== ANALYSIS FOR SYSTEM {user_system_id.strip().upper()} ===")
                context_parts.append("")
            else:
                context_parts.append("=== SAP DOCUMENT ANALYSIS ===")
                context_parts.append("")
            
            for i, doc in enumerate(search_results[:5]):  # Limit to top 5 docs
                doc_content = f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "...\n[Content truncated]"
                
                if total_chars + len(doc_content) > max_context_chars:
                    break
                
                context_parts.append(doc_content)
                total_chars += len(doc_content)
            
            context = "\n\n".join(context_parts)
            
            # Generate analysis
            if OPENAI_API_KEY:
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=CONFIG["temperature"],
                    model="gpt-4o-mini",
                    max_tokens=2000
                )
                
                if user_system_id and user_system_id.strip():
                    prompt = f"""
                    You are a SAP Early Watch Analysis expert. Analyze the following documents for SAP SYSTEM {user_system_id.strip().upper()}.
                    
                    Query: {query}
                    
                    Focus specifically on system: {user_system_id.strip().upper()}
                    
                    Context from SAP documents:
                    {context}
                    
                    Provide detailed analysis for system {user_system_id.strip().upper()} including:
                    - Critical issues specific to {user_system_id.strip().upper()}
                    - Recommendations for {user_system_id.strip().upper()}
                    - Performance insights for {user_system_id.strip().upper()}
                    
                    Format your response with clear sections:
                    ## System {user_system_id.strip().upper()} Analysis
                    [Your analysis here]
                    """
                else:
                    prompt = f"""
                    You are a SAP Early Watch Analysis expert. Analyze the following SAP documents.
                    
                    Query: {query}
                    
                    Context from SAP documents:
                    {context}
                    
                    Provide detailed analysis including:
                    - Critical issues identified
                    - Recommendations
                    - Performance insights
                    """
                
                try:
                    analysis = llm.predict(prompt)
                except Exception as llm_error:
                    if "context_length_exceeded" in str(llm_error):
                        short_context = context[:6000] + "\n[Content truncated due to token limits]"
                        short_prompt = f"SAP Analysis\nQuery: {query}\nContext: {short_context}\nProvide analysis."
                        analysis = llm.predict(short_prompt)
                    else:
                        raise llm_error
                        
            else:
                if user_system_id and user_system_id.strip():
                    analysis = f"""
                    ## System {user_system_id.strip().upper()} Analysis
                    
                    Analysis completed for SAP system: {user_system_id.strip().upper()}. 
                    Query: '{query}'. Found {len(search_results)} relevant documents.
                    
                    Configure OpenAI API key for enhanced detailed analysis.
                    """
                else:
                    analysis = f"""
                    ## SAP Document Analysis
                    
                    Analysis completed for SAP documents. 
                    Query: '{query}'. Found {len(search_results)} relevant documents.
                    
                    Configure OpenAI API key for enhanced detailed analysis.
                    """
            
            state["summary"] = {
                "summary": analysis,
                "critical_findings": ["Analysis completed"],
                "recommendations": ["Configure OpenAI API key for enhanced analysis"] if not OPENAI_API_KEY else ["Review analysis findings"],
                "confidence_score": 0.8 if OPENAI_API_KEY else 0.1,
                "query": query,
                "results_analyzed": len(search_results),
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
    """System output node for user-specified system"""
    start_time = time.time()
    
    try:
        state["workflow_status"] = "system_output"
        state["current_agent"] = "system_output_agent"
        
        search_results = state.get("search_results", [])
        user_system_id = state.get("user_system_id", "")
        
        if search_results and user_system_id and user_system_id.strip():
            summary_generator = SystemSummaryGenerator()
            system_summary = summary_generator.generate_system_summary(search_results, user_system_id.strip().upper())
            state["system_summaries"] = [system_summary]
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

def completion_node(state: WorkflowState) -> WorkflowState:
    """Completion node"""
    state["workflow_status"] = "completed"
    state["current_agent"] = "complete"
    
    total_time = sum(state.get("processing_times", {}).values())
    state["total_processing_time"] = total_time
    
    logger.info(f"✅ Workflow completed in {total_time:.2f} seconds")
    return state

# ===============================
# WORKFLOW CREATION
# ===============================

def create_workflow() -> StateGraph:
    """Create LangGraph workflow"""
    
    try:
        workflow = StateGraph(WorkflowState)
        
        workflow.add_node("pdf_processor", pdf_processing_node)
        workflow.add_node("vector_store_manager", vector_storage_node)
        workflow.add_node("search_agent", search_node)
        workflow.add_node("summary_agent", analysis_node)
        workflow.add_node("system_output_agent", system_output_node)
        workflow.add_node("complete", completion_node)
        
        workflow.set_entry_point("pdf_processor")
        
        workflow.add_edge("pdf_processor", "vector_store_manager")
        
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
        workflow.add_edge("system_output_agent", "complete")
        workflow.add_edge("complete", END)
        
        return workflow.compile()
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        class MockWorkflow:
            pass
        return MockWorkflow()

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
# STREAMLIT UI - FINAL CLEAN VERSION
# ===============================

def main():
    """Main Streamlit application - Final clean version"""
    
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
        <p>AI-Powered Document Analysis with LangGraph | Final Version</p>
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
    
    if "selected_system_id" not in st.session_state:
        st.session_state.selected_system_id = ""
    
    if "search_results_data" not in st.session_state:
        st.session_state.search_results_data = None
    
    if "analysis_completed" not in st.session_state:
        st.session_state.analysis_completed = False
    
    # Email manager
    email_manager = EmailManager()
    
    # Sidebar - Clean version with no system detection display
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System status
        st.subheader("🔋 Current Status")
        
        if OPENAI_API_KEY:
            st.markdown('<div class="status-box success-box">✅ OpenAI API Configured</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ OpenAI API Not Set (Mock Mode)</div>', unsafe_allow_html=True)
        
        # Email status
        st.subheader("📧 Email Status")
        if email_manager.is_configured():
            st.markdown(f'<div class="status-box success-box">✅ {email_manager.provider.title()} Ready</div>', unsafe_allow_html=True)
        else:
            status_msg = email_manager.get_status_message()
            if "disabled" in status_msg:
                st.markdown('<div class="status-box warning-box">ℹ️ Email Disabled</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box error-box">❌ Email Not Configured</div>', unsafe_allow_html=True)
        
        # Processing status - Clean version
        if st.session_state.documents_processed:
            results = st.session_state.processing_results
            st.markdown('<div class="status-box success-box">✅ Documents Processed</div>', unsafe_allow_html=True)
            st.write(f"📄 Files: {len(results.get('processed_files', []))}")
            st.write(f"📚 Chunks: {len(results.get('documents', []))}")
        else:
            st.markdown('<div class="status-box warning-box">⚠️ No Documents Processed</div>', unsafe_allow_html=True)
        
        # Workflow visualization
        st.subheader("🔄 Workflow")
        if st.button("Show Workflow Diagram"):
            try:
                workflow_viz = st.session_state.workflow.get_workflow_visualization()
                if workflow_viz.get("success"):
                    if workflow_viz.get("type") == "png":
                        st.image(workflow_viz["file"], caption="LangGraph Workflow")
                    elif workflow_viz.get("type") == "mermaid":
                        st.code(workflow_viz["code"], language="mermaid")
                    st.success(workflow_viz["message"])
                else:
                    st.error(f"Workflow visualization failed: {workflow_viz.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Workflow visualization error: {str(e)}")
                
        # Show workflow steps
        st.write("**Workflow Steps:**")
        st.write("1. 📄 PDF Processing")
        st.write("2. 🗂️ Vector Storage") 
        st.write("3. 🔍 Document Search")
        st.write("4. 🤖 AI Analysis")
        st.write("5. 📧 Email Results")
    
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
                        user_system_id=""
                    )
                    
                    # Run workflow
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: PDF Processing
                    status_text.text("Step 1/2: Processing PDF files...")
                    progress_bar.progress(50)
                    
                    result_state = pdf_processing_node(initial_state)
                    
                    if result_state.get("error"):
                        st.error(f"❌ {result_state['error']}")
                        st.stop()
                    
                    # Step 2: Vector Storage
                    status_text.text("Step 2/2: Creating vector database...")
                    result_state = vector_storage_node(result_state)
                    
                    if result_state.get("error"):
                        st.error(f"❌ {result_state['error']}")
                        st.stop()
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Store results
                    st.session_state.vector_store = result_state["vector_store"]
                    st.session_state.documents_processed = True
                    st.session_state.processing_results = result_state
                    
                    # Show success
                    st.success(f"✅ Successfully processed {len(result_state['documents'])} document chunks!")
                    
                    # Show processing stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files Processed", len(result_state['processed_files']))
                    with col2:
                        st.metric("Document Chunks", len(result_state['documents']))
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Document processing error: {e}")
    
    # System ID Input Section - USER INPUT ONLY
    if st.session_state.documents_processed:
        st.markdown("---")
        st.header("🖥️ System Selection")
        
        st.markdown('<div class="system-id-box">', unsafe_allow_html=True)
        
        st.write("**Enter System ID to analyze (optional):**")
        selected_system = st.text_input(
            "System ID",
            value=st.session_state.selected_system_id,
            placeholder="Enter system ID (e.g., GDP, P01, PRD, DEV)",
            help="Enter a specific SAP system ID to focus the analysis, or leave empty to analyze all content",
            key="system_id_input"
        )
        
        st.session_state.selected_system_id = selected_system
        
        if selected_system and selected_system.strip():
            st.info(f"🖥️ Analysis will focus on system: **{selected_system.strip().upper()}**")
        else:
            st.info("ℹ️ Analysis will cover all document content")
            
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
                            user_system_id=st.session_state.selected_system_id
                        )
                        
                        # Run search and analysis workflow
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Search
                        status_text.text("Step 1/3: Searching relevant documents...")
                        progress_bar.progress(33)
                        
                        search_state = search_node(search_state)
                        
                        if search_state.get("error"):
                            st.error(f"❌ Search failed: {search_state['error']}")
                            st.stop()
                        
                        # Step 2: Analysis
                        status_text.text("Step 2/3: Generating AI analysis...")
                        progress_bar.progress(66)
                        
                        search_state = analysis_node(search_state)
                        
                        if search_state.get("error"):
                            st.error(f"❌ Analysis failed: {search_state['error']}")
                            st.stop()
                        
                        # Step 3: System-specific analysis
                        status_text.text("Step 3/3: Generating system summary...")
                        progress_bar.progress(90)
                        
                        search_state = system_output_node(search_state)
                        
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        
                        # Store results
                        analysis_results = {
                            "query": enhanced_query,
                            "search_results": search_state.get("search_results", []),
                            "summary": search_state.get("summary", {}),
                            "system_summaries": search_state.get("system_summaries", []),
                            "selected_system_id": st.session_state.selected_system_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.session_state.search_results_data = analysis_results
                        st.session_state.analysis_completed = True
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Search and analysis error: {e}")
                        
    # Display Results Section - CLEAN VERSION WITH ONLY AI ANALYSIS
    if st.session_state.analysis_completed and st.session_state.search_results_data:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        analysis_results = st.session_state.search_results_data
        summary = analysis_results["summary"]
        search_results = analysis_results["search_results"]
        system_summaries = analysis_results["system_summaries"]
        
        # Show query context
        if st.session_state.selected_system_id:
            st.info(f"🖥️ Analysis focused on system: **{st.session_state.selected_system_id}**")
        
        # Main AI Analysis - THIS IS THE USEFUL PART
        st.subheader("🎯 AI Analysis")
        st.write(summary.get("summary", "No analysis available"))
        
        # Show context truncation warning if applicable
        if summary.get("context_truncated"):
            st.warning("⚠️ Large document content was truncated to fit context limits. Analysis is based on available content.")
        
        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Documents Found", len(search_results))
        with col_b:
            st.metric("System Analysis", len(system_summaries))
        with col_c:
            st.metric("Analysis Time", f"{summary.get('processing_time', 0):.1f}s")
        
        # COMMENTED OUT: System Analysis section (showing document metadata instead of useful info)
        # The following section is commented out as it shows document headers instead of meaningful analysis
        # 
        # if system_summaries:
        #     st.markdown("---")
        #     st.header("🖥️ System Analysis")
        #     
        #     for system_summary in system_summaries:
        #         system_id = system_summary.get("system_id", "Unknown")
        #         health_status = system_summary.get("overall_health", "UNKNOWN")
        #         
        #         # Health status styling
        #         if health_status == "CRITICAL":
        #             status_color = "🔴"
        #         elif health_status == "WARNING":
        #             status_color = "🟡"
        #         else:
        #             status_color = "🟢"
        #         
        #         st.subheader(f"{status_color} System {system_id} - {health_status}")
        #         
        #         # Critical alerts
        #         critical_alerts = system_summary.get("critical_alerts", [])
        #         if critical_alerts:
        #             st.write("**🚨 Critical Issues:**")
        #             for alert in critical_alerts:
        #                 st.write(f"• {alert}")
        #         
        #         # Recommendations
        #         recommendations = system_summary.get("recommendations", [])
        #         if recommendations:
        #             st.write("**💡 Recommendations:**")
        #             for rec in recommendations:
        #                 st.write(f"• {rec}")
        #         
        #         # Key metrics
        #         key_metrics = system_summary.get("key_metrics", {})
        #         if key_metrics:
        #             st.write("**📊 Key Metrics:**")
        #             metrics_cols = st.columns(len(key_metrics))
        #             for i, (metric, value) in enumerate(key_metrics.items()):
        #                 with metrics_cols[i]:
        #                     st.metric(metric.replace('_', ' ').title(), value)
        
        # Email functionality
        if email_manager.is_configured():
            st.markdown("---")
            st.header("📧 Email Results")
            
            with st.expander("📧 Send Analysis via Email"):
                # Email form
                col1, col2 = st.columns(2)
                with col1:
                    recipients = st.text_area(
                        "Recipients (one per line)",
                        placeholder="Enter recipient email addresses",
                        height=100
                    )
                with col2:
                    cc_recipients = st.text_area(
                        "CC Recipients (one per line)",
                        placeholder="Enter CC email addresses (optional)",
                        height=100
                    )
                
                email_subject = st.text_input(
                    "Subject",
                    value=f"SAP EWA Analysis Results - {datetime.now().strftime('%Y-%m-%d')}",
                    help="Email subject line"
                )
                
                # Validate recipients
                recipient_list = [r.strip() for r in recipients.split('\n') if r.strip()]
                cc_list = [r.strip() for r in cc_recipients.split('\n') if r.strip()]
                
                valid_recipients = all(validate_email(email) for email in recipient_list) if recipient_list else False
                valid_cc = all(validate_email(email) for email in cc_list) if cc_list else True
                
                if recipient_list and not valid_recipients:
                    st.error("❌ Please enter valid email addresses for recipients")
                if cc_list and not valid_cc:
                    st.error("❌ Please enter valid email addresses for CC recipients")
                
                if st.button("📧 Send Email", disabled=not (valid_recipients and valid_cc)):
                    if recipient_list:
                        # Format email body
                        email_body = format_analysis_email(
                            analysis_results["query"],
                            summary.get("summary", ""),
                            search_results
                        )
                        
                        # Send email
                        result = email_manager.send_email(
                            recipients=recipient_list,
                            subject=email_subject,
                            body=email_body,
                            cc_recipients=cc_list if cc_list else []
                        )
                        
                        if result.get("success"):
                            st.success(f"✅ Email sent successfully to {len(recipient_list)} recipients!")
                        else:
                            st.error(f"❌ Email failed: {result.get('error', 'Unknown error')}")


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    main()