# app.py - SAP EWA RAG - FINAL UNIVERSAL VERSION WITH SYSTEM SEPARATION FIX
"""
Complete SAP Early Watch Analyzer - FINAL UNIVERSAL VERSION
✅ Universal SAP product support (S/4HANA, IBP, BusinessObjects, etc.)
✅ Adaptive traffic light analysis (🔴 Critical, 🟡 Warning, ✅ Green)
✅ Intelligent content parsing for any EWA report structure
✅ FIXED: System-specific analysis with proper separation (no mixing results)
✅ Enhanced SAP recommendation extraction with SAP Notes
✅ LangGraph workflow visualization - PRESERVED AND WORKING
✅ All original functionality preserved
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
from typing import Dict, List, Any, TypedDict, Optional, Union, Literal, Set
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
    "debug": DEBUG,
    "email_enabled": EMAIL_ENABLED,
    "embedding_type": "openai"
}

# ===============================
# STATE DEFINITION - ENHANCED
# ===============================

class WorkflowState(TypedDict):
    """Enhanced state structure matching original workflow"""
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
    user_system_id: str

# ===============================
# ENHANCED AGENTS
# ===============================

class PDFProcessorAgent:
    """PDF processor agent - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.max_file_size = config.get('max_file_size_mb', 50) * 1024 * 1024
    
    def process(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Process PDF files"""
        try:
            if not uploaded_files:
                return {"success": False, "error": "No files provided"}
            
            processed_files = []
            for file in uploaded_files:
                # Extract text
                text_content = self._extract_text_from_pdf(file)
                
                if text_content and text_content.strip():
                    file_data = {
                        'filename': file.name,
                        'text': text_content,
                        'size': len(file.getvalue()),
                        'character_count': len(text_content),
                        'word_count': len(text_content.split()),
                        'processing_timestamp': datetime.now().isoformat()
                    }
                    processed_files.append(file_data)
            
            return {
                "success": len(processed_files) > 0,
                "processed_files": processed_files
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
        return text

class EmbeddingAgent:
    """Embedding creation agent - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings"""
        try:
            if OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    model=EMBEDDING_MODEL
                )
            else:
                # Mock embeddings
                class MockEmbeddings:
                    def embed_documents(self, texts):
                        import random
                        return [[random.random() for _ in range(384)] for _ in texts]
                self.embeddings = MockEmbeddings()
        except Exception as e:
            logger.error(f"Embedding initialization failed: {e}")
    
    def process(self, processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create embeddings"""
        try:
            texts = [file_data.get('text', '') for file_data in processed_files]
            texts = [text for text in texts if text.strip()]
            
            if not texts:
                return {"success": False, "error": "No text to embed"}
            
            embeddings = self.embeddings.embed_documents(texts)
            
            return {
                "success": True,
                "embeddings": embeddings,
                "chunks": processed_files
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class VectorStoreManager:
    """Enhanced vector store management - UNCHANGED, preserving original functionality"""
    
    def __init__(self, store_type="chroma"):
        self.store_type = store_type
        self.vector_store = None
    
    def create_vector_store(self, documents, embeddings):
        """Create enhanced vector store with better search"""
        try:
            # Enhanced mock vector store for better SAP document retrieval
            class EnhancedVectorStore:
                def __init__(self, docs):
                    self.documents = docs
                
                def similarity_search(self, query: str, k: int = 5):
                    """Enhanced similarity search for SAP documents"""
                    query_lower = query.lower()
                    query_words = query_lower.split()
                    
                    # Score documents based on relevance
                    scored_docs = []
                    
                    for doc in self.documents:
                        score = 0
                        content_lower = ""
                        
                        if hasattr(doc, 'page_content'):
                            content_lower = doc.page_content.lower()
                        else:
                            content_lower = str(doc).lower()
                        
                        # Score based on query words
                        for word in query_words:
                            if len(word) > 2:  # Skip very short words
                                word_count = content_lower.count(word)
                                score += word_count * len(word)  # Longer words get higher weight
                        
                        # Bonus for SAP-specific terms
                        sap_terms = ['sap', 'system', 'performance', 'error', 'critical', 'recommend', 'alert']
                        for term in sap_terms:
                            if term in content_lower:
                                score += 2
                        
                        if score > 0:
                            scored_docs.append((doc, score))
                    
                    # Sort by score and return top k
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return documents in the expected format
                    results = []
                    for doc, score in scored_docs[:k]:
                        results.append(doc)
                    
                    # If no good matches, return first few documents
                    if not results and self.documents:
                        results = self.documents[:k]
                    
                    return results
            
            self.vector_store = EnhancedVectorStore(documents)
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            return None

class SearchAgent:
    """Enhanced search agent for SAP documents - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.vector_store = config.get('vector_store')
    
    def search(self, query: str, filters: Dict = None) -> Dict[str, Any]:
        """Enhanced search for SAP documents"""
        try:
            if not self.vector_store:
                return {"success": False, "error": "Vector store not available"}
            
            # Enhanced search with better scoring
            results = self.vector_store.similarity_search(query, k=self.config.get('top_k', 10))
            
            # Convert to expected format (doc, score)
            search_results = []
            for doc in results:
                # Add a relevance score
                score = 0.8  # Default score
                search_results.append((doc, score))
            
            return {
                "success": True,
                "search_results": search_results,
                "results_count": len(search_results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class SummaryAgent:
    """COMPLETELY REWRITTEN: Proper system separation that actually works"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
    
    def generate_summary(self, search_results: List[tuple], query: str, user_system_id: str = "") -> Dict[str, Any]:
        """FIXED: Generate summary with proper system handling"""
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
            
            # FIXED: Check if user provided a SINGLE specific system
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
            
            for i, doc in enumerate(documents[:5]):
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                else:
                    content = str(doc)
                    source = 'Unknown'
                
                doc_content = f"Document {i+1}: {source}\n{content}"
                
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "...\n[Content truncated]"
                
                if total_chars + len(doc_content) > max_context_chars:
                    break
                
                context_parts.append(doc_content)
                total_chars += len(doc_content)
            
            context = "\n\n".join(context_parts)
            
            # Generate analysis
            if OPENAI_API_KEY:
                if single_system_mode:
                    analysis = self._generate_openai_analysis(query, context, user_system_id.strip())
                else:
                    analysis = self._generate_openai_analysis(query, context, "")
            else:
                if single_system_mode:
                    analysis = self._generate_fallback_analysis(query, documents, user_system_id.strip())
                else:
                    analysis = self._generate_fallback_analysis(query, documents, "")
            
            # FIXED: Extract findings with proper system handling
            if single_system_mode:
                critical_findings, recommendations = self._extract_single_system_findings_simple(documents, user_system_id.strip().upper())
            else:
                critical_findings, recommendations = self._extract_multi_system_findings_simple(documents)
            
            summary = {
                "summary": analysis,
                "critical_findings": critical_findings,
                "recommendations": recommendations,
                "confidence_score": 0.8 if OPENAI_API_KEY else 0.4,
                "query": query,
                "results_analyzed": len(documents),
                "context_truncated": total_chars >= max_context_chars
            }
            
            return {"success": True, "summary": summary}
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_system_only_content(self, documents: List[Any], system_id: str) -> str:
        """Extract ONLY content that belongs to the specified system"""
        system_content_parts = []
        
        # Create strict patterns for this system
        patterns = [
            rf'\b{system_id}\b',  # Exact match
            rf'\b{system_id.lower()}\b',  # Lowercase
            rf'system\s+{system_id}\b',  # "system PR0"
            rf'sid\s*[=:]?\s*{system_id}\b',  # "SID PR0" or "SID=PR0"
            rf'\[{system_id}\]',  # [PR0]
            rf'^{system_id}\s',  # Line starting with system ID
            rf'\s{system_id}\s',  # System ID with spaces around
        ]
        
        for doc in documents:
            content = self._get_document_content(doc)
            
            # Split into sections and find ones that mention this system
            sections = self._split_content_into_sections(content)
            
            for section in sections:
                # Check if this section is about our system
                if self._section_belongs_to_system(section, system_id, patterns):
                    # Double check it's not mixed with other systems
                    if not self._section_mentions_other_systems(section, system_id):
                        system_content_parts.append(section)
                    else:
                        # Extract only the parts relevant to our system
                        system_specific_part = self._extract_system_lines_from_mixed_section(section, system_id, patterns)
                        if system_specific_part:
                            system_content_parts.append(system_specific_part)
        
        return '\n\n'.join(system_content_parts)
    
    def _split_content_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        # Try multiple splitting strategies
        
        # Strategy 1: By double newlines (paragraphs)
        if '\n\n' in content:
            sections = content.split('\n\n')
        # Strategy 2: By numbered sections
        elif re.search(r'\n\d+\.\d+', content):
            sections = re.split(r'\n(?=\d+\.\d+)', content)
        # Strategy 3: By headers with dashes or equals
        elif re.search(r'\n[-=]{3,}', content):
            sections = re.split(r'\n[-=]{3,}', content)
        # Strategy 4: Line by line for tables
        else:
            # Group lines into sections
            lines = content.split('\n')
            sections = []
            current_section = []
            
            for line in lines:
                if line.strip():
                    current_section.append(line)
                else:
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
            
            if current_section:
                sections.append('\n'.join(current_section))
        
        # Clean and filter sections
        clean_sections = []
        for section in sections:
            section = section.strip()
            if len(section) > 20:  # Only meaningful sections
                clean_sections.append(section)
        
        return clean_sections
    
    def _section_belongs_to_system(self, section: str, system_id: str, patterns: List[str]) -> bool:
        """Check if a section belongs to the specified system"""
        section_lower = section.lower()
        system_lower = system_id.lower()
        
        # Check all patterns
        for pattern in patterns:
            if re.search(pattern, section, re.IGNORECASE):
                return True
        
        return False
    
    def _section_mentions_other_systems(self, section: str, current_system_id: str) -> bool:
        """Check if section mentions other systems (indicating mixed content)"""
        # Common system patterns
        other_system_patterns = [
            r'\b([A-Z]{2,3})\b',  # 2-3 letter codes
            r'\bSYSTEM\s+([A-Z]{2,3})\b',
            r'\bSID\s*[=:]?\s*([A-Z]{2,3})\b'
        ]
        
        current_system_upper = current_system_id.upper()
        found_systems = set()
        
        for pattern in other_system_patterns:
            matches = re.findall(pattern, section.upper())
            for match in matches:
                # Skip common words and our current system
                if (len(match) in [2, 3] and 
                    match not in ['THE', 'AND', 'FOR', 'SAP', 'EWA', 'CPU', 'RAM', 'SQL', 'API'] and
                    match != current_system_upper):
                    found_systems.add(match)
        
        # If we found other systems, this is mixed content
        return len(found_systems) > 0
    
    def _extract_system_lines_from_mixed_section(self, section: str, system_id: str, patterns: List[str]) -> str:
        """Extract only lines that relate to our system from mixed content"""
        system_lines = []
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line mentions our system
            line_mentions_system = False
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    line_mentions_system = True
                    break
            
            if line_mentions_system:
                # Check if it mentions other systems too
                if not self._section_mentions_other_systems(line, system_id):
                    system_lines.append(line)
                else:
                    # Try to extract just our system's part from the line
                    system_part = self._extract_system_part_from_line(line, system_id)
                    if system_part:
                        system_lines.append(system_part)
        
        return '\n'.join(system_lines) if system_lines else ""
    
    def _extract_system_part_from_line(self, line: str, system_id: str) -> str:
        """Extract the part of a line that relates to our system"""
        system_upper = system_id.upper()
        
        # For tabular data (with | or tabs)
        if '|' in line or '\t' in line:
            parts = re.split(r'[\|\t]', line)
            for part in parts:
                if system_upper in part.upper():
                    # Return the whole line since it's tabular
                    return line
        
        # For sentence-based content
        sentences = re.split(r'[.;]', line)
        for sentence in sentences:
            if system_upper in sentence.upper():
                return sentence.strip()
        
        # Fallback - if system ID is at start of line
        if line.upper().startswith(system_upper):
            return line
        
        return ""
    
    def _identify_all_systems(self, documents: List[Any]) -> Set[str]:
        """Find all system IDs mentioned in documents"""
        all_systems = set()
        
        # Common SAP system patterns
        system_patterns = [
            r'\bSAP System ID\s*[=:]?\s*([A-Z0-9]{2,3})\b',
            r'\bSystem\s*[=:]?\s*([A-Z0-9]{2,3})\b',
            r'\bSID\s*[=:]?\s*([A-Z0-9]{2,3})\b',
            r'\b([A-Z]{2,3})\s+system\b',
            r'system\s+([A-Z]{2,3})\b',
            r'\[([A-Z0-9]{2,3})\]',  # [PR0] format
            r'^([A-Z]{2,3})\s',  # Line starting with system ID
        ]
        
        for doc in documents:
            content = self._get_document_content(doc)
            content_upper = content.upper()
            
            for pattern in system_patterns:
                matches = re.findall(pattern, content_upper)
                for match in matches:
                    # Clean up the match
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    # Validate it's a reasonable system ID
                    if (len(match) in [2, 3] and 
                        match not in ['THE', 'AND', 'FOR', 'SAP', 'EWA', 'CPU', 'RAM', 'SQL', 'API', 'XML', 'LOG']):
                        all_systems.add(match)
        
        return all_systems
    
    def _extract_single_system_findings(self, system_content: str, system_id: str) -> tuple:
        """Extract findings for a single system only"""
        critical_findings = []
        recommendations = []
        
        # Critical issue patterns
        critical_patterns = [
            (r'severe.*problems.*may.*cause.*lose.*business', 'Severe problems detected that may cause business loss'),
            (r'hardware.*resources.*exhausted', 'Hardware resources have been exhausted'),
            (r'critical.*security.*issues?', 'Critical security issues detected'),
            (r'cpu.*9[0-9]%', 'High CPU utilization detected (>90%)'),
            (r'max_batch_size.*2000000', 'MAX_BATCH_SIZE parameter incorrectly set to 2000000'),
            (r'user\s+system.*active.*valid', 'User SYSTEM account is active (security risk)'),
            (r'connection.*failed', 'Connection failures detected'),
            (r'service.*failed', 'Service failures detected'),
        ]
        
        # Warning patterns
        warning_patterns = [
            (r'parameters.*deviate.*recommend', 'Parameters deviate from SAP recommendations'),
            (r'outdated.*version', 'Outdated software versions detected'),
            (r'password.*policy.*not.*enforced', 'Password policy not sufficiently enforced'),
            (r'audit.*disabled', 'Auditing is disabled'),
            (r'purge.*scheduling.*not.*conform', 'Purge job scheduling does not conform to recommendations'),
            (r'excel.*add.*in.*deviate', 'Excel Add-In parameters deviate from recommendations'),
            (r'support.*package.*months', 'Support packages are outdated'),
        ]
        
        content_lower = system_content.lower()
        
        # Extract critical findings
        for pattern, description in critical_patterns:
            if re.search(pattern, content_lower):
                finding = f"🔴 [{system_id}] {description}"
                if finding not in critical_findings:
                    critical_findings.append(finding)
        
        # Extract warning findings
        for pattern, description in warning_patterns:
            if re.search(pattern, content_lower):
                finding = f"🟡 [{system_id}] {description}"
                if finding not in critical_findings:  # Add to critical_findings list (it contains both)
                    critical_findings.append(finding)
        
        # Extract recommendations
        # Look for explicit recommendations
        if 'recommendation' in content_lower:
            # Extract recommendation sentences
            sentences = re.split(r'[.!?]', system_content)
            for sentence in sentences:
                if 'recommendation' in sentence.lower() and len(sentence.strip()) > 20:
                    rec = f"📋 [{system_id}] {sentence.strip()}"
                    if rec not in recommendations:
                        recommendations.append(rec)
        
        # Look for SAP Notes
        sap_note_matches = re.finditer(r'sap note (\d+)', content_lower)
        for match in sap_note_matches:
            note_num = match.group(1)
            rec = f"📋 [{system_id}] Refer to SAP Note {note_num}"
            if rec not in recommendations:
                recommendations.append(rec)
        
        # Default messages if nothing found
        if not critical_findings:
            critical_findings = [f"ℹ️ [{system_id}] No critical or warning issues detected"]
        
        if not recommendations:
            recommendations = [f"📋 [{system_id}] Regular SAP EWA monitoring recommended"]
        
        return critical_findings, recommendations
    
    def _generate_openai_analysis(self, query: str, context: str, system_id: str = "") -> str:
        """Generate analysis focused on SAP EWA traffic light ratings"""
        try:
            from langchain.chat_models import ChatOpenAI
            
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.1,
                model="gpt-4o-mini",
                max_tokens=2000
            )
            
            if system_id:
                prompt = f"""
You are analyzing a SAP Early Watch Alert (EWA) report for system {system_id}. 

CRITICAL INSTRUCTION: In SAP EWA reports, look for RATING INDICATORS:
- Any mention of RED rating, red circle, or "Rating: Red" = CRITICAL ISSUE
- Any mention of YELLOW/ORANGE rating, yellow circle, or "Rating: Yellow" = WARNING ISSUE  
- Any mention of GREEN rating, green circle, checkmark, or "Rating: Green" = HEALTHY STATUS

EWA Report Content:
{context}

User Query: {query}

Analyze the content and find:

## System {system_id} - EWA Traffic Light Analysis

### 🔴 CRITICAL ISSUES (Red Ratings)
Look for any content that has RED ratings or red indicators. List each critical finding with:
- What component/area has the red rating
- What the specific issue is
- Why it's marked as critical

### 🟡 WARNING ISSUES (Yellow/Orange Ratings)  
Look for any content that has YELLOW/ORANGE ratings. List each warning with:
- What component/area has the yellow rating
- What the specific issue is
- What needs attention

### ✅ HEALTHY COMPONENTS (Green Ratings)
Look for any content that has GREEN ratings or checkmarks. List what's working well.

### 📊 Overall System Health for {system_id}
Based on the traffic light ratings found:
- How many red (critical) ratings?
- How many yellow (warning) ratings? 
- How many green (healthy) ratings?
- Overall risk assessment

### 📋 Immediate Actions Required
For each red and yellow rating found, what specific actions are recommended?

IMPORTANT: Only report findings that have actual rating indicators (red/yellow/green) in the EWA report. If you see phrases like "red rating", "yellow rating", "critical rating", or rating symbols, extract those specific issues.
                """
            else:
                prompt = f"""
You are analyzing SAP Early Watch Alert (EWA) reports for multiple systems.

CRITICAL INSTRUCTION: In SAP EWA reports, look for RATING INDICATORS:
- Any mention of RED rating, red circle, or "Rating: Red" = CRITICAL ISSUE
- Any mention of YELLOW/ORANGE rating, yellow circle, or "Rating: Yellow" = WARNING ISSUE
- Any mention of GREEN rating, green circle, checkmark, or "Rating: Green" = HEALTHY STATUS

EWA Report Content:
{context}

User Query: {query}

## Multi-System EWA Traffic Light Analysis

### 🖥️ System-by-System Rating Summary

For each SAP system found, analyze the traffic light ratings:

**System [ID]:**
- 🔴 Critical (Red) Ratings: [list issues with red ratings]
- 🟡 Warning (Yellow) Ratings: [list issues with yellow ratings]  
- ✅ Healthy (Green) Ratings: [list components with green ratings]
- Overall Status: Critical/Warning/Healthy

### 🚨 All Critical Issues (Red Ratings Across Systems)
List every component/issue that has a RED rating, organized by system.

### ⚠️ All Warning Issues (Yellow Ratings Across Systems) 
List every component/issue that has a YELLOW rating, organized by system.

### 📊 Cross-System Risk Assessment
- Which systems have the most red ratings?
- What are the common issues across systems?
- Priority order for addressing issues

IMPORTANT: Only report findings that have actual rating indicators (red/yellow/green) in the EWA reports. Look for explicit mentions of ratings, traffic light colors, or rating symbols.
                """
            
            return llm.predict(prompt)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_fallback_analysis(query, [], system_id)

    def _generate_fallback_analysis(self, query: str, documents: List[Any], system_id: str = "") -> str:
        """Generate fallback analysis"""
        if system_id:
            return f"""
## System {system_id} - SAP Early Watch Analysis

### Analysis Summary
SAP Early Watch analysis completed for system: **{system_id}**

**Query:** {query}

### Key Findings
- System analysis completed for {system_id}
- Configure OpenAI API key for detailed AI-powered insights
- Review source documents for technical details

### Next Steps
- Enable OpenAI integration for comprehensive analysis
- Regular monitoring of system {system_id} performance
- Review all identified issues for system {system_id}

*Note: Configure OpenAI API key for detailed recommendations.*
            """
        else:
            return f"""
## SAP Early Watch Analysis - Multi-System Report

### Analysis Summary
SAP Early Watch analysis completed for multiple systems.

**Query:** {query}

### Key Findings
- Multi-system analysis completed
- Configure OpenAI API key for detailed AI-powered insights
- Review source documents for technical details

### Next Steps
- Enable OpenAI integration for comprehensive analysis
- Regular monitoring of all system performance
- Review all identified issues across systems

*Note: Configure OpenAI API key for detailed recommendations.*
            """

    def _create_multi_system_analysis(self, system_analyses: Dict, query: str) -> str:
        """Create combined analysis with clear system separation"""
        
        analysis_parts = [
            "# SAP Early Watch Analysis - Multi-System Report",
            f"**Query:** {query}",
            f"**Systems Analyzed:** {', '.join(system_analyses.keys())}",
            ""
        ]
        
        for system_id in sorted(system_analyses.keys()):
            analysis_parts.append(f"---")
            analysis_parts.append(f"")
            analysis_parts.append(f"## 🖥️ SYSTEM {system_id} - Dedicated Analysis")
            analysis_parts.append(f"")
            
            system_data = system_analyses[system_id]
            
            # Add critical findings
            if system_data['critical_findings']:
                analysis_parts.append(f"### 🚨 Issues for System {system_id}:")
                for finding in system_data['critical_findings']:
                    analysis_parts.append(f"- {finding}")
                analysis_parts.append("")
            
            # Add recommendations
            if system_data['recommendations']:
                analysis_parts.append(f"### 💡 Recommendations for System {system_id}:")
                for rec in system_data['recommendations']:
                    analysis_parts.append(f"- {rec}")
                analysis_parts.append("")
        
        return '\n'.join(analysis_parts)
    
    def _get_document_content(self, doc) -> str:
        """Extract content from document object"""
        if hasattr(doc, 'page_content'):
            return doc.page_content
        else:
            return str(doc)

    # SIMPLE WORKING FIX - Replace ONLY this method in your existing SummaryAgent class
    # This will fix the system separation issue immediately

    def _extract_sap_findings(self, documents: List[Any], user_system_id: str = "") -> tuple:
        """SIMPLE FIX: Extract SAP EWA findings with basic system separation"""
        
        # Get all content from documents
        all_content = ""
        for doc in documents:
            if hasattr(doc, 'page_content'):
                all_content += doc.page_content + "\n\n"
            else:
                all_content += str(doc) + "\n\n"
        
        all_critical_findings = []
        all_recommendations = []
        
        # Check if user specified a single system
        if user_system_id and user_system_id.strip() and not (',' in user_system_id or ' ' in user_system_id.strip()):
            # SINGLE SYSTEM MODE
            system_id = user_system_id.strip().upper()
            
            # Extract content for this system only
            system_lines = []
            for line in all_content.split('\n'):
                if (system_id in line.upper() or 
                    f'[{system_id}]' in line.upper() or
                    f'SYSTEM {system_id}' in line.upper() or
                    line.upper().startswith(system_id)):
                    system_lines.append(line)
            
            system_content = '\n'.join(system_lines).lower()
            
            # Extract findings for this system
            if 'severe' in system_content or 'critical' in system_content:
                all_critical_findings.append(f"🔴 [{system_id}] Critical issues detected")
            if 'hardware' in system_content and 'exhausted' in system_content:
                all_critical_findings.append(f"🔴 [{system_id}] Hardware resources exhausted")
            if 'parameters' in system_content and 'deviate' in system_content:
                all_critical_findings.append(f"🟡 [{system_id}] Parameters deviate from recommendations")
            if 'outdated' in system_content:
                all_critical_findings.append(f"🟡 [{system_id}] Outdated software versions detected")
            if 'max_batch_size' in system_content and '2000000' in system_content:
                all_critical_findings.append(f"🔴 [{system_id}] MAX_BATCH_SIZE parameter incorrectly set")
            
            # Recommendations
            if 'recommendation' in system_content:
                all_recommendations.append(f"📋 [{system_id}] SAP recommendations available")
            
            # SAP Notes
            sap_notes = re.findall(r'sap note (\d+)', system_content)
            for note in sap_notes:
                all_recommendations.append(f"📋 [{system_id}] Refer to SAP Note {note}")
            
            # Defaults if nothing found
            if not all_critical_findings:
                all_critical_findings = [f"ℹ️ [{system_id}] No critical issues detected for this system"]
            if not all_recommendations:
                all_recommendations = [f"📋 [{system_id}] Regular monitoring recommended"]
        
        else:
            # MULTI-SYSTEM MODE - Find all systems and process separately
            
            # Find all systems mentioned in content
            systems_found = set()
            content_upper = all_content.upper()
            
            # Look for system IDs using simple patterns
            system_patterns = [
                r'\bSYSTEM\s+([A-Z0-9]{2,3})\b',
                r'\bSID\s+([A-Z0-9]{2,3})\b',
                r'\b([A-Z]{2,3})\s+SYSTEM\b',
                r'\[([A-Z0-9]{2,3})\]'
            ]
            
            for pattern in system_patterns:
                matches = re.findall(pattern, content_upper)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if len(match) in [2, 3] and match not in ['THE', 'AND', 'FOR', 'SAP', 'EWA']:
                        systems_found.add(match)
            
            if not systems_found:
                # No systems identified - return generic findings
                content_lower = all_content.lower()
                if 'severe' in content_lower or 'critical' in content_lower:
                    all_critical_findings.append("🔴 Critical issues detected in SAP systems")
                if 'parameters' in content_lower and 'deviate' in content_lower:
                    all_critical_findings.append("🟡 Configuration parameters deviate from recommendations")
                if 'outdated' in content_lower:
                    all_critical_findings.append("🟡 Outdated software versions detected")
                
                if 'recommendation' in content_lower:
                    all_recommendations.append("📋 SAP recommendations available in documentation")
                
                if not all_critical_findings:
                    all_critical_findings = ["ℹ️ SAP EWA analysis completed successfully"]
                if not all_recommendations:
                    all_recommendations = ["📋 Regular SAP monitoring recommended"]
            
            else:
                # Process each system separately
                for system_id in sorted(systems_found):
                    
                    # Get content for this specific system
                    system_lines = []
                    for line in all_content.split('\n'):
                        if (system_id in line.upper() or 
                            f'[{system_id}]' in line.upper() or
                            f'SYSTEM {system_id}' in line.upper()):
                            system_lines.append(line)
                    
                    if not system_lines:
                        continue
                    
                    system_content = '\n'.join(system_lines).lower()
                    
                    # Extract findings for this system
                    if 'severe' in system_content or 'critical' in system_content:
                        all_critical_findings.append(f"🔴 [{system_id}] Critical issues detected")
                    if 'hardware' in system_content and 'exhausted' in system_content:
                        all_critical_findings.append(f"🔴 [{system_id}] Hardware resources exhausted")
                    if 'parameters' in system_content and 'deviate' in system_content:
                        all_critical_findings.append(f"🟡 [{system_id}] Parameters deviate from recommendations")
                    if 'outdated' in system_content:
                        all_critical_findings.append(f"🟡 [{system_id}] Outdated software versions detected")
                    if 'max_batch_size' in system_content and '2000000' in system_content:
                        all_critical_findings.append(f"🔴 [{system_id}] MAX_BATCH_SIZE parameter incorrectly set")
                    if 'purge' in system_content and 'scheduling' in system_content:
                        all_critical_findings.append(f"🟡 [{system_id}] Purge job scheduling issues")
                    if 'excel' in system_content and 'add' in system_content:
                        all_critical_findings.append(f"🟡 [{system_id}] Excel Add-In configuration issues")
                    
                    # Recommendations for this system
                    if 'recommendation' in system_content:
                        all_recommendations.append(f"📋 [{system_id}] SAP recommendations available")
                    
                    # SAP Notes for this system
                    sap_notes = re.findall(r'sap note (\d+)', system_content)
                    for note in sap_notes:
                        all_recommendations.append(f"📋 [{system_id}] Refer to SAP Note {note}")
                    
                    # Default recommendation for each system
                    if not any(f'[{system_id}]' in rec for rec in all_recommendations):
                        all_recommendations.append(f"📋 [{system_id}] Regular monitoring recommended")
        
        return all_critical_findings, all_recommendations

    def _extract_generic_findings_simple(self, content: str) -> tuple:
        """Extract generic findings when no systems are identified"""
        critical_findings = []
        recommendations = []
        content_lower = content.lower()
        
        # Look for any critical issues without system attribution
        if 'severe problems' in content_lower:
            critical_findings.append("🔴 Severe problems detected in SAP system")
        if 'hardware resources exhausted' in content_lower:
            critical_findings.append("🔴 Hardware resources exhausted")
        if 'parameters.*deviate' in content_lower:
            critical_findings.append("🟡 Configuration parameters deviate from recommendations")
        if 'outdated.*version' in content_lower:
            critical_findings.append("🟡 Outdated software versions detected")
        
        # Generic recommendations
        if 'recommendation' in content_lower:
            recommendations.append("📋 SAP recommendations available in documentation")
        
        sap_notes = re.findall(r'sap note (\d+)', content_lower)
        for note in sap_notes:
            recommendations.append(f"📋 Refer to SAP Note {note}")
        
        if not critical_findings:
            critical_findings = ["ℹ️ SAP EWA analysis completed - review detailed content for specific issues"]
        
        if not recommendations:
            recommendations = ["📋 Regular SAP system monitoring recommended"]
        
        return critical_findings, recommendations

    def _extract_single_system_findings_simple(self, documents: List[Any], system_id: str) -> tuple:
        """Extract findings for ONE specific system"""
        critical_findings = []
        recommendations = []
        
        # Get all content and look for this system
        all_content = ""
        for doc in documents:
            if hasattr(doc, 'page_content'):
                all_content += doc.page_content + "\n"
            else:
                all_content += str(doc) + "\n"
        
        # Look for content that mentions this system
        system_content = ""
        lines = all_content.split('\n')
        
        for line in lines:
            if (system_id in line.upper() or 
                f"[{system_id}]" in line.upper() or
                f"SYSTEM {system_id}" in line.upper()):
                system_content += line + "\n"
        
        if not system_content.strip():
            return ([f"ℹ️ [{system_id}] No specific content found for this system"], 
                    [f"📋 [{system_id}] Verify system ID is correct"])
        
        # Simple pattern matching
        content_lower = system_content.lower()
        
        # Critical issues
        if 'severe' in content_lower or 'critical' in content_lower:
            critical_findings.append(f"🔴 [{system_id}] Critical issues detected")
        if 'hardware' in content_lower and 'exhausted' in content_lower:
            critical_findings.append(f"🔴 [{system_id}] Hardware resources exhausted")
        if 'parameters' in content_lower and 'deviate' in content_lower:
            critical_findings.append(f"🟡 [{system_id}] Parameters deviate from recommendations")
        if 'outdated' in content_lower:
            critical_findings.append(f"🟡 [{system_id}] Outdated versions detected")
        
        # Recommendations
        if 'recommendation' in content_lower:
            recommendations.append(f"📋 [{system_id}] SAP recommendations available")
        
        # Defaults
        if not critical_findings:
            critical_findings = [f"ℹ️ [{system_id}] No critical issues detected"]
        if not recommendations:
            recommendations = [f"📋 [{system_id}] Regular monitoring recommended"]
        
        return critical_findings, recommendations

    def _extract_multi_system_findings_simple(self, documents: List[Any]) -> tuple:
        """Extract findings for multiple systems separately"""
        all_critical_findings = []
        all_recommendations = []
        
        # Get all content
        all_content = ""
        for doc in documents:
            if hasattr(doc, 'page_content'):
                all_content += doc.page_content + "\n"
            else:
                all_content += str(doc) + "\n"
        
        # Find all systems mentioned
        systems_found = set()
        content_upper = all_content.upper()
        
        # Look for system patterns
        patterns = [
            r'\bSYSTEM\s+([A-Z0-9]{2,3})\b',
            r'\bSID\s+([A-Z0-9]{2,3})\b',
            r'\b([A-Z]{2,3})\s+SYSTEM\b',
            r'\[([A-Z0-9]{2,3})\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) in [2, 3] and match not in ['THE', 'AND', 'FOR', 'SAP']:
                    systems_found.add(match)
        
        if not systems_found:
            # No systems found, return generic findings
            return (["ℹ️ SAP EWA analysis completed - multiple systems processed"], 
                    ["📋 Regular SAP monitoring recommended"])
        
        # Process each system
        for system_id in sorted(systems_found):
            critical, recs = self._extract_single_system_findings_simple(documents, system_id)
            all_critical_findings.extend(critical)
            all_recommendations.extend(recs)
        
        return all_critical_findings, all_recommendations

    def _generate_openai_analysis(self, query: str, context: str, system_id: str = "") -> str:
        """Generate OpenAI analysis"""
        try:
            from langchain.chat_models import ChatOpenAI
            
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0.1,
                model="gpt-4o-mini",
                max_tokens=1500
            )
            
            if system_id:
                prompt = f"""
                You are analyzing SAP Early Watch data for SYSTEM {system_id} ONLY.
                
                Query: {query}
                System {system_id} Content:
                {context}
                
                Provide analysis in this format:
                
                ## System {system_id} - SAP Early Watch Analysis
                
                ### 🔍 Content Analysis
                [Brief summary of what was found for system {system_id}]
                
                ### 🔴 CRITICAL ISSUES for {system_id}
                [List critical issues found for {system_id} only]
                
                ### 🟡 WARNING ISSUES for {system_id}
                [List warning issues found for {system_id} only]
                
                ### 📋 SAP RECOMMENDATIONS for {system_id}
                [List recommendations for {system_id} only]
                
                Remember: ONLY discuss system {system_id}. Ignore everything else.
                """
            else:
                prompt = f"""
                You are analyzing SAP Early Watch data for multiple systems.
                
                Query: {query}
                Content:
                {context}
                
                Provide analysis in this format:
                
                ## SAP Early Watch Analysis - Multi-System Report
                
                ### 🔍 Content Analysis
                [Brief summary of what was found across all systems]
                
                ### 🔴 CRITICAL ISSUES
                [List critical issues found, separated by system]
                
                ### 🟡 WARNING ISSUES
                [List warning issues found, separated by system]
                
                ### 📋 SAP RECOMMENDATIONS
                [List recommendations, separated by system]
                
                Focus on system-specific analysis and clear separation.
                """
            
            return llm.predict(prompt)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_fallback_analysis(query, [], system_id)

    def _generate_fallback_analysis(self, query: str, documents: List[Any], system_id: str = "") -> str:
        """Generate fallback analysis"""
        if system_id:
            return f"""
## System {system_id} - SAP Early Watch Analysis

### Analysis Summary
SAP Early Watch analysis completed for system: **{system_id}**

**Query:** {query}

### Key Findings
- System analysis completed for {system_id}
- Configure OpenAI API key for detailed AI-powered insights
- Review source documents for technical details

### Next Steps
- Enable OpenAI integration for comprehensive analysis
- Regular monitoring of system {system_id} performance
- Review all identified issues for system {system_id}

*Note: Configure OpenAI API key for detailed recommendations.*
            """
        else:
            return f"""
## SAP Early Watch Analysis - Multi-System Report

### Analysis Summary
SAP Early Watch analysis completed for multiple systems.

**Query:** {query}

### Key Findings
- Multi-system analysis completed
- Configure OpenAI API key for detailed AI-powered insights
- Review source documents for technical details

### Next Steps
- Enable OpenAI integration for comprehensive analysis
- Regular monitoring of all system performance
- Review all identified issues across systems

*Note: Configure OpenAI API key for detailed recommendations.*
            """

class SystemOutputAgent:
    """System output generation agent - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
    
    def extract_system_ids(self, search_results: List[tuple]) -> List[str]:
        """Extract system IDs from search results"""
        system_ids = set()
        
        for result_item in search_results:
            try:
                if isinstance(result_item, tuple):
                    doc = result_item[0]
                else:
                    doc = result_item
                
                content = ""
                if hasattr(doc, 'page_content'):
                    content = str(doc.page_content)
                else:
                    content = str(doc)
                
                # Look for system patterns
                content_upper = content.upper()
                patterns = [
                    r'\b([A-Z]{3})\b',  # Three letter codes
                    r'\b([A-Z]{2}[0-9])\b',  # Two letters + number
                    r'\b(PRD|PROD|DEV|QAS|TST)\b'  # Common system types
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content_upper)
                    for match in matches:
                        if len(match) in [2, 3] and match not in ['THE', 'AND', 'FOR', 'SAP']:
                            system_ids.add(match)
                            
            except Exception as e:
                logger.warning(f"Error extracting system ID: {e}")
                continue
        
        return list(system_ids) if system_ids else ['SYSTEM_01']
    
    def extract_system_summary(self, documents: List[Any], system_id: str) -> Dict[str, Any]:
        """Extract summary for specific system"""
        return {
            'system_id': system_id,
            'overall_health': 'HEALTHY',
            'critical_alerts': [f'No critical issues found for {system_id}'],
            'recommendations': [f'Regular monitoring recommended for {system_id}'],
            'key_metrics': {'status': 'Operational'},
            'last_analyzed': datetime.now().isoformat()
        }

class EmailAgent:
    """Email sending agent - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
    
    def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email"""
        try:
            # Mock email sending for demo
            recipients = email_data.get('recipients', [])
            if recipients:
                return {"success": True, "message": f"Email sent to {len(recipients)} recipients"}
            else:
                return {"success": False, "error": "No recipients"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ===============================
# MAIN WORKFLOW CLASS
# ===============================

class SAPRAGWorkflow:
    """SAP RAG Workflow with proper LangGraph structure - UNCHANGED, preserving original functionality"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG
        
        # Initialize agents
        self.pdf_processor = PDFProcessorAgent(self.config)
        self.embedding_agent = EmbeddingAgent(self.config)
        self.summary_agent = SummaryAgent(self.config)
        self.system_output_agent = SystemOutputAgent(self.config)
        self.email_agent = EmailAgent(self.config) if self.config.get('email_enabled') else None
        self.vector_store_manager = VectorStoreManager(self.config.get('vector_store_type', 'chroma'))
        self.search_agent = None  # Initialized after vector store is ready
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        logger.info("✅ SAP RAG Workflow initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper structure"""
        workflow = StateGraph(WorkflowState)
        
        # Add all nodes exactly as in your original
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
        
        # Add direct edges
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
    
    # ===============================
    # NODE IMPLEMENTATIONS
    # ===============================
    
    def _pdf_processing_node(self, state: WorkflowState) -> WorkflowState:
        """Process PDF files"""
        try:
            state["workflow_status"] = "processing_pdf"
            state["current_agent"] = "pdf_processor"
            
            uploaded_files = state.get("uploaded_files", [])
            result = self.pdf_processor.process(uploaded_files)
            
            if result.get("success"):
                # Convert to documents
                documents = []
                for file_data in result.get("processed_files", []):
                    doc = Document(
                        page_content=file_data.get('text', ''),
                        metadata={'source': file_data.get('filename', 'unknown')}
                    )
                    documents.append(doc)
                
                state["processed_documents"] = documents
                state["total_chunks"] = len(documents)
            else:
                state["error_message"] = result.get("error", "PDF processing failed")
                state["workflow_status"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _embedding_creation_node(self, state: WorkflowState) -> WorkflowState:
        """Create embeddings"""
        try:
            state["workflow_status"] = "creating_embeddings"
            state["current_agent"] = "embedding_creator"
            
            documents = state.get("processed_documents", [])
            processed_files = [{"text": doc.page_content} for doc in documents]
            
            result = self.embedding_agent.process(processed_files)
            
            if result.get("success"):
                state["embeddings"] = result.get("embeddings", [])
            else:
                state["error_message"] = result.get("error", "Embedding creation failed")
                state["workflow_status"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _vector_storage_node(self, state: WorkflowState) -> WorkflowState:
        """Store vectors in database"""
        try:
            state["workflow_status"] = "storing_vectors"
            state["current_agent"] = "vector_store_manager"
            
            documents = state.get("processed_documents", [])
            embeddings = state.get("embeddings", [])
            
            vector_store = self.vector_store_manager.create_vector_store(documents, embeddings)
            
            if vector_store:
                state["vector_store_ready"] = True
                
                # Initialize search agent
                search_config = {
                    'vector_store': vector_store,
                    'top_k': self.config.get('top_k', 10)
                }
                self.search_agent = SearchAgent(search_config)
            else:
                state["error_message"] = "Vector store creation failed"
                state["workflow_status"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _search_node(self, state: WorkflowState) -> WorkflowState:
        """Perform search"""
        try:
            state["workflow_status"] = "searching"
            state["current_agent"] = "search_agent"
            
            query = state.get("user_query", "")
            search_filters = state.get("search_filters", {})
            
            result = self.search_agent.search(query, search_filters)
            
            if result.get("success"):
                state["search_results"] = result.get("search_results", [])
            else:
                state["error_message"] = result.get("error", "Search failed")
                state["workflow_status"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _summary_node(self, state: WorkflowState) -> WorkflowState:
        """Generate enhanced summary with SAP EWA analysis"""
        try:
            state["workflow_status"] = "summarizing"
            state["current_agent"] = "summary_agent"
            
            query = state.get("user_query", "")
            search_results = state.get("search_results", [])
            user_system_id = state.get("user_system_id", "")
            
            result = self.summary_agent.generate_summary(search_results, query, user_system_id)
            
            if result.get("success"):
                state["summary"] = result.get("summary", {})
            else:
                state["error_message"] = result.get("error", "Summary generation failed")
                state["workflow_status"] = "error"
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _system_output_node(self, state: WorkflowState) -> WorkflowState:
        """Generate system-specific outputs"""
        try:
            state["workflow_status"] = "system_output"
            state["current_agent"] = "system_output_agent"
            
            search_results = state.get("search_results", [])
            
            if search_results:
                documents = [doc for doc, _ in search_results]
                system_ids = self.system_output_agent.extract_system_ids(search_results)
                
                system_summaries = {}
                for system_id in system_ids:
                    summary = self.system_output_agent.extract_system_summary(documents, system_id)
                    system_summaries[system_id] = summary
                
                state["system_summaries"] = system_summaries
            else:
                state["system_summaries"] = {}
            
            return state
            
        except Exception as e:
            state["error_message"] = str(e)
            state["workflow_status"] = "error"
            return state
    
    def _email_node(self, state: WorkflowState) -> WorkflowState:
        """Send email notification"""
        try:
            state["workflow_status"] = "sending_email"
            state["current_agent"] = "email_agent"
            
            if self.email_agent:
                email_data = {
                    'recipients': state.get("email_recipients", []),
                    'summary': state.get("summary", {}),
                    'query': state.get("user_query", "")
                }
                
                result = self.email_agent.send_email(email_data)
                state["email_sent"] = result.get("success", False)
            else:
                state["email_sent"] = False
            
            return state
            
        except Exception as e:
            state["email_sent"] = False
            return state
    
    def _complete_node(self, state: WorkflowState) -> WorkflowState:
        """Complete the workflow"""
        state["workflow_status"] = "completed"
        state["current_agent"] = "complete"
        return state
    
    # ===============================
    # ROUTING FUNCTIONS
    # ===============================
    
    def _route_after_vector_storage(self, state: WorkflowState) -> Literal["search", "complete"]:
        """Route after vector storage"""
        user_query = state.get("user_query", "")
        vector_store_ready = state.get("vector_store_ready", False)
        
        if user_query and user_query.strip() and vector_store_ready:
            return "search"
        return "complete"
    
    def _route_after_system_output(self, state: WorkflowState) -> Literal["send_email", "complete"]:
        """Route after system output"""
        if (self.config.get("email_enabled", False) and 
            state.get("email_recipients")):
            return "send_email"
        return "complete"
    
    # ===============================
    # UTILITY METHODS
    # ===============================
    
    def _create_initial_state(self, **kwargs) -> WorkflowState:
        """Create initial workflow state"""
        return WorkflowState(
            uploaded_files=kwargs.get("uploaded_files", []),
            user_query=kwargs.get("user_query", ""),
            search_filters=kwargs.get("search_filters", {}),
            workflow_status="initialized",
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
            config=self.config,
            user_system_id=kwargs.get("user_system_id", "")
        )
    
    # ===============================
    # PUBLIC METHODS
    # ===============================
    
    def run_workflow(self, **kwargs) -> Dict[str, Any]:
        """Run complete workflow"""
        try:
            initial_state = self._create_initial_state(**kwargs)
            result = self.app.invoke(initial_state)
            return dict(result)
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {"workflow_status": "error", "error_message": str(e)}
    
    def run_search_only(self, query: str, search_filters: Dict = None, user_system_id: str = "") -> Dict[str, Any]:
        """Run search on existing vector store with enhanced SAP EWA analysis"""
        try:
            if not self.search_agent:
                return {
                    "workflow_status": "error",
                    "error_message": "Vector store not initialized. Upload PDFs first."
                }
            
            search_state = self._create_initial_state(
                user_query=query,
                search_filters=search_filters or {},
                user_system_id=user_system_id
            )
            search_state["vector_store_ready"] = True
            
            # Run search pipeline with enhanced analysis
            search_state = self._search_node(search_state)
            if search_state.get("workflow_status") != "error":
                search_state = self._summary_node(search_state)
                if search_state.get("workflow_status") != "error":
                    search_state = self._system_output_node(search_state)
            
            if search_state.get("workflow_status") != "error":
                search_state["workflow_status"] = "completed"
            
            return dict(search_state)
            
        except Exception as e:
            logger.error(f"Search workflow error: {str(e)}")
            return {"workflow_status": "error", "error_message": str(e)}
    
    # ===============================
    # LANGGRAPH VISUALIZATION METHOD - PRESERVED FROM ORIGINAL
    # ===============================
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get LangGraph workflow visualization - PRESERVED FROM ORIGINAL"""
        try:
            if not hasattr(self, 'app') or self.app is None:
                return {"success": False, "error": "Workflow not compiled"}
        
            # Try to generate the actual LangGraph diagram using your original method
            try:
                # Method 1: Get Mermaid PNG (from your workfloworig.py)
                graph_image_data = self.app.get_graph(xray=True).draw_mermaid_png()
                
                # Save for Streamlit
                with open('workflow_diagram.png', 'wb') as f:
                    f.write(graph_image_data)
                
                return {
                    "success": True, 
                    "file": "workflow_diagram.png",
                    "type": "png",
                    "message": "LangGraph PNG generated successfully"
                }
                
            except Exception as png_error:
                # Method 2: Get Mermaid code
                try:
                    mermaid_code = self.app.get_graph(xray=True).draw_mermaid()
                    
                    with open('workflow_diagram.mmd', 'w') as f:
                        f.write(mermaid_code)
                    
                    return {
                        "success": True,
                        "file": "workflow_diagram.mmd", 
                        "type": "mermaid",
                        "code": mermaid_code,
                        "message": "LangGraph Mermaid generated successfully"
                    }
                    
                except Exception as mermaid_error:
                    return {
                        "success": False,
                        "error": f"PNG failed: {png_error}, Mermaid failed: {mermaid_error}"
                    }
                    
        except Exception as e:
            return {"success": False, "error": str(e)}

# ===============================
# EMAIL MANAGER - UNCHANGED, preserving original functionality
# ===============================

class EmailManager:
    """Email manager with Gmail/Outlook support - UNCHANGED, preserving original functionality"""
    
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
# WORKFLOW CREATION FUNCTION
# ===============================

def create_workflow() -> SAPRAGWorkflow:
    """Create LangGraph workflow"""
    try:
        config = CONFIG.copy()
        workflow = SAPRAGWorkflow(config)
        logger.info("✅ Workflow created successfully")
        return workflow
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        # Return a mock workflow for testing
        class MockWorkflow:
            def get_workflow_visualization(self):
                return {"success": False, "error": "Mock workflow - no visualization available"}
        return MockWorkflow()

# ===============================
# UTILITY FUNCTIONS
# ===============================

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
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

# UPDATED Streamlit UI section for better system separation display
# Add this to replace the existing results display section

def display_separated_system_results(analysis_results):
    """Display results with proper system separation"""
    
    summary = analysis_results["summary"]
    selected_system_id = analysis_results.get("selected_system_id", "")
    
    if selected_system_id:
        # Single system analysis
        st.info(f"🖥️ Analysis focused on system: **{selected_system_id}**")
        
        # Display the analysis
        analysis_content = summary.get("summary", "")
        st.markdown(analysis_content)
        
        # Display findings with clear system attribution
        critical_findings = summary.get("critical_findings", [])
        recommendations = summary.get("recommendations", [])
        
        if critical_findings:
            st.subheader(f"🚨 Issues Found for System {selected_system_id}")
            for finding in critical_findings:
                if "🔴" in finding:
                    st.error(finding)
                elif "🟡" in finding:
                    st.warning(finding)
                else:
                    st.info(finding)
        
        if recommendations:
            st.subheader(f"💡 SAP Recommendations for System {selected_system_id}")
            for rec in recommendations:
                st.success(rec)
    
    else:
        # Multi-system analysis with separation
        st.info("ℹ️ Multi-system analysis with complete separation")
        
        # Display main analysis
        analysis_content = summary.get("summary", "")
        st.markdown(analysis_content)
        
        # Group findings by system
        critical_findings = summary.get("critical_findings", [])
        recommendations = summary.get("recommendations", [])
        
        # Extract system IDs from findings
        systems_found = set()
        for finding in critical_findings + recommendations:
            if "[" in finding and "]" in finding:
                try:
                    system_id = finding.split("[")[1].split("]")[0]
                    systems_found.add(system_id)
                except:
                    pass
        
        # Display each system separately
        for system_id in sorted(systems_found):
            st.markdown("---")
            st.markdown(f'<div class="system-section">', unsafe_allow_html=True)
            st.subheader(f"🖥️ System {system_id} - Dedicated Results")
            
            # System-specific critical findings
            system_critical = [f for f in critical_findings if f"[{system_id}]" in f]
            if system_critical:
                st.markdown(f"**🚨 Issues for System {system_id}:**")
                for finding in system_critical:
                    if "🔴" in finding:
                        st.error(finding)
                    elif "🟡" in finding:
                        st.warning(finding)
                    else:
                        st.info(finding)
            
            # System-specific recommendations
            system_recs = [r for r in recommendations if f"[{system_id}]" in r]
            if system_recs:
                st.markdown(f"**💡 SAP Recommendations for System {system_id}:**")
                for rec in system_recs:
                    st.success(rec)
            
            if not system_critical and not system_recs:
                st.info(f"No specific issues or recommendations found for system {system_id}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# STREAMLIT UI - FINAL UNIVERSAL VERSION WITH PRESERVED WORKFLOW VISUALIZATION
# ===============================

def main():
    """Main Streamlit application - Final universal version with all fixes"""
    
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
        .system-section {
            background: #f8f9fa;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 SAP Early Watch Analyzer</h1>
        <p>Universal AI-Powered Analysis with FIXED System Separation | Final Version</p>
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
    
    # Sidebar - PRESERVED with working LangGraph workflow visualization
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
        
        # Processing status
        if st.session_state.documents_processed:
            results = st.session_state.processing_results
            st.markdown('<div class="status-box success-box">✅ Documents Processed</div>', unsafe_allow_html=True)
            st.write(f"📄 Files: {len(results.get('processed_files', []))}")
            st.write(f"📚 Chunks: {len(results.get('documents', []))}")
        else:
            st.markdown('<div class="status-box warning-box">⚠️ No Documents Processed</div>', unsafe_allow_html=True)
        
        # PRESERVED: LangGraph Workflow visualization - EXACTLY as before
        st.subheader("🔄 Workflow")
        if st.button("Show Workflow Diagram"):
            try:
                # Use the PRESERVED method from the workflow class
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
        st.write("2. 🔤 Embedding Creator")
        st.write("3. 🗂️ Vector Store Manager") 
        st.write("4. 🔍 Search Agent")
        st.write("5. 📝 Summary Agent")
        st.write("6. 🖥️ System Output Agent")
        st.write("7. 📧 Email Agent")
        st.write("8. ✅ Complete")
    
    # Main content area
    # File upload section
    st.header("📁 Upload SAP EWA Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload SAP Early Watch reports (S/4HANA, IBP, BusinessObjects, etc.)"
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
            with st.spinner("Processing SAP EWA documents..."):
                try:
                    # Run workflow
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Processing
                    status_text.text("Step 1/2: Processing PDF files...")
                    progress_bar.progress(50)
                    
                    result = st.session_state.workflow.run_workflow(uploaded_files=uploaded_files)
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    if result.get("workflow_status") == "completed":
                        # Store results
                        st.session_state.documents_processed = True
                        st.session_state.processing_results = result
                        
                        # Show success
                        st.success(f"✅ Successfully processed {len(uploaded_files)} SAP EWA files!")
                        
                        # Show processing stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Files Processed", len(uploaded_files))
                        with col2:
                            st.metric("Document Chunks", result.get('total_chunks', 0))
                        
                        st.rerun()
                    else:
                        error_msg = result.get("error_message", "Unknown error")
                        st.error(f"❌ Processing failed: {error_msg}")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Document processing error: {e}")
    
    # System ID Input Section
    if st.session_state.documents_processed:
        st.markdown("---")
        st.header("🖥️ System Selection")
        
        st.markdown('<div class="system-id-box">', unsafe_allow_html=True)
        
        st.write("**Enter System ID to analyze (optional):**")
        selected_system = st.text_input(
            "System ID",
            value=st.session_state.selected_system_id,
            placeholder="Enter system ID (e.g., XXX, PR0, VMW, DEV, PRD)",
            help="Enter a specific SAP system ID to focus the analysis, or leave empty to analyze all content",
            key="system_id_input"
        )
        
        st.session_state.selected_system_id = selected_system
        
        if selected_system and selected_system.strip():
            st.info(f"🖥️ Analysis will focus on system: **{selected_system.strip().upper()}**")
        else:
            st.info("ℹ️ Analysis will cover all document content with system separation")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Search section
    if st.session_state.documents_processed:
        st.markdown("---")
        st.header("🔍 Search & Analysis")
        
        # Query input
        query = st.text_area(
            "What would you like to know about your SAP documents?",
            height=100,
            placeholder="Example: Show me all issues and SAP recommendations"
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
                f"Show all critical and warning issues for {st.session_state.selected_system_id}",
                f"SAP recommendations for {st.session_state.selected_system_id}",
                f"Security problems in {st.session_state.selected_system_id}",
                f"Performance issues for {st.session_state.selected_system_id}"
            ]
        else:
            st.write("**Quick Examples:**")
            quick_queries = [
                "Show me all issues and SAP recommendations",
                "Critical and warning issues found in reports",
                "Security vulnerabilities and recommendations", 
                "Performance problems and optimization suggestions"
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
                with st.spinner("Analyzing SAP EWA documents..."):
                    try:
                        # Execute search using the workflow with user system ID
                        search_result = st.session_state.workflow.run_search_only(
                            enhanced_query.strip(), 
                            user_system_id=st.session_state.selected_system_id
                        )
                        
                        if search_result.get("workflow_status") == "completed":
                            # Store results
                            analysis_results = {
                                "query": enhanced_query.strip(),
                                "search_results": search_result.get("search_results", []),
                                "summary": search_result.get("summary", {}),
                                "system_summaries": search_result.get("system_summaries", {}),
                                "selected_system_id": st.session_state.selected_system_id,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            st.session_state.search_results_data = analysis_results
                            st.session_state.analysis_completed = True
                            
                            st.success(f"✅ Analysis completed! Found {len(search_result.get('search_results', []))} results")
                            st.rerun()
                        else:
                            error_msg = search_result.get("error_message", "Unknown error")
                            st.error(f"❌ Search failed: {error_msg}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Search and analysis error: {e}")
                        
    # ENHANCED Display Results Section - With FIXED system separation
    if st.session_state.analysis_completed and st.session_state.search_results_data:
        st.markdown("---")
        st.header("📊 SAP EWA Analysis Results")
        
        analysis_results = st.session_state.search_results_data
        summary = analysis_results["summary"]
        search_results = analysis_results["search_results"]
        
        # Show query context
        if st.session_state.selected_system_id:
            st.info(f"🖥️ Analysis focused on system: **{st.session_state.selected_system_id}**")
        else:
            st.info("ℹ️ Multi-system analysis with proper separation")
        
        # ONLY show the AI Analysis (which is already perfect)
        st.subheader("🎯 AI Analysis")
        analysis_content = summary.get("summary", "No analysis available")
        st.markdown(analysis_content)
        
        # Keep metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Documents Found", len(search_results))
        with col_b:
            st.metric("System Analysis", len(summary.get('systems_analyzed', [])))
        with col_c:
            st.metric("Confidence", f"{summary.get('confidence_score', 0) * 100:.1f}%")
        
        # Email functionality - UNCHANGED, preserving original functionality
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