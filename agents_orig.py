# agents.py - Clean AI Agents for SAP EWA Analysis
"""
SPECIALIZED AI AGENTS FOR SAP EARLY WATCH ANALYSIS WORKFLOW
============================================================

This module implements a collection of specialized AI agents that work together 
in a LangGraph workflow to analyze SAP Early Watch Alert (EWA) documents.

WORKFLOW INTEGRATION:
The agents are designed to be orchestrated by the LangGraph workflow in workflow.py:
1. PDFProcessorAgent → 2. EmbeddingAgent → 3. SearchAgent → 4. SummaryAgent → 5. SystemOutputAgent → 6. EmailAgent

AVAILABLE AGENTS:
- PDFProcessorAgent: Extracts and cleans text from uploaded PDF files
- EmbeddingAgent: Creates vector embeddings from text chunks using OpenAI/SentenceTransformers
- SearchAgent: Performs semantic similarity search on ChromaDB vector store
- SummaryAgent: Generates intelligent summaries with critical findings and recommendations
- EmailAgent: Sends email notifications via Gmail or Outlook SMTP
- SystemOutputAgent: Creates system-specific health analysis outputs

CRITICAL INTEGRATION POINTS:
- All agents inherit from BaseAgent for standardized logging and error handling
- Agents communicate through structured dictionaries returned by their main methods
- The workflow.py orchestrates agent execution and manages state transitions
- Configuration is injected from config.py through the agent_config dictionary
- Vector store integration happens through the SearchAgent with ChromaDB
"""

import json
import os
import re
import time
import logging
import smtplib
import ssl
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from email.message import EmailMessage

# Safe imports with fallbacks
try:
    from models import EmailRecipient, SystemSummary, SearchResult, HealthStatus
except ImportError as e:
    logging.warning(f"Could not import models: {e}")

try:
    from config import get_agent_config
except ImportError as e:
    logging.warning(f"Could not import config: {e}")
    def get_agent_config():
        return {"chunk_size": 1000, "top_k": 10}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# BASE AGENT CLASS - FOUNDATION FOR ALL AGENTS
# ================================

class BaseAgent(ABC):
    """
    ABSTRACT BASE CLASS FOR ALL SAP EWA ANALYSIS AGENTS
    ===================================================
    
    This is the foundation class that all specialized agents inherit from.
    It provides essential shared functionality to ensure consistent behavior
    across all agents in the workflow.
    
    KEY RESPONSIBILITIES:
    1. Standardized logging with agent identification for debugging
    2. Unified error handling and reporting with structured error responses  
    3. Configuration management and validation
    4. Performance metrics tracking for monitoring and optimization
    5. Common utility methods used by all agents
    
    WORKFLOW INTEGRATION:
    - Each agent node in the LangGraph workflow inherits from this class
    - Error handling ensures workflow can gracefully handle agent failures
    - Performance metrics are collected for workflow optimization
    - Logging provides detailed execution traces for debugging
    
    INHERITANCE PATTERN:
    BaseAgent (abstract) → PDFProcessorAgent, EmbeddingAgent, SearchAgent, etc.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base agent with configuration and logging setup.
        
        CRITICAL INTEGRATION POINT:
        - The config parameter comes from config.py via get_agent_config()
        - This config contains OpenAI keys, email settings, processing params
        - All agents receive the same base config but use different parts of it
        
        Args:
            name: Human-readable agent name (used in logs and error tracking)
            config: Configuration dictionary from config.py containing all settings
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")  # Hierarchical logging
        self.start_time = None  # For performance timing
        self.performance_metrics = {}  # Stores timing data for each operation
        
        # Initialize agent-specific defaults (timeout, retry settings, etc.)
        self._setup_agent_defaults()
        self.log_info(f"Agent initialized with config keys: {list(config.keys())}")
    
    def _setup_agent_defaults(self):
        """
        Set up default configuration values that all agents need.
        
        DESIGN PATTERN:
        - Provides sensible defaults for common parameters
        - Allows individual agents to override in their __init__ methods
        - Ensures all agents have basic operational parameters
        """
        defaults = {
            'timeout': 300,        # 5 minutes max execution time
            'retry_attempts': 3,   # How many times to retry failed operations
            'retry_delay': 1.0     # Delay between retries in seconds
        }
        
        # Only set defaults for missing keys (don't override existing config)
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def log_info(self, message: str):
        """
        Log informational message with agent identification.
        
        WORKFLOW INTEGRATION:
        - All log messages include agent name for easy debugging
        - Logs are collected by the workflow for execution tracing
        - Essential for debugging multi-agent workflow issues
        """
        self.logger.info(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        """Log warning message with agent identification."""
        self.logger.warning(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message with agent identification."""
        self.logger.error(f"[{self.name}] {message}")
    
    def start_timer(self):
        """
        Start performance timing for current operation.
        
        PERFORMANCE MONITORING:
        - Each agent operation is timed for workflow optimization
        - Helps identify bottlenecks in the processing pipeline
        - Metrics are aggregated at the workflow level
        """
        self.start_time = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """
        End performance timing and record the duration.
        
        WORKFLOW INTEGRATION:
        - Timing data is stored in performance_metrics dict
        - The workflow collects these metrics for reporting
        - Used for identifying slow agents and optimizing performance
        
        Args:
            operation_name: Name of the operation being timed
            
        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            self.log_warning("Timer not started, cannot measure duration")
            return 0.0
        
        duration = time.time() - self.start_time
        self.performance_metrics[operation_name] = duration
        self.log_info(f"{operation_name} completed in {duration:.2f}s")
        self.start_time = None
        return duration
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get all recorded performance metrics for this agent.
        
        WORKFLOW INTEGRATION:
        - Called by workflow to collect performance data
        - Used for monitoring and optimization
        - Helps identify performance bottlenecks
        
        Returns:
            Dictionary mapping operation names to durations in seconds
        """
        return self.performance_metrics.copy()
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Standardized error handling for all agents.
        
        CRITICAL WORKFLOW INTEGRATION:
        - Returns standardized error format that workflow can process
        - Ensures all agent failures are handled consistently
        - Provides structured error data for debugging and recovery
        - The workflow checks for "success": False to handle errors
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            Standardized error response dictionary with success=False
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.log_error(error_msg)
        
        return {
            "success": False,           # KEY: Workflow checks this field
            "error": error_msg,         # Human-readable error message
            "error_type": type(error).__name__,  # Exception type for debugging
            "agent": self.name,         # Which agent failed
            "timestamp": datetime.now().isoformat()  # When the error occurred
        }


# ================================
# PDF PROCESSOR AGENT - FIRST AGENT IN WORKFLOW
# ================================

class PDFProcessorAgent(BaseAgent):
    """
    FIRST AGENT IN THE LANGGRAPH WORKFLOW PIPELINE
    ==============================================
    
    This agent is the entry point of the SAP EWA analysis workflow.
    It takes uploaded PDF files and converts them into structured text data
    that can be processed by downstream agents.
    
    WORKFLOW POSITION: 1st Agent (Entry Point)
    INPUT: Raw PDF files uploaded via Streamlit interface
    OUTPUT: Structured text data with metadata for EmbeddingAgent
    
    KEY RESPONSIBILITIES:
    1. Validate uploaded PDF files (size, format, accessibility)
    2. Extract text content using multiple PDF processing libraries
    3. Clean and normalize extracted text (remove artifacts, fix encoding)
    4. Structure the text data with metadata for downstream processing
    5. Handle extraction failures gracefully with fallback methods
    
    CRITICAL INTEGRATION POINTS:
    - Receives files from Streamlit file uploader in app.py
    - Output format must match what EmbeddingAgent expects as input
    - Failure here stops the entire workflow, so robust error handling is essential
    - Text quality affects all downstream agents, so cleaning is critical
    
    TECHNICAL APPROACH:
    - Uses multiple PDF libraries (PyPDF2, pdfplumber, PyMuPDF) for reliability
    - Implements fallback extraction methods if primary method fails
    - Preserves document metadata for tracking and debugging
    - Cleans common PDF extraction artifacts and encoding issues
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PDF processor with file handling configuration.
        
        CONFIGURATION INTEGRATION:
        - max_file_size_mb: From config.py processing settings
        - clean_text: Whether to apply text cleaning (default True)
        - supported_encodings: Character encodings to try for text extraction
        
        Args:
            config: Configuration from config.py via get_agent_config()
        """
        super().__init__("PDFProcessor", config)
        
        # Extract PDF-specific configuration from global config
        self.max_file_size = config.get('max_file_size_mb', 50) * 1024 * 1024  # Convert MB to bytes
        self.supported_encodings = config.get('encodings', ['utf-8', 'latin-1', 'cp1252'])
        self.clean_text = config.get('clean_text', True)
    
    def process(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        MAIN PROCESSING METHOD - CALLED BY LANGGRAPH WORKFLOW
        ====================================================
        
        This is the primary method called by the workflow's _pdf_processing_node.
        It processes all uploaded PDF files and returns structured data for the next agent.
        
        WORKFLOW INTEGRATION CRITICAL:
        - Input: uploaded_files from Streamlit file uploader (passed via workflow state)
        - Output: Must return dict with "success" key for workflow error handling
        - If success=True: "processed_files" key contains data for EmbeddingAgent
        - If success=False: "error" key contains error message for workflow
        
        PROCESSING PIPELINE:
        1. Validate each file (size, type, accessibility)
        2. Extract text using multiple methods (fallback on failure)
        3. Clean and normalize extracted text
        4. Structure data with metadata
        5. Collect statistics and metrics
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Dict with structure:
            {
                "success": bool,                    # CRITICAL: Workflow checks this
                "processed_files": [               # INPUT FOR EMBEDDING AGENT
                    {
                        "filename": str,
                        "text": str,               # Cleaned text content
                        "size": int,               # File size in bytes
                        "character_count": int,    # Length of extracted text
                        "word_count": int,         # Number of words
                        "processing_timestamp": str
                    }, ...
                ],
                "failed_files": [...],             # Files that couldn't be processed
                "processing_time": float,          # Performance metrics
                "success_rate": float              # Success percentage
            }
        """
        self.start_timer()  # Start performance monitoring
        
        try:
            self.log_info(f"Starting PDF processing for {len(uploaded_files)} files")
            
            # Validate input - critical for workflow stability
            if not uploaded_files:
                return self.handle_error(ValueError("No PDF files provided"), "PDF Processing")
            
            # Initialize result containers
            processed_files = []  # Successfully processed files → EmbeddingAgent
            failed_files = []     # Failed files for error reporting
            total_size = 0       # Total bytes processed
            
            # Process each file individually to isolate failures
            # DESIGN PRINCIPLE: One bad file shouldn't break the entire batch
            for i, file in enumerate(uploaded_files):
                try:
                    self.log_info(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Step 1: Validate file before processing
                    validation_result = self._validate_file(file)
                    if not validation_result['valid']:
                        failed_files.append({
                            'filename': file.name,
                            'error': validation_result['error']
                        })
                        continue  # Skip this file, continue with others
                    
                    # Step 2: Extract text content using multiple methods
                    text_content = self._extract_text_from_pdf(file)
                    
                    # Step 3: Validate extraction success
                    if text_content and len(text_content.strip()) > 0:
                        # Step 4: Clean and normalize text if enabled
                        if self.clean_text:
                            text_content = self._clean_extracted_text(text_content)
                        
                        # Step 5: Structure data for downstream agents
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
                        # Text extraction failed
                        failed_files.append({
                            'filename': file.name,
                            'error': 'No text content extracted'
                        })
                        
                except Exception as e:
                    # Individual file processing failed - log and continue
                    self.log_error(f"Failed to process {file.name}: {str(e)}")
                    failed_files.append({
                        'filename': file.name,
                        'error': str(e)
                    })
            
            # Calculate metrics and finalize results
            processing_time = self.end_timer("pdf_processing")
            success_rate = len(processed_files) / len(uploaded_files) if uploaded_files else 0
            
            self.log_info(f"PDF processing completed: {len(processed_files)}/{len(uploaded_files)} files successful "
                         f"(success rate: {success_rate:.1%})")
            
            # Return structured results for workflow
            return {
                "success": len(processed_files) > 0,  # Success if at least one file processed
                "processed_files": processed_files,   # → INPUT FOR EMBEDDING AGENT
                "failed_files": failed_files,         # For error reporting
                "total_files": len(uploaded_files),
                "successful_files": len(processed_files),
                "failed_count": len(failed_files),
                "total_size": total_size,
                "processing_time": processing_time,
                "success_rate": success_rate
            }
            
        except Exception as e:
            # Agent-level failure - return standardized error
            return self.handle_error(e, "PDF Processing")
    
    def _validate_file(self, file) -> Dict[str, Any]:
        """
        Validate an uploaded file before processing.
        
        VALIDATION CHECKS:
        1. File size within limits (prevents memory issues)
        2. File type is PDF (prevents processing wrong file types)
        3. File is not empty (prevents wasted processing)
        4. File is accessible (prevents I/O errors)
        
        Args:
            file: Streamlit UploadedFile object
            
        Returns:
            Dict with 'valid' boolean and 'error' message if invalid
        """
        try:
            # Check file size against configured limit
            file_size = len(file.getvalue())
            if file_size > self.max_file_size:
                return {
                    'valid': False,
                    'error': f'File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit '
                            f'({self.max_file_size / 1024 / 1024:.1f}MB)'
                }
            
            # Check file extension
            if not file.name.lower().endswith('.pdf'):
                return {'valid': False, 'error': 'File must be a PDF document'}
            
            # Check for empty files
            if file_size == 0:
                return {'valid': False, 'error': 'File is empty'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'File validation error: {str(e)}'}
    
    def _extract_text_from_pdf(self, file) -> str:
        """
        Extract text content using multiple PDF libraries with fallback.
        
        MULTI-LIBRARY APPROACH FOR RELIABILITY:
        Different PDF libraries handle different PDF types better:
        1. PyPDF2: Most common, handles standard PDFs well
        2. pdfplumber: Better for complex layouts and tables
        3. PyMuPDF (fitz): Handles more PDF variations and formats
        
        FALLBACK STRATEGY:
        - Try each library in order until one succeeds
        - Return first successful extraction
        - If all fail, return error message but don't crash
        
        Args:
            file: PDF file to extract text from
            
        Returns:
            Extracted text content as string (or error message)
        """
        extraction_methods = [
            ('PyPDF2', self._extract_with_pypdf2),
            ('pdfplumber', self._extract_with_pdfplumber),
            ('PyMuPDF', self._extract_with_pymupdf)
        ]
        
        for method_name, extraction_func in extraction_methods:
            try:
                self.log_info(f"Attempting text extraction with {method_name}")
                text = extraction_func(file)
                
                if text and len(text.strip()) > 0:
                    self.log_info(f"✅ Successfully extracted text using {method_name}")
                    return text
                else:
                    self.log_warning(f"⚠️ {method_name} returned empty text")
                    
            except ImportError:
                self.log_warning(f"⚠️ {method_name} library not available")
                continue  # Try next method
            except Exception as e:
                self.log_warning(f"⚠️ {method_name} extraction failed: {str(e)}")
                continue  # Try next method
        
        # All methods failed - return error but don't crash workflow
        self.log_error(f"All text extraction methods failed for {file.name}")
        return f"[Error: Could not extract text from {file.name}]"
    
    def _extract_with_pypdf2(self, file) -> str:
        """
        Extract text using PyPDF2 library (most common PDF processing library).
        
        TECHNICAL APPROACH:
        - Uses PyPDF2.PdfReader to read PDF in memory
        - Extracts text from each page individually
        - Adds page markers for document structure preservation
        - Handles individual page failures gracefully
        """
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text + "\n"
            except Exception as e:
                self.log_warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                continue  # Skip failed page, continue with others
        
        return text_content
    
    def _extract_with_pdfplumber(self, file) -> str:
        """
        Extract text using pdfplumber library (better for complex layouts and tables).
        
        ADVANCED FEATURES:
        - Handles complex PDF layouts better than PyPDF2
        - Can extract tables and preserve table structure
        - Better at handling PDFs with unusual formatting
        """
        import pdfplumber
        import io
        
        text_content = ""
        
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract regular text
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                        
                    # SPECIAL FEATURE: Extract tables if present
                    # This is important for SAP EWA reports which often contain tables
                    tables = page.extract_tables()
                    if tables:
                        text_content += "\n[Tables found on this page]\n"
                        for table in tables:
                            for row in table:
                                if row:
                                    text_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        
                except Exception as e:
                    self.log_warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                    continue
        
        return text_content
    
    def _extract_with_pymupdf(self, file) -> str:
        """
        Extract text using PyMuPDF (fitz) library (handles more PDF variations).
        
        STRENGTHS:
        - Handles a wider variety of PDF formats
        - Better at dealing with corrupted or unusual PDFs
        - Fast performance for large documents
        """
        import fitz  # PyMuPDF
        import io
        
        text_content = ""
        pdf_document = fitz.open(stream=file.getvalue(), filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                if page_text:
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text + "\n"
                    
            except Exception as e:
                self.log_warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                continue
        
        pdf_document.close()
        return text_content
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text content.
        
        CRITICAL FOR DOWNSTREAM PROCESSING:
        - Removes PDF extraction artifacts that confuse embedding models
        - Normalizes whitespace for consistent chunk boundaries
        - Fixes common encoding issues from PDF processing
        - Improves quality of embeddings and search results
        
        CLEANING OPERATIONS:
        1. Normalize excessive whitespace
        2. Remove PDF control characters
        3. Fix common encoding artifacts
        4. Preserve document structure
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines → double newline
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs → single space
        
        # Remove common PDF extraction artifacts
        text = re.sub(r'\x0c', '\n', text)       # Form feed characters → newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Control characters
        
        # Fix common encoding issues from PDF processing
        text = text.replace('â€™', "'")           # Smart quote artifacts
        text = text.replace('â€œ', '"')           # Smart quote artifacts
        text = text.replace('â€\x9d', '"')        # Smart quote artifacts
        text = text.replace('â€"', '-')           # Em dash artifacts
        
        return text.strip()


# ================================
# EMBEDDING AGENT - SECOND AGENT IN WORKFLOW
# ================================

class EmbeddingAgent(BaseAgent):
    """
    SECOND AGENT IN THE LANGGRAPH WORKFLOW PIPELINE
    ===============================================
    
    This agent converts the text data from PDFProcessorAgent into vector embeddings
    that can be stored in ChromaDB and used for semantic similarity search.
    
    WORKFLOW POSITION: 2nd Agent
    INPUT: Processed text data from PDFProcessorAgent
    OUTPUT: Vector embeddings and text chunks for ChromaDB storage
    
    KEY RESPONSIBILITIES:
    1. Split large text documents into optimal-sized chunks
    2. Create vector embeddings using OpenAI or SentenceTransformers
    3. Preserve metadata and document relationships
    4. Handle embedding API rate limits and failures
    5. Prepare data structures for vector database storage
    
    CRITICAL INTEGRATION POINTS:
    - Input format must match PDFProcessorAgent output
    - Output chunks must be compatible with ChromaDB storage format
    - Chunk size and overlap settings affect search quality
    - Embedding model choice affects semantic search accuracy
    - API key configuration comes from config.py
    
    CHUNKING STRATEGY:
    - Uses overlapping text chunks to preserve context across boundaries
    - Splits on sentence boundaries when possible for better semantic coherence
    - Configurable chunk size and overlap for optimization
    - Preserves source document metadata for traceability
    
    EMBEDDING STRATEGY:
    - Primary: OpenAI text-embedding-ada-002 (high quality, requires API key)
    - Fallback: Mock embeddings for development/testing
    - Batch processing for efficiency and API rate limit management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding agent with chunking and embedding configuration.
        
        CONFIGURATION INTEGRATION:
        - chunk_size: From config.py processing settings (default 1000 chars)
        - chunk_overlap: Overlap between chunks (default 200 chars)
        - embedding_model: OpenAI model name (default text-embedding-ada-002)
        - openai_api_key: API key for OpenAI embeddings
        
        Args:
            config: Configuration from config.py via get_agent_config()
        """
        super().__init__("EmbeddingCreator", config)
        
        # Extract embedding-specific configuration
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.embedding_model = config.get('embedding_model', 'text-embedding-ada-002')
        self.batch_size = config.get('embedding_batch_size', 10)
        
        # Initialize OpenAI client for embeddings
        self.embedding_client = self._initialize_embedding_client()
    
    def _initialize_embedding_client(self):
        """
        Initialize the OpenAI embedding client with fallback handling.
        
        CRITICAL INTEGRATION POINT:
        - Uses openai_api_key from config.py
        - If no API key, falls back to mock embeddings for development
        - This affects the quality of semantic search in the workflow
        
        Returns:
            OpenAI client instance or None for mock mode
        """
        try:
            from openai import OpenAI
            api_key = self.config.get('openai_api_key')
            
            if api_key:
                client = OpenAI(api_key=api_key)
                self.log_info(f"OpenAI embedding client initialized with model: {self.embedding_model}")
                return client
            else:
                self.log_warning("No OpenAI API key provided, will use mock embeddings")
                return None
                
        except ImportError:
            self.log_warning("OpenAI library not available, will use mock embeddings")
            return None
        except Exception as e:
            self.log_error(f"Failed to initialize embedding client: {e}")
            return None
    
    def process(self, processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        MAIN PROCESSING METHOD - CALLED BY LANGGRAPH WORKFLOW
        ====================================================
        
        This method is called by the workflow's _embedding_creation_node.
        It takes the structured text data from PDFProcessorAgent and converts
        it into vector embeddings for ChromaDB storage.
        
        WORKFLOW INTEGRATION CRITICAL:
        - Input: processed_files from PDFProcessorAgent.process()
        - Output: Must return dict with "success" key for workflow error handling
        - If success=True: "chunks" and "embeddings" keys contain data for vector storage
        - If success=False: "error" key contains error message for workflow
        
        PROCESSING PIPELINE:
        1. Create text chunks from each processed file
        2. Generate vector embeddings for all chunks
        3. Validate chunk-embedding alignment
        4. Structure data for ChromaDB storage
        5. Return structured results with metrics
        
        Args:
            processed_files: List of file data from PDFProcessorAgent
            
        Returns:
            Dict with structure:
            {
                "success": bool,                    # CRITICAL: Workflow checks this
                "chunks": [                         # TEXT CHUNKS FOR CHROMADB
                    {
                        "text": str,               # Chunk text content
                        "page_content": str,       # LangChain compatibility
                        "chunk_id": int,           # Unique chunk identifier
                        "source": str,             # Source filename
                        "metadata": {...}          # Rich metadata
                    }, ...
                ],
                "embeddings": [[float, ...], ...], # VECTOR EMBEDDINGS FOR CHROMADB
                "chunk_count": int,                 # Number of chunks created
                "embedding_count": int,             # Number of embeddings created
                "processing_time": float            # Performance metrics
            }
        """
        self.start_timer()
        
        try:
            self.log_info(f"Creating embeddings for {len(processed_files)} processed files")
            
            # Validate input from PDFProcessorAgent
            if not processed_files:
                return self.handle_error(ValueError("No processed files provided"), "Embedding Creation")
            
            all_chunks = []      # All text chunks → ChromaDB documents
            all_embeddings = []  # All vector embeddings → ChromaDB embeddings
            
            # Process each file individually to isolate failures
            for file_data in processed_files:
                try:
                    self.log_info(f"Processing embeddings for: {file_data.get('filename', 'unknown')}")
                    
                    # Step 1: Create text chunks from file content
                    file_chunks = self._create_text_chunks(
                        text=file_data.get('text', ''),
                        filename=file_data.get('filename', 'unknown'),
                        metadata=file_data
                    )
                    
                    if not file_chunks:
                        self.log_warning(f"No chunks created for {file_data.get('filename')}")
                        continue  # Skip this file, continue with others
                    
                    # Step 2: Create vector embeddings for chunks
                    chunk_embeddings = self._create_embeddings_for_chunks(file_chunks)
                    
                    # Step 3: Collect results
                    all_chunks.extend(file_chunks)
                    all_embeddings.extend(chunk_embeddings)
                    
                    self.log_info(f"✅ Created {len(file_chunks)} chunks and {len(chunk_embeddings)} embeddings "
                                f"for {file_data.get('filename')}")
                    
                except Exception as e:
                    # Individual file processing failed - log and continue
                    self.log_error(f"Failed to process embeddings for {file_data.get('filename', 'unknown')}: {e}")
                    continue
            
            # Step 4: Validate chunk-embedding alignment (CRITICAL)
            if len(all_chunks) != len(all_embeddings):
                self.log_warning(f"Chunk count ({len(all_chunks)}) != embedding count ({len(all_embeddings)})")
                # Fix alignment by trimming to smaller count
                min_count = min(len(all_chunks), len(all_embeddings))
                all_chunks = all_chunks[:min_count]
                all_embeddings = all_embeddings[:min_count]
            
            processing_time = self.end_timer("embedding_creation")
            
            self.log_info(f"Embedding creation completed: {len(all_chunks)} chunks, "
                         f"{len(all_embeddings)} embeddings in {processing_time:.2f}s")
            
            # Return structured results for workflow
            return {
                "success": True,                     # CRITICAL: Workflow success indicator
                "chunks": all_chunks,                # → INPUT FOR VECTOR STORAGE
                "embeddings": all_embeddings,        # → INPUT FOR VECTOR STORAGE
                "chunk_count": len(all_chunks),
                "embedding_count": len(all_embeddings),
                "processing_time": processing_time,
                "average_chunk_size": sum(len(chunk.get('text', '')) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
            }
            
        except Exception as e:
            # Agent-level failure - return standardized error
            return self.handle_error(e, "Embedding Creation")
    
    def _create_text_chunks(self, text: str, filename: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks suitable for embedding"""
        if not text or len(text.strip()) < self.min_chunk_size:
            self.log_warning(f"Text too short for chunking: {len(text) if text else 0} chars")
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_data = self._create_chunk_data(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    filename=filename,
                    metadata=metadata
                )
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_data = self._create_chunk_data(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                filename=filename,
                metadata=metadata
            )
            chunks.append(chunk_data)
        
        self.log_info(f"Created {len(chunks)} chunks from {filename} "
                     f"(avg size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars)")
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunk boundaries"""
        import re
        
        sentences = re.split(r'([.!?]+\s+)', text)
        
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if sentence.strip():
                result.append(sentence.strip())
        
        return result
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last portion of text for chunk overlap"""
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        first_space = overlap_text.find(' ')
        
        if first_space > 0:
            return overlap_text[first_space:].strip()
        else:
            return overlap_text
    
    def _create_chunk_data(self, text: str, chunk_id: int, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk data structure with text and metadata"""
        return {
            'text': text,
            'page_content': text,
            'chunk_id': chunk_id,
            'source': filename,
            'character_count': len(text),
            'word_count': len(text.split()),
            'metadata': {
                'source': filename,
                'chunk_id': chunk_id,
                'character_count': len(text),
                'word_count': len(text.split()),
                'original_file_size': metadata.get('size', 0),
                'processing_timestamp': datetime.now().isoformat(),
                **metadata
            }
        }
    
    def _create_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create vector embeddings for a list of text chunks"""
        if not chunks:
            return []
        
        if not self.embedding_client:
            return self._create_mock_embeddings(len(chunks))
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            try:
                self.log_info(f"Creating embeddings for batch {i//self.batch_size + 1} "
                             f"({len(batch)} chunks)")
                
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                self.log_info(f"✅ Successfully created {len(batch_embeddings)} embeddings")
                
                if i + self.batch_size < len(chunks):
                    time.sleep(0.1)
                    
            except Exception as e:
                self.log_error(f"Failed to create embeddings for batch {i//self.batch_size + 1}: {e}")
                mock_embeddings = self._create_mock_embeddings(len(batch))
                all_embeddings.extend(mock_embeddings)
        
        return all_embeddings
    
    def _create_mock_embeddings(self, count: int, dimension: int = 1536) -> List[List[float]]:
        """Create mock embeddings for testing"""
        import random
        
        self.log_warning(f"Creating {count} mock embeddings (dimension: {dimension})")
        
        embeddings = []
        for i in range(count):
            vector = [random.gauss(0, 1) for _ in range(dimension)]
            magnitude = sum(x**2 for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x / magnitude for x in vector]
            embeddings.append(vector)
        
        return embeddings


# ================================
# SEARCH AGENT
# ================================

class SearchAgent(BaseAgent):
    """
    Agent responsible for performing similarity search on the vector store.
    
    Handles:
    - Vector similarity search using ChromaDB
    - Query processing and optimization
    - Result filtering and ranking
    - System ID detection from documents
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SearchAgent", config)
        
        self.vector_store = config.get('vector_store')
        self.embedding_agent = config.get('embedding_agent')
        self.top_k = config.get('top_k', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # System ID detection patterns
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',
            r'\bSID[:\s]+([A-Z0-9]{2,4})\b',
            r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\bFOR\s+([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\b([A-Z]{1,3}[0-9]{1,2})\b',
            r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b',
        ]
        
        self.false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL',
            'CPU', 'RAM', 'GB', 'MB', 'KB', 'HTTP', 'URL', 'API', 'GUI', 'UI', 'DB'
        }
        
        self.log_info(f"SearchAgent initialized with vector store: {type(self.vector_store).__name__ if self.vector_store else 'None'}")
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform similarity search on the vector store"""
        self.start_timer()
        
        try:
            self.log_info(f"Performing search for query: '{query}'")
            
            if not query or not query.strip():
                return self.handle_error(ValueError("Empty search query provided"), "Search")
            
            if not self.vector_store:
                self.log_warning("No vector store available, using mock search")
                search_results = self._perform_mock_search(query, filters)
            else:
                search_results = self._perform_vector_search(query, filters)
            
            validated_results = self._validate_search_results(search_results)
            
            processing_time = self.end_timer("search")
            
            self.log_info(f"Search completed: {len(validated_results)} results in {processing_time:.2f}s")
            
            return {
                "success": True,
                "query": query,
                "search_results": validated_results,
                "results_count": len(validated_results),
                "processing_time": processing_time,
                "filters_applied": filters or {},
                "search_metadata": {
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "vector_store_type": type(self.vector_store).__name__ if self.vector_store else "Mock"
                }
            }
            
        except Exception as e:
            return self.handle_error(e, "Search")
    
    def _perform_vector_search(self, query: str, filters: Dict[str, Any] = None) -> List[Any]:
        """Perform actual vector similarity search"""
        try:
            target_systems = filters.get('target_systems', []) if filters else []
            
            self.log_info(f"Vector search with target systems: {target_systems}")
            
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            
            search_results = []
            for doc, score in docs_with_scores:
                try:
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    source = metadata.get('source', 'unknown')
                    
                    system_id = self._extract_system_id(content, metadata)
                    
                    # Apply system filtering
                    if target_systems and system_id not in target_systems:
                        self.log_info(f"Filtering out result for system {system_id} (not in target list)")
                        continue
                    
                    # Apply similarity threshold
                    similarity_score = 1.0 - score if score <= 1.0 else score
                    if similarity_score < self.similarity_threshold:
                        self.log_info(f"Filtering out result with low similarity: {similarity_score:.3f}")
                        continue
                    
                    # Create SearchResult
                    try:
                        from models import SearchResult
                        result = SearchResult(
                            content=content,
                            source=source,
                            system_id=system_id,
                            confidence_score=float(similarity_score),
                            metadata=metadata
                        )
                    except ImportError:
                        # Fallback to dict
                        result = {
                            'content': content,
                            'source': source,
                            'system_id': system_id,
                            'confidence_score': float(similarity_score),
                            'metadata': metadata
                        }
                    
                    search_results.append(result)
                    
                except Exception as doc_error:
                    self.log_error(f"Error processing search result: {doc_error}")
                    continue
            
            self.log_info(f"Vector search returned {len(search_results)} results after filtering")
            return search_results
            
        except Exception as e:
            self.log_error(f"Vector search failed: {e}")
            return self._perform_mock_search(query, filters)
    
    def _perform_mock_search(self, query: str, filters: Dict[str, Any] = None) -> List[Any]:
        """Perform mock search for testing"""
        target_systems = filters.get('target_systems', ['UNKNOWN']) if filters else ['UNKNOWN']
        
        if not target_systems:
            target_systems = ['SYSTEM_01', 'SYSTEM_02']
        
        self.log_info(f"Mock search for target systems: {target_systems}")
        
        mock_results = []
        for i, system in enumerate(target_systems):
            mock_content = self._generate_mock_content(query, system, i)
            
            try:
                from models import SearchResult
                result = SearchResult(
                    content=mock_content,
                    source=f"document_{i+1}.pdf",
                    system_id=system,
                    confidence_score=0.9 - (i * 0.1),
                    metadata={
                        "chunk_id": i,
                        "page": i + 1,
                        "system_id": system,
                        "mock": True,
                        "search_query": query
                    }
                )
            except ImportError:
                result = {
                    'content': mock_content,
                    'source': f"document_{i+1}.pdf",
                    'system_id': system,
                    'confidence_score': 0.9 - (i * 0.1),
                    'metadata': {
                        "chunk_id": i,
                        "page": i + 1,
                        "system_id": system,
                        "mock": True,
                        "search_query": query
                    }
                }
            
            mock_results.append(result)
        
        return mock_results
    
    def _generate_mock_content(self, query: str, system_id: str, index: int) -> str:
        """Generate realistic mock content for search results"""
        templates = [
            f"System {system_id} analysis shows that {query.lower()} requires attention. "
            f"Performance metrics indicate potential optimization opportunities in database operations.",
            
            f"Early Watch Alert for {system_id}: Issues related to {query.lower()} have been detected. "
            f"Recommendations include memory optimization and query tuning.",
            
            f"SAP Basis team report for {system_id} indicates that {query.lower()} monitoring shows "
            f"elevated resource consumption requiring immediate investigation.",
            
            f"Technical analysis of {system_id} reveals {query.lower()} patterns that suggest "
            f"system configuration adjustments may be needed for optimal performance."
        ]
        
        template = templates[index % len(templates)]
        
        if 'performance' in query.lower():
            template += f" CPU utilization in {system_id} is at 85%. Memory usage shows spikes during batch processing."
        elif 'error' in query.lower() or 'critical' in query.lower():
            template += f" Error count in {system_id} has increased by 25% over the past week."
        elif 'recommendation' in query.lower():
            template += f" Recommended actions for {system_id} include index optimization and archiving old data."
        
        return template
    
    def _extract_system_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Extract system ID from document content or metadata"""
        try:
            # Check metadata first
            if 'system_id' in metadata and metadata['system_id']:
                return metadata['system_id'].upper()
            
            # Pattern matching on content
            if content:
                detected_id = self._detect_system_from_content(content)
                if detected_id != 'UNKNOWN':
                    return detected_id
            
            # Check source filename
            source = metadata.get('source', '')
            if source:
                filename_id = self._extract_system_from_filename(source)
                if filename_id != 'UNKNOWN':
                    return filename_id
            
            return 'UNKNOWN'
            
        except Exception as e:
            self.log_error(f"Error extracting system ID: {e}")
            return 'UNKNOWN'
    
    def _detect_system_from_content(self, content: str) -> str:
        """Detect system ID from document content using pattern matching"""
        if not content:
            return 'UNKNOWN'
        
        content_upper = content.upper()
        found_systems = set()
        
        for pattern in self.system_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                
                if (match and 
                    len(match) in [2, 3, 4] and 
                    match not in self.false_positives):
                    found_systems.add(match)
        
        if found_systems:
            result = sorted(list(found_systems))[0]
            return result
        
        return 'UNKNOWN'
    
    def _extract_system_from_filename(self, filename: str) -> str:
        """Extract system ID from filename if present"""
        if not filename:
            return 'UNKNOWN'
        
        filename_upper = filename.upper()
        
        patterns = [
            r'EWA[_\-]([A-Z0-9]{2,4})[_\-]',
            r'([A-Z0-9]{2,4})[_\-](?:EWA|REPORT|ANALYSIS)',
            r'^([A-Z0-9]{2,4})[_\-]',
            r'[_\-]([A-Z0-9]{2,4})[_\-]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_upper)
            if match:
                system_id = match.group(1)
                if system_id not in self.false_positives and len(system_id) in [2, 3, 4]:
                    return system_id
        
        return 'UNKNOWN'
    
    def _validate_search_results(self, results: List[Any]) -> List[Any]:
        """Validate and clean search results"""
        validated = []
        
        for result in results:
            try:
                # Handle both SearchResult objects and dicts
                if hasattr(result, 'content'):
                    content = result.content
                    source = result.source
                    system_id = result.system_id
                    confidence_score = result.confidence_score
                else:
                    content = result.get('content', '')
                    source = result.get('source', 'unknown')
                    system_id = result.get('system_id', 'UNKNOWN')
                    confidence_score = result.get('confidence_score', 0.0)
                
                # Basic validation
                if not content or len(content.strip()) < 10:
                    continue
                
                if not source:
                    if hasattr(result, 'source'):
                        result.source = 'unknown'
                    else:
                        result['source'] = 'unknown'
                
                if not system_id:
                    if hasattr(result, 'system_id'):
                        result.system_id = 'UNKNOWN'
                    else:
                        result['system_id'] = 'UNKNOWN'
                
                # Ensure confidence score is valid
                if not (0.0 <= confidence_score <= 1.0):
                    fixed_score = max(0.0, min(1.0, confidence_score))
                    if hasattr(result, 'confidence_score'):
                        result.confidence_score = fixed_score
                    else:
                        result['confidence_score'] = fixed_score
                
                validated.append(result)
                
            except Exception as e:
                self.log_error(f"Error validating search result: {e}")
                continue
        
        return validated


# ================================
# SUMMARY AGENT - FOURTH AGENT IN WORKFLOW
# ================================

class SummaryAgent(BaseAgent):
    """
    FOURTH AGENT IN THE LANGGRAPH WORKFLOW PIPELINE
    ===============================================
    
    This agent analyzes search results and generates intelligent summaries
    with critical findings, recommendations, and confidence assessments.
    
    WORKFLOW POSITION: 4th Agent
    INPUT: Search results from SearchAgent
    OUTPUT: Structured summary with findings and recommendations
    
    KEY RESPONSIBILITIES:
    1. Analyze search results to extract key insights
    2. Identify critical findings and alerts requiring attention
    3. Generate actionable recommendations
    4. Calculate confidence scores based on result quality
    5. Create executive summaries for decision-making
    
    CRITICAL INTEGRATION POINTS:
    - Input format must match SearchAgent output
    - Output structure used by SystemOutputAgent and EmailAgent
    - Summary quality affects final report value
    - Confidence scores guide user decision-making
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize summary agent with analysis configuration.
        
        Args:
            config: Configuration from config.py via get_agent_config()
        """
        super().__init__("SummaryAgent", config)
        
        # Summary configuration
        self.max_summary_length = config.get('max_summary_length', 500)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        
        # Keywords for identifying different types of content
        self.critical_keywords = [
            'critical', 'error', 'fail', 'down', 'alert', 'urgent', 'severe',
            'exception', 'abort', 'crash', 'timeout', 'unavailable'
        ]
        
        self.recommendation_keywords = [
            'recommend', 'should', 'improve', 'optimize', 'upgrade', 'configure',
            'adjust', 'tune', 'modify', 'consider', 'suggest', 'advise'
        ]
        
        self.performance_keywords = [
            'performance', 'slow', 'response time', 'throughput', 'latency',
            'cpu', 'memory', 'disk', 'bandwidth', 'capacity', 'utilization'
        ]
    
    def generate_summary(self, search_results: List[Any], query: str) -> Dict[str, Any]:
        """
        MAIN PROCESSING METHOD - CALLED BY LANGGRAPH WORKFLOW
        ====================================================
        
        This method is called by the workflow's _summary_node.
        It analyzes search results to create comprehensive summaries.
        
        WORKFLOW INTEGRATION CRITICAL:
        - Input: search_results from SearchAgent.search()
        - Output: Must return dict with "success" key for workflow error handling
        - If success=True: "summary" key contains structured analysis
        - If success=False: "error" key contains error message for workflow
        
        Args:
            search_results: List of search results from SearchAgent
            query: Original search query for context
            
        Returns:
            Dict with structure:
            {
                "success": bool,                    # CRITICAL: Workflow checks this
                "summary": {                        # STRUCTURED SUMMARY DATA
                    "summary": str,                # Executive summary text
                    "critical_findings": [str],    # Critical issues found
                    "recommendations": [str],      # Actionable recommendations
                    "performance_insights": [str], # Performance observations
                    "confidence_score": float,     # Confidence in analysis (0-1)
                    "query": str,                  # Original query
                    "results_analyzed": int,       # Number of results processed
                    "processing_time": float       # Time taken for analysis
                }
            }
        """
        self.start_timer()
        
        try:
            self.log_info(f"Generating summary for {len(search_results)} search results")
            
            if not search_results:
                return self._create_empty_summary(query)
            
            # Extract and analyze content from search results
            analyzed_content = self._analyze_search_results(search_results)
            
            # Generate different types of insights
            critical_findings = self._extract_critical_findings(analyzed_content)
            recommendations = self._extract_recommendations(analyzed_content)
            performance_insights = self._extract_performance_insights(analyzed_content)
            
            # Calculate confidence score based on result quality
            confidence_score = self._calculate_confidence_score(search_results, analyzed_content)
            
            # Generate main summary text
            summary_text = self._generate_summary_text(
                query=query,
                analyzed_content=analyzed_content,
                critical_count=len(critical_findings),
                recommendation_count=len(recommendations)
            )
            
            processing_time = self.end_timer("summary_generation")
            
            result = {
                "success": True,
                "summary": {
                    "summary": summary_text,
                    "critical_findings": critical_findings,
                    "recommendations": recommendations,
                    "performance_insights": performance_insights,
                    "confidence_score": confidence_score,
                    "query": query,
                    "results_analyzed": len(search_results),
                    "processing_time": processing_time
                }
            }
            
            self.log_info(f"Summary generated: {len(critical_findings)} critical findings, "
                         f"{len(recommendations)} recommendations (confidence: {confidence_score:.1%})")
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Summary Generation")
    
    def _create_empty_summary(self, query: str) -> Dict[str, Any]:
        """Create an empty summary when no search results are available"""
        return {
            "success": True,
            "summary": {
                "summary": f"No relevant information found for query: '{query}'. Please try refining your search terms or check if documents are properly processed.",
                "critical_findings": [],
                "recommendations": ["Upload and process relevant SAP documents", "Try broader search terms", "Verify system IDs are correct"],
                "performance_insights": [],
                "confidence_score": 0.0,
                "query": query,
                "results_analyzed": 0,
                "processing_time": 0.0
            }
        }
    
    def _analyze_search_results(self, search_results: List[Any]) -> Dict[str, Any]:
        """Analyze search results to extract structured information"""
        analyzed = {
            'all_content': [],
            'high_confidence_content': [],
            'system_content': {},
            'source_content': {},
            'total_length': 0,
            'avg_confidence': 0.0,
            'systems_found': set(),
            'sources_found': set()
        }
        
        total_confidence = 0.0
        valid_results = 0
        
        for result_item in search_results:
            try:
                # Extract content and metadata from different result formats
                content, confidence, system_id, source = self._extract_result_data(result_item)
                
                if not content:
                    continue
                
                # Store content in various categorizations
                analyzed['all_content'].append(content)
                analyzed['total_length'] += len(content)
                analyzed['systems_found'].add(system_id)
                analyzed['sources_found'].add(source)
                
                # High confidence content (for more reliable insights)
                if confidence >= 0.7:
                    analyzed['high_confidence_content'].append(content)
                
                # Group by system
                if system_id not in analyzed['system_content']:
                    analyzed['system_content'][system_id] = []
                analyzed['system_content'][system_id].append(content)
                
                # Group by source
                if source not in analyzed['source_content']:
                    analyzed['source_content'][source] = []
                analyzed['source_content'][source].append(content)
                
                total_confidence += confidence
                valid_results += 1
                
            except Exception as e:
                self.log_warning(f"Error analyzing result: {e}")
                continue
        
        # Calculate average confidence
        if valid_results > 0:
            analyzed['avg_confidence'] = total_confidence / valid_results
        
        analyzed['systems_found'] = list(analyzed['systems_found'])
        analyzed['sources_found'] = list(analyzed['sources_found'])
        
        return analyzed
    
    def _extract_result_data(self, result_item: Any) -> tuple:
        """Extract content, confidence, system_id, and source from result item"""
        try:
            # Handle tuple format (document, score)
            if isinstance(result_item, tuple) and len(result_item) >= 2:
                doc, score = result_item[0], result_item[1]
                
                # Extract content
                content = getattr(doc, 'page_content', str(doc))
                confidence = 1.0 - score if score <= 1.0 else score  # Convert distance to similarity
                
                # Extract metadata
                metadata = getattr(doc, 'metadata', {})
                system_id = metadata.get('system_id', 'UNKNOWN')
                source = metadata.get('source', 'unknown')
                
                return content, confidence, system_id, source
            
            # Handle SearchResult object format
            elif hasattr(result_item, 'content') and hasattr(result_item, 'confidence_score'):
                return (
                    result_item.content,
                    result_item.confidence_score,
                    getattr(result_item, 'system_id', 'UNKNOWN'),
                    getattr(result_item, 'source', 'unknown')
                )
            
            # Handle dictionary format
            elif isinstance(result_item, dict):
                return (
                    result_item.get('content', ''),
                    result_item.get('confidence_score', 0.5),
                    result_item.get('system_id', 'UNKNOWN'),
                    result_item.get('source', 'unknown')
                )
            
            # Fallback
            else:
                return str(result_item), 0.5, 'UNKNOWN', 'unknown'
                
        except Exception as e:
            self.log_warning(f"Error extracting result data: {e}")
            return '', 0.0, 'UNKNOWN', 'unknown'
    
    def _extract_critical_findings(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract critical findings from analyzed content"""
        critical_findings = []
        
        # Use high confidence content for more reliable findings
        content_to_analyze = analyzed_content.get('high_confidence_content', [])
        if not content_to_analyze:
            content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for critical keywords
                    has_critical = any(keyword in sentence_lower for keyword in self.critical_keywords)
                    
                    if has_critical and len(sentence.strip()) > 15:
                        finding = sentence.strip()
                        if finding and finding not in critical_findings:
                            critical_findings.append(finding)
                        
                        # Limit to prevent overwhelming output
                        if len(critical_findings) >= 10:
                            break
                
                if len(critical_findings) >= 10:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting critical findings: {e}")
                continue
        
        return critical_findings[:5]  # Return top 5 critical findings
    
    def _extract_recommendations(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract recommendations from analyzed content"""
        recommendations = []
        
        content_to_analyze = analyzed_content.get('high_confidence_content', [])
        if not content_to_analyze:
            content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for recommendation keywords
                    has_recommendation = any(keyword in sentence_lower for keyword in self.recommendation_keywords)
                    
                    if has_recommendation and len(sentence.strip()) > 15:
                        rec = sentence.strip()
                        if rec and rec not in recommendations:
                            recommendations.append(rec)
                        
                        # Limit to prevent overwhelming output
                        if len(recommendations) >= 10:
                            break
                
                if len(recommendations) >= 10:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting recommendations: {e}")
                continue
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_performance_insights(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract performance insights from analyzed content"""
        insights = []
        
        content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for performance keywords
                    has_performance = any(keyword in sentence_lower for keyword in self.performance_keywords)
                    
                    if has_performance and len(sentence.strip()) > 15:
                        insight = sentence.strip()
                        if insight and insight not in insights:
                            insights.append(insight)
                        
                        # Limit output
                        if len(insights) >= 5:
                            break
                
                if len(insights) >= 5:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting performance insights: {e}")
                continue
        
        return insights
    
    def _calculate_confidence_score(self, search_results: List[Any], analyzed_content: Dict[str, Any]) -> float:
        """Calculate confidence score based on result quality"""
        try:
            if not search_results:
                return 0.0
            
            # Base confidence from average result confidence
            avg_confidence = analyzed_content.get('avg_confidence', 0.5)
            
            # Boost confidence based on content richness
            total_length = analyzed_content.get('total_length', 0)
            length_factor = min(1.0, total_length / 5000)  # Normalize to 5000 chars
            
            # Boost confidence based on result count
            result_count_factor = min(1.0, len(search_results) / 10)  # Normalize to 10 results
            
            # Boost confidence based on system diversity
            systems_found = len(analyzed_content.get('systems_found', []))
            system_factor = min(1.0, systems_found / 3)  # Normalize to 3 systems
            
            # Calculate weighted confidence
            confidence = (
                avg_confidence * 0.4 +
                length_factor * 0.3 +
                result_count_factor * 0.2 +
                system_factor * 0.1
            )
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.log_warning(f"Error calculating confidence score: {e}")
            return 0.5  # Default moderate confidence
    
    def _generate_summary_text(self, query: str, analyzed_content: Dict[str, Any], 
                             critical_count: int, recommendation_count: int) -> str:
        """Generate the main summary text"""
        try:
            systems_found = analyzed_content.get('systems_found', [])
            avg_confidence = analyzed_content.get('avg_confidence', 0.0)
            total_content = len(analyzed_content.get('all_content', []))
            
            # Create summary based on findings
            if critical_count > 0:
                urgency_level = "CRITICAL ATTENTION REQUIRED" if critical_count >= 3 else "ATTENTION NEEDED"
                summary = f"Analysis of '{query}' reveals {urgency_level}. "
                summary += f"Found {critical_count} critical issues across {len(systems_found)} systems "
                summary += f"requiring immediate investigation. "
            else:
                summary = f"Analysis of '{query}' shows no critical issues. "
                summary += f"Reviewed {total_content} documents across {len(systems_found)} systems. "
            
            if recommendation_count > 0:
                summary += f"Identified {recommendation_count} optimization opportunities. "
            
            # Add system-specific information
            if systems_found:
                if len(systems_found) == 1:
                    summary += f"Analysis focused on system {systems_found[0]}. "
                else:
                    summary += f"Systems analyzed: {', '.join(systems_found)}. "
            
            # Add confidence information
            confidence_pct = avg_confidence * 100
            if confidence_pct >= 80:
                summary += "High confidence in analysis results."
            elif confidence_pct >= 60:
                summary += "Moderate confidence in analysis results."
            else:
                summary += "Low confidence - consider reviewing more documents."
            
            return summary
            
        except Exception as e:
            self.log_warning(f"Error generating summary text: {e}")
            return f"Analysis completed for query '{query}' with {critical_count} critical findings and {recommendation_count} recommendations."


# ================================
# EMAIL AGENT - GMAIL/OUTLOOK SUPPORT
# ================================

class EmailAgent(BaseAgent):
    """
    OPTIONAL AGENT FOR EMAIL NOTIFICATIONS
    =====================================
    
    This agent handles email notifications with support for both Gmail and Outlook.
    It's called conditionally by the workflow based on email configuration.
    
    WORKFLOW POSITION: 6th Agent (Optional)
    INPUT: Analysis results from previous agents
    OUTPUT: Email delivery status
    
    KEY RESPONSIBILITIES:
    1. Format analysis results into professional email content
    2. Support both Gmail and Outlook SMTP servers
    3. Handle authentication (App passwords for Gmail, regular passwords for Outlook)
    4. Retry logic for reliable delivery
    5. Error handling that doesn't break the workflow
    
    CRITICAL INTEGRATION POINTS:
    - Email configuration comes from config.py (provider, credentials)
    - Input data from SummaryAgent and SystemOutputAgent
    - Workflow continues even if email fails (non-blocking)
    - SMTP settings auto-configured based on provider choice
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email agent with provider-specific configuration.
        
        CONFIGURATION INTEGRATION:
        - email_provider: "gmail" or "outlook" from config.py
        - gmail_email, gmail_app_password: Gmail credentials
        - outlook_email, outlook_password: Outlook credentials
        - smtp_timeout: Connection timeout settings
        
        Args:
            config: Configuration from config.py via get_agent_config()
        """
        super().__init__("EmailAgent", config)
        
        # Detect email provider from configuration
        self.email_provider = config.get('email_provider', 'gmail').lower()
        
        # Provider-specific configuration
        if self.email_provider == 'gmail':
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.email_address = config.get('gmail_email')
            self.email_password = config.get('gmail_app_password')
        elif self.email_provider == 'outlook':
            self.smtp_server = 'smtp-mail.outlook.com'
            self.smtp_port = 587
            self.email_address = config.get('outlook_email')
            self.email_password = config.get('outlook_password')
        else:
            # Fallback to Gmail configuration
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.email_address = config.get('gmail_email')
            self.email_password = config.get('gmail_app_password')
        
        self.use_tls = True  # Both Gmail and Outlook use TLS
        self.max_retries = config.get('email_retries', 3)
        self.retry_delay = config.get('email_retry_delay', 5.0)
        self.timeout = config.get('email_timeout', 30)
        
        # Validate configuration
        self._validate_email_config()
    
    def _validate_email_config(self):
        """Validate email configuration and log status"""
        if not self.email_address or not self.email_password:
            self.log_warning(f"{self.email_provider.title()} email credentials not configured - email functionality disabled")
            return False
        
        # Basic email format validation
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}, self.email_address):
            self.log_error(f"Invalid email format: {self.email_address}")
            return False
        
        self.log_info(f"{self.email_provider.title()} email configuration validated for: {self.email_address}")
        return True
    
    def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        MAIN PROCESSING METHOD - CALLED BY LANGGRAPH WORKFLOW
        ====================================================
        
        This method is called by the workflow's _email_node.
        It formats analysis results and sends email notifications.
        
        WORKFLOW INTEGRATION CRITICAL:
        - Input: email_data with analysis results from previous agents
        - Output: Must return dict with "success" key for workflow
        - If email fails, workflow continues (non-blocking failure)
        - Supports both Gmail and Outlook based on configuration
        
        Args:
            email_data: Dict containing:
                - recipients: List of email addresses
                - summary: Analysis summary from SummaryAgent
                - query: Original search query
                - system_summaries: System analysis from SystemOutputAgent
                
        Returns:
            Dict with structure:
            {
                "success": bool,              # Email delivery success
                "recipients_count": int,      # Number of recipients
                "message": str,               # Status message
                "processing_time": float      # Time taken
            }
        """
        self.start_timer()
        
        try:
            self.log_info("Starting email send process")
            
            # Validate email data
            validation_result = self._validate_email_data(email_data)
            if not validation_result['valid']:
                return self.handle_error(ValueError(validation_result['error']), "Email Validation")
            
            recipients = email_data.get('recipients', [])
            if not recipients:
                return self.handle_error(ValueError("No recipients specified"), "Email Send")
            
            # Format email content
            email_content = self._format_email_content(email_data)
            
            # Send email with retry logic
            send_result = self._send_email_with_retry(recipients, email_content)
            
            processing_time = self.end_timer("email_send")
            
            if send_result['success']:
                self.log_info(f"Email sent successfully to {len(recipients)} recipients")
                return {
                    "success": True,
                    "recipients_count": len(recipients),
                    "message": f"Email sent successfully to {len(recipients)} recipients",
                    "processing_time": processing_time,
                    "email_details": {
                        "subject": email_content['subject'],
                        "recipients": recipients,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return self.handle_error(Exception(send_result['error']), "Email Send")
            
        except Exception as e:
            return self.handle_error(e, "Email Send")
    
    def _validate_email_data(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate email data structure and content"""
        try:
            # Check required fields
            if 'recipients' not in email_data:
                return {'valid': False, 'error': 'Missing required field: recipients'}
            
            # Validate recipients
            recipients = email_data['recipients']
            if not isinstance(recipients, list):
                return {'valid': False, 'error': 'Recipients must be a list'}
            
            if not recipients:
                return {'valid': False, 'error': 'Recipients list is empty'}
            
            # Validate email addresses
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
            
            for recipient in recipients:
                email_addr = recipient if isinstance(recipient, str) else recipient.get('email', '')
                if not re.match(email_pattern, email_addr):
                    return {'valid': False, 'error': f'Invalid email address: {email_addr}'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _format_email_content(self, email_data: Dict[str, Any]) -> Dict[str, str]:
        """Format email content including subject and body"""
        try:
            # Extract data for formatting
            summary = email_data.get('summary', {})
            query = email_data.get('query', 'SAP Analysis')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format subject
            critical_count = len(summary.get('critical_findings', []))
            if critical_count > 0:
                urgency = "CRITICAL" if critical_count >= 3 else "ALERT"
                subject = f"[{urgency}] SAP EWA Analysis - {query}"
            else:
                subject = f"SAP EWA Analysis Results - {query}"
            
            # Format body
            body_parts = [
                "SAP Early Watch Analyzer - Analysis Results",
                "=" * 50,
                "",
                f"Query: {query}",
                f"Analysis Time: {timestamp}",
                f"Confidence Score: {summary.get('confidence_score', 0) * 100:.1f}%",
                "",
                "EXECUTIVE SUMMARY:",
                summary.get('summary', 'Analysis completed successfully'),
                ""
            ]
            
            # Add critical findings
            critical_findings = summary.get('critical_findings', [])
            body_parts.extend([
                f"CRITICAL FINDINGS ({len(critical_findings)}):",
                "-" * 30
            ])
            
            if critical_findings:
                for i, finding in enumerate(critical_findings, 1):
                    body_parts.append(f"{i}. {finding}")
            else:
                body_parts.append("✅ No critical issues found")
            
            body_parts.append("")
            
            # Add recommendations
            recommendations = summary.get('recommendations', [])
            body_parts.extend([
                f"RECOMMENDATIONS ({len(recommendations)}):",
                "-" * 30
            ])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    body_parts.append(f"{i}. {rec}")
            else:
                body_parts.append("ℹ️ No specific recommendations at this time")
            
            body_parts.extend([
                "",
                "PERFORMANCE INSIGHTS:",
                "-" * 30
            ])
            
            # Add performance insights if available
            performance_insights = summary.get('performance_insights', [])
            if performance_insights:
                for insight in performance_insights:
                    body_parts.append(f"• {insight}")
            else:
                body_parts.append("• No specific performance insights available")
            
            body_parts.extend([
                "",
                "---",
                "Generated by SAP EWA Analyzer",
                f"Report generated at: {timestamp}",
                "",
                "This is an automated analysis report. For questions or concerns,",
                "please contact your SAP BASIS team or system administrator."
            ])
            
            return {
                'subject': subject,
                'body': '\n'.join(body_parts)
            }
            
        except Exception as e:
            self.log_error(f"Error formatting email content: {e}")
            # Fallback simple format
            return {
                'subject': f"SAP Analysis Results - {email_data.get('query', 'Report')}",
                'body': f"SAP analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nPlease check the system for detailed results."
            }
    
    def _send_email_with_retry(self, recipients: List[str], email_content: Dict[str, str]) -> Dict[str, Any]:
        """Send email with retry logic for improved reliability"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.log_info(f"Email send attempt {attempt + 1}/{self.max_retries}")
                
                result = self._send_email_smtp(recipients, email_content)
                
                if result:
                    return {'success': True}
                else:
                    raise Exception("SMTP send returned False")
                    
            except Exception as e:
                last_error = e
                self.log_warning(f"Email send attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    self.log_info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        return {
            'success': False,
            'error': f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        }
    
    def _send_email_smtp(self, recipients: List[str], email_content: Dict[str, str]) -> bool:
        """Send email using SMTP with provider-specific settings"""
        try:
            # Create email message
            msg = EmailMessage()
            msg['From'] = self.email_address
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = email_content['subject']
            msg.set_content(email_content['body'])
            
            # Create secure SSL context
            context = ssl.create_default_context()
            
            # Connect and send based on provider
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.log_error(f"SMTP send error: {str(e)}")
            return False


# ================================
# SYSTEM OUTPUT AGENT - FIFTH AGENT IN WORKFLOW
# ================================

class SystemOutputAgent(BaseAgent):
    """
    FIFTH AGENT IN THE LANGGRAPH WORKFLOW PIPELINE
    ==============================================
    
    This agent generates system-specific analysis outputs and health assessments.
    It analyzes search results to create detailed reports for each SAP system found.
    
    WORKFLOW POSITION: 5th Agent
    INPUT: Search results from SearchAgent
    OUTPUT: System-specific summaries and health assessments
    
    KEY RESPONSIBILITIES:
    1. Extract unique system IDs from search results
    2. Create system-specific health assessments (HEALTHY/WARNING/CRITICAL)
    3. Generate per-system recommendations and alerts
    4. Calculate system health scores based on findings
    5. Format system reports for stakeholders
    
    CRITICAL INTEGRATION POINTS:
    - Input format must match SearchAgent output
    - Output used by EmailAgent for system-specific notifications
    - Health assessments guide operational decisions
    - System ID detection affects analysis accuracy
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system output agent"""
        super().__init__("SystemOutputAgent", config)
        
        # System analysis configuration
        self.health_thresholds = {
            'critical_alert_threshold': 3,  # 3+ alerts = CRITICAL
            'warning_threshold': 1,         # 1+ alerts = WARNING
            'healthy_threshold': 0          # 0 alerts = HEALTHY
        }
        
        # System ID patterns (consistent with SearchAgent)
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',
            r'\bSID[:\s]+([A-Z0-9]{2,4})\b',
            r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\b([A-Z]{1,3}[0-9]{1,2})\b',
            r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b'
        ]
        
        self.false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL'
        }
    
    def generate_system_outputs(self, search_results: List[Any]) -> Dict[str, Any]:
        """
        MAIN PROCESSING METHOD - CALLED BY LANGGRAPH WORKFLOW
        ====================================================
        
        This method is called by the workflow's _system_output_node.
        It generates system-specific analysis and health assessments.
        
        WORKFLOW INTEGRATION CRITICAL:
        - Input: search_results from SearchAgent.search()
        - Output: Must return dict with "success" key for workflow error handling
        - If success=True: "system_summaries" key contains system analysis
        - If success=False: "error" key contains error message for workflow
        
        Args:
            search_results: List of search results to analyze
            
        Returns:
            Dict with structure:
            {
                "success": bool,                    # CRITICAL: Workflow checks this
                "system_summaries": {              # SYSTEM-SPECIFIC ANALYSIS
                    "SYSTEM_ID": {
                        "system_id": str,          # System identifier
                        "overall_health": str,     # HEALTHY/WARNING/CRITICAL
                        "critical_alerts": [str],  # Critical issues for this system
                        "recommendations": [str],  # System-specific recommendations
                        "key_metrics": {...},      # Performance metrics
                        "last_analyzed": str       # Analysis timestamp
                    }, ...
                },
                "systems_analyzed": int,           # Number of systems found
                "processing_time": float           # Performance metrics
            }
        """
        self.start_timer()
        
        try:
            self.log_info(f"Generating system outputs for {len(search_results)} search results")
            
            if not search_results:
                return self._create_empty_system_output()
            
            # Extract unique system IDs from search results
            system_ids = self.extract_system_ids(search_results)
            self.log_info(f"Found {len(system_ids)} unique systems: {system_ids}")
            
            # Extract documents for analysis
            documents = self._extract_documents_from_results(search_results)
            
            # Generate summary for each system
            system_summaries = {}
            for system_id in system_ids:
                try:
                    summary = self.extract_system_summary(documents, system_id)
                    system_summaries[system_id] = summary
                    
                    # Get health status for logging
                    health = getattr(summary, 'overall_health', 'UNKNOWN')
                    if hasattr(health, 'value'):
                        health = health.value
                    
                    self.log_info(f"Generated summary for {system_id}: {health}")
                except Exception as e:
                    self.log_error(f"Failed to generate summary for {system_id}: {e}")
                    # Create error summary
                    system_summaries[system_id] = self._create_error_system_summary(system_id, str(e))
            
            processing_time = self.end_timer("system_output_generation")
            
            result = {
                "success": True,
                "system_summaries": system_summaries,
                "systems_analyzed": len(system_ids),
                "total_systems_found": len(system_ids),
                "processing_time": processing_time,
                "analysis_metadata": {
                    "documents_analyzed": len(documents),
                    "search_results_processed": len(search_results),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.log_info(f"System analysis completed: {len(system_ids)} systems analyzed")
            return result
            
        except Exception as e:
            return self.handle_error(e, "System Output Generation")
    
    def extract_system_ids(self, search_results: List[Any]) -> List[str]:
        """Extract unique system IDs from search results using multiple strategies"""
        system_ids = set()
        
        for result_item in search_results:
            try:
                # Extract document and metadata
                doc, metadata = self._extract_doc_and_metadata(result_item)
                
                # Strategy 1: Check metadata for explicit system_id
                if isinstance(metadata, dict) and 'system_id' in metadata:
                    system_id = metadata['system_id']
                    if system_id and system_id.upper() not in self.false_positives:
                        system_ids.add(system_id.upper())
                        continue
                
                # Strategy 2: Extract from document content
                if doc:
                    content = self._extract_content_from_doc(doc)
                    if content:
                        detected_id = self._detect_system_from_content(content)
                        if detected_id != 'UNKNOWN':
                            system_ids.add(detected_id)
                            continue
                
                # Strategy 3: Extract from source filename if available
                source = metadata.get('source', '') if isinstance(metadata, dict) else ''
                if source:
                    filename_id = self._extract_system_from_filename(source)
                    if filename_id != 'UNKNOWN':
                        system_ids.add(filename_id)
                        
            except Exception as e:
                self.log_warning(f"Error extracting system ID from result: {e}")
                continue
        
        # Filter out common false positives and ensure valid system IDs
        filtered_systems = []
        for system_id in system_ids:
            if (system_id not in self.false_positives and 
                len(system_id) in [2, 3, 4] and 
                system_id != 'UNKNOWN'):
                filtered_systems.append(system_id)
        
        # If no valid systems found, return default
        if not filtered_systems:
            filtered_systems = ['SYSTEM_01']
        
        return sorted(filtered_systems)  # Sort for consistent ordering
    
    def extract_system_summary(self, documents: List[Any], system_id: str) -> Any:
        """Extract comprehensive summary for a specific system"""
        try:
            self.log_info(f"Creating system summary for {system_id}")
            
            # Initialize summary components
            critical_alerts = []
            recommendations = []
            key_metrics = {}
            
            # Analyze each document for system-specific content
            system_content = []
            for doc in documents:
                content = self._extract_content_from_doc(doc)
                if content and self._is_content_relevant_to_system(content, system_id):
                    system_content.append(content)
            
            self.log_info(f"Found {len(system_content)} relevant documents for {system_id}")
            
            # Extract insights from relevant content
            for content in system_content:
                # Extract critical alerts
                alerts = self._extract_critical_alerts_for_system(content, system_id)
                critical_alerts.extend(alerts)
                
                # Extract recommendations
                recs = self._extract_recommendations_for_system(content, system_id)
                recommendations.extend(recs)
                
                # Extract metrics
                metrics = self._extract_metrics_from_content(content, system_id)
                key_metrics.update(metrics)
            
            # Remove duplicates
            critical_alerts = list(set(critical_alerts))
            recommendations = list(set(recommendations))
            
            # Determine overall health based on findings
            overall_health = self._calculate_system_health(critical_alerts, recommendations, key_metrics)
            
            # Create SystemSummary object or dict
            try:
                from models import SystemSummary
                summary = SystemSummary(
                    system_id=system_id,
                    overall_health=overall_health,
                    critical_alerts=critical_alerts[:5],  # Limit to top 5
                    recommendations=recommendations[:5],   # Limit to top 5
                    key_metrics=key_metrics,
                    last_analyzed=datetime.now().isoformat()
                )
            except ImportError:
                # Fallback to dict
                summary = {
                    'system_id': system_id,
                    'overall_health': overall_health,
                    'critical_alerts': critical_alerts[:5],
                    'recommendations': recommendations[:5],
                    'key_metrics': key_metrics,
                    'last_analyzed': datetime.now().isoformat()
                }
            
            self.log_info(f"System summary for {system_id}: {overall_health}, "
                         f"{len(critical_alerts)} alerts, {len(recommendations)} recommendations")
            
            return summary
            
        except Exception as e:
            self.log_error(f"Error creating system summary for {system_id}: {e}")
            return self._create_error_system_summary(system_id, str(e))
    
    def _extract_doc_and_metadata(self, result_item: Any) -> tuple:
        """Extract document and metadata from different result formats"""
        try:
            if isinstance(result_item, tuple) and len(result_item) >= 2:
                doc, score = result_item[0], result_item[1]
                metadata = getattr(doc, 'metadata', {})
                return doc, metadata
                
            elif hasattr(result_item, 'content') and hasattr(result_item, 'metadata'):
                # SearchResult object
                return result_item, result_item.metadata
                
            else:
                # Fallback
                return result_item, {}
                
        except Exception as e:
            self.log_warning(f"Error extracting doc and metadata: {e}")
            return result_item, {}
    
    def _extract_content_from_doc(self, doc: Any) -> str:
        """Extract text content from document object"""
        try:
            if hasattr(doc, 'page_content'):
                return str(doc.page_content)
            elif hasattr(doc, 'content'):
                return str(doc.content)
            elif isinstance(doc, dict):
                return doc.get('page_content', doc.get('content', str(doc)))
            else:
                return str(doc)
        except Exception:
            return ""
    
    def _detect_system_from_content(self, content: str) -> str:
        """Detect system ID from content using pattern matching"""
        if not content:
            return 'UNKNOWN'
        
        content_upper = content.upper()
        found_systems = set()
        
        for pattern in self.system_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                
                if (match and 
                    len(match) in [2, 3, 4] and 
                    match not in self.false_positives):
                    found_systems.add(match)
        
        return sorted(list(found_systems))[0] if found_systems else 'UNKNOWN'
    
    def _extract_system_from_filename(self, filename: str) -> str:
        """Extract system ID from filename patterns"""
        if not filename:
            return 'UNKNOWN'
        
        filename_upper = filename.upper()
        patterns = [
            r'EWA[_\-]([A-Z0-9]{2,4})[_\-]',
            r'([A-Z0-9]{2,4})[_\-](?:EWA|REPORT|ANALYSIS)',
            r'^([A-Z0-9]{2,4})[_\-]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_upper)
            if match:
                system_id = match.group(1)
                if system_id not in self.false_positives and len(system_id) in [2, 3, 4]:
                    return system_id
        
        return 'UNKNOWN'
    
    def _extract_documents_from_results(self, search_results: List[Any]) -> List[Any]:
        """Extract document objects from search results"""
        documents = []
        
        for result_item in search_results:
            try:
                # Extract document and metadata
                doc, metadata = self._extract_doc_and_metadata(result_item)
                
                if doc:
                    documents.append(doc)
            except Exception as e:
                self.log_warning(f"Error extracting document: {e}")
                continue
        
        return documents
    
    def _create_empty_system_output(self) -> Dict[str, Any]:
        """Create empty system output when no results available"""
        return {
            "success": True,
            "system_summaries": {},
            "systems_analyzed": 0,
            "total_systems_found": 0,
            "processing_time": 0.0,
            "analysis_metadata": {
                "documents_analyzed": 0,
                "search_results_processed": 0,
                "timestamp": datetime.now().isoformat(),
                "message": "No search results to analyze"
            }
        }
    
    def _create_error_system_summary(self, system_id: str, error: str) -> Dict[str, Any]:
        """Create error summary for a system"""
        return {
            'system_id': system_id,
            'overall_health': 'UNKNOWN',
            'critical_alerts': [f"Error analyzing system: {error}"],
            'recommendations': ["Review system configuration", "Check document processing"],
            'key_metrics': {},
            'last_analyzed': datetime.now().isoformat(),
            'error': error
        }
    
    def _is_content_relevant_to_system(self, content: str, system_id: str) -> bool:
        """Check if content is relevant to the specific system"""
        if not content or not system_id:
            return False
        
        content_upper = content.upper()
        system_upper = system_id.upper()
        
        # Check for direct system ID mentions
        if system_upper in content_upper:
            return True
        
        # Check for system patterns
        for pattern in self.system_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                if match == system_upper:
                    return True
        
        return False
    
    def _extract_critical_alerts_for_system(self, content: str, system_id: str) -> List[str]:
        """Extract critical alerts specific to a system"""
        alerts = []
        
        critical_keywords = ['critical', 'error', 'fail', 'alert', 'urgent', 'severe']
        
        try:
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                system_lower = system_id.lower()
                
                # Check if sentence mentions system and has critical keywords
                if (system_lower in sentence_lower and 
                    any(keyword in sentence_lower for keyword in critical_keywords)):
                    
                    alert = sentence.strip()
                    if alert and len(alert) > 20:
                        alerts.append(alert)
                        
                        if len(alerts) >= 3:  # Limit alerts per system
                            break
        except Exception as e:
            self.log_warning(f"Error extracting alerts for {system_id}: {e}")
        
        return alerts
    
    def _extract_recommendations_for_system(self, content: str, system_id: str) -> List[str]:
        """Extract recommendations specific to a system"""
        recommendations = []
        
        rec_keywords = ['recommend', 'should', 'improve', 'optimize', 'consider']
        
        try:
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                system_lower = system_id.lower()
                
                # Check if sentence mentions system and has recommendation keywords
                if (system_lower in sentence_lower and 
                    any(keyword in sentence_lower for keyword in rec_keywords)):
                    
                    rec = sentence.strip()
                    if rec and len(rec) > 20:
                        recommendations.append(rec)
                        
                        if len(recommendations) >= 3:  # Limit recommendations per system
                            break
        except Exception as e:
            self.log_warning(f"Error extracting recommendations for {system_id}: {e}")
        
        return recommendations
    
    def _extract_metrics_from_content(self, content: str, system_id: str) -> Dict[str, Any]:
        """Extract performance metrics from content"""
        metrics = {}
        
        try:
            # Look for common SAP performance metrics
            metric_patterns = [
                (r'cpu[:\s]+(\d+)%', 'cpu_utilization'),
                (r'memory[:\s]+(\d+)%', 'memory_utilization'),
                (r'response\s+time[:\s]+(\d+(?:\.\d+)?)\s*(?:ms|sec)', 'response_time'),
                (r'users[:\s]+(\d+)', 'active_users'),
                (r'sessions[:\s]+(\d+)', 'active_sessions')
            ]
            
            content_lower = content.lower()
            system_lower = system_id.lower()
            
            # Only extract metrics if content mentions the system
            if system_lower in content_lower:
                for pattern, metric_name in metric_patterns:
                    matches = re.findall(pattern, content_lower)
                    if matches:
                        # Take the first match
                        value = matches[0]
                        try:
                            metrics[metric_name] = float(value)
                        except ValueError:
                            metrics[metric_name] = value
        
        except Exception as e:
            self.log_warning(f"Error extracting metrics for {system_id}: {e}")
        
        return metrics
    
    def _calculate_system_health(self, critical_alerts: List[str], 
                                recommendations: List[str], 
                                key_metrics: Dict[str, Any]) -> str:
        """Calculate overall system health based on findings"""
        try:
            # Health determination based on alerts
            alert_count = len(critical_alerts)
            
            if alert_count >= self.health_thresholds['critical_alert_threshold']:
                return 'CRITICAL'
            elif alert_count >= self.health_thresholds['warning_threshold']:
                return 'WARNING'
            elif alert_count == self.health_thresholds['healthy_threshold']:
                return 'HEALTHY'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            self.log_warning(f"Error calculating system health: {e}")
            return 'UNKNOWN'


# ================================
# MODULE EXPORTS AND INITIALIZATION
# ================================

# Export main classes for use by workflow
__all__ = [
    'BaseAgent',
    'PDFProcessorAgent', 
    'EmbeddingAgent',
    'SearchAgent',
    'SummaryAgent', 
    'EmailAgent',
    'SystemOutputAgent'
]

# Log module initialization
logger.info("🚀 SAP EWA Agents module initialized successfully")
logger.info(f"📋 Available agent types: {', '.join(__all__)}")
logger.info("🔗 All agents ready for LangGraph workflow integration")