# agents/pdf_processor.py - PDF Processing Agent
"""
PDF processing agent for extracting and cleaning text from uploaded PDF files.

This agent is the first in the workflow pipeline and handles:
- PDF file validation and preprocessing
- Text extraction using multiple libraries with fallbacks
- Text cleaning and normalization
- Document structure preservation
"""

import re
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent


class PDFProcessorAgent(BaseAgent):
    """
    First agent in the workflow pipeline - processes PDF files and extracts text.
    
    Key Features:
    - Multiple PDF library support (PyPDF2, pdfplumber, PyMuPDF)
    - Fallback extraction methods for reliability
    - Text cleaning and normalization
    - File validation and error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PDF processor with file handling configuration"""
        super().__init__("PDFProcessor", config)
        
        self.max_file_size = config.get('max_file_size_mb', 50) * 1024 * 1024
        self.supported_encodings = config.get('encodings', ['utf-8', 'latin-1', 'cp1252'])
        self.clean_text = config.get('clean_text', True)
    
    def process(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        Main processing method for PDF files.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Dict with processed file data or error information
        """
        self.start_timer()
        
        try:
            self.log_info(f"Starting PDF processing for {len(uploaded_files)} files")
            
            if not uploaded_files:
                return self.handle_error(ValueError("No PDF files provided"), "PDF Processing")
            
            processed_files = []
            failed_files = []
            
            # Process each file individually
            for i, file in enumerate(uploaded_files):
                try:
                    self.log_info(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Validate file
                    validation_result = self._validate_file(file)
                    if not validation_result['valid']:
                        failed_files.append({
                            'filename': file.name,
                            'error': validation_result['error']
                        })
                        continue
                    
                    # Extract text content
                    text_content = self._extract_text_from_pdf(file)
                    
                    if text_content and len(text_content.strip()) > 0:
                        # Clean text if enabled
                        if self.clean_text:
                            text_content = self._clean_extracted_text(text_content)
                        
                        # Create file data structure
                        file_data = {
                            'filename': file.name,
                            'text': text_content,
                            'size': len(file.getvalue()),
                            'character_count': len(text_content),
                            'word_count': len(text_content.split()),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        processed_files.append(file_data)
                        self.log_info(f"✅ Successfully processed {file.name}")
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
            
            processing_time = self.end_timer("pdf_processing")
            success_rate = len(processed_files) / len(uploaded_files) if uploaded_files else 0
            
            self.log_info(f"PDF processing completed: {len(processed_files)}/{len(uploaded_files)} files successful")
            
            return {
                "success": len(processed_files) > 0,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "processing_time": processing_time,
                "success_rate": success_rate
            }
            
        except Exception as e:
            return self.handle_error(e, "PDF Processing")
    
    def _validate_file(self, file) -> Dict[str, Any]:
        """Validate uploaded file before processing"""
        try:
            file_size = len(file.getvalue())
            
            if file_size > self.max_file_size:
                return {
                    'valid': False,
                    'error': f'File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit'
                }
            
            if not file.name.lower().endswith('.pdf'):
                return {'valid': False, 'error': 'File must be a PDF document'}
            
            if file_size == 0:
                return {'valid': False, 'error': 'File is empty'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'File validation error: {str(e)}'}
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text using multiple PDF libraries with fallback"""
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
                continue
            except Exception as e:
                self.log_warning(f"⚠️ {method_name} extraction failed: {str(e)}")
                continue
        
        self.log_error(f"All text extraction methods failed for {file.name}")
        return f"[Error: Could not extract text from {file.name}]"
    
    def _extract_with_pypdf2(self, file) -> str:
        """Extract text using PyPDF2 library"""
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
                continue
        
        return text_content
    
    def _extract_with_pdfplumber(self, file) -> str:
        """Extract text using pdfplumber library"""
        import pdfplumber
        import io
        
        text_content = ""
        
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                        
                    # Extract tables if present
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
        """Extract text using PyMuPDF (fitz) library"""
        import fitz
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
        """Clean and normalize extracted text content"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove PDF artifacts
        text = re.sub(r'\x0c', '\n', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€\x9d', '"')
        text = text.replace('â€"', '-')
        
        return text.strip()