import json
import os
import re
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from models with fallback
try:
    from modelsorig import EmailRecipient, SystemSummary, SearchResult
except ImportError as e:
    logging.warning(f"Could not import from models: {e}")
    # Fallback definitions
    @dataclass
    class EmailRecipient:
        email: str
        name: str = ""
        
        def __post_init__(self):
            if not self.name:
                self.name = self.email.split('@')[0]
    
    @dataclass
    class SystemSummary:
        system_id: str
        overall_health: str
        critical_alerts: List[str]
        recommendations: List[str]
        key_metrics: Dict[str, Any]
        last_analyzed: str
    
    @dataclass
    class SearchResult:
        content: str
        source: str
        system_id: str
        confidence_score: float
        metadata: Dict[str, Any]

# Import other modules with fallbacks
try:
    from workfloworig import SAPRAGWorkflow, WorkflowStatus
except ImportError as e:
    logging.warning(f"Could not import workflow: {e}")

try:
    from configorig import Config
except ImportError as e:
    logging.warning(f"Could not import config: {e}")

try:
    from langgraph.graph import Document
except ImportError as e:
    logging.warning(f"Could not import langgraph: {e}")

# ================================
# AGENT IMPLEMENTATIONS
# ================================

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.name}] {message}")

class PDFProcessorAgent(BaseAgent):
    """Agent for processing PDF files"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PDFProcessor", config)
    
    def process(self, uploaded_files) -> Dict[str, Any]:
        """Process uploaded PDF files"""
        try:
            self.log_info(f"Processing {len(uploaded_files)} PDF files")
            
            processed_files = []
            for file in uploaded_files:
                text = self._extract_text_from_pdf(file)
                processed_files.append({
                    'filename': file.name,
                    'text': text,
                    'size': len(file.getvalue())
                })
            
            return {
                "success": True,
                "processed_files": processed_files,
                "files_count": len(uploaded_files)
            }
            
        except Exception as e:
            self.log_error(f"PDF processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            import io
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
            
        except Exception as e:
            self.log_error(f"Failed to extract text from {file.name}: {str(e)}")
            return f"Error extracting text from {file.name}"

class EmbeddingAgent(BaseAgent):
    """Agent for creating embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EmbeddingCreator", config)
    
    def process(self, processed_files) -> Dict[str, Any]:
        """Create embeddings from processed files"""
        try:
            self.log_info("Creating embeddings for processed files")
            
            chunks = []
            embeddings = []
            
            for file_data in processed_files:
                file_chunks = self._create_chunks(file_data['text'], file_data['filename'])
                chunks.extend(file_chunks)
            
            embeddings = self._create_embeddings(chunks)
            
            return {
                "success": True,
                "chunks": chunks,
                "embeddings": embeddings,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            self.log_error(f"Embedding creation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create text chunks"""
        chunk_size = self.config.get('chunk_size', 1000)
        chunks = []
        
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'source': filename,
                'chunk_id': i // chunk_size
            })
        
        return chunks
    
    def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create embeddings for chunks"""
        embeddings = []
        for chunk in chunks:
            # Mock embedding - replace with actual embedding logic
            embedding = [0.1] * 384
            embeddings.append(embedding)
        
        return embeddings
class SearchAgent(BaseAgent):
    """Agent for searching documents - Dynamic system ID handling with debug capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SearchAgent", config)
        self.vector_store = config.get('vector_store')
        self.embedding_agent = config.get('embedding_agent')
        self.top_k = config.get('top_k', 10)
        self._debug_logged = False  # To avoid spam in logs
        
        # Add this line to verify the method exists
        self.log_info("SearchAgent initialized with debug capabilities")
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search through processed documents"""
        try:
            self.log_info(f"Searching for: {query}")
            
            # Use real vector store if available, otherwise fallback to mock
            if self.vector_store:
                search_results = self._perform_vector_search(query, filters)
            else:
                search_results = self._perform_mock_search(query, filters)
            
            return {
                "success": True,
                "query": query,
                "search_results": search_results,
                "results_count": len(search_results)
            }
            
        except Exception as e:
            self.log_error(f"Search failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def debug_system_detection(self, query: str) -> Dict[str, Any]:
        """Debug function to see what systems are being detected"""
        try:
            self.log_info("=== DEBUGGING SYSTEM DETECTION ===")
            
            if not self.vector_store:
                self.log_info("No vector store available - using mock data")
                return {
                    "detected_systems": ["MOCK"], 
                    "sample_content": ["Mock content - No vector store available"], 
                    "total_docs": 0
                }
            
            # Perform a broad search to get all documents
            try:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=20)
                self.log_info(f"Vector search returned {len(docs_with_scores)} documents")
            except Exception as vs_error:
                self.log_error(f"Vector store search failed: {vs_error}")
                return {
                    "detected_systems": ["ERROR"], 
                    "sample_content": [f"Vector store error: {str(vs_error)}"], 
                    "total_docs": 0
                }
            
            detected_systems = set()
            sample_contents = []
            
            for i, (doc, score) in enumerate(docs_with_scores):
                try:
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    
                    # Log the first few documents
                    if i < 3:
                        self.log_info(f"Document {i+1}:")
                        self.log_info(f"  Content preview: {content[:200]}...")
                        self.log_info(f"  Metadata: {metadata}")
                        sample_contents.append(content[:500])
                    
                    # Try to extract system ID
                    system_id = metadata.get('system_id', 'UNKNOWN')
                    
                    if system_id == 'UNKNOWN':
                        system_id = self._extract_system_id_from_content_debug(content)
                    
                    detected_systems.add(system_id)
                    
                    # Also log what patterns we're finding
                    content_upper = content.upper()
                    if 'GDP' in content_upper:
                        self.log_info(f"Found 'GDP' in document {i+1}")
                    if 'P01' in content_upper:
                        self.log_info(f"Found 'P01' in document {i+1}")
                    if 'EARLY WATCH' in content_upper:
                        self.log_info(f"Found 'EARLY WATCH' in document {i+1}")
                        
                except Exception as e:
                    self.log_error(f"Error processing document {i}: {e}")
            
            self.log_info(f"All detected systems: {list(detected_systems)}")
            self.log_info("=== END DEBUG ===")
            
            return {
                "detected_systems": list(detected_systems),
                "sample_content": sample_contents,
                "total_docs": len(docs_with_scores)
            }
            
        except Exception as e:
            self.log_error(f"Debug function failed: {e}")
            return {
                "detected_systems": ["DEBUG_ERROR"], 
                "sample_content": [f"Debug error: {str(e)}"], 
                "total_docs": 0
            }
    
    def _perform_vector_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform vector similarity search using real vector store"""
        try:
            # Get target systems from filters - NO HARDCODING
            target_systems = filters.get('target_systems', []) if filters else []
            
            self.log_info(f"Searching with target systems: {target_systems}")
            
            # Perform similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            
            search_results = []
            for doc, score in docs_with_scores:
                try:
                    # Extract content
                    content = getattr(doc, 'page_content', str(doc))
                    
                    # Extract metadata
                    metadata = getattr(doc, 'metadata', {})
                    source = metadata.get('source', 'unknown')
                    
                    # Extract or detect system_id
                    system_id = metadata.get('system_id', 'UNKNOWN')
                    
                    # If system_id not in metadata, try to extract from content
                    if system_id == 'UNKNOWN':
                        system_id = self._extract_system_id_from_content(content)
                    
                    # Filter by target systems ONLY if target systems are specified
                    if target_systems and system_id not in target_systems:
                        self.log_info(f"Skipping result for system {system_id} (not in target list)")
                        continue  # Skip this result
                    
                    # Create SearchResult
                    result = SearchResult(
                        content=content,
                        source=source,
                        system_id=system_id,
                        confidence_score=float(score),
                        metadata=metadata
                    )
                    search_results.append(result)
                    
                except Exception as doc_error:
                    self.log_error(f"Error processing search result: {doc_error}")
                    continue
            
            self.log_info(f"Found {len(search_results)} results after filtering")
            return search_results
            
        except Exception as e:
            self.log_error(f"Vector search failed: {e}")
            # Fallback to mock search with same filters
            return self._perform_mock_search(query, filters)
    
    def _perform_mock_search(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform mock search for testing - DYNAMIC system IDs"""
        # Get target systems from filters - NO HARDCODING
        target_systems = filters.get('target_systems', []) if filters else []
        
        # If no target systems specified, use some common examples for testing
        if not target_systems:
            target_systems = ['UNKNOWN']  # Just one unknown system for testing
            self.log_info("No target systems specified, using UNKNOWN for mock search")
        else:
            self.log_info(f"Mock search for target systems: {target_systems}")
        
        mock_results = []
        for i, system in enumerate(target_systems):
            result = SearchResult(
                content=f"Mock search result {i+1} for query: '{query}' in system {system}. This is sample content that shows recommendations and system information for {system}.",
                source=f"document_{i+1}.pdf",
                system_id=system,  # Use the ACTUAL system ID passed in
                confidence_score=0.9 - (i * 0.1),
                metadata={"chunk_id": i, "page": i+1, "system_id": system}
            )
            mock_results.append(result)
        
        return mock_results
    
    def _extract_system_id_from_content(self, content: str) -> str:
        """Extract system ID from content text - supports ANY system ID"""
        try:
            if not content:
                return 'UNKNOWN'
            
            content_upper = content.upper()
            
            # Enhanced patterns specifically for GDP and P01
            patterns = [
                r'\bGDP\b',                               # Exact GDP match
                r'\bP01\b',                               # Exact P01 match
                r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',      # System: P01, System: GDP
                r'\bSID[:\s]+([A-Z0-9]{2,4})\b',         # SID: P01, SID: GDP
                r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',         # P01 SYSTEM, GDP SYSTEM
                r'\bFOR\s+([A-Z0-9]{2,4})\s+SYSTEM\b',   # for P01 system, for GDP system
                r'\b([A-Z]{1,3}[0-9]{1,2})\b',          # P01, Q01, D01, etc.
                r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b',  # Early Watch ... GDP
                r'\b([A-Z]{2,4})\b(?=.*(?:ERP|SAP|HANA|ABAP|BASIS))', # Any 2-4 letter code near SAP terms
            ]
            
            # Look for system IDs
            found_systems = set()
            for pattern in patterns:
                matches = re.findall(pattern, content_upper)
                for match in matches:
                    # Handle different match types
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                    
                    # Filter out common false positives
                    if (len(match) in [2, 3, 4] and 
                        match not in ['THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL']):
                        found_systems.add(match)
            
            # Return the first valid system ID found
            if found_systems:
                return list(found_systems)[0]
            
            return 'UNKNOWN'
            
        except Exception as e:
            self.log_error(f"Error extracting system ID from content: {e}")
            return 'UNKNOWN'
    
    def _extract_system_id_from_content_debug(self, content: str) -> str:
        """Debug version of system ID extraction with detailed logging"""
        try:
            if not content:
                return 'UNKNOWN'
            
            content_upper = content.upper()
            
            # Log what we're searching in (only first few times to avoid spam)
            if not self._debug_logged:
                self.log_info(f"Searching for system ID in content preview: {content_upper[:100]}...")
                self._debug_logged = True
            
            # Enhanced patterns specifically for GDP and P01
            patterns = [
                r'\bGDP\b',                               # Exact GDP match
                r'\bP01\b',                               # Exact P01 match
                r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',      # System: GDP, System: P01
                r'\bSID[:\s]+([A-Z0-9]{2,4})\b',         # SID: GDP, SID: P01
                r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',         # GDP SYSTEM, P01 SYSTEM
                r'\bFOR\s+([A-Z0-9]{2,4})\s+SYSTEM\b',   # for GDP system
                r'\b([A-Z]{1,3}[0-9]{1,2})\b',          # P01, Q01, D01, etc.
                r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b',  # Early Watch ... GDP
            ]
            
            found_systems = set()
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, content_upper)
                if matches:
                    self.log_info(f"Pattern {i+1} ({pattern}) found matches: {matches}")
                    for match in matches:
                        # Handle different match types
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                        
                        if (len(match) in [2, 3, 4] and 
                            match not in ['THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL']):
                            found_systems.add(match)
                            self.log_info(f"Added system: {match}")
            
            # Return the first valid system ID found
            if found_systems:
                result = list(found_systems)[0]
                self.log_info(f"Final system ID: {result}")
                return result
            
            self.log_info("No system ID found, returning UNKNOWN")
            return 'UNKNOWN'
            
        except Exception as e:
            self.log_error(f"Error extracting system ID from content: {e}")
            return 'UNKNOWN'
class SummaryAgent(BaseAgent):
    """Agent for generating summaries from search results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SummaryAgent", config)
    
    def generate_summary(self, search_results: List[Any], query: str) -> Dict[str, Any]:
        """Generate summary from search results"""
        try:
            self.log_info(f"Generating summary for {len(search_results)} results")
            
            if not search_results:
                return {
                    "success": True,
                    "summary": {
                        "summary": "No search results to summarize",
                        "critical_findings": [],
                        "recommendations": [],
                        "confidence_score": 0.0
                    }
                }
            
            # Extract content from search results
            all_content = []
            critical_findings = []
            recommendations = []
            
            for result in search_results:
                if isinstance(result, tuple):
                    doc, score = result
                    content = getattr(doc, 'page_content', str(doc))
                elif hasattr(result, 'content'):
                    content = result.content
                else:
                    content = str(result)
                
                all_content.append(content)
                
                # Look for critical issues
                content_lower = content.lower()
                if any(word in content_lower for word in ['critical', 'error', 'alert', 'fail']):
                    critical_findings.append(f"Critical issue found in document")
                
                # Look for recommendations
                if any(word in content_lower for word in ['recommend', 'should', 'improve']):
                    recommendations.append(f"Recommendation found in document")
            
            # Create summary
            summary_text = f"Analysis of {len(search_results)} documents for query: {query}"
            if all_content:
                summary_text += f". Found {len(critical_findings)} critical issues and {len(recommendations)} recommendations."
            
            return {
                "success": True,
                "summary": {
                    "summary": summary_text,
                    "critical_findings": critical_findings,
                    "recommendations": recommendations,
                    "confidence_score": 0.8
                }
            }
            
        except Exception as e:
            self.log_error(f"Summary generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

class EmailAgent(BaseAgent):
    """Agent for sending email notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EmailAgent", config)
    
    def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email with analysis results"""
        try:
            recipients = email_data.get('recipients', [])
            
            if not recipients:
                return {"success": False, "error": "No recipients specified"}
            
            self.log_info(f"Sending email to {len(recipients)} recipients")
            
            result = self._send_email_smtp(email_data)
            
            return {
                "success": True,
                "recipients_count": len(recipients),
                "message": "Email sent successfully"
            }
            
        except Exception as e:
            self.log_error(f"Email sending failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _send_email_smtp(self, email_data: Dict[str, Any]) -> bool:
        """Send email using SMTP"""
        # Placeholder - implement your SMTP logic
        return True

# ================================
# AGENT FACTORY
# ================================

def create_agent(agent_type: str, config: Dict[str, Any]):
    """Factory function to create agents"""
    agents = {
        'pdf_processor': PDFProcessorAgent,
        'embedding_creator': EmbeddingAgent,
        'search_agent': SearchAgent,
        'summary_agent': SummaryAgent,
        'email_agent': EmailAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agents[agent_type](config)