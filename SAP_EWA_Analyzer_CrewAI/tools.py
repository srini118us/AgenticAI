# tools.py - CrewAI Tools for SAP EWA Analysis
"""
CrewAI tools for SAP Early Watch Alert analysis.
Provides specialized tools for PDF processing, vector search, and health analysis.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

from crewai.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
import pdfplumber
import PyPDF2

from config import config
from models import (
    DocumentMetadata, SAPSystemInfo, SearchResult, 
    HealthAlert, SystemHealthAnalysis, HealthStatus,
    SAPProduct, SystemEnvironment, create_system_info,
    create_health_alert
)

logger = logging.getLogger(__name__)

class PDFProcessorTool(BaseTool):
    """Tool for processing SAP EWA PDF documents"""
    
    name: str = "pdf_processor"
    description: str = "Processes SAP EWA PDF documents and extracts structured text content with metadata"
    
    def _run(self, file_path: str) -> Dict[str, Any]:
        """Process PDF and extract text content with SAP-specific metadata"""
        try:
            extracted_text = ""
            page_count = 0
            
            # Try pdfplumber first for better text extraction
            try:
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                extraction_method = "pdfplumber"
                
            except Exception as e:
                logger.warning(f"Pdfplumber failed, using PyPDF2: {e}")
                
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    page_count = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                extraction_method = "PyPDF2"
            
            # Extract SAP system information
            system_info = self._extract_sap_system_info(extracted_text)
            
            # Get file metadata
            file_stats = os.stat(file_path)
            filename = os.path.basename(file_path)
            
            # Create document metadata
            metadata = DocumentMetadata(
                filename=filename,
                file_size=file_stats.st_size,
                page_count=page_count,
                extraction_method=extraction_method,
                processed_at=datetime.now(),
                system_info=system_info,
                char_count=len(extracted_text),
                word_count=len(extracted_text.split())
            )
            
            return {
                "success": True,
                "text_content": extracted_text,
                "metadata": metadata.to_dict(),
                "system_info": system_info.to_dict(),
                "extraction_stats": {
                    "pages": page_count,
                    "characters": len(extracted_text),
                    "words": len(extracted_text.split()),
                    "method": extraction_method
                }
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text_content": "",
                "metadata": {},
                "system_info": {}
            }
    
    def _extract_sap_system_info(self, text: str) -> SAPSystemInfo:
        """Extract SAP system information from document text"""
        text_upper = text.upper()
        
        # Extract System ID (SID)
        system_id = "UNKNOWN"
        sid_patterns = [
            r'SYSTEM[:\s]+([A-Z0-9]{2,4})',
            r'SID[:\s]+([A-Z0-9]{2,4})',
            r'SAP\s+SYSTEM\s+([A-Z0-9]{2,4})',
            r'SYSTEM\s+ID[:\s]+([A-Z0-9]{2,4})'
        ]
        
        for pattern in sid_patterns:
            match = re.search(pattern, text_upper)
            if match:
                system_id = match.group(1)
                break
        
        # Extract SAP Product
        product = SAPProduct.UNKNOWN
        if 'S/4HANA' in text_upper or 'S4HANA' in text_upper:
            product = SAPProduct.S4HANA
        elif 'ERP' in text_upper and 'SAP ERP' in text_upper:
            product = SAPProduct.ERP
        elif 'IBP' in text_upper:
            product = SAPProduct.IBP
        elif 'BUSINESSOBJECTS' in text_upper or 'BUSINESS OBJECTS' in text_upper:
            product = SAPProduct.BUSINESSOBJECTS
        elif 'HANA DATABASE' in text_upper or 'SAP HANA' in text_upper:
            product = SAPProduct.HANA
        
        # Extract Environment
        environment = SystemEnvironment.UNKNOWN
        if 'PRODUCTIVE' in text_upper or 'PRODUCTION' in text_upper:
            environment = SystemEnvironment.PRODUCTION
        elif 'DEVELOPMENT' in text_upper:
            environment = SystemEnvironment.DEVELOPMENT
        elif 'TEST' in text_upper:
            environment = SystemEnvironment.TEST
        elif 'QUALITY' in text_upper:
            environment = SystemEnvironment.QUALITY
        elif 'SANDBOX' in text_upper:
            environment = SystemEnvironment.SANDBOX
        
        # Extract version if available
        version = None
        version_patterns = [
            r'VERSION[:\s]+([0-9\.]+)',
            r'RELEASE[:\s]+([0-9\.]+)',
            r'SP[:\s]+([0-9]+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text_upper)
            if match:
                version = match.group(1)
                break
        
        # Extract database type
        database_type = None
        if 'HANA DATABASE' in text_upper:
            database_type = "SAP HANA"
        elif 'ORACLE' in text_upper:
            database_type = "Oracle"
        elif 'SQL SERVER' in text_upper:
            database_type = "SQL Server"
        elif 'DB2' in text_upper:
            database_type = "IBM DB2"
        
        return SAPSystemInfo(
            system_id=system_id,
            product=product,
            environment=environment,
            version=version,
            database_type=database_type
        )

class VectorSearchTool(BaseTool):
    """Tool for semantic search using ChromaDB vector store"""
    
    name: str = "vector_search"
    description: str = "Performs semantic similarity search on processed SAP EWA documents"
    
    def __init__(self):
        super().__init__()
        try:
            self._initialize_vector_store()
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            self.client = None
            self.collection = None
            self.embeddings = None
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="sap_ewa_documents",
                metadata={"description": "SAP EWA document embeddings"}
            )
            
            # Initialize OpenAI embeddings
            if config.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=config.OPENAI_API_KEY,
                    model=config.EMBEDDING_MODEL
                )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
    
    def _run(self, query: str, k: int = None, system_filter: str = None) -> Dict[str, Any]:
        """Perform semantic search on the vector store"""
        try:
            if not self.collection or not self.embeddings:
                return {
                    "success": False,
                    "error": "Vector store not properly initialized",
                    "results": []
                }
            
            k = k or config.TOP_K_RESULTS
            
            # Create query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Prepare filter if system specified
            where_filter = {}
            if system_filter:
                where_filter = {"system_id": {"$eq": system_filter}}
            
            # Perform search
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if search_results['documents'] and search_results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
                )):
                    # Convert distance to similarity score (1 - distance for cosine)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Only include results above threshold
                    if similarity_score >= config.SIMILARITY_THRESHOLD:
                        result = SearchResult(
                            content=doc,
                            metadata=metadata,
                            similarity_score=similarity_score,
                            system_id=metadata.get("system_id", "UNKNOWN"),
                            source=metadata.get("source", "unknown")
                        )
                        formatted_results.append(result)
            
            return {
                "success": True,
                "results": [result.to_dict() for result in formatted_results],
                "query": query,
                "total_results": len(formatted_results),
                "search_metadata": {
                    "k": k,
                    "system_filter": system_filter,
                    "similarity_threshold": config.SIMILARITY_THRESHOLD
                }
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        try:
            if not self.collection or not self.embeddings:
                logger.error("Vector store not initialized")
                return False
            
            # Create embeddings for documents
            embeddings = self.embeddings.embed_documents(documents)
            
            # Generate IDs
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

class HealthAnalysisTool(BaseTool):
    """Tool for analyzing SAP system health from EWA content"""
    
    name: str = "health_analyzer"
    description: str = "Analyzes SAP system health indicators and generates recommendations"
    
    def _run(self, content: str, system_id: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze system health from document content"""
        try:
            # Initialize analysis
            analysis = SystemHealthAnalysis(
                system_id=system_id,
                overall_status=HealthStatus.UNKNOWN,
                confidence_score=0.0
            )
            
            content_lower = content.lower()
            
            # Detect critical alerts
            critical_indicators = self._detect_critical_indicators(content_lower, system_id)
            analysis.critical_alerts.extend(critical_indicators)
            
            # Detect warnings
            warning_indicators = self._detect_warning_indicators(content_lower, system_id)
            analysis.warnings.extend(warning_indicators)
            
            # Extract key metrics
            analysis.key_metrics = self._extract_key_metrics(content_lower)
            
            # Generate recommendations
            analysis.recommendations = self._generate_recommendations(
                content_lower, critical_indicators, warning_indicators
            )
            
            # Determine overall health status
            analysis.overall_status = self._calculate_overall_health(
                len(critical_indicators), len(warning_indicators)
            )
            
            # Calculate confidence score
            analysis.confidence_score = self._calculate_confidence_score(
                content_lower, critical_indicators, warning_indicators
            )
            
            return {
                "success": True,
                "analysis": analysis.to_dict(),
                "summary": {
                    "system_id": system_id,
                    "status": analysis.overall_status.value,
                    "critical_count": len(critical_indicators),
                    "warning_count": len(warning_indicators),
                    "confidence": analysis.confidence_score
                }
            }
            
        except Exception as e:
            logger.error(f"Health analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }
    
    def _detect_critical_indicators(self, content: str, system_id: str) -> List[HealthAlert]:
        """Detect critical health indicators"""
        alerts = []
        
        critical_patterns = [
            (r'critical|error|failed|down|offline', 'System Error'),
            (r'memory.*full|out of memory', 'Memory Issue'),
            (r'disk.*full|disk space.*low', 'Storage Issue'),
            (r'tablespace.*full', 'Database Issue'),
            (r'connection.*failed|timeout', 'Connectivity Issue'),
            (r'cpu.*high|cpu.*100%', 'Performance Issue')
        ]
        
        for pattern, category in critical_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                alert = create_health_alert(
                    severity="critical",
                    category=category,
                    message=f"Detected: {match}",
                    system_id=system_id
                )
                alerts.append(alert)
        
        return alerts[:5]  # Limit to top 5 critical alerts
    
    def _detect_warning_indicators(self, content: str, system_id: str) -> List[HealthAlert]:
        """Detect warning health indicators"""
        alerts = []
        
        warning_patterns = [
            (r'warning|caution|attention', 'General Warning'),
            (r'performance.*slow|response.*slow', 'Performance Warning'),
            (r'recommended|should.*consider', 'Recommendation'),
            (r'memory.*high|memory.*80%', 'Memory Warning'),
            (r'cpu.*high|cpu.*[7-9][0-9]%', 'CPU Warning')
        ]
        
        for pattern, category in warning_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                alert = create_health_alert(
                    severity="warning",
                    category=category,
                    message=f"Detected: {match}",
                    system_id=system_id
                )
                alerts.append(alert)
        
        return alerts[:10]  # Limit to top 10 warnings
    
    def _extract_key_metrics(self, content: str) -> Dict[str, Any]:
        """Extract key performance metrics"""
        metrics = {}
        
        metric_patterns = [
            (r'cpu.*?(\d+)%', 'cpu_utilization'),
            (r'memory.*?(\d+)%', 'memory_utilization'),
            (r'disk.*?(\d+)%', 'disk_utilization'),
            (r'response.*?(\d+).*?ms', 'response_time_ms'),
            (r'users.*?(\d+)', 'active_users'),
            (r'sessions.*?(\d+)', 'active_sessions')
        ]
        
        for pattern, metric_name in metric_patterns:
            matches = re.findall(pattern, content)
            if matches:
                try:
                    # Take the first match and convert to number
                    value = matches[0]
                    metrics[metric_name] = float(value)
                except ValueError:
                    metrics[metric_name] = value
        
        return metrics
    
    def _generate_recommendations(self, content: str, critical_alerts: List[HealthAlert], 
                                warnings: List[HealthAlert]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical alert recommendations
        if len(critical_alerts) > 0:
            recommendations.append("🔴 Immediate action required: Address critical system alerts")
            recommendations.append("📞 Consider escalating to SAP support for critical issues")
        
        # Memory recommendations
        if any('memory' in alert.message.lower() for alert in critical_alerts + warnings):
            recommendations.append("💾 Review memory allocation and optimize usage")
            recommendations.append("🔄 Consider implementing memory management strategies")
        
        # Performance recommendations
        if any('performance' in alert.message.lower() for alert in warnings):
            recommendations.append("⚡ Analyze performance bottlenecks and optimize")
            recommendations.append("📊 Set up monitoring for key performance indicators")
        
        # General recommendations
        if len(warnings) > 5:
            recommendations.append("🔍 Conduct comprehensive system health review")
        
        recommendations.append("📋 Schedule regular monitoring and maintenance")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _calculate_overall_health(self, critical_count: int, warning_count: int) -> HealthStatus:
        """Calculate overall system health status"""
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 3:
            return HealthStatus.WARNING
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_confidence_score(self, content: str, critical_alerts: List[HealthAlert], 
                                  warnings: List[HealthAlert]) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = 0.5
        
        # Increase confidence based on content length
        content_length_boost = min(0.2, len(content) / 10000)
        
        # Increase confidence based on detected patterns
        pattern_boost = min(0.2, (len(critical_alerts) + len(warnings)) * 0.05)
        
        # Decrease confidence if no clear indicators
        if len(critical_alerts) == 0 and len(warnings) == 0:
            base_confidence -= 0.2
        
        confidence = base_confidence + content_length_boost + pattern_boost
        return max(0.1, min(0.95, confidence))