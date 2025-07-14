# tools.py - CrewAI Tools for SAP EWA Analysis
"""
CrewAI tools for SAP Early Watch Alert analysis.
Provides specialized tools for PDF processing, vector search, and health analysis.
"""

import os
import re
import logging
import smtplib
import ssl
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

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
        
        # Extract System ID (SID) - Enhanced patterns
        system_id = "UNKNOWN"
        sid_patterns = [
            r'SYSTEM[:\s]+([A-Z0-9]{2,4})',
            r'SID[:\s]+([A-Z0-9]{2,4})',
            r'SAP\s+SYSTEM\s+([A-Z0-9]{2,4})',
            r'SYSTEM\s+ID[:\s]+([A-Z0-9]{2,4})',
            r'SAP\s+SYSTEM\s+ID[:\s]+([A-Z0-9]{2,4})',
            r'EARLY\s+WATCH.*?SYSTEM[:\s]+([A-Z0-9]{2,4})',
            r'DATABASE[:\s]+([A-Z0-9]{2,4})',
            r'CLIENT[:\s]+\d+.*?SYSTEM[:\s]+([A-Z0-9]{2,4})',
            r'([A-Z]{3})\s*(?:SYSTEM|DATABASE|INSTANCE)',  # Common 3-letter SIDs
            r'([A-Z]{2}[0-9])\s*(?:SYSTEM|DATABASE|INSTANCE)',  # 2 letters + 1 number
            r'HOST.*?([A-Z0-9]{2,4})',  # Sometimes in hostname
            r'SERVER.*?([A-Z0-9]{2,4})',  # Sometimes in server name
        ]
        
        for pattern in sid_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                # Filter out common false positives
                if match not in ['EWA', 'SAP', 'PDF', 'PAGE', 'DATE', 'TIME', 'USER', 'SYSTEM', 'ID']:
                    if 2 <= len(match) <= 4:  # Valid SID length
                        system_id = match
                        break
            if system_id != "UNKNOWN":
                break
        
        # If still unknown, try to find in filename or document header
        if system_id == "UNKNOWN":
            # Look for patterns like "EWA_PRD_2024" in the text
            filename_patterns = [
                r'([A-Z0-9]{2,4})[-_](?:EWA|EARLY|WATCH)',
                r'EWA[-_]([A-Z0-9]{2,4})',
                r'([A-Z0-9]{2,4})[-_](?:PROD|DEV|TEST|QAS)',
            ]
            for pattern in filename_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    system_id = match.group(1)
                    break
        
        # Extract SAP Product - Enhanced detection
        product = SAPProduct.UNKNOWN
        if 'S/4HANA' in text_upper or 'S4HANA' in text_upper or 'S4' in text_upper:
            product = SAPProduct.S4HANA
        elif 'ERP' in text_upper and ('SAP ERP' in text_upper or 'ECC' in text_upper):
            product = SAPProduct.ERP
        elif 'IBP' in text_upper or 'INTEGRATED BUSINESS PLANNING' in text_upper:
            product = SAPProduct.IBP
        elif 'BUSINESSOBJECTS' in text_upper or 'BUSINESS OBJECTS' in text_upper or 'BO ' in text_upper:
            product = SAPProduct.BUSINESSOBJECTS
        elif 'HANA DATABASE' in text_upper or 'SAP HANA' in text_upper:
            product = SAPProduct.HANA
        
        # Extract Environment - Enhanced detection
        environment = SystemEnvironment.UNKNOWN
        if any(word in text_upper for word in ['PRODUCTIVE', 'PRODUCTION', 'PROD', 'PRD']):
            environment = SystemEnvironment.PRODUCTION
        elif any(word in text_upper for word in ['DEVELOPMENT', 'DEV', 'DEVELOP']):
            environment = SystemEnvironment.DEVELOPMENT
        elif any(word in text_upper for word in ['TEST', 'TST', 'TESTING']):
            environment = SystemEnvironment.TEST
        elif any(word in text_upper for word in ['QUALITY', 'QAS', 'QA']):
            environment = SystemEnvironment.QUALITY
        elif any(word in text_upper for word in ['SANDBOX', 'SBX', 'SAND']):
            environment = SystemEnvironment.SANDBOX
        
        # Extract version if available
        version = None
        version_patterns = [
            r'VERSION[:\s]+([0-9\.]+)',
            r'RELEASE[:\s]+([0-9\.]+)',
            r'SP[:\s]+([0-9]+)',
            r'SUPPORT\s+PACKAGE[:\s]+([0-9]+)',
            r'KERNEL[:\s]+([0-9\.]+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text_upper)
            if match:
                version = match.group(1)
                break
        
        # Extract database type
        database_type = None
        if 'HANA DATABASE' in text_upper or 'SAP HANA' in text_upper:
            database_type = "SAP HANA"
        elif 'ORACLE' in text_upper:
            database_type = "Oracle"
        elif 'SQL SERVER' in text_upper or 'MSSQL' in text_upper:
            database_type = "SQL Server"
        elif 'DB2' in text_upper:
            database_type = "IBM DB2"
        elif 'MAXDB' in text_upper:
            database_type = "SAP MaxDB"
        
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
    
    def _run(self, query: str, k: int = None, system_filter: str = None) -> Dict[str, Any]:
        """Perform semantic search on the vector store"""
        try:
            # Initialize components locally to avoid field issues
            client = None
            collection = None
            embeddings = None
            
            # Try to initialize ChromaDB
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
                collection = client.get_or_create_collection(
                    name="sap_ewa_documents",
                    metadata={"description": "SAP EWA document embeddings"}
                )
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                return {
                    "success": False,
                    "error": f"Vector store initialization failed: {str(e)}",
                    "results": []
                }
            
            # Try to initialize embeddings
            try:
                if config.OPENAI_API_KEY:
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(
                        openai_api_key=config.OPENAI_API_KEY,
                        model=config.EMBEDDING_MODEL
                    )
                else:
                    return {
                        "success": False,
                        "error": "OpenAI API key not available",
                        "results": []
                    }
            except Exception as e:
                logger.warning(f"Embeddings initialization failed: {e}")
                return {
                    "success": False,
                    "error": f"Embeddings initialization failed: {str(e)}",
                    "results": []
                }
            
            # Perform the search
            if not collection or not embeddings:
                return {
                    "success": False,
                    "error": "Vector store not properly initialized",
                    "results": []
                }
            
            k = k or config.TOP_K_RESULTS
            
            # Create query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Prepare filter if system specified
            where_filter = {}
            if system_filter:
                where_filter = {"system_id": {"$eq": system_filter}}
            
            # Perform search
            search_results = collection.query(
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
            # Initialize components locally
            client = None
            collection = None
            embeddings = None
            
            # Initialize ChromaDB
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
                collection = client.get_or_create_collection(
                    name="sap_ewa_documents",
                    metadata={"description": "SAP EWA document embeddings"}
                )
            except Exception as e:
                logger.error(f"ChromaDB initialization failed: {e}")
                return False
            
            # Initialize embeddings
            try:
                if config.OPENAI_API_KEY:
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(
                        openai_api_key=config.OPENAI_API_KEY,
                        model=config.EMBEDDING_MODEL
                    )
                else:
                    logger.error("OpenAI API key not available")
                    return False
            except Exception as e:
                logger.error(f"Embeddings initialization failed: {e}")
                return False
            
            if not collection or not embeddings:
                logger.error("Vector store not initialized")
                return False
            
            # Create embeddings for documents
            embeddings_list = embeddings.embed_documents(documents)
            
            # Generate IDs
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]
            
            # Add to collection
            collection.add(
                documents=documents,
                embeddings=embeddings_list,
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
            logger.info(f"Starting health analysis for system: {system_id}")
            
            # Initialize analysis with safe defaults
            analysis_result = {
                "system_id": system_id,
                "overall_status": "unknown",
                "confidence_score": 0.5,
                "critical_alerts": [],
                "warnings": [],
                "recommendations": [],
                "key_metrics": {}
            }
            
            if not content or len(content.strip()) == 0:
                logger.warning("No content provided for health analysis")
                return {
                    "success": True,
                    "analysis": analysis_result,
                    "summary": {
                        "system_id": system_id,
                        "status": "unknown",
                        "critical_count": 0,
                        "warning_count": 0,
                        "confidence": 0.1
                    }
                }
            
            content_lower = content.lower()
            logger.info(f"Analyzing content of length: {len(content_lower)}")
            
            # Simple pattern detection
            critical_count = 0
            warning_count = 0
            
            # Check for critical indicators
            critical_patterns = ['critical', 'error', 'failed', 'down', 'offline', 'out of memory']
            for pattern in critical_patterns:
                if pattern in content_lower:
                    critical_count += 1
                    analysis_result["critical_alerts"].append({
                        "severity": "critical",
                        "category": "System Alert",
                        "message": f"Detected: {pattern}",
                        "system_id": system_id
                    })
            
            # Check for warning indicators  
            warning_patterns = ['warning', 'caution', 'attention', 'high', 'slow', 'recommended']
            for pattern in warning_patterns:
                if pattern in content_lower:
                    warning_count += 1
                    analysis_result["warnings"].append({
                        "severity": "warning",
                        "category": "Performance Warning",
                        "message": f"Detected: {pattern}",
                        "system_id": system_id
                    })
            
            # Limit alerts to avoid overwhelming output
            analysis_result["critical_alerts"] = analysis_result["critical_alerts"][:5]
            analysis_result["warnings"] = analysis_result["warnings"][:10]
            
            # Extract basic metrics
            import re
            cpu_matches = re.findall(r'cpu.*?(\d+)%', content_lower)
            if cpu_matches:
                try:
                    analysis_result["key_metrics"]["cpu_utilization"] = float(cpu_matches[0])
                except:
                    pass
            
            memory_matches = re.findall(r'memory.*?(\d+)%', content_lower)
            if memory_matches:
                try:
                    analysis_result["key_metrics"]["memory_utilization"] = float(memory_matches[0])
                except:
                    pass
            
            # Generate simple recommendations
            recommendations = []
            if critical_count > 0:
                recommendations.append("🔴 Immediate action required: Address critical system alerts")
                recommendations.append("📞 Consider escalating to SAP support for critical issues")
            
            if warning_count > 0:
                recommendations.append("⚠️ Review system warnings and plan remediation")
                recommendations.append("📊 Set up monitoring for key performance indicators")
            
            if critical_count == 0 and warning_count == 0:
                recommendations.append("✅ System appears stable - continue regular monitoring")
            
            recommendations.append("📋 Schedule regular system health reviews")
            analysis_result["recommendations"] = recommendations[:6]
            
            # Determine overall health status
            if critical_count > 0:
                analysis_result["overall_status"] = "critical"
            elif warning_count > 3:
                analysis_result["overall_status"] = "warning"
            elif warning_count > 0:
                analysis_result["overall_status"] = "warning"
            else:
                analysis_result["overall_status"] = "healthy"
            
            # Calculate confidence score
            base_confidence = 0.5
            content_boost = min(0.3, len(content_lower) / 5000)
            pattern_boost = min(0.2, (critical_count + warning_count) * 0.1)
            analysis_result["confidence_score"] = min(0.95, base_confidence + content_boost + pattern_boost)
            
            logger.info(f"Health analysis completed: {critical_count} critical, {warning_count} warnings")
            
            return {
                "success": True,
                "analysis": analysis_result,
                "summary": {
                    "system_id": system_id,
                    "status": analysis_result["overall_status"],
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "confidence": analysis_result["confidence_score"]
                }
            }
            
        except Exception as e:
            logger.error(f"Health analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {
                    "system_id": system_id,
                    "overall_status": "unknown",
                    "confidence_score": 0.1,
                    "critical_alerts": [],
                    "warnings": [],
                    "recommendations": ["⚠️ Analysis failed - manual review required"],
                    "key_metrics": {}
                }
            }
class EmailNotificationTool(BaseTool):
    """Tool for sending email notifications with analysis results"""
    
    name: str = "email_notifier"
    description: str = "Sends email notifications with SAP EWA analysis results and reports"
    
    def _run(self, recipients: List[str], subject: str, analysis_results: Dict[str, Any], 
             attachment_data: Optional[str] = None) -> Dict[str, Any]:
        """Send email notification with analysis results"""
        try:
            if not config.EMAIL_ENABLED:
                return {
                    "success": False,
                    "error": "Email notifications are disabled in configuration",
                    "sent_count": 0
                }
            
            # Validate email configuration
            validation = self._validate_email_config()
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Email configuration invalid: {', '.join(validation['errors'])}",
                    "sent_count": 0
                }
            
            # Create email content
            email_content = self._create_email_content(analysis_results)
            
            # Send emails
            sent_count = 0
            errors = []
            
            for recipient in recipients:
                try:
                    result = self._send_email(
                        recipient=recipient,
                        subject=subject,
                        html_content=email_content["html"],
                        text_content=email_content["text"],
                        attachment_data=attachment_data
                    )
                    
                    if result["success"]:
                        sent_count += 1
                    else:
                        errors.append(f"{recipient}: {result['error']}")
                        
                except Exception as e:
                    errors.append(f"{recipient}: {str(e)}")
            
            success = sent_count > 0
            
            return {
                "success": success,
                "sent_count": sent_count,
                "total_recipients": len(recipients),
                "errors": errors,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sent_count": 0
            }
    
    def _validate_email_config(self) -> Dict[str, Any]:
        """Validate email configuration"""
        validation = {"valid": True, "errors": []}
        
        if not config.EMAIL_USER:
            validation["errors"].append("EMAIL_USER not configured")
            validation["valid"] = False
        
        if not config.EMAIL_PASSWORD:
            validation["errors"].append("EMAIL_PASSWORD not configured")
            validation["valid"] = False
        
        if not config.SMTP_SERVER:
            validation["errors"].append("SMTP_SERVER not configured")
            validation["valid"] = False
        
        if config.SMTP_PORT <= 0:
            validation["errors"].append("SMTP_PORT must be positive")
            validation["valid"] = False
        
        return validation
    
    def _create_email_content(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Create HTML and text email content from analysis results"""
        
        # Extract key information
        systems_analyzed = analysis_results.get("systems_analyzed", 0)
        critical_alerts = analysis_results.get("critical_alerts", 0)
        warnings = analysis_results.get("warnings", 0)
        overall_health = analysis_results.get("overall_health", "Unknown")
        timestamp = analysis_results.get("timestamp", datetime.now().isoformat())
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(90deg, #0066CC, #00AA44); color: white; padding: 20px; border-radius: 8px; }}
                .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; border: 1px solid #ddd; }}
                .critical {{ color: #DC3545; font-weight: bold; }}
                .warning {{ color: #FFC107; font-weight: bold; }}
                .healthy {{ color: #28A745; font-weight: bold; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 SAP EWA Analysis Report</h1>
                <p>CrewAI Autonomous Agent Analysis Results</p>
            </div>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p><strong>Analysis completed at:</strong> {timestamp}</p>
                <p><strong>Overall Health Status:</strong> <span class="{overall_health.lower()}">{overall_health}</span></p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>{systems_analyzed}</h3>
                    <p>Systems Analyzed</p>
                </div>
                <div class="metric">
                    <h3 class="critical">{critical_alerts}</h3>
                    <p>Critical Alerts</p>
                </div>
                <div class="metric">
                    <h3 class="warning">{warnings}</h3>
                    <p>Warnings</p>
                </div>
            </div>
            
            <div class="recommendations">
                <h2>🎯 Key Recommendations</h2>
                <ul>
        """
        
        # Add recommendations if available
        recommendations = analysis_results.get("recommendations", [])
        if recommendations:
            for rec in recommendations[:5]:  # Top 5 recommendations
                html_content += f"<li>{rec}</li>"
        else:
            html_content += "<li>No specific recommendations generated</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="footer">
                <p>This report was generated automatically by the CrewAI SAP EWA Analyzer.</p>
                <p>For detailed analysis, please access the full application dashboard.</p>
            </div>
        </body>
        </html>
        """
        
        # Create text content
        text_content = f"""
        SAP EWA ANALYSIS REPORT
        ======================
        
        Analysis completed at: {timestamp}
        Overall Health Status: {overall_health}
        
        SUMMARY:
        - Systems Analyzed: {systems_analyzed}
        - Critical Alerts: {critical_alerts}
        - Warnings: {warnings}
        
        KEY RECOMMENDATIONS:
        """
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                text_content += f"{i}. {rec}\n"
        else:
            text_content += "No specific recommendations generated\n"
        
        text_content += """
        
        This report was generated automatically by the CrewAI SAP EWA Analyzer.
        For detailed analysis, please access the full application dashboard.
        """
        
        return {
            "html": html_content,
            "text": text_content
        }
    
    def _send_email(self, recipient: str, subject: str, html_content: str, 
                   text_content: str, attachment_data: Optional[str] = None) -> Dict[str, Any]:
        """Send individual email"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = config.EMAIL_USER
            msg["To"] = recipient
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Add attachment if provided
            if attachment_data:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(attachment_data.encode())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename=sap_ewa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                msg.attach(attachment)
            
            # Determine SMTP configuration based on email provider or server
            smtp_config = self._get_smtp_config()
            
            # Send email
            with smtplib.SMTP(smtp_config["server"], smtp_config["port"]) as server:
                if smtp_config["use_tls"]:
                    server.starttls(context=ssl.create_default_context())
                
                server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient}")
            
            return {
                "success": True,
                "recipient": recipient,
                "timestamp": datetime.now().isoformat()
            }
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "SMTP Authentication failed. Check email credentials."
            logger.error(f"Email send failed for {recipient}: {error_msg}")
            return {"success": False, "error": error_msg}
            
        except smtplib.SMTPRecipientsRefused:
            error_msg = f"Recipient {recipient} was refused by the server."
            logger.error(f"Email send failed: {error_msg}")
            return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"Email send failed: {str(e)}"
            logger.error(f"Email send failed for {recipient}: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def _get_smtp_config(self) -> Dict[str, Any]:
        """Automatically determine SMTP configuration based on email provider"""
        
        # Auto-detect based on EMAIL_USER domain
        email_domain = config.EMAIL_USER.lower().split('@')[-1] if '@' in config.EMAIL_USER else ""
        
        # Gmail configuration
        if ("gmail.com" in email_domain or 
            config.EMAIL_PROVIDER.lower() == "gmail" or 
            "gmail" in config.SMTP_SERVER.lower()):
            return {
                "server": "smtp.gmail.com",
                "port": 587,
                "use_tls": True
            }
        
        # Outlook/Hotmail configuration
        elif (any(domain in email_domain for domain in ["outlook.com", "hotmail.com", "live.com"]) or
              config.EMAIL_PROVIDER.lower() == "outlook" or
              "outlook" in config.SMTP_SERVER.lower()):
            return {
                "server": "smtp-mail.outlook.com", 
                "port": 587,
                "use_tls": True
            }
        
        # Yahoo configuration
        elif ("yahoo.com" in email_domain or 
              config.EMAIL_PROVIDER.lower() == "yahoo"):
            return {
                "server": "smtp.mail.yahoo.com",
                "port": 587,
                "use_tls": True
            }
        
        # Office 365 / Microsoft 365 configuration
        elif (config.EMAIL_PROVIDER.lower() in ["office365", "microsoft365"] or
              "office365" in config.SMTP_SERVER.lower()):
            return {
                "server": "smtp.office365.com",
                "port": 587,
                "use_tls": True
            }
        
        # Custom/Corporate SMTP
        else:
            return {
                "server": config.SMTP_SERVER,
                "port": config.SMTP_PORT,
                "use_tls": config.EMAIL_USE_TLS
            }
    
    def test_email_connection(self) -> Dict[str, Any]:
        """Test email server connection without sending email"""
        try:
            validation = self._validate_email_config()
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Configuration invalid: {', '.join(validation['errors'])}"
                }
            
            # Get SMTP configuration
            smtp_config = self._get_smtp_config()
            
            # Test connection
            with smtplib.SMTP(smtp_config["server"], smtp_config["port"]) as server:
                if smtp_config["use_tls"]:
                    server.starttls(context=ssl.create_default_context())
                server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
            
            # Determine provider for user feedback
            provider = "Unknown"
            if "gmail" in smtp_config["server"]:
                provider = "Gmail"
            elif "outlook" in smtp_config["server"] or "office365" in smtp_config["server"]:
                provider = "Outlook/Office365"
            elif "yahoo" in smtp_config["server"]:
                provider = "Yahoo"
            else:
                provider = "Custom SMTP"
            
            return {
                "success": True,
                "message": f"Successfully connected to {provider}",
                "provider": provider,
                "server": smtp_config["server"],
                "port": smtp_config["port"],
                "tls_enabled": smtp_config["use_tls"]
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                "success": False,
                "error": "Authentication failed. Check your email credentials."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }