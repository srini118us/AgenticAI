# models.py - Data Models for CrewAI EWA Analyzer
"""
Data models and type definitions for SAP EWA CrewAI Analyzer.
Defines structured data types used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from enum import Enum

# Enums for structured values
class HealthStatus(Enum):
    """SAP system health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class SAPProduct(Enum):
    """SAP product types"""
    S4HANA = "S/4HANA"
    ERP = "ERP"
    IBP = "IBP"
    BUSINESSOBJECTS = "BusinessObjects"
    HANA = "HANA Database"
    UNKNOWN = "Unknown"

class SystemEnvironment(Enum):
    """SAP system environment types"""
    PRODUCTION = "Production"
    DEVELOPMENT = "Development"
    TEST = "Test"
    QUALITY = "Quality"
    SANDBOX = "Sandbox"
    UNKNOWN = "Unknown"

class AgentStatus(Enum):
    """CrewAI agent status"""
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"

# Core Data Models
@dataclass
class SAPSystemInfo:
    """SAP system information extracted from EWA documents"""
    system_id: str = "UNKNOWN"
    product: SAPProduct = SAPProduct.UNKNOWN
    environment: SystemEnvironment = SystemEnvironment.UNKNOWN
    version: Optional[str] = None
    database_type: Optional[str] = None
    hostname: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "product": self.product.value,
            "environment": self.environment.value,
            "version": self.version,
            "database_type": self.database_type,
            "hostname": self.hostname
        }

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    filename: str
    file_size: int
    page_count: int
    extraction_method: str
    processed_at: datetime
    system_info: SAPSystemInfo
    char_count: int = 0
    word_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "extraction_method": self.extraction_method,
            "processed_at": self.processed_at.isoformat(),
            "system_info": self.system_info.to_dict(),
            "char_count": self.char_count,
            "word_count": self.word_count
        }

@dataclass
class SearchResult:
    """Individual search result from vector store"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    system_id: str = "UNKNOWN"
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
            "system_id": self.system_id,
            "source": self.source
        }

@dataclass
class HealthAlert:
    """SAP system health alert"""
    severity: HealthStatus
    category: str
    message: str
    recommendation: Optional[str] = None
    sap_note: Optional[str] = None
    system_id: str = "UNKNOWN"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "recommendation": self.recommendation,
            "sap_note": self.sap_note,
            "system_id": self.system_id
        }

@dataclass
class SystemHealthAnalysis:
    """Complete health analysis for a SAP system"""
    system_id: str
    overall_status: HealthStatus
    confidence_score: float
    critical_alerts: List[HealthAlert] = field(default_factory=list)
    warnings: List[HealthAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "overall_status": self.overall_status.value,
            "confidence_score": self.confidence_score,
            "critical_alerts": [alert.to_dict() for alert in self.critical_alerts],
            "warnings": [alert.to_dict() for alert in self.warnings],
            "recommendations": self.recommendations,
            "key_metrics": self.key_metrics,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }

@dataclass
class AgentCommunication:
    """Communication between CrewAI agents"""
    from_agent: str
    to_agent: str
    message: str
    action: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: AgentStatus = AgentStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message": self.message,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value
        }

@dataclass
class ProcessingResult:
    """Result from document processing"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class AnalysisRequest:
    """Request for EWA analysis"""
    files: List[str]
    search_queries: List[str] = field(default_factory=list)
    system_filter: Optional[str] = None
    include_metrics: bool = True
    include_recommendations: bool = True
    detailed_health: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "search_queries": self.search_queries,
            "system_filter": self.system_filter,
            "include_metrics": self.include_metrics,
            "include_recommendations": self.include_recommendations,
            "detailed_health": self.detailed_health
        }

@dataclass
class CrewExecutionResult:
    """Result from CrewAI execution"""
    success: bool
    analysis_results: List[SystemHealthAnalysis] = field(default_factory=list)
    agent_communications: List[AgentCommunication] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "analysis_results": [result.to_dict() for result in self.analysis_results],
            "agent_communications": [comm.to_dict() for comm in self.agent_communications],
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata
        }

# Type aliases for convenience
SystemID = str
Confidence = float
SearchQuery = str
FilePath = str

# Constants
DEFAULT_SEARCH_QUERIES = [
    "critical system alerts",
    "performance issues",
    "memory utilization",
    "database problems",
    "configuration warnings",
    "SAP recommendations"
]

HEALTH_STATUS_COLORS = {
    HealthStatus.HEALTHY: "#28A745",
    HealthStatus.WARNING: "#FFC107", 
    HealthStatus.CRITICAL: "#DC3545",
    HealthStatus.UNKNOWN: "#6C757D"
}

HEALTH_STATUS_ICONS = {
    HealthStatus.HEALTHY: "✅",
    HealthStatus.WARNING: "⚠️",
    HealthStatus.CRITICAL: "🔴",
    HealthStatus.UNKNOWN: "❓"
}

# Utility functions
def create_system_info(system_id: str = "UNKNOWN", 
                      product: str = "Unknown",
                      environment: str = "Unknown") -> SAPSystemInfo:
    """Create SAPSystemInfo with safe enum conversion"""
    try:
        sap_product = SAPProduct(product)
    except ValueError:
        sap_product = SAPProduct.UNKNOWN
    
    try:
        sys_env = SystemEnvironment(environment)
    except ValueError:
        sys_env = SystemEnvironment.UNKNOWN
    
    return SAPSystemInfo(
        system_id=system_id,
        product=sap_product,
        environment=sys_env
    )

def create_health_alert(severity: str, category: str, message: str,
                       system_id: str = "UNKNOWN") -> HealthAlert:
    """Create HealthAlert with safe enum conversion"""
    try:
        health_status = HealthStatus(severity.lower())
    except ValueError:
        health_status = HealthStatus.UNKNOWN
    
    return HealthAlert(
        severity=health_status,
        category=category,
        message=message,
        system_id=system_id
    )   # Data types and models for SAP EWA Analyzer 