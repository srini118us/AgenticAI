# models.py - Optimized Data Models and Type Definitions
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    INITIALIZED = "initialized"
    PROCESSING_PDF = "processing_pdf"
    CREATING_EMBEDDINGS = "creating_embeddings"
    STORING_VECTORS = "storing_vectors"
    SEARCHING = "searching"
    SUMMARIZING = "summarizing"
    SYSTEM_OUTPUT = "system_output"
    SENDING_EMAIL = "sending_email"
    COMPLETED = "completed"
    ERROR = "error"


class MessageStatus(Enum):
    """Agent message status"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EmailRecipient:
    """Email recipient data model with provider support"""
    email: str
    name: str = ""
    preferred_provider: str = ""  # Optional: "gmail", "outlook", or ""
    
    def __post_init__(self):
        """Auto-generate name from email if not provided"""
        if not self.name:
            self.name = self.email.split('@')[0].replace('.', ' ').title()
        
        # Auto-detect provider from email domain if not specified
        if not self.preferred_provider:
            domain = self.email.split('@')[1].lower() if '@' in self.email else ""
            if 'gmail.com' in domain:
                self.preferred_provider = "gmail"
            elif any(d in domain for d in ['outlook.com', 'hotmail.com', 'live.com']):
                self.preferred_provider = "outlook"
    
    @property
    def is_valid(self) -> bool:
        """Check if email is valid format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, self.email))
    
    @property
    def domain(self) -> str:
        """Get email domain"""
        return self.email.split('@')[1] if '@' in self.email else ""

@dataclass
class EmailConfiguration:
    """Email configuration model for workflow state"""
    enabled: bool = False
    provider: str = "gmail"  # "gmail" or "outlook"
    smtp_server: str = ""
    smtp_port: int = 587
    email_address: str = ""
    authenticated: bool = False
    last_test: Optional[str] = None  # ISO timestamp of last successful test
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EmailConfiguration":
        """Create from configuration dictionary"""
        provider = config.get("email_provider", "gmail").lower()
        
        if provider == "gmail":
            email_address = config.get("gmail_email", "")
            smtp_server = "smtp.gmail.com"
        else:  # outlook
            email_address = config.get("outlook_email", "")
            smtp_server = "smtp-mail.outlook.com"
        
        return cls(
            enabled=config.get("email_enabled", False),
            provider=provider,
            smtp_server=smtp_server,
            smtp_port=config.get("smtp_port", 587),
            email_address=email_address,
            authenticated=bool(email_address and (
                config.get("gmail_app_password") if provider == "gmail" 
                else config.get("outlook_password")
            ))
        )


@dataclass
class SearchResult:
    """Search result data model"""
    content: str
    source: str
    system_id: str
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data"""
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        self.system_id = self.system_id.upper() if self.system_id else "UNKNOWN"


@dataclass
class SystemSummary:
    """System analysis summary data model"""
    system_id: str
    overall_health: Union[HealthStatus, str]
    critical_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    last_analyzed: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Normalize health status"""
        if isinstance(self.overall_health, str):
            try:
                self.overall_health = HealthStatus(self.overall_health.upper())
            except ValueError:
                self.overall_health = HealthStatus.UNKNOWN
    
    def add_alert(self, alert: str) -> None:
        """Add critical alert"""
        if alert and alert not in self.critical_alerts:
            self.critical_alerts.append(alert)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add recommendation"""
        if recommendation and recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    @property
    def alert_count(self) -> int:
        """Get number of critical alerts"""
        return len(self.critical_alerts)
    
    @property
    def recommendation_count(self) -> int:
        """Get number of recommendations"""
        return len(self.recommendations)


@dataclass
class AgentMessage:
    """Agent communication message"""
    agent_name: str
    message: str
    status: Union[MessageStatus, str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Normalize status"""
        if isinstance(self.status, str):
            try:
                self.status = MessageStatus(self.status.lower())
            except ValueError:
                self.status = MessageStatus.PROCESSING


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    total_files: int = 0
    total_chunks: int = 0
    processing_times: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def total_processing_time(self) -> float:
        """Calculate total processing time"""
        return sum(self.processing_times.values())
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.end_time is not None
    
    def add_timing(self, step: str, duration: float) -> None:
        """Add processing step timing"""
        self.processing_times[step] = max(0.0, duration)
    
    def mark_complete(self) -> None:
        """Mark processing as complete"""
        self.end_time = datetime.now()
    
    def get_step_duration(self, step: str) -> float:
        """Get duration for specific step"""
        return self.processing_times.get(step, 0.0)


@dataclass
class WorkflowState:
    """Complete workflow state tracking"""
    # Input parameters
    uploaded_files: List[Any] = field(default_factory=list)
    user_query: str = ""
    search_filters: Dict[str, Any] = field(default_factory=dict)
    email_recipients: List[EmailRecipient] = field(default_factory=list)
    
    # Workflow status
    workflow_status: Union[WorkflowStatus, str] = WorkflowStatus.INITIALIZED
    current_agent: str = ""
    error_message: str = ""
    
    # Processing results
    processed_documents: List[Any] = field(default_factory=list)
    embeddings: List[Any] = field(default_factory=list)
    vector_store_ready: bool = False
    search_results: List[SearchResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    system_summaries: Dict[str, SystemSummary] = field(default_factory=dict)
    
    # Communication
    email_sent: bool = False
    
    # Metrics and messages
    processing_times: Dict[str, float] = field(default_factory=dict)
    agent_messages: List[AgentMessage] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize workflow status"""
        if isinstance(self.workflow_status, str):
            try:
                self.workflow_status = WorkflowStatus(self.workflow_status.lower())
            except ValueError:
                self.workflow_status = WorkflowStatus.INITIALIZED
    
    @property
    def total_chunks(self) -> int:
        """Get total number of document chunks"""
        return len(self.processed_documents)
    
    @property
    def has_results(self) -> bool:
        """Check if workflow has search results"""
        return len(self.search_results) > 0
    
    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete"""
        return self.workflow_status == WorkflowStatus.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Check if workflow has error"""
        return self.workflow_status == WorkflowStatus.ERROR
    
    def add_message(self, agent: str, message: str, status: MessageStatus = MessageStatus.PROCESSING) -> None:
        """Add agent message"""
        agent_msg = AgentMessage(
            agent_name=agent,
            message=message,
            status=status
        )
        self.agent_messages.append(agent_msg)
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get processing metrics"""
        metrics = ProcessingMetrics(
            total_files=len(self.uploaded_files),
            total_chunks=self.total_chunks,
            processing_times=self.processing_times.copy()
        )
        if self.is_complete:
            metrics.mark_complete()
        return metrics


# Type aliases for commonly used types
DocumentList = List[Any]
EmbeddingList = List[Any]
SystemSummaryDict = Dict[str, SystemSummary]
SearchResultList = List[SearchResult]
ConfigDict = Dict[str, Any]


# Factory functions for creating common objects
def create_email_recipient(email: str, name: str = "") -> EmailRecipient:
    """Factory function to create email recipient"""
    return EmailRecipient(email=email, name=name)


def create_search_result(content: str, source: str, system_id: str, 
                        confidence: float, metadata: Dict = None) -> SearchResult:
    """Factory function to create search result"""
    return SearchResult(
        content=content,
        source=source,
        system_id=system_id,
        confidence_score=confidence,
        metadata=metadata or {}
    )


def create_system_summary(system_id: str, health: Union[HealthStatus, str]) -> SystemSummary:
    """Factory function to create system summary"""
    return SystemSummary(
        system_id=system_id,
        overall_health=health
    )


def create_workflow_state(**kwargs) -> WorkflowState:
    """Factory function to create workflow state"""
    return WorkflowState(**kwargs)