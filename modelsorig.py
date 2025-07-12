# models.py - Data classes and models
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class EmailRecipient:
    """Data class for email recipients"""
    email: str
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = self.email.split('@')[0]

@dataclass
class SystemSummary:
    """Data class for system analysis summary"""
    system_id: str
    overall_health: str
    critical_alerts: List[str]
    recommendations: List[str]
    key_metrics: Dict[str, Any]
    last_analyzed: str

@dataclass
class SearchResult:
    """Data class for search results"""
    content: str
    source: str
    system_id: str
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class WorkflowState:
    """Data class for workflow state tracking"""
    current_agent: str
    workflow_status: str
    processing_times: Dict[str, float]
    completed_agents: List[str]
    error_message: Optional[str] = None