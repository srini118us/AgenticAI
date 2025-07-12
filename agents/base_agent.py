# agents/base_agent.py - Foundation Agent Class
"""
Base agent class providing common functionality for all SAP EWA analysis agents.

This abstract base class ensures consistent behavior across all agents including:
- Standardized logging with agent identification
- Unified error handling and reporting
- Performance metrics tracking
- Configuration management
"""

import time
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all SAP EWA analysis agents.
    
    Provides essential shared functionality:
    - Standardized logging and error handling
    - Performance metrics tracking
    - Configuration management
    - Common utility methods
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base agent with configuration.
        
        Args:
            name: Human-readable agent name
            config: Configuration dictionary from config.py
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.start_time = None
        self.performance_metrics = {}
        
        self._setup_agent_defaults()
        self.log_info(f"Agent initialized with config keys: {list(config.keys())}")
    
    def _setup_agent_defaults(self):
        """Set up default configuration values"""
        defaults = {
            'timeout': 300,
            'retry_attempts': 3,
            'retry_delay': 1.0
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def log_info(self, message: str):
        """Log informational message with agent identification"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        """Log warning message with agent identification"""
        self.logger.warning(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message with agent identification"""
        self.logger.error(f"[{self.name}] {message}")
    
    def start_timer(self):
        """Start performance timing for current operation"""
        self.start_time = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """
        End performance timing and record duration.
        
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
        """Get all recorded performance metrics"""
        return self.performance_metrics.copy()
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Standardized error handling for all agents.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Standardized error response dictionary
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.log_error(error_msg)
        
        return {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__,
            "agent": self.name,
            "timestamp": datetime.now().isoformat()
        }