# config.py - Clean Configuration Management
"""
Clean configuration management for SAP EWA Analyzer with:
- OpenAI API settings
- Document processing parameters  
- Email notifications (Gmail & Outlook support)
- ChromaDB vector store settings
- Environment variable loading
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    logging.warning("python-dotenv not available - using system environment variables")

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    """OpenAI service configuration"""
    api_key: str
    model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass  
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    max_file_size_mb: int = 50
    
    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")


@dataclass
class EmailConfig:
    """Email configuration supporting Gmail and Outlook"""
    enabled: bool = False
    provider: str = "gmail"  # "gmail" or "outlook"
    
    # Gmail settings
    gmail_email: str = ""
    gmail_app_password: str = ""
    
    # Outlook settings  
    outlook_email: str = ""
    outlook_password: str = ""
    
    # Common settings
    auto_send_results: bool = False
    default_recipients: List[str] = field(default_factory=list)
    smtp_timeout: int = 30
    
    def __post_init__(self):
        if self.enabled:
            if self.provider == "gmail":
                if not self.gmail_email or not self.gmail_app_password:
                    raise ValueError("Gmail email and app password are required")
            elif self.provider == "outlook":
                if not self.outlook_email or not self.outlook_password:
                    raise ValueError("Outlook email and password are required")
            else:
                raise ValueError("Email provider must be 'gmail' or 'outlook'")
    
    @property
    def is_configured(self) -> bool:
        if not self.enabled:
            return False
        
        if self.provider == "gmail":
            return bool(self.gmail_email and self.gmail_app_password)
        elif self.provider == "outlook":
            return bool(self.outlook_email and self.outlook_password)
        return False
    
    @property
    def email_address(self) -> str:
        if self.provider == "gmail":
            return self.gmail_email
        elif self.provider == "outlook":
            return self.outlook_email
        return ""
    
    @property
    def password(self) -> str:
        if self.provider == "gmail":
            return self.gmail_app_password
        elif self.provider == "outlook":
            return self.outlook_password
        return ""
    
    @property
    def smtp_server(self) -> str:
        if self.provider == "gmail":
            return "smtp.gmail.com"
        elif self.provider == "outlook":
            return "smtp-mail.outlook.com"
        return ""
    
    @property
    def smtp_port(self) -> int:
        return 587  # Both Gmail and Outlook use 587 with TLS
    
    def get_smtp_config(self) -> Dict[str, Any]:
        return {
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "email": self.email_address,
            "password": self.password,
            "use_tls": True,
            "timeout": self.smtp_timeout
        }


@dataclass
class StorageConfig:
    """Storage paths and settings"""
    data_dir: Path
    chroma_path: Path
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_base_dir(cls, base_dir: str = "./data") -> "StorageConfig":
        base_path = Path(base_dir)
        return cls(
            data_dir=base_path,
            chroma_path=base_path / "chroma"
        )


@dataclass
class VectorStoreConfig:
    """ChromaDB vector store configuration"""
    collection_name: str = "sap_documents"
    similarity_threshold: float = 0.7
    max_results: int = 100
    
    def __post_init__(self):
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self._load_and_validate_config()
    
    def _load_and_validate_config(self):
        try:
            # OpenAI Configuration
            self.openai = OpenAIConfig(
                api_key=self._get_required_env("OPENAI_API_KEY"),
                model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
                temperature=self._get_float_env("TEMPERATURE", 0.1),
                max_tokens=self._get_int_env("MAX_TOKENS", None)
            )
            
            # Processing Configuration
            self.processing = ProcessingConfig(
                chunk_size=self._get_int_env("CHUNK_SIZE", 1000),
                chunk_overlap=self._get_int_env("CHUNK_OVERLAP", 200),
                top_k=self._get_int_env("TOP_K", 10),
                min_chunk_size=self._get_int_env("MIN_CHUNK_SIZE", 100),
                max_chunk_size=self._get_int_env("MAX_CHUNK_SIZE", 4000),
                max_file_size_mb=self._get_int_env("MAX_FILE_SIZE_MB", 50)
            )
            
            # Email Configuration (Gmail & Outlook support)
            self.email = EmailConfig(
                enabled=self._get_bool_env("EMAIL_ENABLED", False),
                provider=os.getenv("EMAIL_PROVIDER", "gmail").lower(),
                gmail_email=os.getenv("GMAIL_EMAIL", ""),
                gmail_app_password=os.getenv("GMAIL_APP_PASSWORD", ""),
                outlook_email=os.getenv("OUTLOOK_EMAIL", ""),
                outlook_password=os.getenv("OUTLOOK_PASSWORD", ""),
                auto_send_results=self._get_bool_env("AUTO_SEND_RESULTS", False),
                default_recipients=self._get_list_env("DEFAULT_EMAIL_RECIPIENTS", [])
            )
            
            # Storage Configuration
            self.storage = StorageConfig.from_base_dir(
                os.getenv("DATA_DIR", "./data")
            )
            
            # Vector Store Configuration
            self.vector_store = VectorStoreConfig(
                collection_name=os.getenv("COLLECTION_NAME", "sap_documents"),
                similarity_threshold=self._get_float_env("SIMILARITY_THRESHOLD", 0.7),
                max_results=self._get_int_env("MAX_VECTOR_RESULTS", 100)
            )
            
            logger.info("✅ Configuration loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"❌ Configuration error: {e}")
            raise
    
    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _get_int_env(self, key: str, default: Optional[int]) -> Optional[int]:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_list_env(self, key: str, default: List[str]) -> List[str]:
        value = os.getenv(key)
        if not value:
            return default
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for agent initialization"""
        return {
            # OpenAI settings
            "openai_api_key": self.openai.api_key,
            "llm_model": self.openai.model,
            "embedding_model": self.openai.embedding_model,
            "temperature": self.openai.temperature,
            "max_tokens": self.openai.max_tokens,
            
            # Processing settings
            "chunk_size": self.processing.chunk_size,
            "chunk_overlap": self.processing.chunk_overlap,
            "top_k": self.processing.top_k,
            "min_chunk_size": self.processing.min_chunk_size,
            "max_chunk_size": self.processing.max_chunk_size,
            "max_file_size_mb": self.processing.max_file_size_mb,
            
            # Email settings (Gmail & Outlook)
            "email_enabled": self.email.enabled,
            "email_provider": self.email.provider,
            "gmail_email": self.email.gmail_email,
            "gmail_app_password": self.email.gmail_app_password,
            "outlook_email": self.email.outlook_email,
            "outlook_password": self.email.outlook_password,
            "auto_send_results": self.email.auto_send_results,
            
            # Vector store settings
            "collection_name": self.vector_store.collection_name,
            "similarity_threshold": self.vector_store.similarity_threshold,
            
            # Storage paths
            "data_dir": str(self.storage.data_dir),
            "chroma_path": str(self.storage.chroma_path)
        }
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate the complete setup"""
        return {
            "openai_configured": self.openai.is_configured,
            "email_configured": self.email.is_configured,
            "storage_accessible": self.storage.data_dir.exists()
        }
    
    def get_missing_requirements(self) -> List[str]:
        """Get list of missing configuration requirements"""
        missing = []
        
        if not self.openai.api_key:
            missing.append("OPENAI_API_KEY environment variable")
        
        if self.email.enabled and not self.email.is_configured:
            if self.email.provider == "gmail":
                missing.append("Gmail configuration (GMAIL_EMAIL and GMAIL_APP_PASSWORD)")
            elif self.email.provider == "outlook":
                missing.append("Outlook configuration (OUTLOOK_EMAIL and OUTLOOK_PASSWORD)")
        
        if not self.storage.data_dir.exists():
            missing.append(f"Data directory: {self.storage.data_dir}")
        
        return missing


# Global configuration instance
try:
    config = Config()
except Exception as e:
    logger.error(f"Failed to initialize configuration: {e}")
    # Fallback config for development
    class FallbackConfig:
        def __init__(self):
            self.openai = type('obj', (object,), {
                'api_key': 'mock_key',
                'model': 'gpt-4',
                'embedding_model': 'text-embedding-ada-002',
                'temperature': 0.1,
                'is_configured': False
            })()
            self.processing = type('obj', (object,), {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'top_k': 10,
                'max_file_size_mb': 50
            })()
            self.email = type('obj', (object,), {
                'enabled': False,
                'is_configured': False,
                'provider': 'gmail',
                'auto_send_results': False
            })()
            self.storage = type('obj', (object,), {
                'data_dir': Path('./data'),
                'chroma_path': Path('./data/chroma')
            })()
            self.vector_store = type('obj', (object,), {
                'collection_name': 'sap_documents',
                'similarity_threshold': 0.7,
                'max_results': 100
            })()
        
        def to_dict(self):
            return {
                'openai_api_key': 'mock_key',
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'top_k': 10,
                'email_enabled': False,
                'collection_name': 'sap_documents'
            }
        
        def validate_setup(self):
            return {'all_configured': False}
        
        def get_missing_requirements(self):
            return ['OpenAI API key', 'Email configuration']
    
    config = FallbackConfig()


# Convenience functions for external access
def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration as dictionary"""
    return {
        'api_key': config.openai.api_key,
        'model': config.openai.model,
        'embedding_model': config.openai.embedding_model,
        'temperature': config.openai.temperature
    }


def get_agent_config() -> Dict[str, Any]:
    """Get configuration dictionary for agent initialization"""
    return config.to_dict()


def is_email_enabled() -> bool:
    """Check if email notifications are enabled and configured"""
    return config.email.is_configured


def get_storage_paths() -> Dict[str, str]:
    """Get storage paths as string dictionary"""
    return {
        'data_dir': str(config.storage.data_dir),
        'chroma_path': str(config.storage.chroma_path)
    }


def get_email_config() -> Dict[str, Any]:
    """Get email configuration for both Gmail and Outlook"""
    if not config.email.is_configured:
        return {"enabled": False}
    
    return {
        "enabled": config.email.enabled,
        "provider": config.email.provider,
        "smtp_config": config.email.get_smtp_config(),
        "auto_send_results": config.email.auto_send_results,
        "default_recipients": config.email.default_recipients
    }


# Email templates for notifications
class EmailTemplates:
    """Email templates for different types of notifications"""
    
    SEARCH_RESULTS = {
        "subject": "SAP EWA Analysis Results - {query}",
        "body": """SAP Early Watch Analyzer - Search Results
==========================================

Query: {query}
Timestamp: {timestamp}
Systems Analyzed: {systems}
Results Found: {results_count}

EXECUTIVE SUMMARY:
{summary}

CRITICAL FINDINGS ({critical_count}):
{critical_findings}

RECOMMENDATIONS ({recommendation_count}):
{recommendations}

SYSTEM DETAILS:
{system_details}

---
Generated by SAP EWA Analyzer
Confidence Score: {confidence:.1f}%
Processing Time: {processing_time:.2f}s
"""
    }
    
    ERROR_NOTIFICATION = {
        "subject": "SAP EWA Analysis Error - {error_type}",
        "body": """SAP Early Watch Analyzer - Error Notification
===============================================

An error occurred during analysis:

Error Type: {error_type}
Error Message: {error_message}
Timestamp: {timestamp}
Query: {query}

Please check the system logs for more details.

---
Generated by SAP EWA Analyzer
"""
    }
    
    SYSTEM_ALERT = {
        "subject": "SAP System Alert - {system_id}",
        "body": """SAP System Alert Notification
=============================

System: {system_id}
Alert Level: {alert_level}
Timestamp: {timestamp}

CRITICAL ALERTS:
{alerts}

IMMEDIATE ACTIONS REQUIRED:
{actions}

---
Generated by SAP EWA Analyzer
"""
    }


# Configuration validation on module load
if hasattr(config, 'validate_setup'):
    setup_status = config.validate_setup()
    if not all(setup_status.values()):
        missing = config.get_missing_requirements() if hasattr(config, 'get_missing_requirements') else []
        if missing:
            logger.warning(f"⚠️ Configuration incomplete. Missing: {', '.join(missing)}")
    else:
        logger.info("✅ All configuration validated successfully")

logger.info("📋 Configuration module loaded successfully")