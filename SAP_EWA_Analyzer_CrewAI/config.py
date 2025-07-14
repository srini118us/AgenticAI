# config.py - Configuration for CrewAI EWA Analyzer
"""
Configuration management for SAP EWA CrewAI Analyzer.
Handles environment variables, API keys, and application settings.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables with explicit path
import os
from pathlib import Path

# Get the directory where this config.py file is located
current_dir = Path(__file__).parent
env_file = current_dir / '.env'

# Try multiple ways to load .env
load_dotenv(env_file)  # Try with explicit path
load_dotenv()          # Try with default search
load_dotenv(dotenv_path=str(env_file))  # Try with string path

# Debug: Print if .env was loaded (remove this in production)
import logging
logger = logging.getLogger(__name__)

# Check if .env file exists
if env_file.exists():
    logger.info(f"✅ .env file found at: {env_file}")
    logger.info(f"📏 File size: {env_file.stat().st_size} bytes")
else:
    logger.warning(f"⚠️ .env file not found at: {env_file}")

# Test if API key is loaded
test_key = os.getenv("OPENAI_API_KEY")
if test_key:
    logger.info(f"✅ API key loaded: {test_key[:8]}...")
else:
    logger.warning("⚠️ API key not loaded from environment")

class Config:
    """Application configuration class"""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    
    # Application Settings
    APP_TITLE: str = os.getenv("APP_TITLE", "SAP EWA Analyzer - CrewAI")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # CrewAI Settings
    CREW_MEMORY_ENABLED: bool = os.getenv("CREW_MEMORY_ENABLED", "true").lower() == "true"
    CREW_VERBOSE: bool = os.getenv("CREW_VERBOSE", "true").lower() == "true"
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
    
    # LLM Settings (FIXED - Added these)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    # Mock mode for testing without API calls
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"
    
   
    
    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Search Settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Email Settings (Optional)
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    EMAIL_PROVIDER: str = os.getenv("EMAIL_PROVIDER", "gmail").lower()  # gmail, outlook, custom
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_USER: str = os.getenv("EMAIL_USER", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_USE_TLS: bool = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
    
    # Default email recipients
    DEFAULT_RECIPIENTS: str = os.getenv("DEFAULT_EMAIL_RECIPIENTS", "")
    
    @classmethod
    def get_crew_config(cls) -> Dict[str, Any]:
        """Get CrewAI-specific configuration"""
        return {
            "memory": cls.CREW_MEMORY_ENABLED,
            "verbose": cls.CREW_VERBOSE,
            "max_iterations": cls.MAX_ITERATIONS,
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": cls.OPENAI_API_KEY,
                    "model": cls.EMBEDDING_MODEL
                }
            }
        }
    
    @classmethod
    def get_vector_config(cls) -> Dict[str, Any]:
        """Get vector store configuration"""
        return {
            "persist_directory": cls.CHROMA_PERSIST_DIR,
            "embedding_model": cls.EMBEDDING_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "top_k": cls.TOP_K_RESULTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required API key
        if not cls.OPENAI_API_KEY:
            validation["errors"].append("OPENAI_API_KEY is required")
            validation["valid"] = False
        
        # Check API key format
        if cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("sk-"):
            validation["warnings"].append("OPENAI_API_KEY format may be incorrect")
        
        # Check numeric values
        if cls.CHUNK_SIZE <= 0:
            validation["errors"].append("CHUNK_SIZE must be positive")
            validation["valid"] = False
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            validation["errors"].append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
            validation["valid"] = False
        
        if cls.TOP_K_RESULTS <= 0:
            validation["errors"].append("TOP_K_RESULTS must be positive")
            validation["valid"] = False
        
        if not (0.0 <= cls.SIMILARITY_THRESHOLD <= 1.0):
            validation["errors"].append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
            validation["valid"] = False
        
        # Email configuration validation (if enabled)
        if cls.EMAIL_ENABLED:
            if not cls.EMAIL_USER:
                validation["warnings"].append("EMAIL_USER not set but email is enabled")
            elif "@" not in cls.EMAIL_USER:
                validation["warnings"].append("EMAIL_USER format may be incorrect")
            
            if not cls.EMAIL_PASSWORD:
                validation["warnings"].append("EMAIL_PASSWORD not set but email is enabled")
            
            if cls.SMTP_PORT <= 0 or cls.SMTP_PORT > 65535:
                validation["errors"].append("SMTP_PORT must be between 1 and 65535")
                validation["valid"] = False
        
        return validation
    
    @classmethod
    def is_ready(cls) -> bool:
        """Check if configuration is ready for use"""
        return cls.validate_config()["valid"]

# Create global config instance
config = Config()

# Convenience functions
# Create global config instance
config = Config()

# Convenience functions
def get_openai_api_key() -> str:
    """Get OpenAI API key"""
    return config.OPENAI_API_KEY

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return config.DEBUG_MODE

def get_app_title() -> str:
    """Get application title"""
    return config.APP_TITLE