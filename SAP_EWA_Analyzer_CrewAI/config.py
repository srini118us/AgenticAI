"""
Configuration settings for SAP EWA Analyzer using CrewAI
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for SAP EWA Analyzer"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Settings
    DEFAULT_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    
    # Vector Store Settings
    VECTOR_STORE_PATH = "../data/chroma"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # PDF Processing
    MAX_PDF_SIZE_MB = 50
    SUPPORTED_PDF_EXTENSIONS = [".pdf"]
    
    # CrewAI Settings
    MAX_ITERATIONS = 3
    VERBOSE = True
    
    # File Paths
    DATA_DIR = "../data"
    LOGS_DIR = "../logs"
    
    # SAP EWA Specific Settings
    EWA_SECTIONS = [
        "System Overview",
        "Performance Analysis", 
        "Security Analysis",
        "Database Analysis",
        "Recommendations",
        "Critical Issues"
    ]
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.DEFAULT_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in environment variables")
        return True

# Initialize and validate config
try:
    Config.validate_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please set OPENAI_API_KEY in your .env file") 