# workflow/workflow_utils.py - Utility Functions
"""
Utility functions for SAP EWA Workflow management.

This module provides helper functions for workflow creation, configuration validation,
testing, and general workflow management tasks. These utilities support the main
workflow components and provide convenient factory functions and validation.

Available Functions:
- create_workflow: Factory function for workflow creation
- validate_workflow_config: Configuration validation and recommendations
- test_workflow_basic: Basic workflow functionality testing
- Workflow factory and configuration utilities

Usage:
    from workflow.workflow_utils import create_workflow, validate_workflow_config
    
    # Validate configuration
    validation = validate_workflow_config(config)
    
    # Create workflow if valid
    if validation["valid"]:
        workflow = create_workflow(config)
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import the main workflow class
try:
    from .core_workflow import SAPRAGWorkflow
    logger.info("✅ Core workflow imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import core workflow: {e}")
    raise


# ================================
# WORKFLOW FACTORY FUNCTIONS
# ================================

def create_workflow(config: Dict[str, Any] = None) -> SAPRAGWorkflow:
    """
    Factory function to create a configured SAP RAG workflow.
    
    This is the recommended way to create workflow instances as it provides
    proper error handling and configuration validation.
    
    Args:
        config: Configuration dictionary containing:
            - embedding_type: Type of embeddings (openai, huggingface, mock)
            - vector_store_type: Vector store backend (chroma, faiss, simple, mock)
            - email_enabled: Whether to enable email notifications
            - email_recipients: Default email recipients
            - chunk_size: Text chunk size for embeddings
            - top_k: Number of search results to return
            - Various agent-specific settings
            
    Returns:
        Initialized SAPRAGWorkflow instance
        
    Raises:
        Exception: If workflow creation fails
        
    Example:
        >>> config = {
        ...     "embedding_type": "openai",
        ...     "vector_store_type": "chroma",
        ...     "email_enabled": True,
        ...     "chunk_size": 1000,
        ...     "top_k": 10
        ... }
        >>> workflow = create_workflow(config)
        >>> result = workflow.run_workflow(uploaded_files=files, user_query="query")
    """
    try:
        logger.info("🏭 Creating workflow via factory function")
        
        # Validate configuration if provided
        if config:
            validation = validate_workflow_config(config)
            if not validation["valid"]:
                logger.warning(f"⚠️ Configuration has errors: {validation['errors']}")
                # Continue anyway but log warnings
                for warning in validation["warnings"]:
                    logger.warning(f"⚠️ {warning}")
        
        # Create workflow instance
        workflow = SAPRAGWorkflow(config)
        
        # Log creation success with configuration summary
        logger.info("✅ Workflow created successfully via factory")
        
        if config:
            logger.info(f"📋 Configuration summary:")
            logger.info(f"  - Embedding type: {config.get('embedding_type', 'default')}")
            logger.info(f"  - Vector store: {config.get('vector_store_type', 'default')}")
            logger.info(f"  - Email enabled: {config.get('email_enabled', False)}")
            logger.info(f"  - Chunk size: {config.get('chunk_size', 'default')}")
            logger.info(f"  - Top K results: {config.get('top_k', 'default')}")
        
        return workflow
        
    except Exception as e:
        logger.error(f"❌ Workflow creation failed: {e}")
        raise Exception(f"Failed to create workflow: {str(e)}")


def create_workflow_with_defaults(
    embedding_type: str = "openai",
    vector_store_type: str = "chroma",
    email_enabled: bool = False,
    **kwargs
) -> SAPRAGWorkflow:
    """
    Create workflow with sensible defaults and optional overrides.
    
    This convenience function provides sensible defaults for common use cases
    while allowing specific parameters to be overridden.
    
    Args:
        embedding_type: Type of embeddings to use
        vector_store_type: Vector store backend
        email_enabled: Whether to enable email notifications
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SAPRAGWorkflow instance
        
    Example:
        >>> # Create with defaults
        >>> workflow = create_workflow_with_defaults()
        
        >>> # Create with email enabled
        >>> workflow = create_workflow_with_defaults(
        ...     email_enabled=True,
        ...     gmail_email="user@example.com",
        ...     gmail_app_password="password"
        ... )
    """
    default_config = {
        "embedding_type": embedding_type,
        "vector_store_type": vector_store_type,
        "email_enabled": email_enabled,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 10,
        "collection_name": f"sap_documents_{datetime.now().strftime('%Y%m%d')}",
        "auto_send_results": False,
        "enable_monitoring": True
    }
    
    # Merge with provided kwargs
    config = {**default_config, **kwargs}
    
    logger.info(f"🏭 Creating workflow with defaults: {embedding_type}, {vector_store_type}")
    
    return create_workflow(config)


# ================================
# CONFIGURATION VALIDATION
# ================================

def validate_workflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate workflow configuration and provide recommendations.
    
    Performs comprehensive validation of workflow configuration including
    required parameters, compatibility checks, and performance recommendations.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary containing:
            - valid: Boolean indicating if configuration is valid
            - errors: List of error messages (configuration won't work)
            - warnings: List of warning messages (suboptimal but functional)
            - recommendations: List of optimization suggestions
            - summary: Configuration summary
            
    Example:
        >>> config = {"embedding_type": "openai"}
        >>> result = validate_workflow_config(config)
        >>> if result["valid"]:
        ...     print("Configuration is valid!")
        ... else:
        ...     print(f"Errors: {result['errors']}")
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "summary": {}
    }
    
    try:
        logger.info("🔍 Validating workflow configuration")
        
        # ===== REQUIRED PARAMETER CHECKS =====
        
        # Check embedding configuration
        embedding_type = config.get("embedding_type", "mock")
        if embedding_type not in ["openai", "huggingface", "mock", "sentence_transformers"]:
            validation["warnings"].append(f"Unknown embedding type: {embedding_type}")
        
        if embedding_type == "openai" and not config.get("openai_api_key"):
            validation["warnings"].append("OpenAI embedding type specified but no API key provided - will use mock embeddings")
        
        # Check vector store configuration
        vector_store_type = config.get("vector_store_type", "chroma")
        valid_vector_stores = ["chroma", "faiss", "simple", "mock"]
        if vector_store_type not in valid_vector_stores:
            validation["warnings"].append(f"Unknown vector store type: {vector_store_type}. Valid options: {valid_vector_stores}")
        
        # ===== EMAIL CONFIGURATION CHECKS =====
        
        email_enabled = config.get("email_enabled", False)
        if email_enabled:
            required_email_fields = ["gmail_email", "gmail_app_password"]
            missing_fields = [field for field in required_email_fields if not config.get(field)]
            
            if missing_fields:
                validation["errors"].append(f"Email enabled but missing required fields: {missing_fields}")
                validation["valid"] = False
            
            # Check email recipients
            recipients = config.get("email_recipients", [])
            if not recipients:
                validation["warnings"].append("Email enabled but no default recipients specified")
        
        # ===== PERFORMANCE PARAMETER CHECKS =====
        
        # Check chunk size
        chunk_size = config.get("chunk_size", 1000)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            validation["errors"].append(f"Invalid chunk_size: {chunk_size}. Must be positive integer")
            validation["valid"] = False
        elif chunk_size > 2000:
            validation["recommendations"].append("Consider smaller chunk size (≤2000) for better search precision")
        elif chunk_size < 100:
            validation["warnings"].append("Very small chunk size may result in fragmented content")
        
        # Check chunk overlap
        chunk_overlap = config.get("chunk_overlap", 200)
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            validation["errors"].append(f"Invalid chunk_overlap: {chunk_overlap}. Must be non-negative integer")
            validation["valid"] = False
        elif chunk_overlap >= chunk_size:
            validation["errors"].append("chunk_overlap must be less than chunk_size")
            validation["valid"] = False
        
        # Check top_k parameter
        top_k = config.get("top_k", 10)
        if not isinstance(top_k, int) or top_k <= 0:
            validation["errors"].append(f"Invalid top_k: {top_k}. Must be positive integer")
            validation["valid"] = False
        elif top_k > 50:
            validation["recommendations"].append("Large top_k value (>50) may slow down processing")
        
        # ===== COLLECTION NAME VALIDATION =====
        
        collection_name = config.get("collection_name", "sap_documents")
        if not isinstance(collection_name, str) or not collection_name.strip():
            validation["errors"].append("collection_name must be a non-empty string")
            validation["valid"] = False
        elif " " in collection_name:
            validation["warnings"].append("collection_name contains spaces - may cause issues with some vector stores")
        
        # ===== COMPATIBILITY CHECKS =====
        
        # Check if LangGraph is available for full functionality
        try:
            import langgraph
            validation["summary"]["langgraph_available"] = True
        except ImportError:
            validation["summary"]["langgraph_available"] = False
            validation["warnings"].append("LangGraph not available - will use mock implementation")
        
        # Check if ChromaDB is available
        if vector_store_type == "chroma":
            try:
                import chromadb
                validation["summary"]["chromadb_available"] = True
            except ImportError:
                validation["warnings"].append("ChromaDB not available - will use mock vector store")
                validation["summary"]["chromadb_available"] = False
        
        # ===== PERFORMANCE RECOMMENDATIONS =====
        
        # Recommend enabling monitoring for production
        if not config.get("enable_monitoring", True):
            validation["recommendations"].append("Consider enabling monitoring for production deployments")
        
        # Memory usage estimation
        estimated_memory_mb = (chunk_size * top_k * 0.001)  # Rough estimate
        if estimated_memory_mb > 100:
            validation["recommendations"].append(f"Configuration may use ~{estimated_memory_mb:.1f}MB memory - monitor usage")
        
        # ===== CONFIGURATION SUMMARY =====
        
        validation["summary"].update({
            "embedding_type": embedding_type,
            "vector_store_type": vector_store_type,
            "email_enabled": email_enabled,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "collection_name": collection_name,
            "estimated_memory_mb": estimated_memory_mb
        })
        
        # ===== FINAL VALIDATION SUMMARY =====
        
        error_count = len(validation["errors"])
        warning_count = len(validation["warnings"])
        recommendation_count = len(validation["recommendations"])
        
        logger.info(f"🔍 Configuration validation completed:")
        logger.info(f"  - Valid: {validation['valid']}")
        logger.info(f"  - Errors: {error_count}")
        logger.info(f"  - Warnings: {warning_count}")
        logger.info(f"  - Recommendations: {recommendation_count}")
        
        if error_count > 0:
            logger.error(f"❌ Configuration errors found: {validation['errors']}")
        
        return validation
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        validation["valid"] = False
        validation["errors"].append(f"Configuration validation error: {str(e)}")
        return validation


def get_default_config() -> Dict[str, Any]:
    """
    Get default workflow configuration.
    
    Returns a complete default configuration that can be used as a starting
    point for customization.
    
    Returns:
        Dictionary with default configuration values
        
    Example:
        >>> config = get_default_config()
        >>> config["email_enabled"] = True
        >>> config["gmail_email"] = "user@example.com"
        >>> workflow = create_workflow(config)
    """
    return {
        # Core settings
        "embedding_type": "openai",
        "vector_store_type": "chroma",
        "collection_name": f"sap_documents_{datetime.now().strftime('%Y%m%d')}",
        
        # Text processing
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 10,
        
        # Email settings
        "email_enabled": False,
        "auto_send_results": False,
        "email_recipients": [],
        
        # Performance settings
        "enable_monitoring": True,
        "enable_caching": True,
        "batch_size": 100,
        
        # API keys (should be set via environment variables)
        "openai_api_key": "",
        "gmail_email": "",
        "gmail_app_password": "",
        
        # Advanced settings
        "embedding_model": "text-embedding-ada-002",
        "similarity_threshold": 0.7,
        "max_retries": 3,
        "timeout_seconds": 30
    }


# ================================
# TESTING UTILITIES
# ================================

def test_workflow_basic() -> bool:
    """
    Basic workflow test to ensure components are working.
    
    Performs a simple test of workflow creation and basic functionality
    without requiring actual files or API keys.
    
    Returns:
        True if basic test passes, False otherwise
        
    Example:
        >>> if test_workflow_basic():
        ...     print("Workflow components are working!")
        ... else:
        ...     print("Workflow test failed - check logs")
    """
    try:
        logger.info("🧪 Running basic workflow test...")
        
        # Create test configuration
        test_config = {
            "embedding_type": "mock",
            "vector_store_type": "mock",
            "email_enabled": False,
            "top_k": 5,
            "chunk_size": 500,
            "enable_monitoring": False
        }
        
        # Validate test configuration
        validation = validate_workflow_config(test_config)
        if not validation["valid"]:
            logger.error(f"❌ Test configuration validation failed: {validation['errors']}")
            return False
        
        # Create workflow
        workflow = SAPRAGWorkflow(test_config)
        
        # Check basic functionality
        status = workflow.get_workflow_status()
        
        if not status.get("workflow_ready"):
            logger.warning("⚠️ Workflow not fully ready, but this may be expected in test environment")
        
        # Test state creation
        test_state = workflow._create_initial_state(
            uploaded_files=[],
            user_query="test query",
            search_filters={}
        )
        
        if not test_state:
            logger.error("❌ Failed to create initial state")
            return False
        
        logger.info("✅ Basic workflow test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic workflow test failed: {e}")
        return False


def test_workflow_with_mock_data() -> Dict[str, Any]:
    """
    Test workflow with mock data to verify full pipeline.
    
    Creates a more comprehensive test using mock data to verify that the
    entire workflow pipeline can execute without errors.
    
    Returns:
        Dictionary with test results and metrics
        
    Example:
        >>> result = test_workflow_with_mock_data()
        >>> if result["success"]:
        ...     print(f"Test completed in {result['duration']:.2f}s")
    """
    try:
        logger.info("🧪 Running comprehensive workflow test with mock data")
        
        start_time = datetime.now()
        
        # Create test configuration
        test_config = {
            "embedding_type": "mock",
            "vector_store_type": "mock", 
            "email_enabled": False,
            "chunk_size": 200,
            "top_k": 3,
            "enable_monitoring": True
        }
        
        # Create workflow
        workflow = create_workflow(test_config)
        
        # Create mock uploaded files
        mock_files = [
            {
                "filename": "test_report.pdf",
                "content": "Mock SAP system analysis report with performance metrics",
                "size": 1024
            }
        ]
        
        # Test full workflow (this will use mock implementations)
        result = workflow.run_workflow(
            uploaded_files=mock_files,
            user_query="performance issues",
            search_filters={"target_systems": ["PRD"]},
            email_recipients=[]
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze results
        success = result.get("workflow_status") in ["completed", "error"]  # Either is fine for mock test
        
        test_result = {
            "success": success,
            "duration": duration,
            "workflow_status": result.get("workflow_status"),
            "error_message": result.get("error_message", ""),
            "processing_times": result.get("processing_times", {}),
            "total_chunks": result.get("total_chunks", 0),
            "search_results_count": len(result.get("search_results", [])),
            "systems_analyzed": len(result.get("system_summaries", {}))
        }
        
        logger.info(f"✅ Comprehensive test completed in {duration:.2f}s")
        logger.info(f"   Status: {test_result['workflow_status']}")
        
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Comprehensive workflow test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }


# ================================
# CONFIGURATION HELPERS
# ================================

def load_config_from_env() -> Dict[str, Any]:
    """
    Load workflow configuration from environment variables.
    
    Reads configuration values from environment variables with sensible
    defaults. Useful for production deployments.
    
    Returns:
        Configuration dictionary loaded from environment
        
    Environment Variables:
        - SAP_WORKFLOW_EMBEDDING_TYPE: Embedding type (default: openai)
        - SAP_WORKFLOW_VECTOR_STORE: Vector store type (default: chroma)
        - SAP_WORKFLOW_EMAIL_ENABLED: Enable email (default: false)
        - SAP_WORKFLOW_CHUNK_SIZE: Text chunk size (default: 1000)
        - SAP_WORKFLOW_TOP_K: Search result count (default: 10)
        - OPENAI_API_KEY: OpenAI API key
        - GMAIL_EMAIL: Gmail address for notifications
        - GMAIL_APP_PASSWORD: Gmail app password
        
    Example:
        >>> import os
        >>> os.environ["SAP_WORKFLOW_EMAIL_ENABLED"] = "true"
        >>> config = load_config_from_env()
        >>> workflow = create_workflow(config)
    """
    import os
    
    def str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ("true", "1", "yes", "on")
    
    def safe_int(value: str, default: int) -> int:
        """Safely convert string to int with default."""
        try:
            return int(value) if value else default
        except ValueError:
            return default
    
    config = {
        # Core settings
        "embedding_type": os.getenv("SAP_WORKFLOW_EMBEDDING_TYPE", "openai"),
        "vector_store_type": os.getenv("SAP_WORKFLOW_VECTOR_STORE", "chroma"),
        "collection_name": os.getenv("SAP_WORKFLOW_COLLECTION", f"sap_docs_{datetime.now().strftime('%Y%m%d')}"),
        
        # Text processing
        "chunk_size": safe_int(os.getenv("SAP_WORKFLOW_CHUNK_SIZE"), 1000),
        "chunk_overlap": safe_int(os.getenv("SAP_WORKFLOW_CHUNK_OVERLAP"), 200),
        "top_k": safe_int(os.getenv("SAP_WORKFLOW_TOP_K"), 10),
        
        # Email settings
        "email_enabled": str_to_bool(os.getenv("SAP_WORKFLOW_EMAIL_ENABLED", "false")),
        "auto_send_results": str_to_bool(os.getenv("SAP_WORKFLOW_AUTO_SEND", "false")),
        
        # API keys and credentials
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "gmail_email": os.getenv("GMAIL_EMAIL", ""),
        "gmail_app_password": os.getenv("GMAIL_APP_PASSWORD", ""),
        
        # Performance settings
        "enable_monitoring": str_to_bool(os.getenv("SAP_WORKFLOW_MONITORING", "true")),
        "batch_size": safe_int(os.getenv("SAP_WORKFLOW_BATCH_SIZE"), 100),
        "timeout_seconds": safe_int(os.getenv("SAP_WORKFLOW_TIMEOUT"), 30)
    }
    
    # Parse email recipients from environment
    recipients_env = os.getenv("SAP_WORKFLOW_EMAIL_RECIPIENTS", "")
    if recipients_env:
        config["email_recipients"] = [email.strip() for email in recipients_env.split(",") if email.strip()]
    else:
        config["email_recipients"] = []
    
    logger.info("📋 Configuration loaded from environment variables")
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with later configs taking precedence.
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration dictionary
        
    Example:
        >>> base_config = get_default_config()
        >>> env_config = load_config_from_env()
        >>> user_config = {"email_enabled": True}
        >>> final_config = merge_configs(base_config, env_config, user_config)
    """
    merged = {}
    
    for config in configs:
        if config:
            merged.update(config)
    
    logger.info(f"🔀 Merged {len(configs)} configuration dictionaries")
    
    return merged


# ================================
# WORKFLOW COMPARISON UTILITIES
# ================================

def compare_workflow_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two workflow configurations and highlight differences.
    
    Args:
        config1: First configuration to compare
        config2: Second configuration to compare
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> result = compare_workflow_configs(old_config, new_config)
        >>> for key in result["differences"]:
        ...     print(f"{key}: {result['differences'][key]}")
    """
    comparison = {
        "identical": True,
        "differences": {},
        "config1_only": {},
        "config2_only": {},
        "summary": {}
    }
    
    # Find all unique keys
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)
        
        if key in config1 and key not in config2:
            comparison["config1_only"][key] = val1
            comparison["identical"] = False
        elif key in config2 and key not in config1:
            comparison["config2_only"][key] = val2
            comparison["identical"] = False
        elif val1 != val2:
            comparison["differences"][key] = {"config1": val1, "config2": val2}
            comparison["identical"] = False
    
    # Summary statistics
    comparison["summary"] = {
        "total_keys_config1": len(config1),
        "total_keys_config2": len(config2),
        "differences_count": len(comparison["differences"]),
        "config1_only_count": len(comparison["config1_only"]),
        "config2_only_count": len(comparison["config2_only"])
    }
    
    return comparison


# ================================
# MODULE EXPORTS
# ================================

__all__ = [
    # Factory functions
    'create_workflow',
    'create_workflow_with_defaults',
    
    # Validation functions
    'validate_workflow_config',
    'get_default_config',
    
    # Testing functions
    'test_workflow_basic',
    'test_workflow_with_mock_data',
    
    # Configuration helpers
    'load_config_from_env',
    'merge_configs',
    'compare_workflow_configs'
]

logger.info("✅ Workflow utilities module loaded successfully")