# vector_store.py - ChromaDB Vector Store Implementation
"""
Clean ChromaDB vector store implementation for SAP EWA Analyzer.
Provides efficient document storage and similarity search capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

@dataclass
class VectorConfig:
    """Configuration for ChromaDB vector store"""
    collection_name: str = "sap_documents"
    persist_directory: str = "./chroma_db"
    embedding_function: str = "sentence-transformers"  # or "openai"
    model_name: str = "all-MiniLM-L6-v2"
    distance_metric: str = "cosine"
    max_results: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChromaVectorStore:
    """ChromaDB-based vector store for document storage and retrieval"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_client()
        self._setup_embedding_function()
        self._get_or_create_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB persistent client"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at {self.config.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _setup_embedding_function(self):
        """Setup the embedding function based on configuration"""
        try:
            if self.config.embedding_function == "openai":
                # Use OpenAI embeddings if API key is available
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("No OpenAI API key found, falling back to SentenceTransformers")
                    self.config.embedding_function = "sentence-transformers"
                else:
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name=self.config.model_name or "text-embedding-ada-002"
                    )
                    logger.info(f"OpenAI embedding function setup: {self.config.model_name}")
                    return
            
            # Default to SentenceTransformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.model_name
            )
            logger.info(f"SentenceTransformer embedding function setup: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup embedding function: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {self.config.collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of unique IDs for each document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{hash(doc) % 100000}" for i, doc in enumerate(documents)]
            
            # Ensure metadatas is provided
            if metadatas is None:
                metadatas = [{"source": f"doc_{i}", "text_length": len(doc)} for i, doc in enumerate(documents)]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: Optional[int] = None,
                         where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            where: Filter conditions
            
        Returns:
            List of search results with documents, scores, and metadata
        """
        try:
            k = k or self.config.max_results
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'score': results['distances'][0][i] if results['distances'] else None,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Search for similar documents and return (document, score) tuples
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (document_object, distance_score) tuples
        """
        try:
            k = k or self.config.max_results
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Create document objects and return tuples
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    # Create a document-like object
                    document = DocumentLike(
                        page_content=doc_text,
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                    score = results['distances'][0][i] if results['distances'] else 0.0
                    formatted_results.append((document, score))
            
            logger.info(f"Found {len(formatted_results)} documents with scores")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def delete_collection(self) -> bool:
        """Delete the current collection"""
        try:
            self.client.delete_collection(name=self.config.collection_name)
            logger.info(f"Deleted collection: {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """Peek at some documents in the collection"""
        try:
            return self.collection.peek(limit=limit)
        except Exception as e:
            logger.error(f"Failed to peek at collection: {e}")
            return {}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            return {
                'name': self.collection.name,
                'count': self.collection.count(),
                'metadata': self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


class DocumentLike:
    """Simple document-like object for compatibility with LangChain"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata
    
    def __str__(self):
        return self.page_content


# Convenience functions for easy setup
def create_vector_store(collection_name: str = "sap_documents",
                       persist_directory: str = "./chroma_db",
                       embedding_model: str = "all-MiniLM-L6-v2",
                       use_openai: bool = False) -> ChromaVectorStore:
    """
    Create a ChromaDB vector store with specified settings
    
    Args:
        collection_name: Name for the document collection
        persist_directory: Directory to store the database
        embedding_model: Model name for embeddings
        use_openai: Whether to use OpenAI embeddings (requires API key)
        
    Returns:
        Configured ChromaVectorStore instance
    """
    config = VectorConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function="openai" if use_openai else "sentence-transformers",
        model_name="text-embedding-ada-002" if use_openai else embedding_model
    )
    return ChromaVectorStore(config)


def create_openai_vector_store(collection_name: str = "sap_documents_openai",
                              persist_directory: str = "./chroma_db") -> ChromaVectorStore:
    """
    Create a ChromaDB vector store specifically configured for OpenAI embeddings
    
    Args:
        collection_name: Name for the document collection
        persist_directory: Directory to store the database
        
    Returns:
        ChromaVectorStore configured for OpenAI embeddings
    """
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("No OpenAI API key found, falling back to SentenceTransformers")
        return create_vector_store(collection_name, persist_directory)
    
    config = VectorConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function="openai",
        model_name="text-embedding-ada-002"
    )
    return ChromaVectorStore(config)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Testing ChromaDB Vector Store...")
    
    # Create vector store
    vector_store = create_vector_store("test_collection")
    
    # Add some test documents
    test_documents = [
        "ChromaDB is a vector database for AI applications",
        "SAP EWA provides system health monitoring and analysis",
        "Vector search enables semantic document retrieval"
    ]
    
    test_metadatas = [
        {"category": "database", "type": "description"},
        {"category": "sap", "type": "monitoring"},
        {"category": "search", "type": "technology"}
    ]
    
    # Add documents
    success = vector_store.add_documents(test_documents, test_metadatas)
    if success:
        print(f"✅ Added {len(test_documents)} test documents")
        
        # Search for similar documents
        results = vector_store.similarity_search("What is ChromaDB?", k=2)
        
        print("\n🔍 Search Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Document: {result['document'][:50]}...")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Metadata: {result['metadata']}")
            print()
        
        # Get collection info
        info = vector_store.get_collection_info()
        print(f"📊 Collection Info: {info}")
    else:
        print("❌ Failed to add test documents")