# vector_store.py
import os
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain.schema import Document
import logging

# Safe imports with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available")

# Safe config import
def get_config_value(key: str, default: Any = None):
    """Safely get config values"""
    try:
        from configorig import Config
        return getattr(Config, key, default)
    except (ImportError, AttributeError):
        return default

logger = logging.getLogger(__name__)

# ================================
# BASE VECTOR STORE
# ================================
class BaseVectorStore:
    """Base class for vector stores"""
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray] = None):
        raise NotImplementedError
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        raise NotImplementedError
    
    def similarity_search(self, query: str, k: int = 10, filter: Dict = None) -> List[Document]:
        results = self.similarity_search_with_score(query, k, filter)
        return [doc for doc, _ in results]

# ================================
# DUMMY VECTOR STORE (FALLBACK)
# ================================
class DummyVectorStore(BaseVectorStore):
    """Fallback vector store for testing"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray] = None):
        """Add documents to dummy store"""
        self.documents.extend(documents)
        if embeddings:
            self.embeddings.extend(embeddings)
        logger.info(f"Added {len(documents)} documents to dummy vector store")
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Dummy search that returns stored documents with fake scores"""
        if not self.documents:
            return []
        
        # Return up to k documents with decreasing fake scores
        results = []
        for i, doc in enumerate(self.documents[:k]):
            # Create fake similarity score
            score = 0.9 - (i * 0.1)
            score = max(score, 0.5)
            results.append((doc, score))
        
        return results

# ================================
# CHROMA VECTOR STORE
# ================================
class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation"""
    
    def __init__(self, collection_name: str = "sap_documents"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = get_config_value('CHROMA_PATH', './chroma_db')
        
        # Create directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Chroma
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            
            # Create or get collection
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new Chroma collection: {collection_name}")
            except Exception:
                # Collection exists
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Using existing Chroma collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray] = None):
        """Add documents to Chroma"""
        try:
            if not documents:
                return
            
            # Prepare data
            timestamp = int(datetime.now().timestamp())
            ids = [f"doc_{i}_{timestamp}" for i in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]
            
            if embeddings:
                embeddings_list = [
                    emb.tolist() if isinstance(emb, np.ndarray) else emb 
                    for emb in embeddings
                ]
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings_list
                )
            else:
                # Let Chroma generate embeddings
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas, 
                    ids=ids
                )
            
            logger.info(f"Added {len(documents)} documents to Chroma")
            
        except Exception as e:
            logger.error(f"Chroma add error: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search with scores"""
        try:
            where_clause = self._build_where_clause(filter) if filter else None
            
            # Get collection count safely
            try:
                collection_count = self.collection.count()
            except:
                collection_count = 1000  # Default fallback
            
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, max(1, collection_count)),
                where=where_clause
            )
            
            documents_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'][0] else {}
                    )
                    # Convert distance to similarity
                    distance = results['distances'][0][i] if results['distances'] else 0.5
                    similarity_score = max(0, 1 - distance)
                    documents_with_scores.append((doc, similarity_score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Chroma search error: {str(e)}")
            return []
    
    def _build_where_clause(self, filter: Dict) -> Dict:
        """Build Chroma where clause"""
        where_clause = {}
        for key, value in filter.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = {"$eq": value}
        return where_clause
    
    def delete_collection(self):
        """Delete collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted Chroma collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")

# ================================
# FAISS VECTOR STORE
# ================================
class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, embedding_dimension: int = 1536):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.embedding_dimension = embedding_dimension
        self.persist_directory = get_config_value('FAISS_PATH', './faiss_db')
        self.index_file = os.path.join(self.persist_directory, "index.faiss")
        self.metadata_file = os.path.join(self.persist_directory, "metadata.pkl")
        
        # Create directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner product
        self.documents = []
        self.metadatas = []
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing FAISS data"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load index
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                
                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Could not load FAISS data: {str(e)}")
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents to FAISS"""
        try:
            if not embeddings:
                raise ValueError("FAISS requires embeddings")
            
            # Normalize embeddings
            embedding_matrix = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embedding_matrix)
            
            # Add to index
            self.index.add(embedding_matrix)
            
            # Store documents
            self.documents.extend(documents)
            self.metadatas.extend([doc.metadata for doc in documents])
            
            # Persist
            self._persist_data()
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            
        except Exception as e:
            logger.error(f"FAISS add error: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search using query (requires pre-computed embedding)"""
        logger.warning("FAISS similarity_search_with_score requires pre-computed embeddings")
        return []
    
    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search using pre-computed embedding"""
        try:
            if len(self.documents) == 0:
                return []
            
            # Normalize query
            query_emb = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_emb)
            
            # Search
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_emb, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Apply filter
                    if filter and not self._matches_filter(doc.metadata, filter):
                        continue
                    
                    results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return []
    
    def _matches_filter(self, metadata: Dict, filter: Dict) -> bool:
        """Check if metadata matches filter"""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def _persist_data(self):
        """Save FAISS data"""
        try:
            # Save index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            data = {
                'documents': self.documents,
                'metadatas': self.metadatas
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"FAISS persist error: {str(e)}")
    
    def delete_collection(self):
        """Delete FAISS data"""
        try:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents = []
            self.metadatas = []
            
        except Exception as e:
            logger.error(f"FAISS delete error: {str(e)}")

# ================================
# SIMPLE NUMPY VECTOR STORE
# ================================
class SimpleVectorStore(BaseVectorStore):
    """Simple vector store using numpy operations"""
    
    def __init__(self):
        self.persist_directory = get_config_value('SIMPLE_STORE_PATH', './simple_store')
        self.embeddings_file = os.path.join(self.persist_directory, "embeddings.npy")
        self.documents_file = os.path.join(self.persist_directory, "documents.pkl")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.embeddings_matrix = None
        self.documents = []
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing data"""
        try:
            if os.path.exists(self.embeddings_file) and os.path.exists(self.documents_file):
                self.embeddings_matrix = np.load(self.embeddings_file)
                
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded Simple store with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Could not load Simple store data: {str(e)}")
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents"""
        try:
            if not embeddings:
                # For testing, create dummy embeddings
                embeddings = [np.random.rand(1536) for _ in documents]
            
            new_embeddings = np.array(embeddings)
            
            if self.embeddings_matrix is None:
                self.embeddings_matrix = new_embeddings
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
            
            self.documents.extend(documents)
            
            # Persist
            self._persist_data()
            
            logger.info(f"Added {len(documents)} documents to Simple store")
            
        except Exception as e:
            logger.error(f"Simple store add error: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search (requires pre-computed embedding)"""
        # For testing, return dummy results
        if not self.documents:
            return []
        
        results = []
        for i, doc in enumerate(self.documents[:k]):
            # Apply filter
            if filter and not self._matches_filter(doc.metadata, filter):
                continue
            
            # Fake similarity score
            score = 0.9 - (i * 0.05)
            score = max(score, 0.5)
            results.append((doc, score))
        
        return results
    
    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 10, filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search using embedding"""
        try:
            if self.embeddings_matrix is None or len(self.documents) == 0:
                return []
            
            # Calculate similarities using cosine similarity
            if SKLEARN_AVAILABLE:
                similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
            else:
                # Manual cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                embeddings_norm = np.linalg.norm(self.embeddings_matrix, axis=1)
                similarities = np.dot(self.embeddings_matrix, query_embedding) / (embeddings_norm * query_norm + 1e-8)
            
            # Get top-k
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_k_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Apply filter
                    if filter and not self._matches_filter(doc.metadata, filter):
                        continue
                    
                    score = float(similarities[idx])
                    results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Simple store search error: {str(e)}")
            return []
    
    def _matches_filter(self, metadata: Dict, filter: Dict) -> bool:
        """Check filter match"""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def _persist_data(self):
        """Save data"""
        try:
            if self.embeddings_matrix is not None:
                np.save(self.embeddings_file, self.embeddings_matrix)
            
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.error(f"Simple store persist error: {str(e)}")
    
    def delete_collection(self):
        """Delete data"""
        try:
            if os.path.exists(self.embeddings_file):
                os.remove(self.embeddings_file)
            if os.path.exists(self.documents_file):
                os.remove(self.documents_file)
            
            self.embeddings_matrix = None
            self.documents = []
            
        except Exception as e:
            logger.error(f"Simple store delete error: {str(e)}")

# ================================
# VECTOR STORE MANAGER
# ================================
class VectorStoreManager:
    """Factory for vector stores with fallbacks"""
    
    def __init__(self, store_type: str = "chroma"):
        self.store_type = store_type.lower()
        self.vector_store = None
        
        # Determine available store type
        if self.store_type == "chroma" and not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, falling back to simple store")
            self.store_type = "simple"
        elif self.store_type == "faiss" and not FAISS_AVAILABLE:
            logger.warning("FAISS not available, falling back to simple store")
            self.store_type = "simple"
    
    def create_vector_store(self, documents: List[Document], embeddings: List[np.ndarray] = None) -> BaseVectorStore:
        """Create and populate vector store"""
        try:
            # Initialize store
            if self.store_type == "chroma" and CHROMADB_AVAILABLE:
                self.vector_store = ChromaVectorStore()
            elif self.store_type == "faiss" and FAISS_AVAILABLE:
                embedding_dim = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 1536
                self.vector_store = FAISSVectorStore(embedding_dim)
            elif self.store_type == "simple":
                self.vector_store = SimpleVectorStore()
            else:
                # Ultimate fallback
                logger.warning("Using dummy vector store")
                self.vector_store = DummyVectorStore()
            
            # Add documents
            if documents:
                self.vector_store.add_documents(documents, embeddings)
            
            logger.info(f"Created {self.store_type} vector store with {len(documents)} documents")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            # Final fallback
            logger.warning("Using dummy vector store as final fallback")
            self.vector_store = DummyVectorStore()
            if documents:
                self.vector_store.add_documents(documents, embeddings)
            return self.vector_store
    
    def get_vector_store(self) -> BaseVectorStore:
        """Get current vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store
    
    def clear_vector_store(self):
        """Clear vector store"""
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None