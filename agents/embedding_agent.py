# agents/embedding_agent.py - Embedding Creation Agent
"""
Embedding agent for creating vector embeddings from processed text documents.

This agent handles:
- Text chunking with configurable size and overlap
- Vector embedding generation using OpenAI or fallback models
- Batch processing for efficiency
- Embedding validation and formatting
"""

import time
import random
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent


class EmbeddingAgent(BaseAgent):
    """
    Second agent in the workflow pipeline - creates vector embeddings from text.
    
    Key Features:
    - Intelligent text chunking with overlap
    - OpenAI embedding integration with fallbacks
    - Batch processing for API efficiency
    - Chunk-embedding alignment validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding agent with chunking and embedding configuration"""
        super().__init__("EmbeddingCreator", config)
        
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.embedding_model = config.get('embedding_model', 'text-embedding-ada-002')
        self.batch_size = config.get('embedding_batch_size', 10)
        
        self.embedding_client = self._initialize_embedding_client()
    
    def _initialize_embedding_client(self):
        """Initialize OpenAI embedding client with fallback handling"""
        try:
            from openai import OpenAI
            api_key = self.config.get('openai_api_key')
            
            if api_key:
                client = OpenAI(api_key=api_key)
                self.log_info(f"OpenAI embedding client initialized with model: {self.embedding_model}")
                return client
            else:
                self.log_warning("No OpenAI API key provided, will use mock embeddings")
                return None
                
        except ImportError:
            self.log_warning("OpenAI library not available, will use mock embeddings")
            return None
        except Exception as e:
            self.log_error(f"Failed to initialize embedding client: {e}")
            return None
    
    def process(self, processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main processing method for creating embeddings.
        
        Args:
            processed_files: List of file data from PDFProcessorAgent
            
        Returns:
            Dict with chunks and embeddings or error information
        """
        self.start_timer()
        
        try:
            self.log_info(f"Creating embeddings for {len(processed_files)} processed files")
            
            if not processed_files:
                return self.handle_error(ValueError("No processed files provided"), "Embedding Creation")
            
            all_chunks = []
            all_embeddings = []
            
            # Process each file individually
            for file_data in processed_files:
                try:
                    self.log_info(f"Processing embeddings for: {file_data.get('filename', 'unknown')}")
                    
                    # Create text chunks
                    file_chunks = self._create_text_chunks(
                        text=file_data.get('text', ''),
                        filename=file_data.get('filename', 'unknown'),
                        metadata=file_data
                    )
                    
                    if not file_chunks:
                        self.log_warning(f"No chunks created for {file_data.get('filename')}")
                        continue
                    
                    # Create embeddings for chunks
                    chunk_embeddings = self._create_embeddings_for_chunks(file_chunks)
                    
                    all_chunks.extend(file_chunks)
                    all_embeddings.extend(chunk_embeddings)
                    
                    self.log_info(f"✅ Created {len(file_chunks)} chunks and {len(chunk_embeddings)} embeddings")
                    
                except Exception as e:
                    self.log_error(f"Failed to process embeddings for {file_data.get('filename', 'unknown')}: {e}")
                    continue
            
            # Validate chunk-embedding alignment
            if len(all_chunks) != len(all_embeddings):
                self.log_warning(f"Chunk count ({len(all_chunks)}) != embedding count ({len(all_embeddings)})")
                min_count = min(len(all_chunks), len(all_embeddings))
                all_chunks = all_chunks[:min_count]
                all_embeddings = all_embeddings[:min_count]
            
            processing_time = self.end_timer("embedding_creation")
            
            self.log_info(f"Embedding creation completed: {len(all_chunks)} chunks, {len(all_embeddings)} embeddings")
            
            return {
                "success": True,
                "chunks": all_chunks,
                "embeddings": all_embeddings,
                "chunk_count": len(all_chunks),
                "embedding_count": len(all_embeddings),
                "processing_time": processing_time
            }
            
        except Exception as e:
            return self.handle_error(e, "Embedding Creation")
    
    def _create_text_chunks(self, text: str, filename: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks suitable for embedding"""
        if not text or len(text.strip()) < self.min_chunk_size:
            self.log_warning(f"Text too short for chunking: {len(text) if text else 0} chars")
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_data = self._create_chunk_data(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    filename=filename,
                    metadata=metadata
                )
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_data = self._create_chunk_data(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                filename=filename,
                metadata=metadata
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunk boundaries"""
        import re
        
        sentences = re.split(r'([.!?]+\s+)', text)
        
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if sentence.strip():
                result.append(sentence.strip())
        
        return result
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last portion of text for chunk overlap"""
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        first_space = overlap_text.find(' ')
        
        if first_space > 0:
            return overlap_text[first_space:].strip()
        else:
            return overlap_text
    
    def _create_chunk_data(self, text: str, chunk_id: int, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk data structure with text and metadata"""
        return {
            'text': text,
            'page_content': text,
            'chunk_id': chunk_id,
            'source': filename,
            'character_count': len(text),
            'word_count': len(text.split()),
            'metadata': {
                'source': filename,
                'chunk_id': chunk_id,
                'character_count': len(text),
                'word_count': len(text.split()),
                'original_file_size': metadata.get('size', 0),
                'processing_timestamp': datetime.now().isoformat(),
                **metadata
            }
        }
    
    def _create_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Create vector embeddings for a list of text chunks"""
        if not chunks:
            return []
        
        if not self.embedding_client:
            return self._create_mock_embeddings(len(chunks))
        
        all_embeddings = []
        
        # Process in batches for API efficiency
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            try:
                self.log_info(f"Creating embeddings for batch {i//self.batch_size + 1}")
                
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                self.log_info(f"✅ Successfully created {len(batch_embeddings)} embeddings")
                
                # Rate limiting
                if i + self.batch_size < len(chunks):
                    time.sleep(0.1)
                    
            except Exception as e:
                self.log_error(f"Failed to create embeddings for batch {i//self.batch_size + 1}: {e}")
                mock_embeddings = self._create_mock_embeddings(len(batch))
                all_embeddings.extend(mock_embeddings)
        
        return all_embeddings
    
    def _create_mock_embeddings(self, count: int, dimension: int = 1536) -> List[List[float]]:
        """Create mock embeddings for testing"""
        self.log_warning(f"Creating {count} mock embeddings (dimension: {dimension})")
        
        embeddings = []
        for i in range(count):
            vector = [random.gauss(0, 1) for _ in range(dimension)]
            magnitude = sum(x**2 for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x / magnitude for x in vector]
            embeddings.append(vector)
        
        return embeddings