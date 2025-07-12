# agents/search_agent.py - Search Agent
"""
Search agent for performing similarity search on the vector store.

This agent handles:
- Vector similarity search using ChromaDB
- Query processing and optimization
- Result filtering and ranking
- System ID detection from documents
"""

import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .base_agent import BaseAgent


class SearchAgent(BaseAgent):
    """
    Agent responsible for performing similarity search on the vector store.
    
    Key Features:
    - ChromaDB vector similarity search
    - System ID detection and filtering
    - Result validation and ranking
    - Mock search fallbacks for development
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize search agent with vector store and configuration"""
        super().__init__("SearchAgent", config)
        
        self.vector_store = config.get('vector_store')
        self.embedding_agent = config.get('embedding_agent')
        self.top_k = config.get('top_k', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # System ID detection patterns
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',
            r'\bSID[:\s]+([A-Z0-9]{2,4})\b',
            r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\bFOR\s+([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\b([A-Z]{1,3}[0-9]{1,2})\b',
            r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b',
        ]
        
        self.false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL',
            'CPU', 'RAM', 'GB', 'MB', 'KB', 'HTTP', 'URL', 'API', 'GUI', 'UI', 'DB'
        }
        
        self.log_info(f"SearchAgent initialized with vector store: {type(self.vector_store).__name__ if self.vector_store else 'None'}")
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            filters: Optional search filters (e.g., target_systems)
            
        Returns:
            Dict with search results or error information
        """
        self.start_timer()
        
        try:
            self.log_info(f"Performing search for query: '{query}'")
            
            if not query or not query.strip():
                return self.handle_error(ValueError("Empty search query provided"), "Search")
            
            if not self.vector_store:
                self.log_warning("No vector store available, using mock search")
                search_results = self._perform_mock_search(query, filters)
            else:
                search_results = self._perform_vector_search(query, filters)
            
            validated_results = self._validate_search_results(search_results)
            processing_time = self.end_timer("search")
            
            self.log_info(f"Search completed: {len(validated_results)} results in {processing_time:.2f}s")
            
            return {
                "success": True,
                "query": query,
                "search_results": validated_results,
                "results_count": len(validated_results),
                "processing_time": processing_time,
                "filters_applied": filters or {},
                "search_metadata": {
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "vector_store_type": type(self.vector_store).__name__ if self.vector_store else "Mock"
                }
            }
            
        except Exception as e:
            return self.handle_error(e, "Search")
    
    def _perform_vector_search(self, query: str, filters: Dict[str, Any] = None) -> List[Any]:
        """Perform actual vector similarity search"""
        try:
            target_systems = filters.get('target_systems', []) if filters else []
            
            self.log_info(f"Vector search with target systems: {target_systems}")
            
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            
            search_results = []
            for doc, score in docs_with_scores:
                try:
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    source = metadata.get('source', 'unknown')
                    
                    system_id = self._extract_system_id(content, metadata)
                    
                    # Apply system filtering
                    if target_systems and system_id not in target_systems:
                        self.log_info(f"Filtering out result for system {system_id}")
                        continue
                    
                    # Apply similarity threshold
                    similarity_score = 1.0 - score if score <= 1.0 else score
                    if similarity_score < self.similarity_threshold:
                        self.log_info(f"Filtering out result with low similarity: {similarity_score:.3f}")
                        continue
                    
                    # Create SearchResult
                    try:
                        from models import SearchResult
                        result = SearchResult(
                            content=content,
                            source=source,
                            system_id=system_id,
                            confidence_score=float(similarity_score),
                            metadata=metadata
                        )
                    except ImportError:
                        result = {
                            'content': content,
                            'source': source,
                            'system_id': system_id,
                            'confidence_score': float(similarity_score),
                            'metadata': metadata
                        }
                    
                    search_results.append(result)
                    
                except Exception as doc_error:
                    self.log_error(f"Error processing search result: {doc_error}")
                    continue
            
            self.log_info(f"Vector search returned {len(search_results)} results after filtering")
            return search_results
            
        except Exception as e:
            self.log_error(f"Vector search failed: {e}")
            return self._perform_mock_search(query, filters)
    
    def _perform_mock_search(self, query: str, filters: Dict[str, Any] = None) -> List[Any]:
        """Perform mock search for testing"""
        target_systems = filters.get('target_systems', ['UNKNOWN']) if filters else ['UNKNOWN']
        
        if not target_systems:
            target_systems = ['SYSTEM_01', 'SYSTEM_02']
        
        self.log_info(f"Mock search for target systems: {target_systems}")
        
        mock_results = []
        for i, system in enumerate(target_systems):
            mock_content = self._generate_mock_content(query, system, i)
            
            try:
                from models import SearchResult
                result = SearchResult(
                    content=mock_content,
                    source=f"document_{i+1}.pdf",
                    system_id=system,
                    confidence_score=0.9 - (i * 0.1),
                    metadata={
                        "chunk_id": i,
                        "page": i + 1,
                        "system_id": system,
                        "mock": True,
                        "search_query": query
                    }
                )
            except ImportError:
                result = {
                    'content': mock_content,
                    'source': f"document_{i+1}.pdf",
                    'system_id': system,
                    'confidence_score': 0.9 - (i * 0.1),
                    'metadata': {
                        "chunk_id": i,
                        "page": i + 1,
                        "system_id": system,
                        "mock": True,
                        "search_query": query
                    }
                }
            
            mock_results.append(result)
        
        return mock_results
    
    def _generate_mock_content(self, query: str, system_id: str, index: int) -> str:
        """Generate realistic mock content for search results"""
        templates = [
            f"System {system_id} analysis shows that {query.lower()} requires attention. "
            f"Performance metrics indicate potential optimization opportunities in database operations.",
            
            f"Early Watch Alert for {system_id}: Issues related to {query.lower()} have been detected. "
            f"Recommendations include memory optimization and query tuning.",
            
            f"SAP Basis team report for {system_id} indicates that {query.lower()} monitoring shows "
            f"elevated resource consumption requiring immediate investigation.",
            
            f"Technical analysis of {system_id} reveals {query.lower()} patterns that suggest "
            f"system configuration adjustments may be needed for optimal performance."
        ]
        
        template = templates[index % len(templates)]
        
        if 'performance' in query.lower():
            template += f" CPU utilization in {system_id} is at 85%. Memory usage shows spikes during batch processing."
        elif 'error' in query.lower() or 'critical' in query.lower():
            template += f" Error count in {system_id} has increased by 25% over the past week."
        elif 'recommendation' in query.lower():
            template += f" Recommended actions for {system_id} include index optimization and archiving old data."
        
        return template
    
    def _extract_system_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Extract system ID from document content or metadata"""
        try:
            # Check metadata first
            if 'system_id' in metadata and metadata['system_id']:
                return metadata['system_id'].upper()
            
            # Pattern matching on content
            if content:
                detected_id = self._detect_system_from_content(content)
                if detected_id != 'UNKNOWN':
                    return detected_id
            
            # Check source filename
            source = metadata.get('source', '')
            if source:
                filename_id = self._extract_system_from_filename(source)
                if filename_id != 'UNKNOWN':
                    return filename_id
            
            return 'UNKNOWN'
            
        except Exception as e:
            self.log_error(f"Error extracting system ID: {e}")
            return 'UNKNOWN'
    
    def _detect_system_from_content(self, content: str) -> str:
        """Detect system ID from document content using pattern matching"""
        if not content:
            return 'UNKNOWN'
        
        content_upper = content.upper()
        found_systems = set()
        
        for pattern in self.system_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                
                if (match and 
                    len(match) in [2, 3, 4] and 
                    match not in self.false_positives):
                    found_systems.add(match)
        
        if found_systems:
            result = sorted(list(found_systems))[0]
            return result
        
        return 'UNKNOWN'
    
    def _extract_system_from_filename(self, filename: str) -> str:
        """Extract system ID from filename if present"""
        if not filename:
            return 'UNKNOWN'
        
        filename_upper = filename.upper()
        
        patterns = [
            r'EWA[_\-]([A-Z0-9]{2,4})[_\-]',
            r'([A-Z0-9]{2,4})[_\-](?:EWA|REPORT|ANALYSIS)',
            r'^([A-Z0-9]{2,4})[_\-]',
            r'[_\-]([A-Z0-9]{2,4})[_\-]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_upper)
            if match:
                system_id = match.group(1)
                if system_id not in self.false_positives and len(system_id) in [2, 3, 4]:
                    return system_id
        
        return 'UNKNOWN'
    
    def _validate_search_results(self, results: List[Any]) -> List[Any]:
        """Validate and clean search results"""
        validated = []
        
        for result in results:
            try:
                # Handle both SearchResult objects and dicts
                if hasattr(result, 'content'):
                    content = result.content
                    source = result.source
                    system_id = result.system_id
                    confidence_score = result.confidence_score
                else:
                    content = result.get('content', '')
                    source = result.get('source', 'unknown')
                    system_id = result.get('system_id', 'UNKNOWN')
                    confidence_score = result.get('confidence_score', 0.0)
                
                # Basic validation
                if not content or len(content.strip()) < 10:
                    continue
                
                if not source:
                    if hasattr(result, 'source'):
                        result.source = 'unknown'
                    else:
                        result['source'] = 'unknown'
                
                if not system_id:
                    if hasattr(result, 'system_id'):
                        result.system_id = 'UNKNOWN'
                    else:
                        result['system_id'] = 'UNKNOWN'
                
                # Ensure confidence score is valid
                if not (0.0 <= confidence_score <= 1.0):
                    fixed_score = max(0.0, min(1.0, confidence_score))
                    if hasattr(result, 'confidence_score'):
                        result.confidence_score = fixed_score
                    else:
                        result['confidence_score'] = fixed_score
                
                validated.append(result)
                
            except Exception as e:
                self.log_error(f"Error validating search result: {e}")
                continue
        
        return validated