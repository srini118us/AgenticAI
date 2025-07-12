# agents/system_output_agent.py - System Analysis Agent
"""
System output agent for generating system-specific analysis and health assessments.

This agent handles:
- System ID extraction and identification
- Health status assessment (HEALTHY/WARNING/CRITICAL)
- System-specific alerts and recommendations
- Performance metrics extraction
"""

import re
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent


class SystemOutputAgent(BaseAgent):
    """
    Agent responsible for generating system-specific analysis outputs and health assessments.
    
    Key Features:
    - System ID detection and analysis
    - Health status classification
    - System-specific recommendations
    - Performance metrics extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system output agent"""
        super().__init__("SystemOutputAgent", config)
        
        # System analysis configuration
        self.health_thresholds = {
            'critical_alert_threshold': 3,  # 3+ alerts = CRITICAL
            'warning_threshold': 1,         # 1+ alerts = WARNING
            'healthy_threshold': 0          # 0 alerts = HEALTHY
        }
        
        # System ID patterns (consistent with SearchAgent)
        self.system_patterns = [
            r'\bSYSTEM[:\s]+([A-Z0-9]{2,4})\b',
            r'\bSID[:\s]+([A-Z0-9]{2,4})\b',
            r'\b([A-Z0-9]{2,4})\s+SYSTEM\b',
            r'\b([A-Z]{1,3}[0-9]{1,2})\b',
            r'\bEARLY\s+WATCH.*?([A-Z0-9]{2,4})\b'
        ]
        
        self.false_positives = {
            'THE', 'AND', 'FOR', 'SAP', 'ERP', 'SYS', 'LOG', 'ALL', 'PDF', 'XML', 'SQL'
        }
    
    def generate_system_outputs(self, search_results: List[Any]) -> Dict[str, Any]:
        """
        Main processing method for generating system-specific outputs.
        
        Args:
            search_results: List of search results to analyze
            
        Returns:
            Dict with system summaries or error information
        """
        self.start_timer()
        
        try:
            self.log_info(f"Generating system outputs for {len(search_results)} search results")
            
            if not search_results:
                return self._create_empty_system_output()
            
            # Extract unique system IDs from search results
            system_ids = self.extract_system_ids(search_results)
            self.log_info(f"Found {len(system_ids)} unique systems: {system_ids}")
            
            # Extract documents for analysis
            documents = self._extract_documents_from_results(search_results)
            
            # Generate summary for each system
            system_summaries = {}
            for system_id in system_ids:
                try:
                    summary = self.extract_system_summary(documents, system_id)
                    system_summaries[system_id] = summary
                    
                    # Get health status for logging
                    health = getattr(summary, 'overall_health', 'UNKNOWN')
                    if hasattr(health, 'value'):
                        health = health.value
                    
                    self.log_info(f"Generated summary for {system_id}: {health}")
                except Exception as e:
                    self.log_error(f"Failed to generate summary for {system_id}: {e}")
                    system_summaries[system_id] = self._create_error_system_summary(system_id, str(e))
            
            processing_time = self.end_timer("system_output_generation")
            
            result = {
                "success": True,
                "system_summaries": system_summaries,
                "systems_analyzed": len(system_ids),
                "total_systems_found": len(system_ids),
                "processing_time": processing_time,
                "analysis_metadata": {
                    "documents_analyzed": len(documents),
                    "search_results_processed": len(search_results),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.log_info(f"System analysis completed: {len(system_ids)} systems analyzed")
            return result
            
        except Exception as e:
            return self.handle_error(e, "System Output Generation")
    
    def extract_system_ids(self, search_results: List[Any]) -> List[str]:
        """Extract unique system IDs from search results using multiple strategies"""
        system_ids = set()
        
        for result_item in search_results:
            try:
                # Extract document and metadata
                doc, metadata = self._extract_doc_and_metadata(result_item)
                
                # Strategy 1: Check metadata for explicit system_id
                if isinstance(metadata, dict) and 'system_id' in metadata:
                    system_id = metadata['system_id']
                    if system_id and system_id.upper() not in self.false_positives:
                        system_ids.add(system_id.upper())
                        continue
                
                # Strategy 2: Extract from document content
                if doc:
                    content = self._extract_content_from_doc(doc)
                    if content:
                        detected_id = self._detect_system_from_content(content)
                        if detected_id != 'UNKNOWN':
                            system_ids.add(detected_id)
                            continue
                
                # Strategy 3: Extract from source filename if available
                source = metadata.get('source', '') if isinstance(metadata, dict) else ''
                if source:
                    filename_id = self._extract_system_from_filename(source)
                    if filename_id != 'UNKNOWN':
                        system_ids.add(filename_id)
                        
            except Exception as e:
                self.log_warning(f"Error extracting system ID from result: {e}")
                continue
        
        # Filter out common false positives and ensure valid system IDs
        filtered_systems = []
        for system_id in system_ids:
            if (system_id not in self.false_positives and 
                len(system_id) in [2, 3, 4] and 
                system_id != 'UNKNOWN'):
                filtered_systems.append(system_id)
        
        # If no valid systems found, return default
        if not filtered_systems:
            filtered_systems = ['SYSTEM_01']
        
        return sorted(filtered_systems)  # Sort for consistent ordering
    
    def extract_system_summary(self, documents: List[Any], system_id: str) -> Any:
        """Extract comprehensive summary for a specific system"""
        try:
            self.log_info(f"Creating system summary for {system_id}")
            
            # Initialize summary components
            critical_alerts = []
            recommendations = []
            key_metrics = {}
            
            # Analyze each document for system-specific content
            system_content = []
            for doc in documents:
                content = self._extract_content_from_doc(doc)
                if content and self._is_content_relevant_to_system(content, system_id):
                    system_content.append(content)
            
            self.log_info(f"Found {len(system_content)} relevant documents for {system_id}")
            
            # Extract insights from relevant content
            for content in system_content:
                # Extract critical alerts
                alerts = self._extract_critical_alerts_for_system(content, system_id)
                critical_alerts.extend(alerts)
                
                # Extract recommendations
                recs = self._extract_recommendations_for_system(content, system_id)
                recommendations.extend(recs)
                
                # Extract metrics
                metrics = self._extract_metrics_from_content(content, system_id)
                key_metrics.update(metrics)
            
            # Remove duplicates
            critical_alerts = list(set(critical_alerts))
            recommendations = list(set(recommendations))
            
            # Determine overall health based on findings
            overall_health = self._calculate_system_health(critical_alerts, recommendations, key_metrics)
            
            # Create SystemSummary object or dict
            try:
                from models import SystemSummary
                summary = SystemSummary(
                    system_id=system_id,
                    overall_health=overall_health,
                    critical_alerts=critical_alerts[:5],  # Limit to top 5
                    recommendations=recommendations[:5],   # Limit to top 5
                    key_metrics=key_metrics,
                    last_analyzed=datetime.now().isoformat()
                )
            except ImportError:
                # Fallback to dict
                summary = {
                    'system_id': system_id,
                    'overall_health': overall_health,
                    'critical_alerts': critical_alerts[:5],
                    'recommendations': recommendations[:5],
                    'key_metrics': key_metrics,
                    'last_analyzed': datetime.now().isoformat()
                }
            
            self.log_info(f"System summary for {system_id}: {overall_health}, "
                         f"{len(critical_alerts)} alerts, {len(recommendations)} recommendations")
            
            return summary
            
        except Exception as e:
            self.log_error(f"Error creating system summary for {system_id}: {e}")
            return self._create_error_system_summary(system_id, str(e))
    
    def _extract_doc_and_metadata(self, result_item: Any) -> tuple:
        """Extract document and metadata from different result formats"""
        try:
            if isinstance(result_item, tuple) and len(result_item) >= 2:
                doc, score = result_item[0], result_item[1]
                metadata = getattr(doc, 'metadata', {})
                return doc, metadata
                
            elif hasattr(result_item, 'content') and hasattr(result_item, 'metadata'):
                # SearchResult object
                return result_item, result_item.metadata
                
            else:
                # Fallback
                return result_item, {}
                
        except Exception as e:
            self.log_warning(f"Error extracting doc and metadata: {e}")
            return result_item, {}
    
    def _extract_content_from_doc(self, doc: Any) -> str:
        """Extract text content from document object"""
        try:
            if hasattr(doc, 'page_content'):
                return str(doc.page_content)
            elif hasattr(doc, 'content'):
                return str(doc.content)
            elif isinstance(doc, dict):
                return doc.get('page_content', doc.get('content', str(doc)))
            else:
                return str(doc)
        except Exception:
            return ""
    
    def _detect_system_from_content(self, content: str) -> str:
        """Detect system ID from content using pattern matching"""
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
        
        return sorted(list(found_systems))[0] if found_systems else 'UNKNOWN'
    
    def _extract_system_from_filename(self, filename: str) -> str:
        """Extract system ID from filename patterns"""
        if not filename:
            return 'UNKNOWN'
        
        filename_upper = filename.upper()
        patterns = [
            r'EWA[_\-]([A-Z0-9]{2,4})[_\-]',
            r'([A-Z0-9]{2,4})[_\-](?:EWA|REPORT|ANALYSIS)',
            r'^([A-Z0-9]{2,4})[_\-]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_upper)
            if match:
                system_id = match.group(1)
                if system_id not in self.false_positives and len(system_id) in [2, 3, 4]:
                    return system_id
        
        return 'UNKNOWN'
    
    def _extract_documents_from_results(self, search_results: List[Any]) -> List[Any]:
        """Extract document objects from search results"""
        documents = []
        
        for result_item in search_results:
            try:
                doc, metadata = self._extract_doc_and_metadata(result_item)
                if doc:
                    documents.append(doc)
            except Exception as e:
                self.log_warning(f"Error extracting document: {e}")
                continue
        
        return documents
    
    def _create_empty_system_output(self) -> Dict[str, Any]:
        """Create empty system output when no results available"""
        return {
            "success": True,
            "system_summaries": {},
            "systems_analyzed": 0,
            "total_systems_found": 0,
            "processing_time": 0.0,
            "analysis_metadata": {
                "documents_analyzed": 0,
                "search_results_processed": 0,
                "timestamp": datetime.now().isoformat(),
                "message": "No search results to analyze"
            }
        }
    
    def _create_error_system_summary(self, system_id: str, error: str) -> Dict[str, Any]:
        """Create error summary for a system"""
        return {
            'system_id': system_id,
            'overall_health': 'UNKNOWN',
            'critical_alerts': [f"Error analyzing system: {error}"],
            'recommendations': ["Review system configuration", "Check document processing"],
            'key_metrics': {},
            'last_analyzed': datetime.now().isoformat(),
            'error': error
        }
    
    def _is_content_relevant_to_system(self, content: str, system_id: str) -> bool:
        """Check if content is relevant to the specific system"""
        if not content or not system_id:
            return False
        
        content_upper = content.upper()
        system_upper = system_id.upper()
        
        # Check for direct system ID mentions
        if system_upper in content_upper:
            return True
        
        # Check for system patterns
        for pattern in self.system_patterns:
            matches = re.findall(pattern, content_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                if match == system_upper:
                    return True
        
        return False
    
    def _extract_critical_alerts_for_system(self, content: str, system_id: str) -> List[str]:
        """Extract critical alerts specific to a system"""
        alerts = []
        critical_keywords = ['critical', 'error', 'fail', 'alert', 'urgent', 'severe']
        
        try:
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                system_lower = system_id.lower()
                
                # Check if sentence mentions system and has critical keywords
                if (system_lower in sentence_lower and 
                    any(keyword in sentence_lower for keyword in critical_keywords)):
                    
                    alert = sentence.strip()
                    if alert and len(alert) > 20:
                        alerts.append(alert)
                        
                        if len(alerts) >= 3:  # Limit alerts per system
                            break
        except Exception as e:
            self.log_warning(f"Error extracting alerts for {system_id}: {e}")
        
        return alerts
    
    def _extract_recommendations_for_system(self, content: str, system_id: str) -> List[str]:
        """Extract recommendations specific to a system"""
        recommendations = []
        rec_keywords = ['recommend', 'should', 'improve', 'optimize', 'consider']
        
        try:
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                system_lower = system_id.lower()
                
                # Check if sentence mentions system and has recommendation keywords
                if (system_lower in sentence_lower and 
                    any(keyword in sentence_lower for keyword in rec_keywords)):
                    
                    rec = sentence.strip()
                    if rec and len(rec) > 20:
                        recommendations.append(rec)
                        
                        if len(recommendations) >= 3:  # Limit recommendations per system
                            break
        except Exception as e:
            self.log_warning(f"Error extracting recommendations for {system_id}: {e}")
        
        return recommendations
    
    def _extract_metrics_from_content(self, content: str, system_id: str) -> Dict[str, Any]:
        """Extract performance metrics from content"""
        metrics = {}
        
        try:
            # Look for common SAP performance metrics
            metric_patterns = [
                (r'cpu[:\s]+(\d+)%', 'cpu_utilization'),
                (r'memory[:\s]+(\d+)%', 'memory_utilization'),
                (r'response\s+time[:\s]+(\d+(?:\.\d+)?)\s*(?:ms|sec)', 'response_time'),
                (r'users[:\s]+(\d+)', 'active_users'),
                (r'sessions[:\s]+(\d+)', 'active_sessions')
            ]
            
            content_lower = content.lower()
            system_lower = system_id.lower()
            
            # Only extract metrics if content mentions the system
            if system_lower in content_lower:
                for pattern, metric_name in metric_patterns:
                    matches = re.findall(pattern, content_lower)
                    if matches:
                        # Take the first match
                        value = matches[0]
                        try:
                            metrics[metric_name] = float(value)
                        except ValueError:
                            metrics[metric_name] = value
        
        except Exception as e:
            self.log_warning(f"Error extracting metrics for {system_id}: {e}")
        
        return metrics
    
    def _calculate_system_health(self, critical_alerts: List[str], 
                                recommendations: List[str], 
                                key_metrics: Dict[str, Any]) -> str:
        """Calculate overall system health based on findings"""
        try:
            # Health determination based on alerts
            alert_count = len(critical_alerts)
            
            if alert_count >= self.health_thresholds['critical_alert_threshold']:
                return 'CRITICAL'
            elif alert_count >= self.health_thresholds['warning_threshold']:
                return 'WARNING'
            elif alert_count == self.health_thresholds['healthy_threshold']:
                return 'HEALTHY'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            self.log_warning(f"Error calculating system health: {e}")
            return 'UNKNOWN'