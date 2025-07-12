# agents/summary_agent.py - Summary Agent
"""
Summary agent for generating intelligent summaries from search results.

This agent handles:
- Analysis of search results to extract key insights
- Identification of critical findings and alerts
- Generation of actionable recommendations
- Confidence scoring based on result quality
"""

from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    """
    Agent responsible for analyzing search results and generating intelligent summaries.
    
    Key Features:
    - Critical findings identification
    - Actionable recommendations generation
    - Performance insights extraction
    - Confidence scoring based on data quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize summary agent with analysis configuration"""
        super().__init__("SummaryAgent", config)
        
        self.max_summary_length = config.get('max_summary_length', 500)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        
        # Keywords for identifying different types of content
        self.critical_keywords = [
            'critical', 'error', 'fail', 'down', 'alert', 'urgent', 'severe',
            'exception', 'abort', 'crash', 'timeout', 'unavailable'
        ]
        
        self.recommendation_keywords = [
            'recommend', 'should', 'improve', 'optimize', 'upgrade', 'configure',
            'adjust', 'tune', 'modify', 'consider', 'suggest', 'advise'
        ]
        
        self.performance_keywords = [
            'performance', 'slow', 'response time', 'throughput', 'latency',
            'cpu', 'memory', 'disk', 'bandwidth', 'capacity', 'utilization'
        ]
    
    def generate_summary(self, search_results: List[Any], query: str) -> Dict[str, Any]:
        """
        Main processing method for generating summaries.
        
        Args:
            search_results: List of search results from SearchAgent
            query: Original search query for context
            
        Returns:
            Dict with structured summary data or error information
        """
        self.start_timer()
        
        try:
            self.log_info(f"Generating summary for {len(search_results)} search results")
            
            if not search_results:
                return self._create_empty_summary(query)
            
            # Extract and analyze content
            analyzed_content = self._analyze_search_results(search_results)
            
            # Generate different types of insights
            critical_findings = self._extract_critical_findings(analyzed_content)
            recommendations = self._extract_recommendations(analyzed_content)
            performance_insights = self._extract_performance_insights(analyzed_content)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(search_results, analyzed_content)
            
            # Generate main summary text
            summary_text = self._generate_summary_text(
                query=query,
                analyzed_content=analyzed_content,
                critical_count=len(critical_findings),
                recommendation_count=len(recommendations)
            )
            
            processing_time = self.end_timer("summary_generation")
            
            result = {
                "success": True,
                "summary": {
                    "summary": summary_text,
                    "critical_findings": critical_findings,
                    "recommendations": recommendations,
                    "performance_insights": performance_insights,
                    "confidence_score": confidence_score,
                    "query": query,
                    "results_analyzed": len(search_results),
                    "processing_time": processing_time
                }
            }
            
            self.log_info(f"Summary generated: {len(critical_findings)} critical findings, "
                         f"{len(recommendations)} recommendations (confidence: {confidence_score:.1%})")
            
            return result
            
        except Exception as e:
            return self.handle_error(e, "Summary Generation")
    
    def _create_empty_summary(self, query: str) -> Dict[str, Any]:
        """Create an empty summary when no search results are available"""
        return {
            "success": True,
            "summary": {
                "summary": f"No relevant information found for query: '{query}'. Please try refining your search terms or check if documents are properly processed.",
                "critical_findings": [],
                "recommendations": ["Upload and process relevant SAP documents", "Try broader search terms", "Verify system IDs are correct"],
                "performance_insights": [],
                "confidence_score": 0.0,
                "query": query,
                "results_analyzed": 0,
                "processing_time": 0.0
            }
        }
    
    def _analyze_search_results(self, search_results: List[Any]) -> Dict[str, Any]:
        """Analyze search results to extract structured information"""
        analyzed = {
            'all_content': [],
            'high_confidence_content': [],
            'system_content': {},
            'source_content': {},
            'total_length': 0,
            'avg_confidence': 0.0,
            'systems_found': set(),
            'sources_found': set()
        }
        
        total_confidence = 0.0
        valid_results = 0
        
        for result_item in search_results:
            try:
                content, confidence, system_id, source = self._extract_result_data(result_item)
                
                if not content:
                    continue
                
                # Store content in various categorizations
                analyzed['all_content'].append(content)
                analyzed['total_length'] += len(content)
                analyzed['systems_found'].add(system_id)
                analyzed['sources_found'].add(source)
                
                # High confidence content for more reliable insights
                if confidence >= 0.7:
                    analyzed['high_confidence_content'].append(content)
                
                # Group by system
                if system_id not in analyzed['system_content']:
                    analyzed['system_content'][system_id] = []
                analyzed['system_content'][system_id].append(content)
                
                # Group by source
                if source not in analyzed['source_content']:
                    analyzed['source_content'][source] = []
                analyzed['source_content'][source].append(content)
                
                total_confidence += confidence
                valid_results += 1
                
            except Exception as e:
                self.log_warning(f"Error analyzing result: {e}")
                continue
        
        # Calculate average confidence
        if valid_results > 0:
            analyzed['avg_confidence'] = total_confidence / valid_results
        
        analyzed['systems_found'] = list(analyzed['systems_found'])
        analyzed['sources_found'] = list(analyzed['sources_found'])
        
        return analyzed
    
    def _extract_result_data(self, result_item: Any) -> tuple:
        """Extract content, confidence, system_id, and source from result item"""
        try:
            # Handle tuple format (document, score)
            if isinstance(result_item, tuple) and len(result_item) >= 2:
                doc, score = result_item[0], result_item[1]
                content = getattr(doc, 'page_content', str(doc))
                confidence = 1.0 - score if score <= 1.0 else score
                metadata = getattr(doc, 'metadata', {})
                system_id = metadata.get('system_id', 'UNKNOWN')
                source = metadata.get('source', 'unknown')
                return content, confidence, system_id, source
            
            # Handle SearchResult object format
            elif hasattr(result_item, 'content') and hasattr(result_item, 'confidence_score'):
                return (
                    result_item.content,
                    result_item.confidence_score,
                    getattr(result_item, 'system_id', 'UNKNOWN'),
                    getattr(result_item, 'source', 'unknown')
                )
            
            # Handle dictionary format
            elif isinstance(result_item, dict):
                return (
                    result_item.get('content', ''),
                    result_item.get('confidence_score', 0.5),
                    result_item.get('system_id', 'UNKNOWN'),
                    result_item.get('source', 'unknown')
                )
            
            # Fallback
            else:
                return str(result_item), 0.5, 'UNKNOWN', 'unknown'
                
        except Exception as e:
            self.log_warning(f"Error extracting result data: {e}")
            return '', 0.0, 'UNKNOWN', 'unknown'
    
    def _extract_critical_findings(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract critical findings from analyzed content"""
        critical_findings = []
        
        # Use high confidence content for more reliable findings
        content_to_analyze = analyzed_content.get('high_confidence_content', [])
        if not content_to_analyze:
            content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for critical keywords
                    has_critical = any(keyword in sentence_lower for keyword in self.critical_keywords)
                    
                    if has_critical and len(sentence.strip()) > 15:
                        finding = sentence.strip()
                        if finding and finding not in critical_findings:
                            critical_findings.append(finding)
                        
                        if len(critical_findings) >= 10:
                            break
                
                if len(critical_findings) >= 10:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting critical findings: {e}")
                continue
        
        return critical_findings[:5]  # Return top 5 critical findings
    
    def _extract_recommendations(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract recommendations from analyzed content"""
        recommendations = []
        
        content_to_analyze = analyzed_content.get('high_confidence_content', [])
        if not content_to_analyze:
            content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for recommendation keywords
                    has_recommendation = any(keyword in sentence_lower for keyword in self.recommendation_keywords)
                    
                    if has_recommendation and len(sentence.strip()) > 15:
                        rec = sentence.strip()
                        if rec and rec not in recommendations:
                            recommendations.append(rec)
                        
                        if len(recommendations) >= 10:
                            break
                
                if len(recommendations) >= 10:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting recommendations: {e}")
                continue
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _extract_performance_insights(self, analyzed_content: Dict[str, Any]) -> List[str]:
        """Extract performance insights from analyzed content"""
        insights = []
        
        content_to_analyze = analyzed_content.get('all_content', [])
        
        for content in content_to_analyze:
            try:
                sentences = content.split('.')
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    
                    # Check for performance keywords
                    has_performance = any(keyword in sentence_lower for keyword in self.performance_keywords)
                    
                    if has_performance and len(sentence.strip()) > 15:
                        insight = sentence.strip()
                        if insight and insight not in insights:
                            insights.append(insight)
                        
                        if len(insights) >= 5:
                            break
                
                if len(insights) >= 5:
                    break
                    
            except Exception as e:
                self.log_warning(f"Error extracting performance insights: {e}")
                continue
        
        return insights
    
    def _calculate_confidence_score(self, search_results: List[Any], analyzed_content: Dict[str, Any]) -> float:
        """Calculate confidence score based on result quality"""
        try:
            if not search_results:
                return 0.0
            
            # Base confidence from average result confidence
            avg_confidence = analyzed_content.get('avg_confidence', 0.5)
            
            # Boost confidence based on content richness
            total_length = analyzed_content.get('total_length', 0)
            length_factor = min(1.0, total_length / 5000)  # Normalize to 5000 chars
            
            # Boost confidence based on result count
            result_count_factor = min(1.0, len(search_results) / 10)  # Normalize to 10 results
            
            # Boost confidence based on system diversity
            systems_found = len(analyzed_content.get('systems_found', []))
            system_factor = min(1.0, systems_found / 3)  # Normalize to 3 systems
            
            # Calculate weighted confidence
            confidence = (
                avg_confidence * 0.4 +
                length_factor * 0.3 +
                result_count_factor * 0.2 +
                system_factor * 0.1
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.log_warning(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_summary_text(self, query: str, analyzed_content: Dict[str, Any], 
                             critical_count: int, recommendation_count: int) -> str:
        """Generate the main summary text"""
        try:
            systems_found = analyzed_content.get('systems_found', [])
            avg_confidence = analyzed_content.get('avg_confidence', 0.0)
            total_content = len(analyzed_content.get('all_content', []))
            
            # Create summary based on findings
            if critical_count > 0:
                urgency_level = "CRITICAL ATTENTION REQUIRED" if critical_count >= 3 else "ATTENTION NEEDED"
                summary = f"Analysis of '{query}' reveals {urgency_level}. "
                summary += f"Found {critical_count} critical issues across {len(systems_found)} systems. "
            else:
                summary = f"Analysis of '{query}' shows no critical issues. "
                summary += f"Reviewed {total_content} documents across {len(systems_found)} systems. "
            
            if recommendation_count > 0:
                summary += f"Identified {recommendation_count} optimization opportunities. "
            
            # Add system-specific information
            if systems_found:
                if len(systems_found) == 1:
                    summary += f"Analysis focused on system {systems_found[0]}. "
                else:
                    summary += f"Systems analyzed: {', '.join(systems_found)}. "
            
            # Add confidence information
            confidence_pct = avg_confidence * 100
            if confidence_pct >= 80:
                summary += "High confidence in analysis results."
            elif confidence_pct >= 60:
                summary += "Moderate confidence in analysis results."
            else:
                summary += "Low confidence - consider reviewing more documents."
            
            return summary
            
        except Exception as e:
            self.log_warning(f"Error generating summary text: {e}")
            return f"Analysis completed for query '{query}' with {critical_count} critical findings and {recommendation_count} recommendations."