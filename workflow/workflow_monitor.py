# workflow/workflow_monitor.py - Monitoring & Debugging Classes
"""
Monitoring and debugging utilities for SAP EWA Workflow.

This module provides comprehensive monitoring, debugging, and recovery capabilities
for workflow execution. It includes real-time monitoring, performance analysis,
error recovery strategies, and debugging tools for development and production.

Available Classes:
- WorkflowMonitor: Real-time monitoring for workflow execution
- WorkflowDebugger: Debugging utilities for development and troubleshooting
- WorkflowRecoveryManager: Error recovery and retry logic
- PerformanceAnalyzer: Performance metrics and optimization recommendations

Usage:
    from workflow.workflow_monitor import WorkflowMonitor, WorkflowDebugger
    
    # Monitor workflow execution
    monitor = WorkflowMonitor()
    monitor.start_monitoring("workflow_1", workflow)
    
    # Debug workflow issues
    debugger = WorkflowDebugger(workflow)
    analysis = debugger.analyze_workflow_state(state)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import time
import json

# Configure logging
logger = logging.getLogger(__name__)

# Import workflow components
try:
    from .core_workflow import DetailedWorkflowState, SAPRAGWorkflow
    from models import WorkflowStatus
    logger.info("✅ Workflow components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import workflow components: {e}")
    
    # Fallback classes
    class WorkflowStatus:
        INITIALIZED = "initialized"
        PROCESSING_PDF = "processing_pdf"
        CREATING_EMBEDDINGS = "creating_embeddings"
        STORING_VECTORS = "storing_vectors"
        SEARCHING = "searching"
        SUMMARIZING = "summarizing"
        SYSTEM_OUTPUT = "system_output"
        SENDING_EMAIL = "sending_email"
        COMPLETED = "completed"
        ERROR = "error"
    
    # Mock state type
    DetailedWorkflowState = Dict[str, Any]


# ================================
# MONITORING DATA STRUCTURES
# ================================

@dataclass
class WorkflowExecutionMetrics:
    """Data class for workflow execution metrics."""
    
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    current_agent: str = ""
    
    # Performance metrics
    total_duration: float = 0.0
    agent_durations: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    
    # Processing metrics
    files_processed: int = 0
    documents_created: int = 0
    embeddings_created: int = 0
    search_results_count: int = 0
    systems_analyzed: int = 0
    
    # Error information
    error_message: str = ""
    error_agent: str = ""
    retry_count: int = 0
    
    # Additional metadata
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    agent_messages: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for monitoring alerts."""
    
    max_total_duration: float = 300.0  # 5 minutes
    max_agent_duration: float = 120.0  # 2 minutes per agent
    max_memory_usage_mb: float = 1024.0  # 1GB
    min_search_results: int = 1
    max_error_rate: float = 0.1  # 10%


# ================================
# WORKFLOW MONITOR
# ================================

class WorkflowMonitor:
    """
    Real-time monitoring for workflow execution.
    
    Tracks workflow progress, performance metrics, and health indicators
    during execution for operational monitoring and alerting.
    """
    
    def __init__(self, performance_thresholds: PerformanceThresholds = None):
        """
        Initialize workflow monitor.
        
        Args:
            performance_thresholds: Custom performance thresholds for alerts
        """
        self.active_workflows: Dict[str, WorkflowExecutionMetrics] = {}
        self.execution_history: List[WorkflowExecutionMetrics] = []
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        
        # Global performance metrics
        self.global_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "peak_memory_usage": 0.0,
            "total_processing_time": 0.0
        }
        
        # Alert system
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[callable] = []
        
        logger.info("📊 WorkflowMonitor initialized")
    
    def start_monitoring(self, workflow_id: str, workflow: 'SAPRAGWorkflow', 
                        config_snapshot: Dict[str, Any] = None):
        """
        Start monitoring a workflow execution.
        
        Args:
            workflow_id: Unique identifier for this execution
            workflow: Workflow instance to monitor
            config_snapshot: Snapshot of workflow configuration
        """
        metrics = WorkflowExecutionMetrics(
            workflow_id=workflow_id,
            start_time=datetime.now(),
            config_snapshot=config_snapshot or {}
        )
        
        self.active_workflows[workflow_id] = metrics
        
        logger.info(f"📊 Started monitoring workflow: {workflow_id}")
        self._check_active_workflow_limits()
    
    def update_workflow_progress(self, workflow_id: str, state: DetailedWorkflowState):
        """
        Update progress for a monitored workflow.
        
        Args:
            workflow_id: Workflow identifier
            state: Current workflow state
        """
        if workflow_id not in self.active_workflows:
            logger.warning(f"⚠️ Workflow {workflow_id} not found in active monitoring")
            return
        
        metrics = self.active_workflows[workflow_id]
        
        # Update basic status
        metrics.status = state.get("workflow_status", "unknown")
        metrics.current_agent = state.get("current_agent", "")
        
        # Update processing metrics
        metrics.files_processed = len(state.get("uploaded_files", []))
        metrics.documents_created = len(state.get("processed_documents", []))
        metrics.embeddings_created = len(state.get("embeddings", []))
        metrics.search_results_count = len(state.get("search_results", []))
        metrics.systems_analyzed = len(state.get("system_summaries", {}))
        
        # Update agent durations
        processing_times = state.get("processing_times", {})
        metrics.agent_durations.update(processing_times)
        
        # Update agent messages
        agent_messages = state.get("agent_messages", [])
        if len(agent_messages) > len(metrics.agent_messages):
            # Add new messages
            new_messages = agent_messages[len(metrics.agent_messages):]
            metrics.agent_messages.extend(new_messages)
        
        # Update error information if present
        if state.get("error_message"):
            metrics.error_message = state.get("error_message", "")
            metrics.error_agent = state.get("current_agent", "")
        
        # Calculate current duration
        metrics.total_duration = (datetime.now() - metrics.start_time).total_seconds()
        
        # Check for performance alerts
        self._check_performance_alerts(workflow_id, metrics)
        
        logger.debug(f"📊 Updated progress for workflow {workflow_id}: {metrics.status}")
    
    def finish_monitoring(self, workflow_id: str, final_state: DetailedWorkflowState):
        """
        Finish monitoring and record final results.
        
        Args:
            workflow_id: Workflow identifier  
            final_state: Final workflow state
        """
        if workflow_id not in self.active_workflows:
            logger.warning(f"⚠️ Workflow {workflow_id} not found for completion")
            return
        
        metrics = self.active_workflows[workflow_id]
        metrics.end_time = datetime.now()
        metrics.total_duration = (metrics.end_time - metrics.start_time).total_seconds()
        
        # Final status update
        metrics.status = final_state.get("workflow_status", "unknown")
        metrics.error_message = final_state.get("error_message", "")
        
        # Update final processing metrics
        self.update_workflow_progress(workflow_id, final_state)
        
        # Record in execution history
        self.execution_history.append(metrics)
        
        # Update global metrics
        self._update_global_metrics(metrics)
        
        # Clean up active workflows
        del self.active_workflows[workflow_id]
        
        logger.info(f"📊 Finished monitoring workflow {workflow_id}: "
                   f"{metrics.total_duration:.2f}s, status: {metrics.status}")
        
        # Generate completion alert if configured
        self._generate_completion_alert(metrics)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get current monitoring summary.
        
        Returns:
            Summary of all monitored workflows and performance metrics
        """
        current_time = datetime.now()
        
        # Active workflow details
        active_details = {}
        for workflow_id, metrics in self.active_workflows.items():
            runtime = (current_time - metrics.start_time).total_seconds()
            active_details[workflow_id] = {
                "status": metrics.status,
                "current_agent": metrics.current_agent,
                "runtime_seconds": runtime,
                "files_processed": metrics.files_processed,
                "search_results": metrics.search_results_count,
                "error_message": metrics.error_message
            }
        
        # Recent execution summary
        recent_executions = self.execution_history[-10:] if self.execution_history else []
        recent_summary = [
            {
                "workflow_id": m.workflow_id,
                "duration": m.total_duration,
                "status": m.status,
                "files_processed": m.files_processed,
                "end_time": m.end_time.isoformat() if m.end_time else None
            }
            for m in recent_executions
        ]
        
        return {
            "timestamp": current_time.isoformat(),
            "active_workflows": {
                "count": len(self.active_workflows),
                "details": active_details
            },
            "global_metrics": self.global_metrics,
            "recent_executions": recent_summary,
            "alerts": {
                "total_alerts": len(self.alerts),
                "recent_alerts": self.alerts[-5:] if self.alerts else []
            },
            "performance_thresholds": {
                "max_duration": self.performance_thresholds.max_total_duration,
                "max_memory": self.performance_thresholds.max_memory_usage_mb,
                "max_error_rate": self.performance_thresholds.max_error_rate
            }
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate performance report for specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Detailed performance analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent executions
        recent_executions = [
            m for m in self.execution_history 
            if m.end_time and m.end_time >= cutoff_time
        ]
        
        if not recent_executions:
            return {
                "period_hours": hours,
                "executions_found": 0,
                "message": "No executions found in specified period"
            }
        
        # Calculate statistics
        total_executions = len(recent_executions)
        successful_executions = len([m for m in recent_executions if m.status == WorkflowStatus.COMPLETED])
        failed_executions = total_executions - successful_executions
        
        durations = [m.total_duration for m in recent_executions]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Agent performance analysis
        agent_stats = {}
        for metrics in recent_executions:
            for agent, duration in metrics.agent_durations.items():
                if agent not in agent_stats:
                    agent_stats[agent] = []
                agent_stats[agent].append(duration)
        
        agent_performance = {}
        for agent, durations in agent_stats.items():
            agent_performance[agent] = {
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "execution_count": len(durations)
            }
        
        return {
            "period_hours": hours,
            "period_start": cutoff_time.isoformat(),
            "period_end": datetime.now().isoformat(),
            "execution_summary": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0
            },
            "duration_statistics": {
                "average_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "total_processing_time": sum(durations)
            },
            "agent_performance": agent_performance,
            "performance_issues": self._identify_performance_issues(recent_executions)
        }
    
    def add_alert_callback(self, callback: callable):
        """
        Add callback function for alert notifications.
        
        Args:
            callback: Function to call when alerts are generated
        """
        self.alert_callbacks.append(callback)
        logger.info(f"📢 Added alert callback: {callback.__name__}")
    
    def _check_performance_alerts(self, workflow_id: str, metrics: WorkflowExecutionMetrics):
        """Check for performance threshold violations."""
        alerts = []
        
        # Check total duration
        if metrics.total_duration > self.performance_thresholds.max_total_duration:
            alerts.append({
                "type": "duration_exceeded",
                "message": f"Workflow {workflow_id} running for {metrics.total_duration:.1f}s (threshold: {self.performance_thresholds.max_total_duration}s)",
                "severity": "warning"
            })
        
        # Check agent durations
        for agent, duration in metrics.agent_durations.items():
            if duration > self.performance_thresholds.max_agent_duration:
                alerts.append({
                    "type": "agent_duration_exceeded",
                    "message": f"Agent {agent} in workflow {workflow_id} took {duration:.1f}s (threshold: {self.performance_thresholds.max_agent_duration}s)",
                    "severity": "warning"
                })
        
        # Check search results
        if (metrics.status == WorkflowStatus.COMPLETED and 
            metrics.search_results_count < self.performance_thresholds.min_search_results):
            alerts.append({
                "type": "low_search_results",
                "message": f"Workflow {workflow_id} returned only {metrics.search_results_count} search results",
                "severity": "info"
            })
        
        # Process alerts
        for alert in alerts:
            alert.update({
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id
            })
            self.alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"❌ Alert callback failed: {e}")
    
    def _check_active_workflow_limits(self):
        """Check if too many workflows are running concurrently."""
        max_concurrent = 5  # Configurable limit
        
        if len(self.active_workflows) > max_concurrent:
            alert = {
                "type": "concurrent_limit_exceeded",
                "message": f"Too many concurrent workflows: {len(self.active_workflows)} (limit: {max_concurrent})",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"❌ Alert callback failed: {e}")
    
    def _update_global_metrics(self, metrics: WorkflowExecutionMetrics):
        """Update global performance metrics."""
        self.global_metrics["total_executions"] += 1
        self.global_metrics["total_processing_time"] += metrics.total_duration
        
        if metrics.status == WorkflowStatus.COMPLETED:
            self.global_metrics["successful_executions"] += 1
        else:
            self.global_metrics["failed_executions"] += 1
        
        # Update average execution time
        total_executions = self.global_metrics["total_executions"]
        self.global_metrics["average_execution_time"] = (
            self.global_metrics["total_processing_time"] / total_executions
        )
        
        # Update peak memory usage
        if metrics.memory_usage_mb > self.global_metrics["peak_memory_usage"]:
            self.global_metrics["peak_memory_usage"] = metrics.memory_usage_mb
    
    def _generate_completion_alert(self, metrics: WorkflowExecutionMetrics):
        """Generate alert for workflow completion."""
        if metrics.status == WorkflowStatus.ERROR:
            alert = {
                "type": "workflow_failed",
                "message": f"Workflow {metrics.workflow_id} failed: {metrics.error_message}",
                "severity": "error",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": metrics.workflow_id
            }
        elif metrics.total_duration > self.performance_thresholds.max_total_duration:
            alert = {
                "type": "slow_completion",
                "message": f"Workflow {metrics.workflow_id} completed slowly: {metrics.total_duration:.1f}s",
                "severity": "warning",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": metrics.workflow_id
            }
        else:
            # Success alert (optional)
            alert = {
                "type": "workflow_completed",
                "message": f"Workflow {metrics.workflow_id} completed successfully in {metrics.total_duration:.1f}s",
                "severity": "info",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": metrics.workflow_id
            }
        
        self.alerts.append(alert)
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"❌ Alert callback failed: {e}")
    
    def _identify_performance_issues(self, executions: List[WorkflowExecutionMetrics]) -> List[Dict[str, Any]]:
        """Identify performance issues from execution history."""
        issues = []
        
        # Check for consistently slow agents
        agent_durations = {}
        for metrics in executions:
            for agent, duration in metrics.agent_durations.items():
                if agent not in agent_durations:
                    agent_durations[agent] = []
                agent_durations[agent].append(duration)
        
        for agent, durations in agent_durations.items():
            avg_duration = sum(durations) / len(durations)
            if avg_duration > self.performance_thresholds.max_agent_duration * 0.8:  # 80% of threshold
                issues.append({
                    "type": "slow_agent",
                    "agent": agent,
                    "average_duration": avg_duration,
                    "recommendation": f"Agent {agent} is consistently slow - consider optimization"
                })
        
        # Check error rate
        failed_count = len([m for m in executions if m.status == WorkflowStatus.ERROR])
        error_rate = failed_count / len(executions) if executions else 0
        
        if error_rate > self.performance_thresholds.max_error_rate:
            issues.append({
                "type": "high_error_rate",
                "error_rate": error_rate,
                "failed_executions": failed_count,
                "recommendation": "High error rate detected - investigate common failure causes"
            })
        
        return issues


# ================================
# WORKFLOW DEBUGGER
# ================================

class WorkflowDebugger:
    """
    Utility class for debugging workflow execution issues.
    
    Provides tools for analyzing workflow state, performance bottlenecks,
    and component interactions during development and troubleshooting.
    """
    
    def __init__(self, workflow: 'SAPRAGWorkflow'):
        """
        Initialize debugger with workflow instance.
        
        Args:
            workflow: SAPRAGWorkflow instance to debug
        """
        self.workflow = workflow
        self.debug_logs: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, Any] = {}
        
        logger.info(f"🐛 WorkflowDebugger initialized for workflow")
    
    def analyze_workflow_state(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """
        Analyze current workflow state for debugging.
        
        Args:
            state: Current workflow state
            
        Returns:
            Analysis results with debugging information
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "state_completeness": {},
            "performance_analysis": {},
            "component_status": {},
            "potential_issues": [],
            "recommendations": []
        }
        
        try:
            # Check state completeness
            required_fields = [
                "workflow_status", "uploaded_files", "user_query",
                "processed_documents", "embeddings", "vector_store_ready",
                "search_results", "summary", "system_summaries"
            ]
            
            for field in required_fields:
                value = state.get(field)
                analysis["state_completeness"][field] = {
                    "present": value is not None,
                    "type": type(value).__name__,
                    "length": len(value) if hasattr(value, '__len__') else None,
                    "empty": not bool(value) if value is not None else True
                }
            
            # Analyze performance
            processing_times = state.get("processing_times", {})
            if processing_times:
                total_time = sum(processing_times.values())
                slowest_step = max(processing_times.items(), key=lambda x: x[1])
                fastest_step = min(processing_times.items(), key=lambda x: x[1])
                
                analysis["performance_analysis"] = {
                    "total_time": total_time,
                    "step_count": len(processing_times),
                    "average_step_time": total_time / len(processing_times),
                    "slowest_step": {
                        "name": slowest_step[0],
                        "duration": slowest_step[1]
                    },
                    "fastest_step": {
                        "name": fastest_step[0],
                        "duration": fastest_step[1]
                    },
                    "step_details": processing_times
                }
            
            # Check component status
            analysis["component_status"] = {
                "pdf_processor": self.workflow.pdf_processor is not None,
                "embedding_agent": self.workflow.embedding_agent is not None,
                "search_agent": self.workflow.search_agent is not None,
                "summary_agent": self.workflow.summary_agent is not None,
                "email_agent": self.workflow.email_agent is not None,
                "system_output_agent": self.workflow.system_output_agent is not None,
                "vector_store": self.workflow.vector_store is not None
            }
            
            # Identify potential issues
            issues = []
            
            if not state.get("vector_store_ready") and state.get("user_query"):
                issues.append("Query provided but vector store not ready")
            
            if len(state.get("processed_documents", [])) == 0 and len(state.get("uploaded_files", [])) > 0:
                issues.append("Files uploaded but no documents processed")
            
            if state.get("workflow_status") == WorkflowStatus.ERROR:
                issues.append(f"Workflow in error state: {state.get('error_message', 'Unknown error')}")
            
            if len(state.get("search_results", [])) == 0 and state.get("user_query"):
                issues.append("Search query provided but no results found")
            
            if state.get("embeddings") and len(state.get("embeddings", [])) != len(state.get("processed_documents", [])):
                issues.append("Mismatch between embeddings and documents count")
            
            analysis["potential_issues"] = issues
            
            # Generate recommendations
            recommendations = []
            
            if processing_times:
                slowest_duration = max(processing_times.values())
                if slowest_duration > 60:  # 1 minute
                    slowest_agent = max(processing_times.items(), key=lambda x: x[1])[0]
                    recommendations.append(f"Consider optimizing {slowest_agent} - taking {slowest_duration:.1f}s")
            
            if not analysis["component_status"]["search_agent"]:
                recommendations.append("Initialize search agent after vector store is populated")
            
            if state.get("user_query") and len(state.get("user_query", "")) < 3:
                recommendations.append("Search query is very short - consider longer, more specific queries")
            
            analysis["recommendations"] = recommendations
            
            # Cache analysis for comparison
            state_hash = str(hash(json.dumps(state, sort_keys=True, default=str)))
            self.analysis_cache[state_hash] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing workflow state: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_states(self, state1: DetailedWorkflowState, state2: DetailedWorkflowState) -> Dict[str, Any]:
        """
        Compare two workflow states to identify changes.
        
        Args:
            state1: First workflow state
            state2: Second workflow state
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "changes": {},
            "performance_delta": {},
            "status_change": {},
            "summary": {}
        }
        
        try:
            # Compare status
            status1 = state1.get("workflow_status")
            status2 = state2.get("workflow_status")
            
            if status1 != status2:
                comparison["status_change"] = {
                    "from": status1,
                    "to": status2,
                    "progression": self._analyze_status_progression(status1, status2)
                }
            
            # Compare processing times
            times1 = state1.get("processing_times", {})
            times2 = state2.get("processing_times", {})
            
            for agent in set(times1.keys()) | set(times2.keys()):
                time1 = times1.get(agent, 0)
                time2 = times2.get(agent, 0)
                
                if time1 != time2:
                    comparison["performance_delta"][agent] = {
                        "before": time1,
                        "after": time2,
                        "delta": time2 - time1
                    }
            
            # Compare key metrics
            metrics_to_compare = [
                "total_chunks", "vector_store_ready", "email_sent"
            ]
            
            for metric in metrics_to_compare:
                val1 = state1.get(metric)
                val2 = state2.get(metric)
                
                if val1 != val2:
                    comparison["changes"][metric] = {
                        "from": val1,
                        "to": val2
                    }
            
            # Compare list lengths
            list_fields = ["processed_documents", "embeddings", "search_results", "agent_messages"]
            
            for field in list_fields:
                len1 = len(state1.get(field, []))
                len2 = len(state2.get(field, []))
                
                if len1 != len2:
                    comparison["changes"][f"{field}_count"] = {
                        "from": len1,
                        "to": len2,
                        "delta": len2 - len1
                    }
            
            # Summary
            comparison["summary"] = {
                "total_changes": len(comparison["changes"]),
                "performance_changes": len(comparison["performance_delta"]),
                "status_changed": bool(comparison["status_change"]),
                "overall_progress": len2 > len1 if "agent_messages" in locals() else "unknown"
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error comparing workflow states: {e}")
            return {"error": str(e)}
    
    def generate_debug_report(self, state: DetailedWorkflowState) -> str:
        """
        Generate a human-readable debug report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Formatted debug report string
        """
        try:
            analysis = self.analyze_workflow_state(state)
            
            report_lines = [
                "=" * 60,
                "WORKFLOW DEBUG REPORT",
                f"Generated: {analysis['timestamp']}",
                "=" * 60,
                "",
                "📊 WORKFLOW STATUS:",
                f"  Status: {state.get('workflow_status', 'unknown')}",
                f"  Current Agent: {state.get('current_agent', 'none')}",
                f"  Error Message: {state.get('error_message', 'none')}",
                "",
                "📋 STATE COMPLETENESS:"
            ]
            
            for field, info in analysis["state_completeness"].items():
                status = "✅" if info["present"] and not info["empty"] else "❌" if not info["present"] else "⚠️"
                length_info = f" (length: {info['length']})" if info["length"] is not None else ""
                report_lines.append(f"  {status} {field}: {info['type']}{length_info}")
            
            if analysis["performance_analysis"]:
                perf = analysis["performance_analysis"]
                report_lines.extend([
                    "",
                    "⏱️ PERFORMANCE ANALYSIS:",
                    f"  Total Time: {perf['total_time']:.2f}s",
                    f"  Average Step Time: {perf['average_step_time']:.2f}s",
                    f"  Slowest Step: {perf['slowest_step']['name']} ({perf['slowest_step']['duration']:.2f}s)",
                    f"  Fastest Step: {perf['fastest_step']['name']} ({perf['fastest_step']['duration']:.2f}s)"
                ])
            
            report_lines.extend([
                "",
                "🔧 COMPONENT STATUS:"
            ])
            
            for component, available in analysis["component_status"].items():
                status = "✅" if available else "❌"
                report_lines.append(f"  {status} {component}")
            
            if analysis["potential_issues"]:
                report_lines.extend([
                    "",
                    "⚠️ POTENTIAL ISSUES:"
                ])
                for issue in analysis["potential_issues"]:
                    report_lines.append(f"  • {issue}")
            
            if analysis["recommendations"]:
                report_lines.extend([
                    "",
                    "💡 RECOMMENDATIONS:"
                ])
                for rec in analysis["recommendations"]:
                    report_lines.append(f"  • {rec}")
            
            report_lines.extend([
                "",
                "=" * 60
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"❌ Error generating debug report: {e}"
    
    def log_debug_event(self, event_type: str, message: str, context: Dict[str, Any] = None):
        """
        Log a debug event for later analysis.
        
        Args:
            event_type: Type of debug event
            message: Debug message
            context: Additional context information
        """
        debug_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "context": context or {}
        }
        
        self.debug_logs.append(debug_event)
        logger.debug(f"🐛 Debug event: {event_type} - {message}")
    
    def _analyze_status_progression(self, from_status: str, to_status: str) -> str:
        """Analyze if status change represents normal progression."""
        normal_progressions = [
            ("initialized", "processing_pdf"),
            ("processing_pdf", "creating_embeddings"),
            ("creating_embeddings", "storing_vectors"),
            ("storing_vectors", "searching"),
            ("searching", "summarizing"),
            ("summarizing", "system_output"),
            ("system_output", "sending_email"),
            ("sending_email", "completed"),
            ("system_output", "completed")  # Skip email
        ]
        
        if (from_status, to_status) in normal_progressions:
            return "normal"
        elif to_status == "error":
            return "error"
        elif from_status == to_status:
            return "no_change"
        else:
            return "unexpected"


# ================================
# WORKFLOW RECOVERY MANAGER
# ================================

class WorkflowRecoveryManager:
    """
    Handles workflow error recovery and retry logic.
    
    Provides intelligent recovery strategies for different types of
    workflow failures, including partial state recovery and retry mechanisms.
    """
    
    def __init__(self, workflow: 'SAPRAGWorkflow'):
        """
        Initialize recovery manager.
        
        Args:
            workflow: Workflow instance to manage recovery for
        """
        self.workflow = workflow
        self.recovery_strategies = {
            "pdf_processor": self._recover_pdf_processing,
            "embedding_creator": self._recover_embedding_creation,
            "vector_storage": self._recover_vector_storage,
            "search_agent": self._recover_search,
            "summary_agent": self._recover_summary,
            "email_agent": self._recover_email
        }
        
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_retries = 3
        
        logger.info("🔧 WorkflowRecoveryManager initialized")
    
    def attempt_recovery(self, failed_state: DetailedWorkflowState) -> Dict[str, Any]:
        """
        Attempt to recover from workflow failure.
        
        Args:
            failed_state: State where workflow failed
            
        Returns:
            Recovery result and next steps
        """
        try:
            current_agent = failed_state.get("current_agent", "unknown")
            error_message = failed_state.get("error_message", "")
            
            logger.info(f"🔧 Attempting recovery from {current_agent} failure: {error_message}")
            
            # Check retry count
            retry_count = failed_state.get("retry_count", 0)
            if retry_count >= self.max_retries:
                return {
                    "success": False,
                    "error": f"Maximum retries ({self.max_retries}) exceeded",
                    "recommendations": ["Manual intervention required", "Check system logs for detailed error information"]
                }
            
            # Choose recovery strategy based on failed component
            recovery_func = self.recovery_strategies.get(current_agent)
            
            if recovery_func:
                recovery_result = recovery_func(failed_state)
                
                # Record recovery attempt
                recovery_record = {
                    "timestamp": datetime.now().isoformat(),
                    "failed_agent": current_agent,
                    "error_message": error_message,
                    "retry_count": retry_count + 1,
                    "recovery_strategy": recovery_func.__name__,
                    "success": recovery_result.get("success", False)
                }
                self.recovery_history.append(recovery_record)
                
                if recovery_result["success"]:
                    logger.info(f"✅ Recovery successful for {current_agent}")
                else:
                    logger.warning(f"⚠️ Recovery failed for {current_agent}: {recovery_result.get('error')}")
                
                return recovery_result
            else:
                return {
                    "success": False,
                    "error": f"No recovery strategy available for {current_agent}",
                    "recommendations": ["Manual intervention required", "Check logs for detailed error information"]
                }
                
        except Exception as e:
            logger.error(f"❌ Recovery attempt failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": ["Contact system administrator"]
            }
    
    def _recover_pdf_processing(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for PDF processing failures."""
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            # Strategy 1: Try with smaller file subset
            if len(uploaded_files) > 1:
                return {
                    "success": True,
                    "strategy": "partial_processing",
                    "action": "Process files individually",
                    "recommendations": [
                        "Check file integrity",
                        "Try with smaller files first",
                        "Verify PDF format compatibility"
                    ]
                }
            
            # Strategy 2: Alternative PDF processing method
            return {
                "success": True,
                "strategy": "alternative_processor",
                "action": "Try different PDF processing library",
                "recommendations": [
                    "Check PDF file format",
                    "Try converting PDF to text manually",
                    "Verify file is not corrupted"
                ]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_embedding_creation(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for embedding creation failures."""
        try:
            # Strategy 1: Fall back to mock embeddings
            if self.workflow.agent_config.get("embedding_type") != "mock":
                return {
                    "success": True,
                    "strategy": "fallback_embeddings",
                    "action": "Use mock embeddings for testing",
                    "recommendations": [
                        "Check OpenAI API key validity",
                        "Verify network connectivity",
                        "Check API rate limits"
                    ]
                }
            
            # Strategy 2: Reduce chunk size
            current_chunk_size = self.workflow.agent_config.get("chunk_size", 1000)
            if current_chunk_size > 500:
                return {
                    "success": True,
                    "strategy": "reduce_chunk_size",
                    "action": f"Reduce chunk size from {current_chunk_size} to {current_chunk_size // 2}",
                    "recommendations": ["Smaller chunks may process more reliably"]
                }
            
            return {
                "success": False,
                "error": "All embedding recovery strategies exhausted"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_vector_storage(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for vector storage failures."""
        try:
            # Strategy 1: Switch to different vector store type
            current_type = self.workflow.config.get("vector_store_type", "chroma")
            fallback_types = ["simple", "mock"]
            
            for fallback_type in fallback_types:
                if fallback_type != current_type:
                    return {
                        "success": True,
                        "strategy": "fallback_vector_store",
                        "action": f"Switch from {current_type} to {fallback_type} vector store",
                        "recommendations": [
                            f"Consider using {fallback_type} for development",
                            "Check ChromaDB installation and permissions"
                        ]
                    }
            
            # Strategy 2: Reinitialize vector store
            return {
                "success": True,
                "strategy": "reinitialize_vector_store",
                "action": "Reinitialize vector store with fresh configuration",
                "recommendations": [
                    "Clear any existing vector store data",
                    "Check disk space availability"
                ]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_search(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for search failures."""
        try:
            # Strategy 1: Reinitialize search agent
            if not state.get("vector_store_ready"):
                return {
                    "success": True,
                    "strategy": "reinitialize_search",
                    "action": "Reinitialize search agent after ensuring vector store is ready",
                    "recommendations": ["Verify vector store is properly populated"]
                }
            
            # Strategy 2: Simplify search query
            query = state.get("user_query", "")
            if len(query) > 50:
                simplified_query = " ".join(query.split()[:5])  # Take first 5 words
                return {
                    "success": True,
                    "strategy": "simplify_query",
                    "action": f"Simplify query from '{query}' to '{simplified_query}'",
                    "recommendations": ["Try shorter, more specific queries"]
                }
            
            return {
                "success": False,
                "error": "Search agent configuration issue - manual intervention needed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_summary(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for summary generation failures."""
        try:
            search_results = state.get("search_results", [])
            
            # Strategy 1: Generate basic summary if no results
            if not search_results:
                return {
                    "success": True,
                    "strategy": "empty_summary",
                    "action": "Generate basic summary indicating no relevant results found",
                    "recommendations": ["Review search query and filters"]
                }
            
            # Strategy 2: Simplified summary with fewer results
            if len(search_results) > 10:
                return {
                    "success": True,
                    "strategy": "simplified_summary",
                    "action": "Generate summary using top 5 search results only",
                    "recommendations": ["Large result sets may cause summary complexity issues"]
                }
            
            # Strategy 3: Basic text extraction summary
            return {
                "success": True,
                "strategy": "basic_summary",
                "action": "Generate basic summary without advanced analysis",
                "recommendations": ["Review summary agent configuration and API limits"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recover_email(self, state: DetailedWorkflowState) -> Dict[str, Any]:
        """Recovery strategy for email failures."""
        try:
            # Strategy 1: Skip email if not critical
            return {
                "success": True,
                "strategy": "skip_email",
                "action": "Continue workflow without sending email",
                "recommendations": [
                    "Check email configuration",
                    "Verify Gmail app password",
                    "Email functionality is optional for workflow completion"
                ]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get history of all recovery attempts."""
        return self.recovery_history.copy()
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts."""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0}
        
        total_attempts = len(self.recovery_history)
        successful_attempts = len([r for r in self.recovery_history if r["success"]])
        
        # Count failures by agent
        failure_counts = {}
        for record in self.recovery_history:
            agent = record["failed_agent"]
            failure_counts[agent] = failure_counts.get(agent, 0) + 1
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts,
            "failure_counts_by_agent": failure_counts,
            "most_problematic_agent": max(failure_counts.items(), key=lambda x: x[1])[0] if failure_counts else None
        }


# ================================
# PERFORMANCE ANALYZER
# ================================

class PerformanceAnalyzer:
    """
    Advanced performance analysis for workflow optimization.
    
    Analyzes workflow execution patterns, identifies bottlenecks,
    and provides optimization recommendations.
    """
    
    def __init__(self, monitor: WorkflowMonitor):
        """
        Initialize performance analyzer.
        
        Args:
            monitor: WorkflowMonitor instance to analyze data from
        """
        self.monitor = monitor
        self.optimization_rules = {
            "slow_pdf_processing": self._analyze_pdf_performance,
            "slow_embedding_creation": self._analyze_embedding_performance,
            "slow_search": self._analyze_search_performance,
            "memory_usage": self._analyze_memory_usage,
            "concurrent_processing": self._analyze_concurrency
        }
        
        logger.info("📈 PerformanceAnalyzer initialized")
    
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance trends over specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance trend analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter relevant executions
        recent_executions = [
            m for m in self.monitor.execution_history
            if m.end_time and m.end_time >= cutoff_date
        ]
        
        if len(recent_executions) < 2:
            return {
                "period_days": days,
                "executions_analyzed": len(recent_executions),
                "trend": "insufficient_data"
            }
        
        # Calculate trends
        durations = [m.total_duration for m in recent_executions]
        success_rates = []
        
        # Group by day for trend analysis
        daily_stats = {}
        for execution in recent_executions:
            day_key = execution.end_time.strftime('%Y-%m-%d')
            if day_key not in daily_stats:
                daily_stats[day_key] = {"durations": [], "successes": 0, "total": 0}
            
            daily_stats[day_key]["durations"].append(execution.total_duration)
            daily_stats[day_key]["total"] += 1
            if execution.status == WorkflowStatus.COMPLETED:
                daily_stats[day_key]["successes"] += 1
        
        # Calculate daily averages
        daily_averages = []
        for day, stats in sorted(daily_stats.items()):
            avg_duration = sum(stats["durations"]) / len(stats["durations"])
            success_rate = stats["successes"] / stats["total"]
            daily_averages.append({
                "date": day,
                "avg_duration": avg_duration,
                "success_rate": success_rate,
                "execution_count": stats["total"]
            })
        
        # Determine trends
        if len(daily_averages) >= 2:
            duration_trend = "improving" if daily_averages[-1]["avg_duration"] < daily_averages[0]["avg_duration"] else "degrading"
            success_trend = "improving" if daily_averages[-1]["success_rate"] > daily_averages[0]["success_rate"] else "degrading"
        else:
            duration_trend = "stable"
            success_trend = "stable"
        
        return {
            "period_days": days,
            "executions_analyzed": len(recent_executions),
            "overall_trends": {
                "duration_trend": duration_trend,
                "success_trend": success_trend
            },
            "daily_statistics": daily_averages,
            "performance_summary": {
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_executions": len(recent_executions),
                "successful_executions": len([m for m in recent_executions if m.status == WorkflowStatus.COMPLETED])
            },
            "optimization_opportunities": self._identify_optimization_opportunities(recent_executions)
        }
    
    def _identify_optimization_opportunities(self, executions: List[WorkflowExecutionMetrics]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        for rule_name, analyzer_func in self.optimization_rules.items():
            try:
                result = analyzer_func(executions)
                if result:
                    opportunities.append({
                        "type": rule_name,
                        "analysis": result
                    })
            except Exception as e:
                logger.error(f"❌ Error in optimization rule {rule_name}: {e}")
        
        return opportunities
    
    def _analyze_pdf_performance(self, executions: List[WorkflowExecutionMetrics]) -> Optional[Dict[str, Any]]:
        """Analyze PDF processing performance."""
        pdf_durations = []
        
        for execution in executions:
            pdf_duration = execution.agent_durations.get("pdf_processor")
            if pdf_duration:
                pdf_durations.append(pdf_duration)
        
        if not pdf_durations:
            return None
        
        avg_duration = sum(pdf_durations) / len(pdf_durations)
        
        if avg_duration > 30:  # 30 seconds threshold
            return {
                "issue": "slow_pdf_processing",
                "avg_duration": avg_duration,
                "recommendation": "Consider optimizing PDF processing or using faster PDF libraries",
                "affected_executions": len(pdf_durations)
            }
        
        return None
    
    def _analyze_embedding_performance(self, executions: List[WorkflowExecutionMetrics]) -> Optional[Dict[str, Any]]:
        """Analyze embedding creation performance."""
        embedding_durations = []
        
        for execution in executions:
            embedding_duration = execution.agent_durations.get("embedding_creator")
            if embedding_duration:
                embedding_durations.append(embedding_duration)
        
        if not embedding_durations:
            return None
        
        avg_duration = sum(embedding_durations) / len(embedding_durations)
        
        if avg_duration > 60:  # 1 minute threshold
            return {
                "issue": "slow_embedding_creation",
                "avg_duration": avg_duration,
                "recommendation": "Consider smaller chunk sizes or batching embedding requests",
                "affected_executions": len(embedding_durations)
            }
        
        return None
    
    def _analyze_search_performance(self, executions: List[WorkflowExecutionMetrics]) -> Optional[Dict[str, Any]]:
        """Analyze search performance."""
        low_result_executions = []
        
        for execution in executions:
            if execution.search_results_count < 3:  # Less than 3 results
                low_result_executions.append(execution)
        
        if len(low_result_executions) > len(executions) * 0.3:  # More than 30% have low results
            return {
                "issue": "consistently_low_search_results",
                "low_result_rate": len(low_result_executions) / len(executions),
                "recommendation": "Review search queries, chunk sizes, or vector store configuration",
                "affected_executions": len(low_result_executions)
            }
        
        return None
    
    def _analyze_memory_usage(self, executions: List[WorkflowExecutionMetrics]) -> Optional[Dict[str, Any]]:
        """Analyze memory usage patterns."""
        memory_usages = [e.memory_usage_mb for e in executions if e.memory_usage_mb > 0]
        
        if not memory_usages:
            return None
        
        avg_memory = sum(memory_usages) / len(memory_usages)
        max_memory = max(memory_usages)
        
        if max_memory > 2048:  # 2GB threshold
            return {
                "issue": "high_memory_usage",
                "avg_memory_mb": avg_memory,
                "peak_memory_mb": max_memory,
                "recommendation": "Consider reducing batch sizes or implementing memory optimization",
                "affected_executions": len([m for m in memory_usages if m > 1024])
            }
        
        return None
    
    def _analyze_concurrency(self, executions: List[WorkflowExecutionMetrics]) -> Optional[Dict[str, Any]]:
        """Analyze concurrent execution patterns."""
        # This would analyze if multiple workflows running concurrently affect performance
        # For now, return None as this requires more complex analysis
        return None


# ================================
# GLOBAL INSTANCES AND UTILITIES
# ================================

# Create global monitor instance
workflow_monitor = WorkflowMonitor()

# Registry for debugger instances
workflow_debugger_registry: Dict[int, WorkflowDebugger] = {}

def get_workflow_debugger(workflow: 'SAPRAGWorkflow') -> WorkflowDebugger:
    """
    Get or create debugger instance for a workflow.
    
    Args:
        workflow: Workflow instance
        
    Returns:
        WorkflowDebugger instance
    """
    workflow_id = id(workflow)
    if workflow_id not in workflow_debugger_registry:
        workflow_debugger_registry[workflow_id] = WorkflowDebugger(workflow)
    return workflow_debugger_registry[workflow_id]

def setup_console_alert_handler():
    """Setup simple console-based alert handler for development."""
    def console_alert_handler(alert: Dict[str, Any]):
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌"
        }
        icon = severity_icons.get(alert.get("severity", "info"), "📢")
        print(f"{icon} ALERT: {alert.get('message', 'Unknown alert')}")
    
    workflow_monitor.add_alert_callback(console_alert_handler)
    logger.info("📢 Console alert handler configured")

def setup_file_alert_handler(log_file: str = "workflow_alerts.log"):
    """Setup file-based alert handler for production."""
    def file_alert_handler(alert: Dict[str, Any]):
        try:
            with open(log_file, 'a') as f:
                alert_line = f"{alert.get('timestamp', datetime.now().isoformat())} - {alert.get('severity', 'INFO').upper()} - {alert.get('message', 'Unknown alert')}\n"
                f.write(alert_line)
        except Exception as e:
            logger.error(f"❌ Failed to write alert to file: {e}")
    
    workflow_monitor.add_alert_callback(file_alert_handler)
    logger.info(f"📁 File alert handler configured: {log_file}")


# ================================
# MODULE EXPORTS
# ================================

__all__ = [
    # Core monitoring classes
    'WorkflowMonitor',
    'WorkflowDebugger', 
    'WorkflowRecoveryManager',
    'PerformanceAnalyzer',
    
    # Data structures
    'WorkflowExecutionMetrics',
    'PerformanceThresholds',
    
    # Global instances and utilities
    'workflow_monitor',
    'get_workflow_debugger',
    'setup_console_alert_handler',
    'setup_file_alert_handler'
]

logger.info("✅ Workflow monitoring and debugging module loaded successfully")