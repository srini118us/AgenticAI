from typing import Any, Dict, List, Optional
from .base_node import BaseNode, NodeState

class SupervisorNode(BaseNode):
    """Supervisor node that orchestrates the workflow between different nodes"""
    
    def __init__(self, node_id: str = "supervisor"):
        super().__init__(node_id)
        self.nodes: Dict[str, BaseNode] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_node(self, node: BaseNode) -> None:
        """Register a node with the supervisor"""
        self.nodes[node.node_id] = node
    
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute the workflow orchestration"""
        try:
            self.update_state(status="running")
            
            # Get the initial task type from input
            task_type = input_data.get("task_type", "unknown")
            current_node_id = self._determine_next_node(task_type, input_data)
            
            while current_node_id:
                # Execute the current node
                node = self.nodes[current_node_id]
                node_state = await node.execute(input_data)
                
                # Record execution
                self.execution_history.append({
                    "node_id": current_node_id,
                    "state": node_state.dict(),
                    "timestamp": input_data.get("timestamp")
                })
                
                # Check if validation is needed
                if current_node_id != "validator" and node_state.status == "completed":
                    # Route to validator
                    validator = self.nodes.get("validator")
                    if validator:
                        validation_state = await validator.execute({
                            "input": node_state.result,
                            "task_type": task_type
                        })
                        
                        if validation_state.status == "failed":
                            # If validation fails, determine next node based on failure reason
                            current_node_id = self._determine_next_node(
                                task_type,
                                {"validation_error": validation_state.error}
                            )
                            continue
                
                # If we reach here with validation passed or no validation needed
                if node_state.status == "completed":
                    self.update_state(
                        status="completed",
                        result=node_state.result,
                        metadata={"execution_history": self.execution_history}
                    )
                    break
                
                # Determine next node
                current_node_id = self._determine_next_node(task_type, node_state.dict())
            
            return self.state
            
        except Exception as e:
            self.update_state(status="failed", error=str(e))
            return self.state
    
    def _determine_next_node(self, task_type: str, context: Dict[str, Any]) -> Optional[str]:
        """Determine which node should be executed next based on task type and context"""
        if "validation_error" in context:
            # If validation failed, route back to appropriate node
            error = context["validation_error"]
            if "llm" in error.lower():
                return "llm"
            elif "rag" in error.lower():
                return "rag"
            elif "web" in error.lower():
                return "web_scraper"
        
        # Initial routing based on task type
        if task_type == "llm":
            return "llm"
        elif task_type == "rag":
            return "rag"
        elif task_type == "web_search":
            return "web_scraper"
        
        return None
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history of the workflow"""
        return self.execution_history 