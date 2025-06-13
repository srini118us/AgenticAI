from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

class NodeState(BaseModel):
    """Base state model for all nodes"""
    node_id: str
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseNode(ABC):
    """Base class for all nodes in the system"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.state = NodeState(node_id=node_id)
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute the node's main logic"""
        pass
    
    def get_state(self) -> NodeState:
        """Get the current state of the node"""
        return self.state
    
    def update_state(self, **kwargs) -> None:
        """Update the node's state"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def reset(self) -> None:
        """Reset the node's state"""
        self.state = NodeState(node_id=self.node_id) 