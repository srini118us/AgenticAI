from typing import Any, Dict
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .base_node import BaseNode, NodeState

class LLMNode(BaseNode):
    """Node for handling LLM interactions"""
    
    def __init__(self, node_id: str = "llm", model_name: str = "gpt-3.5-turbo"):
        super().__init__(node_id)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.system_prompt = """You are a helpful AI assistant. Provide accurate and relevant responses 
        based on the given input. If you're unsure about something, acknowledge the uncertainty."""
    
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute LLM processing"""
        try:
            self.update_state(status="running")
            
            # Extract the query from input data
            query = input_data.get("query", "")
            if not query:
                raise ValueError("No query provided for LLM processing")
            
            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ]
            
            # Get response from LLM
            response = await self.llm.agenerate([messages])
            result = response.generations[0][0].text
            
            self.update_state(
                status="completed",
                result=result,
                metadata={"model": self.llm.model_name}
            )
            
            return self.state
            
        except Exception as e:
            self.update_state(status="failed", error=str(e))
            return self.state 