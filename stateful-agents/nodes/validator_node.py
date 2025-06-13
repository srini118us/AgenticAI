from typing import Any, Dict
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .base_node import BaseNode, NodeState

class ValidatorNode(BaseNode):
    """Node for validating outputs from other nodes"""
    
    def __init__(self, node_id: str = "validator"):
        super().__init__(node_id)
        self.llm = ChatOpenAI(temperature=0)
        self.validation_criteria = {
            "llm": """Validate the LLM response for:
            1. Relevance to the query
            2. Completeness of the answer
            3. Factual accuracy
            4. Clarity and coherence
            If any of these criteria are not met, specify which one failed.""",
            
            "rag": """Validate the RAG response for:
            1. Relevance to the query
            2. Proper use of retrieved context
            3. Completeness of the answer
            4. Accuracy of information
            If any of these criteria are not met, specify which one failed.""",
            
            "web_scraper": """Validate the web scraping results for:
            1. Successful retrieval of content
            2. Relevance of scraped content
            3. Completeness of information
            4. Clean and readable text
            If any of these criteria are not met, specify which one failed."""
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute validation"""
        try:
            self.update_state(status="running")
            
            task_type = input_data.get("task_type")
            if not task_type or task_type not in self.validation_criteria:
                raise ValueError(f"Invalid task type for validation: {task_type}")
            
            content = input_data.get("input")
            if not content:
                raise ValueError("No content provided for validation")
            
            # Prepare validation prompt
            system_prompt = self.validation_criteria[task_type]
            human_prompt = f"""Please validate the following {task_type} output:
            
            Content to validate:
            {content}
            
            Provide a validation result in the following format:
            Status: [PASS/FAIL]
            Reason: [If FAIL, specify which criteria failed and why]
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Get validation result
            response = await self.llm.agenerate([messages])
            validation_result = response.generations[0][0].text
            
            # Parse validation result
            status = "completed" if "Status: PASS" in validation_result else "failed"
            error = None if status == "completed" else validation_result
            
            self.update_state(
                status=status,
                result=validation_result,
                metadata={
                    "task_type": task_type,
                    "validation_criteria": self.validation_criteria[task_type]
                }
            )
            
            return self.state
            
        except Exception as e:
            self.update_state(status="failed", error=str(e))
            return self.state 