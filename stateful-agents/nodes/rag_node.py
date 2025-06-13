from typing import Any, Dict, List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from .base_node import BaseNode, NodeState

class RAGNode(BaseNode):
    """Node for handling Retrieval Augmented Generation"""
    
    def __init__(self, node_id: str = "rag"):
        super().__init__(node_id)
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.llm = ChatOpenAI(temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the vector store"""
        texts = self.text_splitter.split_text("\n".join(documents))
        if not self.vector_store:
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_texts(texts)
    
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute RAG processing"""
        try:
            self.update_state(status="running")
            
            if not self.vector_store:
                raise ValueError("No documents have been added to the RAG system")
            
            query = input_data.get("query", "")
            if not query:
                raise ValueError("No query provided for RAG processing")
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            # Get response
            result = await qa_chain.arun(query)
            
            self.update_state(
                status="completed",
                result=result,
                metadata={
                    "num_documents": len(self.vector_store.get()["ids"]),
                    "query": query
                }
            )
            
            return self.state
            
        except Exception as e:
            self.update_state(status="failed", error=str(e))
            return self.state 