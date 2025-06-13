import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import spacy
from tqdm import tqdm

from llm import get_llm_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFRAG:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "vector_store"
    ):
        """Initialize the RAG system with embedding model and vector store.
        
        Args:
            model_name: Name of the embedding model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist vector store
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def load_pdf(self, pdf_path: str) -> None:
        """Load and process a PDF file into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
        """
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from PDF.")
            
            # Process and clean text
            processed_docs = []
            for page in tqdm(pages, desc="Processing pages"):
                # Clean and process text using spaCy
                doc = self.nlp(page.page_content)
                cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
                
                # Create new document with metadata
                processed_doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_path,
                        "page": page.metadata.get("page", 0),
                        "processed_date": datetime.now().isoformat()
                    }
                )
                processed_docs.append(processed_doc)
            logger.info(f"Processed {len(processed_docs)} documents after cleaning.")
            
            # Split into chunks
            logger.info("Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(processed_docs)
            logger.info(f"Split into {len(chunks)} chunks.")
            
            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new vector store...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                logger.info("Updating existing vector store...")
                self.vector_store.add_documents(chunks)
            
            # Save vector store
            self.save_vector_store()
            logger.info(f"Successfully processed and stored {len(chunks)} chunks from {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            traceback.print_exc()
            raise

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.pdf") -> None:
        """Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            glob_pattern: Pattern to match PDF files
        """
        try:
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=PyPDFLoader
            )
            pdf_files = loader.load()
            
            for pdf_file in pdf_files:
                self.load_pdf(pdf_file.metadata["source"])
                
        except Exception as e:
            logger.error(f"Error loading directory: {str(e)}")
            raise

    def save_vector_store(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store is not None:
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"Vector store saved to {self.persist_directory}")

    def load_vector_store(self) -> None:
        """Load the vector store from disk."""
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings
                )
                logger.info(f"Vector store loaded from {self.persist_directory}")
            else:
                logger.warning(f"No existing vector store found at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def search(
        self,
        query: str,
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search the vector store for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_criteria: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise Exception("No documents loaded. Please load a PDF first.")
        
        try:
            # Perform similarity search with optional filtering
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_criteria
            )
            return docs
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def get_response(
        self,
        query: str,
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get a response using RAG with the loaded documents.
        
        Args:
            query: User query
            k: Number of relevant documents to retrieve
            filter_criteria: Optional metadata filters
            
        Returns:
            Generated response
        """
        try:
            # Search for relevant documents
            relevant_docs = self.search(query, k=k, filter_criteria=filter_criteria)
            
            if not relevant_docs:
                return "No relevant context found for the query."

            # Combine relevant documents with metadata
            context_parts = []
            for doc in relevant_docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                context_parts.append(f"Source: {source} (Page {page})\\n{doc.page_content}")
            
            context = "\\n\\n---\\n\\n".join(context_parts)
            
            # Get LLM response with context
            prompt = f"""Context: {context}

Question: {query}

Please provide a comprehensive answer based on the given context. If the context doesn\'t contain enough information to answer the question, please say so. Include relevant page numbers and sources in your response when applicable."""
            
            return get_llm_response(prompt)
        except Exception as e:
            logger.error(f"Error in RAG operation: {str(e)}")
            return f"Error generating response: {str(e)}"\

def get_rag_response(
    query: str,
    pdf_path: Optional[str] = None,
    directory_path: Optional[str] = None,
    documents: Optional[List[str]] = None
) -> str:
    """Convenience function for backward compatibility and simple usage.
    
    Args:
        query: User query
        pdf_path: Optional path to a single PDF file
        directory_path: Optional path to directory containing PDFs
        documents: Optional list of text documents for simple matching
        
    Returns:
        Generated response
    """
    try:
        rag = PDFRAG()
        
        if pdf_path:
            rag.load_pdf(pdf_path)
            return rag.get_response(query)
        elif directory_path:
            rag.load_directory(directory_path)
            return rag.get_response(query)
        elif documents:
            # Fallback to simple string matching for backward compatibility
            relevant_docs = [doc for doc in documents if query.lower() in doc.lower()]
            context = "\\n".join(relevant_docs)
            if not context:
                return "No relevant context found for the query."
            prompt = f"Context: {context}\\n\\nQuestion: {query}"
            return get_llm_response(prompt)
        else:
            return "Please provide either a PDF path, directory path, or a list of documents."
    except Exception as e:
        logger.error(f"Error in RAG operation: {str(e)}")
        return f"Error: {str(e)}"\

if __name__ == "__main__":
    # Example usage
    print("--- RAG PDF Example ---")
    
    # Example with a single PDF
    pdf_path = "stateful-agents-lab/attention.pdf"  # Replace with your PDF path
    if os.path.exists(pdf_path):
        rag = PDFRAG()
        rag.load_pdf(pdf_path)
        
        # Example queries
        queries = [
            "Explain the fundamental concept of attention in neural networks. What problem does it solve and how does it differ from traditional approaches?",
            "Describe the mathematical formulation of attention mechanism. Include the key equations and explain what each component represents.",
            "What are the different types of attention mechanisms mentioned in the document? Compare and contrast their approaches and use cases.",
            "How does self-attention work in transformer models? Explain the process of computing attention scores and the role of queries, keys, and values.",
            "What are the computational and memory requirements of attention mechanisms? Discuss the challenges and potential solutions for scaling attention.",
            "Explain the relationship between attention and sequence modeling. How does attention help in capturing long-range dependencies?",
            "What are the practical applications of attention mechanisms beyond transformers? Provide specific examples from the document.",
            "How does attention contribute to the interpretability of neural networks? Discuss any visualization or analysis techniques mentioned."
        ]
        
        for query in queries:
            print(f"\\n{'='*100}")
            print(f"Query: {query}")
            print('-'*100)
            response = rag.get_response(
                query,
                k=4,  # Increased to 4 chunks for more comprehensive answers
                filter_criteria=None  # No specific filtering, search across all content
            )
            print(f"Response: {response}")
            print(f"{'='*100}\\n")
    else:
        print(f"PDF file not found: {pdf_path}")
        
        # Fallback to simple example
        print("\\n--- Simple RAG Example ---")
        documents = [
            "Python is a versatile programming language.",
            "It is widely used for web development, data analysis, artificial intelligence, and scientific computing.",
            "Python\'s simplicity and extensive libraries make it popular for beginners and experts alike."
        ]
        rag_query = "What are the common uses of Python?"
        rag_response = get_rag_response(rag_query, documents=documents)
        print("RAG Query:", rag_query)
        print("RAG Response:", rag_response) 