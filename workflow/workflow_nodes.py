# workflow/workflow_nodes.py - All Node Implementations
"""
Workflow node implementations for SAP Early Watch Analysis.

This module contains all the individual workflow nodes that process different
stages of the SAP EWA analysis pipeline. Each node is responsible for a specific
task and follows the LangGraph node pattern.

Node Types:
- PDF Processing: Extract text from uploaded PDF files
- Embedding Creation: Generate vector embeddings from text
- Vector Storage: Store embeddings in vector database
- Search: Perform similarity search on stored vectors
- Summary: Generate intelligent summaries
- System Output: Create system-specific analysis
- Email: Send email notifications
- Completion: Finalize workflow execution

All nodes use the WorkflowNodeMixin for standardized error handling and timing.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Literal, Union

# Configure logging
logger = logging.getLogger(__name__)

# Import required modules with fallbacks
try:
    from agents import SearchAgent
    from models import WorkflowStatus
    logger.info("✅ Agent modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import agent modules: {e}")
    
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

# Import the state and mixin from core_workflow
from .core_workflow import DetailedWorkflowState, WorkflowNodeMixin


class WorkflowNodes(WorkflowNodeMixin):
    """
    Container class for all workflow node implementations.
    
    This class contains all the individual workflow nodes and routing logic.
    It inherits from WorkflowNodeMixin to get standardized error handling,
    timing, and state management capabilities.
    """
    
    def __init__(self, workflow_instance):
        """
        Initialize workflow nodes with reference to main workflow.
        
        Args:
            workflow_instance: Reference to the main SAPRAGWorkflow instance
        """
        self.workflow = workflow_instance
        logger.info("🔧 WorkflowNodes initialized")
    
    # ================================
    # CORE PROCESSING NODES
    # ================================
    
    def pdf_processing_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Process uploaded PDF files and extract text content.
        
        This node handles:
        - Loading and validating uploaded PDF files
        - Extracting text using multiple PDF processing libraries
        - Text cleaning and normalization
        - Creating document objects for further processing
        
        Args:
            state: Current workflow state containing uploaded_files
            
        Returns:
            Updated state with processed_documents and metrics
        """
        start_time = self.start_node_timer(state, "pdf_processor")
        
        try:
            # Update workflow status
            state["workflow_status"] = WorkflowStatus.PROCESSING_PDF
            state["current_agent"] = "pdf_processor"
            
            # Get uploaded files from state
            uploaded_files = state.get("uploaded_files", [])
            if not uploaded_files:
                raise ValueError("No PDF files provided for processing")
            
            logger.info(f"📄 Processing {len(uploaded_files)} PDF files")
            
            # Use PDF processor agent to extract text
            result = self.workflow.pdf_processor.process(uploaded_files)
            
            if not result.get("success"):
                raise Exception(f"PDF processing failed: {result.get('error', 'Unknown error')}")
            
            processed_files = result.get("processed_files", [])
            if not processed_files:
                raise Exception("No text content extracted from PDF files")
            
            # Convert processed files to document objects
            documents = self._create_document_objects(processed_files)
            
            # Update state with results
            state["processed_documents"] = documents
            state["total_chunks"] = len(documents)
            
            self.end_node_timer(state, "pdf_processor", start_time, True)
            
            logger.info(f"✅ PDF processing completed: {len(documents)} documents created")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "pdf_processor", start_time, False)
            return self.handle_node_error(state, "pdf_processor", e, "PDF processing")
    
    def embedding_creation_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Create vector embeddings from processed documents.
        
        This node handles:
        - Text chunking with configurable size and overlap
        - Vector embedding generation using configured model
        - Batch processing for efficiency
        - Embedding validation and formatting
        
        Args:
            state: Current workflow state with processed_documents
            
        Returns:
            Updated state with embeddings and chunks
        """
        start_time = self.start_node_timer(state, "embedding_creator")
        
        try:
            state["workflow_status"] = WorkflowStatus.CREATING_EMBEDDINGS
            state["current_agent"] = "embedding_creator"
            
            # Get processed documents
            documents = state.get("processed_documents", [])
            if not documents:
                raise ValueError("No processed documents available for embedding")
            
            logger.info(f"🔤 Creating embeddings for {len(documents)} documents")
            
            # Convert documents to format expected by embedding agent
            processed_files = self._documents_to_processed_files(documents)
            
            # Create embeddings using embedding agent
            result = self.workflow.embedding_agent.process(processed_files)
            
            if not result.get("success"):
                raise Exception(f"Embedding creation failed: {result.get('error', 'Unknown error')}")
            
            embeddings = result.get("embeddings", [])
            chunks = result.get("chunks", [])
            
            if len(embeddings) != len(chunks):
                logger.warning(f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})")
                # Trim to match smaller count
                min_count = min(len(embeddings), len(chunks))
                embeddings = embeddings[:min_count]
                chunks = chunks[:min_count]
            
            # Update state
            state["embeddings"] = embeddings
            state["processed_documents"] = self._chunks_to_documents(chunks)
            state["total_chunks"] = len(chunks)
            
            self.end_node_timer(state, "embedding_creator", start_time, True)
            
            logger.info(f"✅ Embedding creation completed: {len(embeddings)} embeddings created")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "embedding_creator", start_time, False)
            return self.handle_node_error(state, "embedding_creator", e, "Embedding creation")
    
    def vector_storage_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Store vector embeddings in the vector database and initialize search agent.
        
        This is a critical node that:
        - Creates the vector store with documents and embeddings
        - Initializes the search agent with the populated vector store
        - Validates that search functionality is working
        - Sets up the system for similarity search operations
        
        Args:
            state: Current workflow state with embeddings and documents
            
        Returns:
            Updated state with vector_store_ready flag and search agent
        """
        start_time = self.start_node_timer(state, "vector_storage")
        
        try:
            state["workflow_status"] = WorkflowStatus.STORING_VECTORS
            state["current_agent"] = "vector_storage"
            
            # Get documents and embeddings
            documents = state.get("processed_documents", [])
            embeddings = state.get("embeddings", [])
            
            if not documents:
                raise ValueError("No documents available for vector storage")
            
            logger.info(f"🗄️ Storing {len(documents)} documents and {len(embeddings)} embeddings")
            
            # Add documents to ChromaDB vector store
            self.workflow.vector_store.add_documents(documents, metadatas=None, ids=None)
            
            logger.info(f"✅ Documents added to ChromaDB vector store")
            
            # Initialize search agent with the populated vector store
            search_agent_config = {
                **self.workflow.agent_config,
                'vector_store': self.workflow.vector_store,
                'embedding_agent': self.workflow.embedding_agent
            }
            
            self.workflow.search_agent = SearchAgent(search_agent_config)
            
            # Validate search agent setup
            if not hasattr(self.workflow.search_agent, 'vector_store') or not self.workflow.search_agent.vector_store:
                raise Exception("Search agent was not properly initialized with vector store")
            
            # Test search functionality with a simple query
            test_result = self.workflow.search_agent.search("test", {})
            if not test_result.get("success"):
                logger.warning("⚠️ Search agent test failed, but continuing")
            
            # Mark vector store as ready
            state["vector_store_ready"] = True
            
            self.end_node_timer(state, "vector_storage", start_time, True)
            
            logger.info("✅ Vector storage completed, search agent ready")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "vector_storage", start_time, False)
            return self.handle_node_error(state, "vector_storage", e, "Vector storage")
    
    # ================================
    # SEARCH AND ANALYSIS NODES
    # ================================
    
    def search_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Perform similarity search on the vector store.
        
        This node executes the core search functionality:
        - Takes user query and search filters from state
        - Performs vector similarity search
        - Filters and ranks results
        - Formats results for downstream processing
        
        Args:
            state: Current workflow state with user_query and search_filters
            
        Returns:
            Updated state with search_results
        """
        start_time = self.start_node_timer(state, "search_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SEARCHING
            state["current_agent"] = "search_agent"
            
            # Better query extraction and validation
            query = state.get("user_query", "").strip()
            search_filters = state.get("search_filters", {})
            
            # Debug logging
            logger.info(f"Raw query from state: '{state.get('user_query', 'NOT_FOUND')}'")
            logger.info(f"Cleaned query: '{query}'")
            logger.info(f"Search filters: {search_filters}")
            logger.info(f"State type: {type(state)}")
            logger.info(f"State keys: {list(state.keys()) if hasattr(state, 'keys') else 'No keys method'}")
            
            if not query:
                # Check alternative query locations in state
                alt_query = state.get("query", "").strip()
                if alt_query:
                    query = alt_query
                    logger.info(f"Using alternative query: '{query}'")
                else:
                    # Additional debugging
                    logger.error(f"State content: {state}")
                    logger.error(f"State user_query: {state.get('user_query', 'NOT_FOUND')}")
                    logger.error(f"State query: {state.get('query', 'NOT_FOUND')}")
                    raise ValueError("No search query provided in state")
            
            if len(query) < 3:
                raise ValueError(f"Search query too short: '{query}' (minimum 3 characters)")
            
            if not self.workflow.search_agent:
                raise Exception("Search agent not initialized - vector store may not be ready")
            
            logger.info(f"🔍 Searching for: '{query}' with filters: {search_filters}")
            
            # Perform the search
            result = self.workflow.search_agent.search(query, search_filters)
            
            if not result.get("success"):
                raise Exception(f"Search failed: {result.get('error', 'Unknown error')}")
            
            search_results = result.get("search_results", [])
            
            # Store results in state
            state["search_results"] = search_results
            
            self.end_node_timer(state, "search_agent", start_time, True)
            
            logger.info(f"✅ Search completed: {len(search_results)} results found")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "search_agent", start_time, False)
            return self.handle_node_error(state, "search_agent", e, "Search execution")
    
    def summary_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Generate intelligent summary from search results.
        
        This node creates comprehensive analysis including:
        - Executive summary of findings
        - Critical issues identification
        - Actionable recommendations
        - Confidence scoring based on result quality
        
        Args:
            state: Current workflow state with search_results
            
        Returns:
            Updated state with summary data
        """
        start_time = self.start_node_timer(state, "summary_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SUMMARIZING
            state["current_agent"] = "summary_agent"
            
            # Get search results and query
            search_results = state.get("search_results", [])
            query = state.get("user_query", "")
            
            logger.info(f"📝 Generating summary for {len(search_results)} search results")
            
            # Generate summary using summary agent
            result = self.workflow.summary_agent.generate_summary(search_results, query)
            
            if not result.get("success"):
                raise Exception(f"Summary generation failed: {result.get('error', 'Unknown error')}")
            
            summary = result.get("summary", {})
            
            # Store summary in state
            state["summary"] = summary
            
            self.end_node_timer(state, "summary_agent", start_time, True)
            
            # Log summary statistics
            critical_count = len(summary.get("critical_findings", []))
            recommendation_count = len(summary.get("recommendations", []))
            confidence = summary.get("confidence_score", 0) * 100
            
            logger.info(f"✅ Summary generated: {critical_count} critical findings, "
                       f"{recommendation_count} recommendations (confidence: {confidence:.1f}%)")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "summary_agent", start_time, False)
            return self.handle_node_error(state, "summary_agent", e, "Summary generation")
    
    def system_output_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Generate system-specific analysis outputs.
        
        This node creates detailed analysis for each SAP system found:
        - System health assessments (HEALTHY/WARNING/CRITICAL)
        - System-specific alerts and recommendations
        - Performance metrics extraction
        - Individual system reports
        
        Args:
            state: Current workflow state with search_results
            
        Returns:
            Updated state with system_summaries
        """
        start_time = self.start_node_timer(state, "system_output_agent")
        
        try:
            state["workflow_status"] = WorkflowStatus.SYSTEM_OUTPUT
            state["current_agent"] = "system_output_agent"
            
            search_results = state.get("search_results", [])
            
            logger.info(f"🖥️ Generating system outputs for {len(search_results)} search results")
            
            # Generate system-specific outputs
            result = self.workflow.system_output_agent.generate_system_outputs(search_results)
            
            if not result.get("success"):
                raise Exception(f"System output generation failed: {result.get('error', 'Unknown error')}")
            
            system_summaries = result.get("system_summaries", {})
            
            # Store system summaries in state
            state["system_summaries"] = system_summaries
            
            self.end_node_timer(state, "system_output_agent", start_time, True)
            
            # Log system analysis statistics
            systems_analyzed = len(system_summaries)
            critical_systems = sum(1 for s in system_summaries.values() 
                                 if hasattr(s, 'overall_health') and 
                                 str(s.overall_health).upper() == 'CRITICAL')
            
            logger.info(f"✅ System analysis completed: {systems_analyzed} systems analyzed, "
                       f"{critical_systems} critical systems identified")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "system_output_agent", start_time, False)
            return self.handle_node_error(state, "system_output_agent", e, "System output generation")
    
    # ================================
    # COMMUNICATION AND COMPLETION NODES
    # ================================
    
    def email_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Send email notifications with analysis results.
        
        This optional node handles:
        - Email content formatting with analysis results
        - SMTP delivery with retry logic
        - Multiple recipient support
        - Email delivery status tracking
        
        Args:
            state: Current workflow state with summary and system_summaries
            
        Returns:
            Updated state with email_sent status
        """
        start_time = self.start_node_timer(state, "email_agent")
        
        try:
            # Check if email agent is available and configured
            if not self.workflow.email_agent:
                self._add_agent_message(state, "email_agent", "📧 Email not configured - skipping", "skipped")
                state["email_sent"] = False
                self.end_node_timer(state, "email_agent", start_time, True)
                return state
            
            state["workflow_status"] = WorkflowStatus.SENDING_EMAIL
            state["current_agent"] = "email_agent"
            
            # Get email recipients
            recipients = state.get("email_recipients", [])
            if not recipients:
                self._add_agent_message(state, "email_agent", "📧 No recipients specified - skipping", "skipped")
                state["email_sent"] = False
                self.end_node_timer(state, "email_agent", start_time, True)
                return state
            
            logger.info(f"📧 Sending email to {len(recipients)} recipients")
            
            # Prepare email data
            email_data = {
                'recipients': recipients,
                'summary': state.get("summary", {}),
                'query': state.get("user_query", ""),
                'system_summaries': state.get("system_summaries", {})
            }
            
            # Send email using email agent
            result = self.workflow.email_agent.send_email(email_data)
            
            email_sent = result.get("success", False)
            state["email_sent"] = email_sent
            
            self.end_node_timer(state, "email_agent", start_time, email_sent)
            
            if email_sent:
                logger.info(f"✅ Email sent successfully to {len(recipients)} recipients")
            else:
                logger.warning(f"⚠️ Email sending failed: {result.get('error', 'Unknown error')}")
            
            return state
            
        except Exception as e:
            self.end_node_timer(state, "email_agent", start_time, False)
            state["email_sent"] = False
            self._add_agent_message(state, "email_agent", f"📧 Email failed: {str(e)}", "error")
            # Don't fail the entire workflow for email errors
            logger.warning(f"⚠️ Email node error (continuing): {e}")
            return state
    
    def completion_node(self, state: DetailedWorkflowState) -> DetailedWorkflowState:
        """
        Complete the workflow and finalize metrics.
        
        This final node:
        - Sets workflow status to completed
        - Calculates total processing time
        - Logs final statistics and summary
        - Prepares final state for return
        
        Args:
            state: Current workflow state
            
        Returns:
            Final completed workflow state
        """
        try:
            state["workflow_status"] = WorkflowStatus.COMPLETED
            state["current_agent"] = "complete"
            
            # Calculate total processing time
            processing_times = state.get("processing_times", {})
            total_time = sum(processing_times.values())
            
            # Update workflow statistics
            self.workflow.execution_count += 1
            self.workflow.total_processing_time += total_time
            self.workflow.last_execution_time = total_time
            
            # Log completion statistics
            logger.info("🎉 Workflow completed successfully!")
            logger.info(f"📊 Total processing time: {total_time:.2f}s")
            logger.info(f"📄 Documents processed: {state.get('total_chunks', 0)}")
            logger.info(f"🔍 Search results: {len(state.get('search_results', []))}")
            logger.info(f"🖥️ Systems analyzed: {len(state.get('system_summaries', {}))}")
            logger.info(f"📧 Email sent: {'Yes' if state.get('email_sent', False) else 'No'}")
            
            # Add final completion message
            self._add_agent_message(
                state, 
                "workflow", 
                f"🎉 Workflow completed successfully in {total_time:.2f}s",
                "completed"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Completion node error: {e}")
            return self.handle_node_error(state, "complete", e, "Workflow completion")
    
    # ================================
    # ROUTING FUNCTIONS
    # ================================
    
    def route_after_vector_storage(self, state: DetailedWorkflowState) -> Literal["search", "complete"]:
        """
        Conditional routing after vector storage is complete.
        
        Determines whether to proceed with search or complete the workflow:
        - If user_query exists and vector store is ready → proceed to search
        - If no query or vector store not ready → complete workflow
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "search" or "complete"
        """
        try:
            user_query = state.get("user_query", "").strip()
            vector_store_ready = state.get("vector_store_ready", False)
            has_search_agent = self.workflow.search_agent is not None
            
            logger.info(f"🔀 Routing decision:")
            logger.info(f"  - Query: '{user_query}' (exists: {bool(user_query)})")
            logger.info(f"  - Vector store ready: {vector_store_ready}")
            logger.info(f"  - Search agent ready: {has_search_agent}")
            
            # Only proceed to search if we have a query and everything is ready
            if user_query and vector_store_ready and has_search_agent:
                logger.info("🔀 Routing to: search")
                return "search"
            else:
                logger.info("🔀 Routing to: complete (no search needed)")
                return "complete"
                
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
            return "complete"  # Safe default
    
    def route_after_system_output(self, state: DetailedWorkflowState) -> Literal["send_email", "complete"]:
        """
        Conditional routing after system output generation.
        
        Determines whether to send email notifications:
        - If email is enabled, configured, and recipients exist → send email
        - Otherwise → complete workflow
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "send_email" or "complete"
        """
        try:
            email_enabled = self.workflow.config.get("email_enabled", False)
            auto_send = self.workflow.config.get("auto_send_results", False)
            has_recipients = bool(state.get("email_recipients"))
            has_email_agent = self.workflow.email_agent is not None
            
            logger.info(f"🔀 Email routing decision:")
            logger.info(f"  - Email enabled: {email_enabled}")
            logger.info(f"  - Auto send: {auto_send}")
            logger.info(f"  - Has recipients: {has_recipients}")
            logger.info(f"  - Email agent ready: {has_email_agent}")
            
            # Send email if all conditions are met
            if email_enabled and auto_send and has_recipients and has_email_agent:
                logger.info("🔀 Routing to: send_email")
                return "send_email"
            else:
                logger.info("🔀 Routing to: complete (no email needed)")
                return "complete"
                
        except Exception as e:
            logger.error(f"❌ Email routing error: {e}")
            return "complete"  # Safe default
    
    # ================================
    # UTILITY FUNCTIONS
    # ================================
    
    def _create_document_objects(self, processed_files: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert processed file data to document objects.
        
        Creates document objects that are compatible with the vector store
        and contain proper metadata for downstream processing.
        
        Args:
            processed_files: List of processed file dictionaries
            
        Returns:
            List of document objects
        """
        documents = []
        
        for file_data in processed_files:
            try:
                # Try to use LangChain Document if available
                try:
                    from langchain.schema import Document
                    doc = Document(
                        page_content=file_data.get('text', ''),
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0),
                            'character_count': file_data.get('character_count', 0),
                            'word_count': file_data.get('word_count', 0),
                            'processing_timestamp': file_data.get('processing_timestamp', datetime.now().isoformat())
                        }
                    )
                except ImportError:
                    # Fallback to simple object
                    class DocumentLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    doc = DocumentLike(
                        page_content=file_data.get('text', ''),
                        metadata={
                            'source': file_data.get('filename', 'unknown'),
                            'size': file_data.get('size', 0)
                        }
                    )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error creating document object: {e}")
                continue
        
        return documents
    
    def _documents_to_processed_files(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert document objects back to processed file format for embedding agent.
        
        Args:
            documents: List of document objects
            
        Returns:
            List of processed file dictionaries
        """
        processed_files = []
        
        for doc in documents:
            try:
                file_data = {
                    'text': getattr(doc, 'page_content', str(doc)),
                    'filename': getattr(doc, 'metadata', {}).get('source', 'unknown'),
                    'size': getattr(doc, 'metadata', {}).get('size', 0)
                }
                processed_files.append(file_data)
            except Exception as e:
                logger.warning(f"Error converting document: {e}")
                continue
        
        return processed_files
    
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert text chunks back to document objects.
        
        Args:
            chunks: List of text chunk dictionaries
            
        Returns:
            List of document objects
        """
        documents = []
        
        for chunk in chunks:
            try:
                # Try LangChain Document first
                try:
                    from langchain.schema import Document
                    doc = Document(
                        page_content=chunk.get('page_content', chunk.get('text', '')),
                        metadata=chunk.get('metadata', {})
                    )
                except ImportError:
                    # Fallback
                    class DocumentLike:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    doc = DocumentLike(
                        page_content=chunk.get('page_content', chunk.get('text', '')),
                        metadata=chunk.get('metadata', {})
                    )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error converting chunk to document: {e}")
                continue
        
        return documents


# ================================
# MODULE EXPORTS
# ================================

__all__ = [
    'WorkflowNodes'
]

logger.info("✅ WorkflowNodes module loaded successfully")