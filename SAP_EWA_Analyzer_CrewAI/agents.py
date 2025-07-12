# agents.py - CrewAI Agents for SAP EWA Analysis
"""
CrewAI agent definitions for SAP Early Watch Alert analysis.
Defines four specialized agents that collaborate to analyze SAP system health.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from config import config
from tools import PDFProcessorTool, VectorSearchTool, HealthAnalysisTool
from models import (
    CrewExecutionResult, AgentCommunication, AgentStatus,
    SystemHealthAnalysis, AnalysisRequest
)

logger = logging.getLogger(__name__)

class SAPEWAAgents:
    """Factory class for creating SAP EWA analysis agents"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model for agents"""
        return ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=2000
        )
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all tools for agents"""
        return {
            "pdf_processor": PDFProcessorTool(),
            "vector_search": VectorSearchTool(),
            "health_analyzer": HealthAnalysisTool()
        }
    
    def create_document_processor_agent(self) -> Agent:
        """Create the Document Processing Agent"""
        return Agent(
            role="SAP Document Processing Specialist",
            goal="Extract and structure information from SAP EWA PDF documents with maximum accuracy",
            backstory="""You are an expert in SAP system documentation with years of experience 
            processing Early Watch Alert reports. You understand SAP terminology, system identifiers, 
            and document structures. Your expertise ensures no critical information is lost during extraction.
            
            You excel at:
            - Identifying SAP system IDs (SIDs) and product types
            - Extracting system environment information (Production, Test, Development)
            - Preserving document structure and metadata
            - Handling various PDF formats and quality levels
            - Detecting SAP-specific patterns and conventions""",
            verbose=config.CREW_VERBOSE,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.tools["pdf_processor"]],
            max_iter=config.MAX_ITERATIONS
        )
    
    def create_vector_manager_agent(self) -> Agent:
        """Create the Vector Database Manager Agent"""
        return Agent(
            role="Vector Database and Search Manager",
            goal="Efficiently manage document embeddings and perform intelligent semantic searches",
            backstory="""You are a specialist in vector databases and semantic search technologies. 
            You understand how to chunk documents optimally for SAP content, create meaningful embeddings, 
            and perform precise searches that return the most relevant information for system analysis.
            
            Your expertise includes:
            - Optimal text chunking strategies for SAP documentation
            - Creating high-quality embeddings using OpenAI models
            - Managing ChromaDB collections and metadata
            - Performing semantic similarity searches with filters
            - Optimizing search parameters for SAP-specific queries""",
            verbose=config.CREW_VERBOSE,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.tools["vector_search"]],
            max_iter=config.MAX_ITERATIONS
        )
    
    def create_health_analyst_agent(self) -> Agent:
        """Create the SAP System Health Analyst Agent"""
        return Agent(
            role="SAP System Health Analyst",
            goal="Analyze SAP system health indicators and provide actionable recommendations",
            backstory="""You are a senior SAP Basis consultant with deep expertise in system health 
            monitoring and Early Watch Alert interpretation. You can quickly identify critical issues, 
            assess system performance, and provide prioritized recommendations for system optimization.
            
            Your specializations include:
            - SAP system performance analysis and tuning
            - Early Watch Alert pattern recognition
            - Critical alert prioritization and escalation
            - Memory, CPU, and database performance analysis
            - SAP Note recommendations and best practices
            - Risk assessment and mitigation strategies""",
            verbose=config.CREW_VERBOSE,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.tools["health_analyzer"]],
            max_iter=config.MAX_ITERATIONS
        )
    
    def create_report_coordinator_agent(self) -> Agent:
        """Create the Report Coordinator Agent"""
        return Agent(
            role="SAP EWA Report Coordinator",
            goal="Coordinate the analysis workflow and compile comprehensive system health reports",
            backstory="""You are a project coordinator with expertise in SAP system management. 
            You ensure all aspects of the EWA analysis are completed thoroughly and that findings 
            are presented in a clear, actionable format for SAP administrators and management.
            
            Your responsibilities include:
            - Coordinating workflow between specialized agents
            - Ensuring comprehensive analysis coverage
            - Compiling findings into executive-ready reports
            - Prioritizing recommendations by business impact
            - Quality assurance and validation of results
            - Stakeholder communication and reporting""",
            verbose=config.CREW_VERBOSE,
            allow_delegation=True,
            llm=self.llm,
            tools=[],  # Coordinator delegates to other agents
            max_iter=config.MAX_ITERATIONS
        )

class SAPEWATasks:
    """Factory class for creating SAP EWA analysis tasks"""
    
    @staticmethod
    def create_document_processing_task(agent: Agent, pdf_files: List[str]) -> Task:
        """Create document processing task"""
        return Task(
            description=f"""Process the uploaded SAP EWA PDF documents and extract structured information.
            
            Files to process: {pdf_files}
            
            Requirements:
            1. Extract all text content from each PDF file
            2. Identify SAP system information (SID, product type, environment)
            3. Preserve document structure and metadata
            4. Handle any PDF processing errors gracefully
            5. Provide detailed extraction statistics
            6. Create document chunks suitable for vector embedding
            
            For each document, ensure you capture:
            - System ID (SID) - typically 3 characters
            - SAP product (S/4HANA, ERP, IBP, BusinessObjects, etc.)
            - Environment type (Production, Development, Test, Quality)
            - Document metadata (page count, file size, processing method)
            - Any version or release information found
            
            Expected Output: Structured document data with extracted text, system metadata, 
            and processing statistics for each file.""",
            agent=agent,
            expected_output="JSON object containing extracted text, system metadata, and processing statistics for each PDF file"
        )
    
    @staticmethod
    def create_vector_management_task(agent: Agent) -> Task:
        """Create vector store management task"""
        return Task(
            description="""Create embeddings and manage the vector database for semantic search.
            
            Requirements:
            1. Chunk processed documents appropriately for SAP content (1000-1500 characters)
            2. Create high-quality embeddings using OpenAI text-embedding-ada-002
            3. Store embeddings in ChromaDB with proper metadata including:
               - System ID for filtering
               - Document source and page information
               - Content type (alert, recommendation, metric, etc.)
            4. Ensure system-specific filtering capabilities
            5. Optimize chunk overlap for context preservation
            6. Validate embedding quality and storage success
            
            The vector store should enable efficient searches for:
            - System-specific health indicators
            - Cross-system pattern analysis
            - Semantic similarity for related issues
            - Performance metrics and trends
            
            Expected Output: Vector store status confirmation with metadata about stored embeddings, 
            collection statistics, and search readiness verification.""",
            agent=agent,
            expected_output="Confirmation of vector store creation with embedding statistics, metadata summary, and search capability verification"
        )
    
    @staticmethod
    def create_health_analysis_task(agent: Agent, search_queries: List[str]) -> Task:
        """Create health analysis task"""
        return Task(
            description=f"""Perform comprehensive SAP system health analysis using the processed data.
            
            Search queries to analyze: {search_queries}
            
            Requirements:
            1. Search for system-specific health indicators using semantic search
            2. Identify and categorize critical alerts by severity:
               - Critical: System down, errors, failures
               - Warning: Performance issues, recommendations
               - Info: General status and metrics
            3. Assess overall system health status for each detected system
            4. Extract key performance metrics (CPU, memory, disk, response times)
            5. Generate prioritized recommendations with SAP Note references where applicable
            6. Provide confidence scores for each assessment
            7. Identify patterns across multiple systems if applicable
            
            For each system found, analyze:
            - Critical alerts requiring immediate attention
            - Performance bottlenecks and capacity issues
            - Configuration deviations from SAP best practices
            - Recommended optimizations and maintenance actions
            - Risk factors and potential business impact
            
            Expected Output: Comprehensive health analysis report with system-specific findings, 
            prioritized recommendations, confidence scores, and executive summary.""",
            agent=agent,
            expected_output="Detailed health analysis report with system status, critical alerts, performance metrics, prioritized recommendations, and confidence assessments"
        )
    
    @staticmethod
    def create_report_compilation_task(agent: Agent) -> Task:
        """Create report compilation task"""
        return Task(
            description="""Coordinate the complete analysis workflow and compile final reports.
            
            Requirements:
            1. Ensure all analysis tasks are completed successfully
            2. Validate results consistency across different agents
            3. Compile findings into a comprehensive executive report including:
               - Executive summary with key findings
               - System-by-system health analysis
               - Critical alerts requiring immediate action
               - Performance optimization recommendations
               - Risk assessment and business impact analysis
            4. Format results for different audiences:
               - Technical details for SAP Basis teams
               - Executive summary for management
               - Action items with priorities and timelines
            5. Include workflow statistics and agent collaboration summary
            6. Provide quality assurance validation of all findings
            
            The final report should enable stakeholders to:
            - Quickly understand overall SAP landscape health
            - Identify and prioritize critical issues
            - Plan remediation activities with clear next steps
            - Understand business risk and impact
            
            Expected Output: Complete SAP EWA analysis report with executive summary, 
            detailed findings, prioritized action items, and appendices with supporting data.""",
            agent=agent,
            expected_output="Final comprehensive SAP EWA analysis report with executive summary, detailed system analysis, prioritized recommendations, and supporting documentation"
        )

def create_sap_ewa_crew(pdf_files: List[str], 
                       search_queries: Optional[List[str]] = None) -> Crew:
    """Create and configure the complete SAP EWA analysis crew"""
    
    if not search_queries:
        search_queries = [
            "critical system alerts",
            "performance issues", 
            "memory utilization",
            "database problems",
            "configuration warnings"
        ]
    
    # Initialize agent factory
    agent_factory = SAPEWAAgents()
    
    # Create agents
    document_processor = agent_factory.create_document_processor_agent()
    vector_manager = agent_factory.create_vector_manager_agent()
    health_analyst = agent_factory.create_health_analyst_agent()
    report_coordinator = agent_factory.create_report_coordinator_agent()
    
    # Create tasks
    doc_task = SAPEWATasks.create_document_processing_task(document_processor, pdf_files)
    vector_task = SAPEWATasks.create_vector_management_task(vector_manager)
    health_task = SAPEWATasks.create_health_analysis_task(health_analyst, search_queries)
    report_task = SAPEWATasks.create_report_compilation_task(report_coordinator)
    
    # Create and configure crew
    crew = Crew(
        agents=[document_processor, vector_manager, health_analyst, report_coordinator],
        tasks=[doc_task, vector_task, health_task, report_task],
        process=Process.sequential,
        verbose=config.CREW_VERBOSE,
        memory=config.CREW_MEMORY_ENABLED,
        embedder=config.get_crew_config().get("embedder")
    )
    
    return crew

def execute_sap_ewa_analysis(request: AnalysisRequest) -> CrewExecutionResult:
    """Execute complete SAP EWA analysis using CrewAI"""
    
    start_time = datetime.now()
    agent_communications = []
    
    try:
        logger.info("🚀 Starting CrewAI SAP EWA Analysis")
        
        # Create the crew
        crew = create_sap_ewa_crew(
            pdf_files=request.files,
            search_queries=request.search_queries
        )
        
        # Prepare execution inputs
        crew_inputs = {
            "pdf_files": request.files,
            "search_queries": request.search_queries,
            "system_filter": request.system_filter,
            "analysis_options": {
                "include_metrics": request.include_metrics,
                "include_recommendations": request.include_recommendations,
                "detailed_health": request.detailed_health
            },
            "timestamp": start_time.isoformat()
        }
        
        # Execute the crew
        logger.info("🔄 Executing CrewAI workflow...")
        result = crew.kickoff(inputs=crew_inputs)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Simulate agent communications (in real implementation, these would come from crew execution)
        agent_communications = [
            AgentCommunication(
                from_agent="Document Processor",
                to_agent="Vector Manager", 
                message=f"Processed {len(request.files)} PDF files successfully",
                action="documents_processed"
            ),
            AgentCommunication(
                from_agent="Vector Manager",
                to_agent="Health Analyst",
                message="Vector embeddings created and stored in ChromaDB",
                action="embeddings_ready"
            ),
            AgentCommunication(
                from_agent="Health Analyst", 
                to_agent="Report Coordinator",
                message="Health analysis completed with recommendations",
                action="analysis_complete"
            ),
            AgentCommunication(
                from_agent="Report Coordinator",
                to_agent="All Agents",
                message="Final report compiled and validated",
                action="workflow_complete"
            )
        ]
        
        logger.info("✅ CrewAI analysis completed successfully")
        
        # Create successful result
        return CrewExecutionResult(
            success=True,
            analysis_results=[],  # Would be populated from actual crew execution
            agent_communications=agent_communications,
            execution_time=execution_time,
            metadata={
                "crew_config": config.get_crew_config(),
                "files_processed": len(request.files),
                "queries_executed": len(request.search_queries),
                "completion_time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ CrewAI analysis failed: {e}")
        
        # Create error result
        return CrewExecutionResult(
            success=False,
            agent_communications=agent_communications,
            execution_time=execution_time,
            error=str(e),
            metadata={
                "error_time": datetime.now().isoformat(),
                "files_attempted": len(request.files)
            }
        )

# Convenience function for simple execution
def analyze_sap_ewa_documents(pdf_files: List[str], 
                            search_queries: Optional[List[str]] = None) -> CrewExecutionResult:
    """Simple function to analyze SAP EWA documents"""
    
    request = AnalysisRequest(
        files=pdf_files,
        search_queries=search_queries or [],
        include_metrics=True,
        include_recommendations=True,
        detailed_health=True
    )
    
    return execute_sap_ewa_analysis(request)# All 4 CrewAI agents for SAP EWA Analyzer 