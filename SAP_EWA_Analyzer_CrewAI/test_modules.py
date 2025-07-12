# test_modules.py - Simple Test Script for CrewAI EWA Analyzer
"""
Simple test script to verify all modules can be imported and basic functionality works.
Run this after setting up the project to ensure everything is configured correctly.
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test all module imports"""
    print("🧪 Testing module imports...")
    
    try:
        # Test config
        from config import config, get_openai_api_key
        print("✅ config.py - OK")
        
        # Test models
        from models import (
            HealthStatus, SAPProduct, SystemEnvironment,
            SAPSystemInfo, AnalysisRequest, CrewExecutionResult
        )
        print("✅ models.py - OK")
        
        # Test tools
        from tools import PDFProcessorTool, VectorSearchTool, HealthAnalysisTool
        print("✅ tools.py - OK")
        
        # Test agents
        from agents import SAPEWAAgents, create_sap_ewa_crew, analyze_sap_ewa_documents
        print("✅ agents.py - OK")
        
        print("🎉 All modules imported successfully!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    print("🧪 Testing configuration...")
    
    try:
        from config import config
        
        # Test configuration validation
        validation = config.validate_config()
        
        print(f"Configuration valid: {validation['valid']}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  ❌ {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  ⚠️ {warning}")
        
        # Test specific config values
        print(f"App Title: {config.APP_TITLE}")
        print(f"Chunk Size: {config.CHUNK_SIZE}")
        print(f"Top K Results: {config.TOP_K_RESULTS}")
        print(f"Debug Mode: {config.DEBUG_MODE}")
        
        api_key = get_openai_api_key()
        if api_key:
            print(f"OpenAI API Key: {'✅ Set' if api_key.startswith('sk-') else '⚠️ Set but format unclear'}")
        else:
            print("OpenAI API Key: ❌ Not set")
        
        print("✅ Configuration test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_models():
    """Test data models"""
    print("🧪 Testing data models...")
    
    try:
        from models import (
            HealthStatus, SAPProduct, SystemEnvironment,
            SAPSystemInfo, AnalysisRequest, create_system_info,
            create_health_alert
        )
        
        # Test enum creation
        health = HealthStatus.HEALTHY
        product = SAPProduct.S4HANA
        env = SystemEnvironment.PRODUCTION
        print(f"Health Status: {health.value}")
        print(f"SAP Product: {product.value}")
        print(f"Environment: {env.value}")
        
        # Test system info creation
        system_info = create_system_info("PRD", "S/4HANA", "Production")
        print(f"System Info: {system_info.to_dict()}")
        
        # Test health alert creation
        alert = create_health_alert("critical", "Memory", "High memory usage detected", "PRD")
        print(f"Health Alert: {alert.to_dict()}")
        
        # Test analysis request
        request = AnalysisRequest(
            files=["test.pdf"],
            search_queries=["test query"],
            include_metrics=True
        )
        print(f"Analysis Request: {request.to_dict()}")
        
        print("✅ Data models test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False

def test_tools():
    """Test CrewAI tools (basic initialization)"""
    print("🧪 Testing CrewAI tools...")
    
    try:
        from tools import PDFProcessorTool, VectorSearchTool, HealthAnalysisTool
        
        # Test tool initialization
        pdf_tool = PDFProcessorTool()
        print(f"PDF Tool: {pdf_tool.name} - {pdf_tool.description}")
        
        vector_tool = VectorSearchTool()
        print(f"Vector Tool: {vector_tool.name} - {vector_tool.description}")
        
        health_tool = HealthAnalysisTool()
        print(f"Health Tool: {health_tool.name} - {health_tool.description}")
        
        # Test health analysis with sample content
        sample_content = """
        SAP System ID: PRD
        Product: SAP S/4HANA
        Status: Productive
        Critical: Memory utilization high
        Warning: Performance degradation detected
        """
        
        health_result = health_tool._run(sample_content, "PRD")
        if health_result["success"]:
            analysis = health_result["analysis"]
            print(f"Health Analysis - System: {analysis['system_id']}, Status: {analysis['overall_status']}")
        else:
            print(f"Health analysis failed: {health_result.get('error', 'Unknown error')}")
        
        print("✅ Tools test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False

def test_agents():
    """Test CrewAI agents (basic initialization)"""
    print("🧪 Testing CrewAI agents...")
    
    try:
        from agents import SAPEWAAgents
        
        # Test agent factory
        agent_factory = SAPEWAAgents()
        
        # Test LLM initialization
        print(f"LLM Model: {agent_factory.llm.model_name}")
        
        # Test tools initialization
        tools = agent_factory.tools
        print(f"Available tools: {list(tools.keys())}")
        
        # Test agent creation (without actually creating crew to avoid API calls)
        doc_agent = agent_factory.create_document_processor_agent()
        print(f"Document Agent: {doc_agent.role}")
        
        vector_agent = agent_factory.create_vector_manager_agent()
        print(f"Vector Agent: {vector_agent.role}")
        
        health_agent = agent_factory.create_health_analyst_agent()
        print(f"Health Agent: {health_agent.role}")
        
        coordinator_agent = agent_factory.create_report_coordinator_agent()
        print(f"Coordinator Agent: {coordinator_agent.role}")
        
        print("✅ Agents test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Agents test failed: {e}")
        return False

def test_dependencies():
    """Test external dependencies"""
    print("🧪 Testing external dependencies...")
    
    dependencies = [
        ("streamlit", "Streamlit web framework"),
        ("crewai", "CrewAI framework"),
        ("langchain_openai", "LangChain OpenAI integration"),
        ("chromadb", "ChromaDB vector database"),
        ("pdfplumber", "PDF processing"),
        ("python-dotenv", "Environment variables"),
        ("pandas", "Data manipulation")
    ]
    
    missing_deps = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
        except ImportError:
            print(f"❌ {dep} - {description} (MISSING)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("✅ All dependencies available\n")
        return True

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting CrewAI EWA Analyzer Tests")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Models", test_models),
        ("Tools", test_tools),
        ("Agents", test_agents)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        
        print("-" * 30)
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your CrewAI EWA Analyzer is ready to use.")
        print("Next step: Run 'streamlit run app.py' to start the application.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        print("Common fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Set OpenAI API key in .env file")
        print("3. Check Python version (3.8+ required)")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)