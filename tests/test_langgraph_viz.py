# test_langgraph_viz.py
# Test if LangGraph visualization works outside Jupyter

def test_basic_langgraph():
    """Test basic LangGraph functionality"""
    print("🧪 Testing Basic LangGraph...")
    
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        # Create a simple workflow
        class SimpleState(TypedDict):
            message: str
            count: int
        
        def node1(state):
            return {"message": "Hello from node1", "count": state.get("count", 0) + 1}
        
        def node2(state):
            return {"message": "Hello from node2", "count": state.get("count", 0) + 1}
        
        # Build graph
        graph = StateGraph(SimpleState)
        graph.add_node("node1", node1)
        graph.add_node("node2", node2)
        graph.set_entry_point("node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", END)
        
        # Compile
        app = graph.compile()
        print("✅ LangGraph basic functionality works")
        
        # Test the exact same visualization code from your Jupyter
        print("🎨 Testing visualization (same as Jupyter)...")
        
        # This is EXACTLY your Jupyter code:
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Save it
        with open("test_langgraph_viz.png", "wb") as f:
            f.write(graph_image)
        
        print(f"✅ SUCCESS! Visualization works outside Jupyter!")
        print(f"📁 File saved: test_langgraph_viz.png ({len(graph_image)} bytes)")
        
        return True, "test_langgraph_viz.png"
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install with: pip install langgraph[viz]")
        return False, str(e)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)

def test_your_workflow():
    """Test your actual workflow visualization"""
    print("\n🔧 Testing Your Actual Workflow...")
    
    try:
        from workfloworig import SAPRAGWorkflow
        
        # Create your workflow
        config = {
            "embedding_type": "openai",
            "vector_store_type": "chroma", 
            "email_enabled": False,
            "top_k": 5,
            "chunk_size": 1000,
            "temperature": 0.1
        }
        
        workflow = SAPRAGWorkflow(config)
        
        # Check what we got
        print(f"Workflow type: {type(workflow).__name__}")
        print(f"Has app: {hasattr(workflow, 'app')}")
        print(f"App type: {type(getattr(workflow, 'app', None)).__name__}")
        
        if not hasattr(workflow, 'app') or workflow.app is None:
            print("❌ No workflow app - check workflow.py")
            return False, "No app"
        
        app_type = type(workflow.app).__name__
        if app_type == 'MockApp':
            print("⚠️ Using MockApp - not real LangGraph")
            print("💡 Check your imports in workflow.py")
            return False, "MockApp detected"
        
        # Try the same visualization code
        print("🎨 Generating from your workflow...")
        
        # EXACT same code as Jupyter:
        workflow_app = workflow.app  # This is like your workflow_app variable
        graph_image = workflow_app.get_graph().draw_mermaid_png()
        
        # Save it
        with open("your_workflow_viz.png", "wb") as f:
            f.write(graph_image)
        
        print(f"✅ SUCCESS! Your workflow visualization works!")
        print(f"📁 File saved: your_workflow_viz.png ({len(graph_image)} bytes)")
        
        return True, "your_workflow_viz.png"
        
    except ImportError as e:
        print(f"❌ Can't import workflow: {e}")
        return False, str(e)
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    print("=" * 60)
    print("🧪 Testing LangGraph Visualization Outside Jupyter")
    print("=" * 60)
    
    # Test 1: Basic LangGraph
    basic_success, basic_result = test_basic_langgraph()
    
    if basic_success:
        print(f"\n🎉 Basic test passed! File: {basic_result}")
        
        # Test 2: Your workflow
        workflow_success, workflow_result = test_your_workflow()
        
        if workflow_success:
            print(f"\n🎉 Your workflow test passed! File: {workflow_result}")
            print("\n✅ CONCLUSION: LangGraph visualization works perfectly outside Jupyter!")
            print("💡 The issue in Streamlit is likely in the integration, not LangGraph itself")
        else:
            print(f"\n❌ Your workflow failed: {workflow_result}")
            print("💡 Check workflow.py - it might be using MockApp")
    else:
        print(f"\n❌ Basic test failed: {basic_result}")
        print("💡 Install LangGraph with: pip install langgraph[viz] pygraphviz")

if __name__ == "__main__":
    main()