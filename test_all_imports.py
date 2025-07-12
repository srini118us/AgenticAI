# test_all_imports.py
def test_langgraph_imports():
    """Test all possible StateGraph import combinations"""
    
    import_tests = [
        # Test 1: Check if it's in graph module
        {
            'name': 'langgraph.graph',
            'code': 'from langgraph.graph import StateGraph, END',
            'test': 'StateGraph'
        },
        
        # Test 2: Check if it's in pregel
        {
            'name': 'langgraph.pregel', 
            'code': 'from langgraph.pregel import StateGraph; END = "__end__"',
            'test': 'StateGraph'
        },
        
        # Test 3: Check main langgraph  
        {
            'name': 'langgraph main',
            'code': 'from langgraph import StateGraph, END',
            'test': 'StateGraph'
        },
        
        # Test 4: Check constants
        {
            'name': 'langgraph with constants',
            'code': 'from langgraph import StateGraph; from langgraph.constants import END',
            'test': 'StateGraph'
        },
        
        # Test 5: Alternative names
        {
            'name': 'alternative names',
            'code': 'from langgraph.graph import Graph as StateGraph; END = "__end__"',
            'test': 'StateGraph'
        },
        
        # Test 6: Check prebuilt
        {
            'name': 'langgraph.prebuilt',
            'code': 'from langgraph.prebuilt import StateGraph; END = "__end__"',
            'test': 'StateGraph'
        }
    ]
    
    successful_imports = []
    
    for test in import_tests:
        print(f"\n🧪 Testing {test['name']}: {test['code']}")
        
        try:
            # Create a local scope for testing
            local_scope = {}
            exec(test['code'], globals(), local_scope)
            
            # Check if the required object exists
            if test['test'] in local_scope:
                StateGraph = local_scope[test['test']]
                print(f"  ✓ {test['test']} imported successfully")
                
                # Try to create an instance
                try:
                    from typing import TypedDict
                    
                    class TestState(TypedDict):
                        message: str
                    
                    sg = StateGraph(TestState)
                    print(f"  ✓ StateGraph instance created successfully")
                    print(f"  ✓ StateGraph type: {type(StateGraph)}")
                    print(f"  ✓ Available methods: {[m for m in dir(sg) if not m.startswith('_')][:5]}")
                    
                    successful_imports.append({
                        'method': test['name'],
                        'code': test['code'],
                        'StateGraph': StateGraph
                    })
                    
                except Exception as e:
                    print(f"  ✗ Failed to create StateGraph instance: {e}")
                    
            else:
                print(f"  ✗ {test['test']} not found in imported module")
                
        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(successful_imports)} successful import methods found")
    
    for i, success in enumerate(successful_imports, 1):
        print(f"{i}. {success['method']}: {success['code']}")
    
    return successful_imports

if __name__ == "__main__":
    results = test_langgraph_imports()
    
    if results:
        print(f"\n🎉 Use this import in your code:")
        print(f"   {results[0]['code']}")
    else:
        print(f"\n❌ No working imports found. LangGraph may need to be reinstalled.")