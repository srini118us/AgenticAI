from graphviz import Digraph

def create_workflow_graph():
    dot = Digraph(comment='Stateful Agent System Workflow', format='png')
    dot.attr(rankdir='TB', size='10,10')

    # Nodes
    dot.node('start', '_start_', shape='oval', style='filled', fillcolor='#D0C8F5')
    dot.node('supervisor', 'Supervisor', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('llm_call', 'LLM Call', shape='box', style='filled', fillcolor='#E0E0E0')
    dot.node('rag_call', 'RAG Call', shape='box', style='filled', fillcolor='#E0E0E0')
    dot.node('web_call', 'Web Call', shape='box', style='filled', fillcolor='#E0E0E0')
    dot.node('llm', 'LLM', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('rag', 'RAG', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('web', 'Web', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('validation', 'Validation', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('revoked', 'revoked', shape='box', style='filled', fillcolor='#E0E0E0')
    dot.node('accepted', 'accepted', shape='box', style='filled', fillcolor='#E0E0E0')
    dot.node('end_node', 'End', shape='box', style='filled', fillcolor='#D0C8F5')
    dot.node('end', '_end_', shape='oval', style='filled', fillcolor='#D0C8F5')

    # Edges
    dot.edge('start', 'supervisor')
    dot.edge('supervisor', 'llm_call', style='dotted')
    dot.edge('supervisor', 'rag_call', style='dotted')
    dot.edge('supervisor', 'web_call', style='dotted')
    dot.edge('llm_call', 'llm', style='dotted')
    dot.edge('rag_call', 'rag', style='dotted')
    dot.edge('web_call', 'web', style='dotted')

    dot.edge('llm', 'validation')
    dot.edge('rag', 'validation')
    dot.edge('web', 'validation')
    
    # Decision edges from Supervisor to Validation (revoked) - adjusted based on image
    # The 'revoked' text looks like an edge label from supervisor to validation directly
    # or a separate node that leads back to supervisor. Based on the provided image, 
    # it seems to be a conditional path from Validation back to Supervisor.
    dot.edge('validation', 'supervisor', label='revoked', style='dotted')
    
    dot.edge('validation', 'accepted', style='dotted')
    dot.edge('accepted', 'end_node')
    dot.edge('end_node', 'end')

    # Render the graph
    dot.render('stateful_agent_workflow', view=False)
    print("Workflow graph 'stateful_agent_workflow.png' generated successfully!")

if __name__ == '__main__':
    create_workflow_graph() 