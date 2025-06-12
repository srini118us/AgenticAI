from graphviz import Digraph

def create_node(dot, name, label, shape='box', fillcolor='#D0C8F5', style='filled'):
    dot.node(name, label, shape=shape, style=style, fillcolor=fillcolor)

def create_workflow_graph():
    dot = Digraph(comment='Stateful Agent System Workflow', format='png')
    dot.attr(rankdir='TB', size='10,10')

    # === Nodes ===
    # Start & End
    create_node(dot, 'start', '_start_', shape='oval')
    create_node(dot, 'end', '_end_', shape='oval')

    # Core process
    create_node(dot, 'supervisor', 'Supervisor')
    create_node(dot, 'validation', 'Validation')
    create_node(dot, 'end_process', 'End')

    # Call wrappers
    call_nodes = ['llm_call', 'rag_call', 'web_call']
    for node in call_nodes:
        create_node(dot, node, node.replace('_', ' ').title(), fillcolor='#E0E0E0')

    # Agents
    agent_nodes = ['llm', 'rag', 'web']
    for node in agent_nodes:
        create_node(dot, node, node.upper())

    # Outcomes
    create_node(dot, 'revoked', 'revoked', fillcolor='#E0E0E0')
    create_node(dot, 'accepted', 'accepted', fillcolor='#E0E0E0')

    # === Edges ===
    dot.edge('start', 'supervisor')

    # Supervisor dispatch to call nodes
    for call_node in call_nodes:
        dot.edge('supervisor', call_node, style='dotted')

    # Call nodes to corresponding agents (using zip for cleaner mapping)
    for call_node, agent in zip(call_nodes, agent_nodes):
        dot.edge(call_node, agent, style='dotted')

    # Agent responses to validation
    for agent in agent_nodes:
        dot.edge(agent, 'validation')

    # Validation outcomes
    dot.edge('validation', 'supervisor', label='revoked', style='dotted')
    dot.edge('validation', 'accepted', style='dotted')
    dot.edge('accepted', 'end_process')
    dot.edge('end_process', 'end')

    # Render
    dot.render('stateful_agent_workflow', view=False)
    print("Workflow graph 'stateful_agent_workflow.png' generated successfully!")

if __name__ == '__main__':
    create_workflow_graph()
