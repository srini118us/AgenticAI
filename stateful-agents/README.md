# Stateful Agent System

A modular and extensible system for building stateful AI agents with specialized nodes for different tasks.

## System Overview

The Stateful Agent System is designed to create intelligent agents that can:
- Process information using language models
- Retrieve and analyze documents
- Scrape and process web content
- Validate outputs for quality and relevance
- Coordinate multiple tasks through a supervisor

## Components

### Core Nodes
- `base_node.py`: Foundation class with state management
- `llm_node.py`: Language model interactions
- `rag_node.py`: Document retrieval and analysis
- `web_scraper_node.py`: Web content fetching
- `validator_node.py`: Output validation
- `supervisor_node.py`: Node coordination

### Demonstration Files
- `stateful_agent_system.ipynb`: Interactive Jupyter notebook tutorial
- `stateful_agent_cli.py`: Command-line interface tutorial
- `main.py`: Basic usage examples
- `research_example.py`: Complex research assistant example

## Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Required packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### Running the System

#### Interactive Tutorial (Recommended)
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `stateful_agent_system.ipynb`
3. Run cells sequentially to learn the system

#### Command Line Interface
1. Run the CLI tutorial:
   ```bash
   python stateful_agent_cli.py
   ```
2. Follow the interactive menu

#### Quick Examples
1. Basic usage:
   ```bash
   python main.py
   ```
2. Research assistant:
   ```bash
   python research_example.py
   ```

## System Architecture

### Node Types
1. **Base Node**
   - State management
   - Error handling
   - Common interface

2. **LLM Node**
   - OpenAI API integration
   - Query processing
   - Response generation

3. **RAG Node**
   - Document storage
   - Semantic search
   - Context-aware responses

4. **Web Scraper Node**
   - URL processing
   - Content extraction
   - Error handling

5. **Validator Node**
   - Output validation
   - Quality checks
   - Relevance assessment

6. **Supervisor Node**
   - Node coordination
   - Task management
   - Execution history

### Workflow
1. Supervisor receives task
2. Appropriate nodes are activated
3. Results are validated
4. Final output is generated

## Examples

### Basic Usage
```python
from nodes.supervisor_node import SupervisorNode
from nodes.llm_node import LLMNode

# Initialize nodes
supervisor = SupervisorNode()
llm_node = LLMNode()

# Register node
supervisor.register_node(llm_node)

# Execute task
result = await supervisor.execute({
    "task_type": "llm",
    "query": "What is artificial intelligence?"
})
```

### Research Assistant
See `research_example.py` for a complete example of a research assistant that:
- Scrapes multiple sources
- Processes information
- Generates summaries
- Validates outputs

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenAI for the language model API
- Contributors and maintainers
- Open source community 