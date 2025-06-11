# Stateful Agents Lab

This directory contains a set of examples demonstrating the implementation of stateful agents for various AI tasks.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Workflow Visualization](#workflow-visualization)
- [Running Examples](#running-examples)
- [Key Files](#key-files)

## Project Overview

This project explores the concept of stateful agents, where agents can maintain and update their internal state based on interactions and observations. The examples provided here demonstrate different capabilities of these agents, including:

-   **LLM Integration**: Agents interacting with Large Language Models.
-   **RAG (Retrieval Augmented Generation)**: Agents leveraging external knowledge bases.
-   **Web Scraping**: Agents designed to extract information from the web.
-   **Validation**: Agents responsible for validating information or decisions.
-   **Orchestration**: A supervisor agent managing the overall workflow.

## Setup

To run the examples in this lab, you need to set up your Python environment and install the necessary dependencies.

1.  **Navigate to the directory**:
    ```bash
    cd stateful-agents-lab
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Graphviz**:
    Graphviz is required to visualize the agent workflows. Download and install it from the official [Graphviz website](https://graphviz.org/download/) or using your system's package manager. Ensure the `dot` command is available in your system's PATH.

## Workflow Visualization

The `graph_workflow.py` script generates a visual representation of the stateful agent system's high-level flow.

To generate the workflow graph (`stateful_agent_workflow.png`):

```bash
python graph_workflow.py
```

After running, the `stateful_agent_workflow.png` file will be created in this directory.

## Running Examples

To run the main stateful agent workflow, execute the `main.py` script:

```bash
python main.py
```

### Example Output

When you run `main.py`, you will see output similar to the following, demonstrating the interaction of different agents:

```
--- Starting Simple Workflow for Task: llm_query ---
Calling LLM with query: 'Explain quantum computing in simple terms.'
LLM Raw Result: Sure! Quantum computing is a type of computing that uses the principles of quantum mechanics to perf...
LLM Result Valid: True
--- Workflow for Task: llm_query Finished ---

Workflow Completed Successfully!
Final Validated Output: Sure! Quantum computing is a type of computing that uses the principles of quantum mechanics to perform operations. In traditional computing, information is stored in bits, which can be either a 0 or a 1. Quantum computing uses quantum bits, or qubits, which can be in a state of 0, 1, or both at the same time due to a phenomenon called superposition.

This ability to be in multiple states simultaneously allows quantum computers to process a vast amount of information much more quickly than classical computers. Quantum computers can also take advantage of another quantum phenomenon called entanglement, which allows qubits to be linked together in a way that the state of one qubit is dependent on the state of another, even if they are far apart.

In simple terms, quantum computing is a new way of computing that leverages the strange behavior of quantum mechanics to solve complex problems much faster than traditional computers.

--- Starting Simple Workflow for Task: rag_query ---
Calling RAG with query: 'What are the key concepts of quantum computing?' and 3 documents.
RAG Raw Result: No relevant context found for the query....
RAG Result Valid: True
--- Workflow for Task: rag_query Finished ---

Workflow Completed Successfully!
Final Validated Output: No relevant context found for the query.

--- Starting Simple Workflow for Task: web_scrape ---
Calling Web Scraper for URL: http://example.com
Web Scraper Raw Result: <!doctype html>
<html>
<head>
    <title>Example Domain</title>
...
Web Scraper Result Valid: True
--- Workflow for Task: web_scrape Finished ---

Workflow Completed Successfully!
Final Validated Output: <!doctype html>
<html>
<head>
    <title>Example Domain</title>
...
```

*(Further instructions for specific examples (LLM, RAG, Web Scraper, Validator) will be added here soon.)*

## Key Files

-   `main.py`: The main entry point for running the stateful agent workflow.
-   `graph_workflow.py`: Generates the visual workflow graph.
-   `llm.py`: Contains the logic for LLM interactions.
-   `rag.py`: Implements Retrieval Augmented Generation capabilities.
-   `web_scraper.py`: Handles web scraping functionalities.
-   `validator.py`: Contains validation logic.
-   `stateful_agent_workflow.png`: The generated workflow visualization.
-   `requirements.txt`: Lists all Python dependencies. 