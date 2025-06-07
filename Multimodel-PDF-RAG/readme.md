# Multimodal PDF Retrieval-Augmented Generation (RAG) Pipeline with OpenSearch

This project implements a PDF-based semantic search and question-answering pipeline using **LangChain**, **OpenAI embeddings**, and **OpenSearch** as the vector database.

---

## Overview

This pipeline allows you to:

- Extract text from PDF documents
- Perform semantic chunking on extracted text
- Generate embeddings using OpenAI's `text-embedding-ada-002` model
- Store and index embeddings in OpenSearch
- Perform semantic similarity search on OpenSearch vector store
- Generate natural language answers using GPT-4o based on retrieved document chunks
- Save the generated answer into a `.docx` file

---

## Features

- **PDF Text Extraction:** Uses `PyMuPDFLoader` for accurate text extraction.
- **Semantic Chunking:** Splits text into meaningful chunks for better embedding quality.
- **Embedding and Indexing:** Uses OpenAI embeddings and stores vectors in OpenSearch.
- **Semantic Search:** Fast and scalable similarity search with OpenSearch.
- **LLM Answer Generation:** Uses GPT-4o via LangChain to generate context-aware answers.
- **Response Export:** Save answers as Word documents (`.docx`).

---

## Prerequisites

- Python 3.8+
- OpenAI API Key
- OpenSearch running locally or remotely
- Required Python packages (`requirements.txt` available)

---

## Setup

1. **Clone the repo:**

```bash
git clone https://github.com/srini118us/AgenticAI.git
cd AgenticAI

2. **Create a .env file with the following environment variables:**
    OPENAI_API_KEY=your_openai_api_key
    OPENSEARCH_HOST=localhost
    OPENSEARCH_PORT=9200
    OPENSEARCH_INDEX=pdf_chunks
3. **Install dependencies:**
    pip install -r requirements.txt
## Running the Pipeline
python app.py
(Modify pdf_path and user_query inside the script or call run_pipeline() with your own parameters.)
## Requirements
    langchain
    langchain-community
    openai
    opensearch-py
    python-docx
    python-dotenv
    PyMuPDF
