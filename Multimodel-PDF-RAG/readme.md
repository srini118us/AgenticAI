# 📄 Multimodal PDF RAG Pipeline

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline that extracts text from PDFs, semantically chunks and embeds the content, stores it in OpenSearch, and generates intelligent answers to user queries using OpenAI's GPT-4o model. The final response is saved as a DOCX document.

## 🚀 Features

- 📥 **PDF Text Extraction**: Loads academic papers and extracts their content.
- ✂️ **Semantic Chunking**: Uses language-aware chunking to preserve context.
- 🧠 **Embedding**: Uses `text-embedding-ada-002` for semantic representation.
- 🔍 **OpenSearch Vector Storage**: Stores and indexes embeddings for fast retrieval.
- 💬 **LLM-Powered Q&A**: GPT-4o generates accurate, structured responses.
- 📄 **DOCX Output**: Saves generated answers in a Word document.

## 🧾 Project Structure

```
Multimodel-PDF-RAG/
├── .env                       # API keys and config
├── app.py                    # Main pipeline script
├── attention.pdf             # Sample input PDF
├── multimodel_pdfRAG.ipynb   # Jupyter exploration notebook
├── readme.md                 # 📘 You’re here!
├── requirements.txt          # Python dependencies
└── response.docx             # Generated LLM output
```

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Multimodel-PDF-RAG.git
cd Multimodel-PDF-RAG
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Create `.env` file

```ini
OPENAI_API_KEY=your_openai_key
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=pdf_chunks
```

### 4. Start OpenSearch via Docker

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword \
  opensearchproject/opensearch:2.13.0
```

### 5. Run the pipeline

```bash
python app.py
```

## 🧪 Example Query

The current pipeline uses the paper **"Attention Is All You Need"** and answers:

> _"Can you explain the core idea behind the self-attention mechanism proposed in the paper?"_

Result is saved to `response.docx`.

## 🛠️ Dependencies

Key libraries used:
- `langchain`
- `openai`
- `opensearch-py`
- `python-docx`
- `PyMuPDF`
- `python-dotenv`

Install them all via:

```bash
pip install -r requirements.txt
```

## 📌 TODO

- [ ] Add UI using Streamlit or Gradio
- [ ] Add chart/image extraction support
- [ ] Support for multi-file uploads
- [ ] Add unit tests

## 📄 License

MIT License


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

## Setup

1. **Clone the repo:**

    ```bash
    git clone https://github.com/srini118us/AgenticAI.git
    cd AgenticAI
    ```

2. **Create a `.env` file with the following environment variables:**

    ```env
    OPENAI_API_KEY=your_openai_api_key
    OPENSEARCH_HOST=localhost
    OPENSEARCH_PORT=9200
    OPENSEARCH_INDEX=pdf_chunks
    ```

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
