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

