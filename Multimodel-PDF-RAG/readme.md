Multimodal PDF RAG Pipeline
An end-to-end Retrieval-Augmented Generation (RAG) pipeline that extracts and semantically analyzes research papers in PDF format, retrieves contextually relevant information using vector search, and generates structured, high-quality answers using GPT-4.

Overview
This pipeline automates the process of:

Extracting content from academic or technical PDFs

Splitting the content into meaningful semantic chunks

Embedding and storing those chunks in OpenSearch

Performing semantic search using natural language queries

Generating AI-powered responses using GPT-4

Exporting results to a DOCX file for easy sharing


Requirements
Install dependencies using:
pip install -r requirements.txt

requirements.txt

# PDF & OCR
PyMuPDF
PyPDF2
pdf2image
pytesseract
Pillow

# Vector DB
pymilvus
opensearch-py>=2.4.0

# LangChain Ecosystem
langchain
langchain-core
langchain-openai
langchain-community

# LLM + Utility
openai
python-docx
python-dotenv
tqdm
Environment Variables
Create a .env file in your project root with:
OPENAI_API_KEY=your_openai_api_key
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=pdf_chunks

Example Usage
python app.py

Example query:
"Can you explain the core idea behind the self-attention mechanism proposed in the paper?"

The result will be saved as response.docx.

Output
response.docx: Generated answer with references from the PDF document.

Console logs for each phase execution.




