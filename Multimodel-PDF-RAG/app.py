# multimodal_pdf_rag_pipeline.py

# =========================
# 🔹 Imports and Setup
# =========================
import os
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from dotenv import load_dotenv
from opensearchpy import OpenSearch

# =========================
# 🔹 Load Environment Variables
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "pdf_chunks")

# =========================
# 🔹 Step 1: Data Ingestion from PDF using PyMuPDFLoader
# =========================
def extract_text_and_charts(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
    print(f"✅ Extracted {len(texts)} pages from PDF")
    return texts

# =========================
# 🔹 Step 2: Semantic Chunking
# =========================
def semantic_chunking(texts, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    print(f"✅ Generated {len(chunks)} semantic chunks")
    return chunks

# =========================
# 🔹 Step 3: Embedding and Storing in OpenSearch
# =========================
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def embed_and_store_chunks_opensearch(chunks):
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True
    )
    vectorstore = OpenSearchVectorSearch(
    index_name=OPENSEARCH_INDEX,
    embedding_function=embedding_model,  # Pass the full embedding model
    opensearch_url=f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
    http_auth=None,
    use_ssl=False,
    )
    vectorstore.add_texts(chunks)
    print(f"✅ Embedded and stored {len(chunks)} chunks in OpenSearch")
    return vectorstore

# =========================
# 🔹 Step 4: Semantic Search with OpenSearch
# =========================
def search_opensearch(vectorstore, query, top_k=5):
    start = time.time()
    results = vectorstore.similarity_search(query, k=top_k)
    duration = time.time() - start
    print(f"✅ Search completed in {duration:.2f} seconds, top {top_k} results retrieved")
    return results, duration

# =========================
# 🔹 Step 5: Prompt Template and LLM Output Generation
# =========================
def generate_response(context, query):
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "You are an AI assistant helping to summarize and explain research papers.\n"
            "Based on the following context extracted from the paper 'Attention Is All You Need':\n\n"
            "{context}\n\n"
            "Please provide a clear and concise answer to the following question:\n"
            "{query}\n"
            "Your response should be accurate, well-structured, and use appropriate technical terminology."
        )
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": context, "query": query})
    print("✅ LLM response generated")
    return response

# =========================
# 🔹 Step 6: Save LLM Response to DOCX
# =========================
def save_response_to_docx(response_text, filename="response.docx"):
    doc = Document()
    doc.add_paragraph(response_text)
    doc.save(filename)
    print(f"✅ Response saved to {filename}")

# =========================
# 🔹 Step 7: Full Pipeline Execution
# =========================
def run_pipeline(pdf_path, user_query):
    print("🔹 Extracting data from PDF...")
    combined_texts = extract_text_and_charts(pdf_path)

    print("🔹 Performing semantic chunking...")
    chunks = semantic_chunking(combined_texts)

    print("🔹 Embedding chunks and storing in OpenSearch...")
    vectorstore = embed_and_store_chunks_opensearch(chunks)

    print("🔹 Performing similarity search...")
    results, _ = search_opensearch(vectorstore, user_query)

    top_context = "\n".join([doc.page_content for doc in results])

    print("🔹 Generating response using LLM...")
    response = generate_response(top_context, user_query)

    save_response_to_docx(response)
    print(response)

# =========================
# 🔹 Run Example
# =========================
if __name__ == "__main__":
    run_pipeline(
        pdf_path="attention.pdf",
        user_query="Can you explain the core idea behind the self-attention mechanism proposed in the paper?"
    )
