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
