from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

# Load .env file from the project root
load_dotenv(find_dotenv())

def get_llm_response(query: str, model: str = "gpt-3.5-turbo") -> str:
    """Sends a query to the LLM and returns the response."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in LLM call: {str(e)}"

if __name__ == "__main__":
    print("--- LLM Simple Example ---")
    response = get_llm_response("What is the capital of France?")
    print("LLM Response:", response) 