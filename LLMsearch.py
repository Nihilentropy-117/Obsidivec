import requests
import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
SERVER_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your_fastapi_key_here")  # Key for your FastAPI server
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", None)

# --- Functions ---
def test_search(query_text, top_k=3):
    """
    Sends a search query to the FastAPI server and prints the results.
    """
    returnable = ""
    search_endpoint = f"{SERVER_URL}/search"

    payload = {"query": query_text, "n_results": top_k}
    headers = {"X-API-Key": API_KEY}

    print(f"--- Querying for: '{query_text}' ---")

    try:
        response = requests.post(search_endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            results = response.json()
            if not results.get('results'):
                return "Query successful, but no matching documents found."

            for i, res in enumerate(results['results']):
                returnable += f"\nResult {i + 1}:\n"
                returnable += f"File: {res['filepath']}\n"
                returnable += f"Chunk: {res['chunk_index']}\n"
                returnable += f"Text: \"{res['document']}\"\n"

        elif response.status_code == 503:
            returnable += "Server is still initializing. Please wait a moment and try again."
        elif response.status_code == 401:
            returnable += "Unauthorized — invalid or missing API key."
        else:
            returnable += f"Error: {response.status_code}\nDetails: {response.text}"

    except requests.exceptions.ConnectionError:
        returnable += f"Error: Could not connect to server at {SERVER_URL}."
    except Exception as e:
        returnable += f"Unexpected error: {e}"

    return returnable


def generate_summary(query, results_text):
    """
    Sends the query + search results to OpenRouter for summarization.
    """
    user_message = f"""
Here is a user query and relevant text chunks. 
Step 1: Summarize user question in simpler words. 
Step 2: Decide which retrieved text chunks directly apply. 
Step 3: Combine those chunks into an outline. 
Step 4: Draft a single, coherent answer. 
Show all steps, in [THINKING][/THINKING] tags, then provide the final refined answer in [ANSWER][/ANSWER] tags.

-----
RESULTS
-----
{results_text}

-----
User Query
-----
{query}
"""

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": user_message}]
    )

    response = completion.choices[0].message.content
    if "[ANSWER]" in response:
        if "[/ANSWER]" in response:
            return response.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()
        return response.split("[ANSWER]")[1].strip()
    return response

def query(query_text):
    results = test_search(query_text)
    summary = generate_summary(query_text, results)
    return summary

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's Kas' Garage Code?"
    results = test_search(query)
    print(generate_summary(query, results))
