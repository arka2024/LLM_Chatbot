import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURATION === #
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_API_URL = "https://api.groq.com/v1/chat"  # Replace with the actual Groq API URL if different

FAISS_INDEX_FILE = "district_index.faiss"
METADATA_FILE = "parsed_districts.json"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with desired embedding model

# === LOAD FAISS INDEX AND METADATA === #
index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, 'r') as f:
    district_metadata = json.load(f)

# === EMBEDDING FUNCTIONS === #
def embed_query(query):
    """Generate embedding for the query."""
    return model.encode(query)

def search_faiss(query, top_k=5):
    """Search the FAISS index for most similar districts."""
    query_embedding = np.array([embed_query(query)])
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

# === FORMAT RESULTS === #
def format_results(indices, distances):
    """Retrieve and format district metadata for results."""
    results = []
    for idx, dist in zip(indices, distances):
        if idx < 0:  # Invalid result
            continue
        district_info = district_metadata[idx]
        district_info['similarity'] = 1 - dist  # Approximate similarity
        results.append(district_info)
    return results

# === GROQ API CALL === #
def query_groq_llm(prompt, context):
    """Send a prompt and context to the Groq LLM API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are an assistant providing district insights."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        raise Exception(f"Groq API error: {response.status_code}, {response.text}")

# === QUERY SYSTEM WITH LLM AUGMENTATION === #
def query_system():
    """Interactive query system with FAISS and LLM augmentation."""
    print("Welcome to the District Query System with LLM augmentation!")
    print("Type your query (e.g., 'Which districts have high water reserves, medkits, and ammo?'):")

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting. Thank you!")
            break
        
        print("\nSearching for districts...\n")
        try:
            # Search FAISS index
            distances, indices = search_faiss(user_query, top_k=5)
            results = format_results(indices, distances)
            
            if results:
                print("\nTop relevant districts:")
                context = ""
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"District Name: {result['district_name']}")
                    print(f"Water Reserves: {result['water']}")
                    print(f"Medkits: {result['medkits']}")
                    print(f"Food Rations: {result['food_rations']}")
                    print(f"Ammo Count: {result['ammo']}")
                    print(f"Camps Exist: {result['camp_exists']}")
                    print(f"Similarity Score: {result['similarity']:.4f}")
                    # Add district info to context for LLM
                    context += (
                        f"District {result['district_name']}: "
                        f"Water={result['water']}, Medkits={result['medkits']}, "
                        f"Food Rations={result['food_rations']}, Ammo={result['ammo']}, "
                        f"Camps Exist={result['camp_exists']}.\n"
                    )
                
                # Query the Groq LLM with the user's query and context
                print("\nEnhancing results using Groq LLM...")
                llm_response = query_groq_llm(user_query, context)
                print("\nGroq LLM Response:")
                print(llm_response)
            else:
                print("No matching districts found.")
        except Exception as e:
            print(f"An error occurred: {e}")

# === MAIN === #
if __name__ == "__main__":
    query_system()
