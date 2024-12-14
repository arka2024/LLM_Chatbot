import json
import requests
import numpy as np
import faiss

# Load the JSON file with district data
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Extract features from JSON
districts_data = data.get('features', [])

# Load metadata and embeddings (assuming these are stored as a list of lists)
with open('district_embeddings_with_metadata.json', 'r') as f:
    embedding_data = json.load(f)

embeddings_np = np.array(embedding_data['embeddings'])
metadata = embedding_data['metadata']

# Create a FAISS index
index = faiss.IndexFlatL2(len(embeddings_np[0]))

# Add embeddings to the index
index.add(embeddings_np)

# User's query
query = "I need information on districts with high water reserves and medkits."

# Generate query embedding using Groq
query_response = requests.post(
    "https:///embedding",
    json={"text": query}
)
query_embedding = np.array(query_response.json()['embedding'])

# Find the most similar embeddings
D, I = index.search(query_embedding.reshape(1, -1), k=5)  # 'k' is the number of nearest neighbors to retrieve

# Get the relevant districts
relevant_districts = [districts_data[i] for i in I[0]]

# Generate the response
response = f"Based on your query, here are the relevant districts: {', '.join([district['properties']['dist_name'] for district in relevant_districts])}. For detailed information, check their water reserves, medkits, and other resources."

print(response)
