from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

# Load the JSON file with district data
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Extract features from JSON
districts_data = data.get('features', [])

# Load embeddings and metadata
with open('district_embeddings_with_metadata.json', 'r') as f:
    embedding_data = json.load(f)

embeddings_np = np.array(embedding_data['embeddings'])
metadata = embedding_data['metadata']

# Create a FAISS index
index = faiss.IndexFlatL2(len(embeddings_np[0]))

# Add embeddings to the index
index.add(embeddings_np)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# User's query
query = "I need information on districts which have temperature less than 15."
query_embedding = model.encode(query)

# Find the most similar embeddings
D, I = index.search(query_embedding.reshape(1, -1), k=5)  # 'k' is the number of nearest neighbors to retrieve

# Get the relevant districts
relevant_districts = [districts_data[i] for i in I[0]]

# Generate the response
response = f"Based on your query, here are the relevant districts: {', '.join([district['properties']['dist_name'] for district in relevant_districts])}. For detailed information, check their water reserves, medkits, and other resources."

print(response)
