import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Step 1: Load Data and Generate Embeddings
def generate_embeddings(data_file, model_name='all-MiniLM-L6-v2'):
    # Load JSON data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract district descriptions for embedding generation
    descriptions = [
        f"District: {item['district_name']}, Water: {item['water']}, Medkits: {item['medkits']}"
        for item in data
    ]

    # Load embedding model
    model = SentenceTransformer(model_name)
    embeddings = np.array([model.encode(desc) for desc in descriptions])

    return embeddings, data

# Step 2: Build and Save FAISS Index (Cosine Similarity)
def build_faiss_index(embeddings, index_file):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # Use Inner Product for Cosine Similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize embeddings
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index

# Step 3: Query FAISS Index
def query_faiss_index(query, model, index, metadata, top_k=5):
    query_embedding = model.encode(query).reshape(1, -1)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)  # Normalize query
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve results
    results = [
        {"metadata": metadata[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])
    ]
    return results

# Main Execution
if __name__ == '__main__':
    # File paths
    data_file = 'parsed_districts.json'  # Replace with your data file
    index_file = 'district_index.faiss'

    # Generate embeddings and load metadata
    print("Generating embeddings...")
    embeddings, metadata = generate_embeddings(data_file)

    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings, index_file)
    print(f"FAISS index saved to {index_file}")

    # Load embedding model (for queries)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # User query
    user_query = "District with high water reserves and medkits"
    print(f"Querying FAISS index for: '{user_query}'")
    results = query_faiss_index(user_query, model, index, metadata)

    # Display results
    print("Top Results:")
    for result in results:
        print(f"District: {result['metadata']['district_name']}, Water: {result['metadata']['water']}, Medkits: {result['metadata']['medkits']}, Similarity: {result['distance']:.4f}")
