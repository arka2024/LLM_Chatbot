import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with the embedding model you're using

# File paths
faiss_index_file = 'district_index.faiss'  # FAISS index file
metadata_file = 'parsed_districts.json'   # Metadata file containing district information

# Load FAISS index
index = faiss.read_index(faiss_index_file)

# Load metadata (district information corresponding to embeddings)
with open(metadata_file, 'r') as f:
    district_metadata = json.load(f)

# Function to embed a user query
def embed_query(query):
    """Generate embeddings for the user query."""
    return model.encode(query)

# Function to perform a search in FAISS index
def search_faiss(query, top_k=5):
    """Search the FAISS index for the most similar districts."""
    query_embedding = embed_query(query)
    query_embedding = np.array([query_embedding])  # FAISS requires embeddings as 2D array
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

# Function to format and retrieve search results
def format_results(indices, distances):
    """Format results for user-friendly display."""
    results = []
    for idx, dist in zip(indices, distances):
        if idx < 0:  # FAISS returns -1 for invalid results
            continue
        district_info = district_metadata[idx]
        district_info['similarity'] = 1 - dist  # Convert L2 distance to similarity (approximation)
        results.append(district_info)
    return results

# Query System Interface
def query_system():
    """Interactive query system for district search."""
    print("Welcome to the District Query System!")
    print("Type your query (e.g., 'Find districts with high water reserves and medkits'):")
    
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting. Thank you!")
            break
        
        print("\nSearching for districts...\n")
        try:
            # Search in the FAISS index
            distances, indices = search_faiss(user_query, top_k=5)
            
            # Format results
            results = format_results(indices, distances)
            
            if results:
                print("Top results:")
                for i, result in enumerate(results):
                    print(f"\nResult {i+1}:")
                    print(f"District Name: {result['district_name']}")
                    print(f"Water Reserves: {result['water']}")
                    print(f"Medkits: {result['medkits']}")
                    print(f"Food Rations: {result['food_rations']}")
                    print(f"Camps Exist: {result['camp_exists']}")
                    print(f"Similarity Score: {result['similarity']:.4f}")
            else:
                print("No matching districts found.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the query system
if __name__ == "__main__":
    query_system()
