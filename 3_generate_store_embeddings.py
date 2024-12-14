from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

# Load the JSON file with district data
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Extract features from JSON
districts_data = data.get('features', [])

# Create descriptions for each district
descriptions = []
for district in districts_data:
    district_name = district['properties'].get('dist_name', 'Unknown')
    description = f"{district_name} has {district['properties']['water']} liters of water, {district['properties']['food_rations']} food rations, {district['properties']['medkits']} medkits, and {district['properties']['ammo']} ammo."
    descriptions.append(description)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(descriptions)

# Convert embeddings to numpy array for FAISS
embeddings_np = np.array(embeddings)

# Create a FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))

# Add embeddings to the index
index.add(embeddings_np)

# Store embeddings and their metadata in JSON format
embedding_data = {
    'metadata': [{'dist_name': district['properties'].get('dist_name', 'Unknown'), 'dist_code': district['properties'].get('dist_code', 'Unknown')} for district in districts_data],
    'embeddings': embeddings_np.tolist()  # Convert numpy array to list for JSON compatibility
}

with open('district_embeddings_with_metadata.json', 'w') as f:
    json.dump(embedding_data, f, indent=4)

print("Embeddings and their metadata have been stored in 'district_embeddings_with_metadata.json'.")