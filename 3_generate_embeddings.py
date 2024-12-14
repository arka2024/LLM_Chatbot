from sentence_transformers import SentenceTransformer
import json

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

# Store embeddings
with open('district_embeddings.json', 'w') as f:
    json.dump(embeddings.tolist(), f)  # Convert numpy array to list for JSON compatibility

print("Embeddings have been saved to 'district_embeddings.json'.")
