import json
import requests
import numpy as np
import faiss

# Load the JSON file with district data
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Extract district information
districts_data = data.get('features', [])

# Gemini API details
GEMINI_API_URL = "https://ai.google.dev/api/rest"  # Replace with the correct endpoint
GEMINI_API_KEY = ""  # Add your API key

# Generate embeddings for each district
embeddings = []
metadata = []

for district in districts_data:
    district_name = district['properties']['dist_name']
    district_description = (
        f"District: {district_name}, "
        f"Water: {district['properties']['water']}, "
        f"Food Rations: {district['properties']['food_rations']}, "
        f"Medkits: {district['properties']['medkits']}, "
        f"Ammo: {district['properties']['ammo']}, "
        f"Camp Exists: {district['properties']['camp_exists']}"
    )
    
    # Send a request to Gemini API
    response = requests.post(
        GEMINI_API_URL,
        headers={
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"text": district_description}
    )
    
    if response.status_code == 200:
        embedding = response.json().get("embedding", [])
        embeddings.append(embedding)
        metadata.append({"name": district_name, "description": district_description})
    else:
        print(f"Error generating embedding for {district_name}: {response.text}")

# Save embeddings and metadata
with open('district_embeddings_with_metadata.json', 'w') as f:
    json.dump({"embeddings": embeddings, "metadata": metadata}, f, indent=4)
