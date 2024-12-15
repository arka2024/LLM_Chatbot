import json

# Load the JSON file
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Initialize a list to store parsed districts
parsed_data = []

# Extract the 'features' list
features = data.get('features', [])

# Parse each feature
for feature in features:
    properties = feature.get('properties', {})
    geometry = feature.get('geometry', {})
    
    # Extract relevant fields
    parsed_data.append({
        'district_name': properties.get('dist_name', 'Unknown'),
        'district_code': properties.get('dist_code', 'Unknown'),
        'water': properties.get('water', 0),
        'food_rations': properties.get('food_rations', 0),
        'medkits': properties.get('medkits', 0),
        'ammo': properties.get('ammo', 0),
        'camp_exists': properties.get('camp_exists', False),
        'coordinates': geometry.get('coordinates', [])
    })

# Save parsed data into a new JSON file
with open('parsed_districts.json', 'w') as f:
    json.dump(parsed_data, f, indent=4)