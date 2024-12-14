import folium
import json

# Load the JSON file with district data
with open('resources_data.json', 'r') as f:
    data = json.load(f)

# Extract features from JSON
features = data.get('features', [])

# Collect all coordinates
latitudes = []
longitudes = []

for feature in features:
    coordinates = feature['geometry'].get('coordinates', [[]])[0]  # Ensure it's a list
    for coord in coordinates:
        if isinstance(coord, list) and len(coord) == 2:  # Check if it's a valid pair
            longitudes.append(coord[0])  # Longitude
            latitudes.append(coord[1])  # Latitude

# Calculate the geographic center
if latitudes and longitudes:
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
else:
    raise ValueError("No valid coordinates found in the dataset.")

# Create a map centered at the computed geographic center with an appropriate zoom level
m = folium.Map(location=[center_lat, center_lon], zoom_start=5)  # Adjust zoom level as necessary

# Add polygons for each district
for feature in features:
    district_name = feature['properties'].get('dist_name', 'Unknown')
    coordinates = feature['geometry'].get('coordinates', [[]])[0]  # Use the outer boundary for the Polygon
    
    folium.Polygon(
        locations=coordinates,
        color='blue',
        weight=2,
        fill=True,
        fill_color='cyan',
        fill_opacity=0.4,
        popup=f"District: {district_name}",
        tooltip=district_name
    ).add_to(m)

# Save the map to an HTML file
m.save('generalized_districts_map.html')

print("Map has been saved as 'generalized_districts_map.html'. Open it in your browser to view.")
