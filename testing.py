import requests
import json

def fetch_and_store_data(api_url, output_file):
    try:
        # Fetch data from the API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Parse the JSON response
        data = response.json()

        # Save the data to a file
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Data successfully fetched from {api_url} and stored in {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")

# URL of the API endpoint
api_url = "https://api.mlsakiit.com/resources"
# Output file path
output_file = "resources_data.json"

# Call the function to fetch and store data
fetch_and_store_data(api_url, output_file)
