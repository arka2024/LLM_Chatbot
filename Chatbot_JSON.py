import os
import json
from openai import OpenAI

# Set API key manually
GROQ_API_KEY = "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"  # Replace with your actual API key

def load_json_data():
    """Load the JSON dataset containing district information."""
    try:
        with open('parsed_districts.json') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: 'parsed_districts.json' file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON data.")
        return None

def query_dataset(question, data):
    """
    Query the dataset based on the user's question.

    Parameters:
    - question: str, user input question
    - data: list of dict, dataset loaded from JSON

    Returns:
    - str: Details about the matching district if found, otherwise None
    """
    # Convert question to lowercase for case insensitive matching
    question_lower = question.lower()

    # Check for specific keywords and find matching district data
    if "water" in question_lower:
        max_water = max(data, key=lambda x: x['water'])
        return f"District with the most water: {max_water['district_name']}, Water - {max_water['water']} liters"

    if "food rations" in question_lower:
        max_food_rations = max(data, key=lambda x: x['food_rations'])
        return f"District with the most food rations: {max_food_rations['district_name']}, Food Rations - {max_food_rations['food_rations']} kg"

    if "medkits" in question_lower:
        max_medkits = max(data, key=lambda x: x['medkits'])
        return f"District with the most medkits: {max_medkits['district_name']}, Medkits - {max_medkits['medkits']} units"

    if "ammo" in question_lower:
        max_ammo = max(data, key=lambda x: x['ammo'])
        return f"District with the most ammo: {max_ammo['district_name']}, Ammo - {max_ammo['ammo']}"

    # Search through the dataset for other related questions
    for district in data:
        if district['district_name'].lower() in question_lower:
            return f"Details about {district['district_name']}: Water - {district['water']} liters, Food Rations - {district['food_rations']} kg, Medkits - {district['medkits']} units, Ammo - {district['ammo']}"

    return None

def chatbot():
    print("Welcome to the Groq LLM Chatbot! Type 'exit' to quit.")
    conversation_history = []
    
    # Initialize the OpenAI client with Groq's API base and API key
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # Load the dataset
    data = load_json_data()

    if not data:
        return  # Exit if there is an error loading the data

    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Check if the question is related to the dataset
        dataset_response = query_dataset(user_input, data)
        if dataset_response:
            print(f"Chatbot: {dataset_response}")
            continue  # Move to next iteration

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # Call Groq API for a response
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Example model (supports LLaMA, Mistral, etc.)
                messages=conversation_history,
                max_tokens=300,
                temperature=0.7,
            )

            # Extract response and display
            reply = response.choices[0].message.content
            print(f"Chatbot: {reply}")

            # Add bot response to conversation history
            conversation_history.append({"role": "assistant", "content": reply})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot()
