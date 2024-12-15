import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def chatbot():
    print("Welcome to the Groq LLM Chatbot! Type 'exit' to quit.")
    conversation_history = []

    # Initialize the OpenAI client with Groq's API base
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

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