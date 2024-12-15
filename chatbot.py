import json
from groq import Groq
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import numpy as np
import os

# Placeholder embedding class (to replace with a proper embedding generator later)
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Replace with actual embedding generation logic
        return [np.random.rand(384) for _ in texts]

    def embed_query(self, text):
        # Replace with actual embedding generation logic
        return np.random.rand(384)

# Step 1: Load and Parse JSON File
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Step 2: Preprocess District Data into Documents
def preprocess_data(district_data):
    documents = []
    for district in district_data:
        content = (
            f"District Name: {district['district_name']}\n"
            f"District Code: {district['district_code']}\n"
            f"Water Available: {district['water']}\n"
            f"Food Rations: {district['food_rations']}\n"
            f"Medical Kits: {district['medkits']}\n"
            f"Ammo Stock: {district['ammo']}\n"
            f"Camp Exists: {district['camp_exists']}\n"
            f"Coordinates: {district['coordinates']}"
        )
        documents.append(Document(page_content=content))
    return documents

# Step 3: Build Retrieval System
def build_retriever(documents):
    # Convert documents to raw text for embedding
    texts = [doc.page_content for doc in documents]

    # Generate embeddings
    embeddings = DummyEmbeddings()
    faiss_index = FAISS.from_texts(texts, embeddings)  # Embed and build FAISS
    return faiss_index.as_retriever()

# Step 4: Query Groq API
def query_groq_llm(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant providing district insights."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"  # Replace with your specific Groq model if needed
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred communicating with Groq API: {e}")
        return "An error occurred while fetching a response."

# Step 5: Set Up Chatbot
def create_chatbot(retriever, max_context_length=1000):
    def chatbot(query):
        # Retrieve context from FAISS retriever
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Query Groq LLM with context
        prompt = f"{query}\n\nContext:\n{context}"
        return query_groq_llm(prompt)

    return chatbot

# Step 6: Chatbot Interaction
def interact_with_chatbot(chatbot):
    print("Chatbot is ready! Ask your questions or type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Stay safe!")
            break
        try:
            response = chatbot(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

# Main Execution
if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Please set your Groq API key in the .env file.")

    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    # Load and preprocess data
    file_path = "parsed_districts.json"  # Path to the uploaded JSON file
    district_data = load_data(file_path)
    documents = preprocess_data(district_data)

    # Build retriever and chatbot
    retriever = build_retriever(documents)
    chatbot = create_chatbot(retriever)

    # Start interaction
    interact_with_chatbot(chatbot)
