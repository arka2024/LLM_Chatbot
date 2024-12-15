import json
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.schema import Document
from dotenv import load_dotenv
import os

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
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever()

# Step 4: Set Up Chatbot
def create_chatbot(retriever):
    llm = OpenAI(model="gpt-4")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

# Step 5: Chatbot Interaction
def interact_with_chatbot(chatbot):
    print("Chatbot is ready! Ask your questions or type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Stay safe!")
            break
        try:
            response = chatbot.run(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

# Main Execution
if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set your OpenAI API key in the .env file.")

    # Load and preprocess data
    file_path = "parsed_districts.json"  # Path to the uploaded JSON file
    district_data = load_data(file_path)
    documents = preprocess_data(district_data)

    # Build retriever and chatbot
    retriever = build_retriever(documents)
    chatbot = create_chatbot(retriever)

    # Start interaction
    interact_with_chatbot(chatbot)
