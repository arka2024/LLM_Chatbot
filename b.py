from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

# Step 1: Load Survival Data
def load_survival_data(file_path):
    loader = TextLoader(file_path)  # Load survival guide or similar file
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Step 2: Build Vector Store for Document Retrieval
def build_vector_store(docs):
    embeddings = OpenAIEmbeddings()  # Use OpenAI's embeddings (e.g., text-embedding-ada-002)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Step 3: Create the Chatbot (RAG Pipeline)
def create_survival_chatbot(vector_store):
    llm = OpenAI(model="gpt-4")  # Large language model for answering queries
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 4: Main Chatbot Functionality
def chatbot_response(chatbot, query):
    response = chatbot.run(query)
    return response

# Main Code
if __name__ == "__main__":
    # Step 1: Load survival-related text file (e.g., "survival_guide.txt")
    print("Loading survival data...")
    docs = load_survival_data("survival_guide.txt")

    # Step 2: Build vector store for document retrieval
    print("Building vector store...")
    vector_store = build_vector_store(docs)

    # Step 3: Create the chatbot
    print("Creating chatbot...")
    survival_chatbot = create_survival_chatbot(vector_store)

    # Interactive Chatbot
    print("Survival Chatbot is ready! Type your queries below:")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Stay safe!")
            break
        response = chatbot_response(survival_chatbot, user_query)
        print(f"Bot: {response}")
