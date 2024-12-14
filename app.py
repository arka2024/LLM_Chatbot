from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Load documents and create embeddings
documents =   # Replace with your document loader
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Set up Retriever
retriever = vector_store.as_retriever()

# LLM setup
llm = OpenAI(model="gpt-4")

# Create the RetrievalQA pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # Concatenate documents with user query
)

# Chatbot interaction
query = "What is the company's refund policy?"
response = qa_chain.run(query)
print(response)
