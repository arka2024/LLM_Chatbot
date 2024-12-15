from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load text documents
loader = TextLoader("C:\Users\KIIT0001\Downloads\mlsa file\map_description.txt")  # Replace with your text file path
documents = loader.load()

# Process as before
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
