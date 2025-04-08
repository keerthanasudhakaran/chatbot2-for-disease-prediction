from openai import OpenAI
import os
from dotenv import load_dotenv

import nltk
nltk.download('wordnet')

from streamlit_chat import message
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS



# Load the PDF files from the path
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)


# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

print("embedding : ",embeddings)
# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

import pickle

# Assuming you've already created the vector store using FAISS
# Serialize the vector store
with open("vector_store.pickle", "wb") as f:
    pickle.dump(vector_store, f)