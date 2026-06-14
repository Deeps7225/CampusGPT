from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

# Load all PDFs
all_documents = []

for file in os.listdir("data"):

    if file.endswith(".pdf"):

        pdf_path = os.path.join("data", file)

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        all_documents.extend(documents)

print("Documents Loaded:", len(all_documents))

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(all_documents)

print("Chunks Created:", len(split_docs))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS
vector_store = FAISS.from_documents(
    split_docs,
    embeddings
)

# Save FAISS
vector_store.save_local("vector_db")

print("Vector Database Saved Successfully!")