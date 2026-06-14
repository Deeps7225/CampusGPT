# CampusGPT

An AI-powered academic assistant built using RAG (Retrieval-Augmented Generation), FAISS, LangChain, Streamlit, and Groq.

## Features

* University Question Answering
* PDF Upload and Analysis
* Conversational Memory
* Citation Support
* General Knowledge Fallback
* Vector Database using FAISS

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Groq LLM

## Project Workflow

1. Upload university documents
2. Create embeddings using HuggingFace
3. Store embeddings in FAISS Vector Database
4. Retrieve relevant chunks
5. Generate answers using Groq LLM
6. Display citations with answers

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Author

Dipendra singh
