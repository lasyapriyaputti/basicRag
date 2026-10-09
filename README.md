# Basic RAG PDF Question Answering System

A simple Retrieval-Augmented Generation (RAG) system that allows an LLM to answer questions using information retrieved from PDF documents.

## Architecture

PDF
↓
Document Loading
↓
Text Chunking
↓
Embeddings
↓
ChromaDB
↓
Query Embedding
↓
Similarity Search
↓
Retrieved Context
↓
Prompt
↓
Groq LLM
↓
Answer

## Technologies

- Python
- LangChain
- PyPDF
- Sentence Transformers
- ChromaDB
- Groq
- NumPy

## Project Structure

```text
rag-project/
│
├── data/
│   └── pdfs/
│
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_pipeline.py
│
├── main.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md