# Technical Test: Simple RAG System

## Overview

Build a RAG (Retrieval-Augmented Generation) system in Python using:
- **ChromaDB** for vector storage
- **all-MiniLM-L6-v2** for embeddings  
- **OpenRouter** for LLM responses

## Three Simple Tasks

### 1. Add Documents
**POST /add_document**
- Take text input
- Generate embedding with all-MiniLM-L6-v2
- Store in ChromaDB

### 2. Search  
**GET /search?query=...**
- Generate query embedding
- Find similar documents in ChromaDB
- Return matches with scores

### 3. RAG Chat
**POST /chat**
- Search for relevant context
- Send context + question to OpenRouter
- Return answer with sources

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Setup .env
OPENROUTER_API_KEY=your_key_here
CHROMADB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
MODEL_SLUG=openai/gpt-3.5-turbo

# Run
uvicorn app.main:app --reload
```

## What to Implement

Each file has TODO comments. Main tasks:

1. **database.py**: Initialize ChromaDB, implement add/search
2. **embeddings.py**: Load model, generate embeddings  
3. **rag.py**: OpenRouter client, build prompt, get answer