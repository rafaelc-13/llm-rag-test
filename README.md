# Technical Test: Simple RAG System

Welcome! The goal of this challenge is to build a simple Retrieval-Augmented Generation (RAG) system from scratch. You'll create a basic API that can store documents, search for relevant information, and answer questions based on the stored content.

## Core Technologies

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

-----

## Your Task & Evaluation

  * **Implementation**: Your primary task is to complete the logic in the files marked with `TODO` comments (`database.py`, `embeddings.py`, `rag.py`).
  * **Evaluation**: Your submission will be evaluated on **correctness**, **code quality**, and the application of **software engineering best practices** (e.g., clarity, modularity, error handling, guardrails).
  * **Freedom**: You are free to add any dependencies you see fit. We want you to use your best judgment as you would on a real project.

-----

## Submission

To submit, create a new public repository containing your solution and share the link with us. Good luck! 😄