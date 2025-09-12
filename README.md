# Technical Test: Simple RAG System

Welcome! The goal of this challenge is to build a simple Retrieval-Augmented Generation (RAG) system from scratch. You'll create a basic API that can store documents, search for relevant information, and answer questions based on the stored content.


## Core Technologies

- **ChromaDB** for vector storage
- **all-MiniLM-L6-v2** for embeddings  
- **OpenRouter** for LLM responses

## Required Functionalities

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

# Run Backend
uvicorn app.main:app --reload

#Run Interface
streamlit run streamlit_app.py
```

## Your Task & Evaluation

  * **Implementation**: Your primary task is to complete the logic in the files marked with `TODO` comments (`database.py`, `embeddings.py`, `rag.py`).
  * **Evaluation**: Your submission will be evaluated on **correctness**, **code quality**, and the application of **software engineering best practices** (e.g., clarity, modularity, error handling, guardrails).
  * **Freedom**: You are free to add any dependencies you see fit. We want you to use your best judgment as you would on a real project.

## Submission

To submit, create a new public repository containing your solution and share the link with us. Good luck! 😄

---

## Additional Notes & Engineering Decisions

- **Pinned openai version:**  
  The project uses `openai==0.28.1` to ensure compatibility with OpenRouter's OpenAI-compatible API.  
  Newer versions of the `openai` library (>=1.0.0) are not compatible with this integration.  
  If you upgrade, you must refactor the LLM call logic.

- **Project Structure:**
    ```
    llm-rag-test/
    ├── app/
    │   ├── main.py
    |   ├── .env
    │   ├── database.py
    │   ├── config.py
    │   ├── embeddings.py
    │   ├── rag.py
    │   ├── models.py
    │   └── ...
    ├── streamlit_app.py
    ├── requirements.txt
    └── README.md
    ```

- **Quick API Test Examples:**
    ```bash
    # Add a document
    curl -X POST "http://localhost:8000/add_document" -H "Content-Type: application/json" -d '{"text": "Brazil é dos Brasileiros", "metadata": {"contexto": "País", "date": "2024-06-01"}}'

    # Search for documents
    curl "http://localhost:8000/search?query=Brazil&limit=3"

    # RAG Chat
    curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"question": "Quem é o dono do Brazil?", "max_results": 2}'
    ```

- **Engineering Notes:**
    - Modular, extensible, and clean codebase.
    - Robust error handling and logging throughout the stack.
    - All endpoints are fully documented and testable via Swagger (`/docs`).
    - Streamlit frontend guides the user through all flows.
    - Ready for extension: add authentication, new vector DBs, or LLM providers easily.

- **Possible Extensions:**
    - Add authentication and user management.
    - Support for multiple embedding or LLM models.
    - Deploy with Docker or on cloud platforms.
    - Add monitoring and analytics for usage.

---

**For any questions or suggestions, feel free to open an issue or contact the author.**