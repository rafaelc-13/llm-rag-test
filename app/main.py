from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .models import AddDocumentRequest, SearchResult, ChatRequest, ChatResponse
from .database import ChromaDBManager
from .embeddings import EmbeddingGenerator
from .rag import RAGPipeline
from .config import OPENROUTER_API_KEY, CHROMADB_PATH, EMBEDDING_MODEL, MODEL_SLUG

app = FastAPI(title="RAG System API")

# Enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = ChromaDBManager(CHROMADB_PATH)
embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL)
rag_pipeline = RAGPipeline(db_manager, embedding_generator, OPENROUTER_API_KEY, MODEL_SLUG)


@app.get("/")
def root():
    return {"name": "RAG System", "version": "1.0"}


@app.post("/add_document")
async def add_document(request: AddDocumentRequest):
    """
    Add a document to the database.

    This endpoint allows you to store a document in the database, 
    it generates embeddings for the provided text and metadata and stores them
    along with the metadata.

    Args:
        The request body containing the text and optional metadata.

        - 'text' (str): The text content of the document, with a limit of 5000 characters.
        - 'metadata' (dict, optional): Additional metadata to store with the document.
            - 'author' (str, optional): The author of the document.
            - 'date' (str, optional): The date the document was created.
            - 'source' (str, optional): The source of the document.
    
    Example:
        {
            "text": "This is a sample document.",
            "metadata": {
                "author": "John Doe",
                "date": "2023-10-01",
                "source": "Sample Source"
            }
        }

    Returns:
        dict: A HTTP response indicating success and the document ID.
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text input exceeds the maximum length of 5000 characters.")

        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings([request.text])

        # Store in ChromaDB
        doc_id = db_manager.add_document(request.text, embeddings[0], request.metadata)

        return {"success": True, "id": doc_id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str = Query(...),
    limit: int = Query(default=5)
):
    # TODO: Generate query embedding and search ChromaDB
    return []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: Implement RAG
    result = rag_pipeline.generate_answer(
        request.question,
        request.max_results
    )
    
    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)