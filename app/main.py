from fastapi import FastAPI, Query, HTTPException, logger
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
        raise HTTPException(status_code=500, detail= f"An unexpected error occurred.{e}")


@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str = Query(..., description="Query string to search for similar documents."),
    limit: int = Query(default=5, description="Number of top results to return.")
):
    """
    Search for similar documents using a query string.

    This endpoint generates an embedding for the query, searches for similar documents in ChromaDB,
    and returns the top matches with their similarity scores and metadata.

    Args:
        query (str): The query string to search for.
        limit (int): The number of top results to return.

    Returns:
        List[SearchResult]: List of search results with content, score, and metadata.
    """
    try:
        # 1. Gerar embedding da query
        query_embedding = embedding_generator.generate_embeddings([query])[0]

        # 2. Buscar documentos similares
        results = db_manager.search(query_embedding, n_results=limit)

        # 3. Formatar resultados para o modelo SearchResult
        search_results = []
        docs = results.get("documents", [[]])
        scores = results.get("distances", [[]])
        metadatas = results.get("metadatas", [[]])

        for doc, score, meta in zip(docs[0], scores[0], metadatas[0]):
            search_results.append(SearchResult(
                content=doc,
                score=score,
                metadata=meta or {}
            ))

        if not search_results:
            raise HTTPException(
                status_code=404,
                detail="Nenhum documento relevante encontrado para a consulta fornecida."
            )

        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")


@app.post("/chat", response_model=ChatResponse, summary="RAG Chat", tags=["RAG"])
async def chat(request: ChatRequest):
    """
    Does a RAG query: retrieves relevant context, sends it to the LLM, and returns the response with sources.

    - **question**: User's question.
    - **max_results**: Maximum number of context documents to retrieve (default: 3).

    Example payload:
    {
        "question": "What is the capital of Japan?",
        "max_results": 3
    }
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="O campo 'question' não pode ser vazio.")
        if request.max_results < 1 or request.max_results > 10:
            raise HTTPException(status_code=400, detail="O campo 'max_results' deve estar entre 1 e 10.")

        # Logging para rastreabilidade
        logger.info(f"Recebida pergunta para RAG: '{request.question}' (max_results={request.max_results})")

        result = rag_pipeline.generate_answer(
            request.question,
            request.max_results
        )

        # Checagem de resposta do pipeline
        if not result.get("answer"):
            raise HTTPException(status_code=500, detail="Falha ao gerar resposta.")

        return ChatResponse(**result)

    except HTTPException as he:
        logger.warning(f"Erro de input no endpoint /chat: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Erro inesperado no endpoint /chat: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a requisição do chat.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)