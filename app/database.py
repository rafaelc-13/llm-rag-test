import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    A class to manage the connection, insertion, and search operations in ChromaDB.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB client and create a default collection.

        Args:
            persist_directory (str): Directory to persist the ChromaDB data.
        """
        try:
            logger.info(f"Initializing ChromaDB client with persistence directory: {persist_directory}")
            self.client = chromadb.Client(Settings(persist_directory=persist_directory))
            self.collection = self.client.get_or_create_collection(name="documents")
            logger.info("ChromaDB client initialized and default collection created.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Error initializing ChromaDB client: {e}")

    def add_document(self, text: str, embedding: List[float], metadata: Dict = {}) -> str:
        """
        Add a document to the ChromaDB collection.

        Args:
            text (str): The text content of the document.
            embedding (List[float]): The embedding vector of the document.
            metadata (Dict): Additional metadata to store with the document.

        Returns:
            str: The unique ID of the added document.
        """
        try:
            doc_id = str(uuid.uuid4())
            logger.info(f"Adding document with ID: {doc_id}")
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            logger.info("Document added successfully.")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add document to ChromaDB: {e}")
            raise RuntimeError(f"Error adding document to ChromaDB: {e}")

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the ChromaDB collection.

        Args:
            query_embedding (List[float]): The embedding vector to search for.
            n_results (int): The number of top results to return.

        Returns:
            Dict[str, Any]: A dictionary containing the matched documents, distances, and metadata.
        """
        try:
            logger.info(f"Searching for top {n_results} similar documents.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.info("Search completed successfully.")
            return {
                "documents": results.get("documents", []),
                "distances": results.get("distances", []),
                "metadatas": results.get("metadatas", [])
            }
        except Exception as e:
            logger.error(f"Failed to search in ChromaDB: {e}")
            raise RuntimeError(f"Error searching in ChromaDB: {e}")