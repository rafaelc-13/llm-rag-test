from sentence_transformers import SentenceTransformer, util
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    A class responsible for generating embeddings using the SentenceTransformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator by loading the SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to load.
        """
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.

        Args:
            texts (List[str]): A list of strings to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.

        Raises:
            ValueError: If the input is not a list of strings or is empty.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            logger.error("Input must be a list of strings.")
            raise ValueError("Input must be a list of strings.")
        
        if not texts:
            logger.warning("Received an empty list of texts. Returning an empty list of embeddings.")
            return []

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts.")
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            logger.info("Embeddings generated successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RuntimeError(f"Error generating embeddings: {e}")