from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingGenerator:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # TODO: Load sentence transformer model
        self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # TODO: Generate embeddings
        return [[0.0] * 384 for _ in texts]