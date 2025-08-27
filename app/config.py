import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_SLUG = os.getenv("MODEL_SLUG", "openai/gpt-3.5-turbo")