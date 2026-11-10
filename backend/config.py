
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM Model
LLM_MODEL = "openrouter/free"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS Configuration
VECTOR_DB_PATH = "data/faiss_index"

# PDF Configuration
UPLOAD_FOLDER = "data"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Text Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
