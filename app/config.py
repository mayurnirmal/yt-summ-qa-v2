"""
app/config.py

Central configuration for the YouTube RAG LangChain project.

Why this exists:
    Every other module (ingestion, RAG, database) imports from here instead
    of reading env vars or hardcoding paths directly. This keeps API keys,
    model choices, chunking parameters, and storage paths in one place so
    they can be tuned/benchmarked (see spec section 13, 21) without touching
    business logic.

What it does:
    - Loads secrets/config from .env via python-dotenv
    - Exposes chunking defaults (CHUNK_SIZE, CHUNK_OVERLAP) as starting
      values to be benchmarked later, not final numbers
    - Defines the on-disk data layout (audio/transcripts/vectorstores/db)
      so ingestion and RAG modules agree on where things live
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# paths
DATA_DIR = "data"
AUDIO_DIR = f"{DATA_DIR}/audio"
TRANSCRIPT_DIR = f"{DATA_DIR}/transcripts"
VECTORSTORE_DIR = f"{DATA_DIR}/vectorstores"
DB_PATH = f"{DATA_DIR}/database/youtube_rag.db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5