# app/config.py
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