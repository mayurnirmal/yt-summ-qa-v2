"""
Wraps the HuggingFace sentence-transformer embedding model as a single shared instance.
Lazy-loaded singleton, same pattern as the whisper model — avoids reloading weights per call.
"""
import logging
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings