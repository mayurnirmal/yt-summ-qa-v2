"""
Builds, saves, loads, and merges per-video FAISS indexes.
Each video gets its own index on disk so re-embedding one video never touches another's data.
"""
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.config import VECTORSTORE_DIR
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _index_path(video_id: str) -> Path:
    return Path(VECTORSTORE_DIR) / video_id


def build_vectorstore(documents: list[Document], video_id: str) -> FAISS:
    """Embed documents and save a new FAISS index for this video."""
    if not documents:
        raise ValueError(f"No documents to embed for {video_id}")

    embeddings = get_embeddings()
    store = FAISS.from_documents(documents, embeddings)

    path = _index_path(video_id)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

    logger.info(f"Saved FAISS index for {video_id} ({len(documents)} chunks) -> {path}")
    return store


def load_vectorstore(video_id: str) -> FAISS:
    """Load an existing FAISS index for one video. Raises FileNotFoundError if it doesn't exist."""
    path = _index_path(video_id)
    if not path.exists():
        raise FileNotFoundError(f"No vectorstore found for {video_id} at {path}")

    embeddings = get_embeddings()
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def load_multiple_vectorstores(video_ids: list[str]) -> FAISS:
    """Load and merge indexes for several videos into one — needed for multi-video retrieval (spec Milestone 4)."""
    if not video_ids:
        raise ValueError("video_ids must not be empty")

    merged = load_vectorstore(video_ids[0])
    for vid in video_ids[1:]:
        merged.merge_from(load_vectorstore(vid))

    return merged