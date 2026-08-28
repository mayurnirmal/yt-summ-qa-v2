"""
Wraps a FAISS vectorstore into a LangChain retriever with top-k config from settings.
Thin wrapper now so retrieval strategy (MMR, filters, etc.) can change in one place later.
"""
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from app.config import TOP_K

logger = logging.getLogger(__name__)


def get_retriever(vectorstore: FAISS, k: int = TOP_K) -> BaseRetriever:
    """Returns a similarity-search retriever over the given vectorstore."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})