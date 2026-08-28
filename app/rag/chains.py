"""
Builds the Gemini-powered RAG chain: retrieved chunks + question -> grounded answer with sources.
This is what main.py/Streamlit UI actually calls to answer a user's question about a video.
"""
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below, which is transcribed from a YouTube video.
If the context doesn't contain the answer, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    """Joins retrieved chunks into one context string, each tagged with its timestamp for traceability."""
    return "\n\n".join(
        f"[{doc.metadata.get('start_timestamp', 0):.0f}s] {doc.page_content}"
        for doc in docs
    )


# app/rag/chains.py — updated build_rag_chain()
def build_rag_chain(retriever: BaseRetriever):
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GEMINI_API_KEY)
    # no temperature param — Gemini 3.x ignores it now and will error on it in future releases

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(retriever: BaseRetriever, question: str) -> dict:
    """Runs the RAG chain and also returns the source chunks used, for citation display in the UI."""
    chain = build_rag_chain(retriever)
    answer = chain.invoke(question)
    sources = retriever.invoke(question)

    return {
        "answer": answer,
        "sources": [
            {"text": doc.page_content, "timestamp": doc.metadata.get("start_timestamp"), "video_id": doc.metadata.get("video_id")}
            for doc in sources
        ],
    }