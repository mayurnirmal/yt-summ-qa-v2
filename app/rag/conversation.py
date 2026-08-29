"""
Adds conversational memory to the RAG chain — follow-up questions get reformulated using chat history before retrieval.
Builds on chains.py's single-turn ask(); this is what the Streamlit UI will call for multi-turn chat.
"""
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, rephrase the question "
               "into a standalone question understandable without the chat history. "
               "Do NOT answer it — just reformulate if needed, otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the context below, transcribed from a YouTube video. "
               "If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_conversational_chain(retriever: BaseRetriever):
    """History-aware RAG chain: reformulate question w/ history -> retrieve -> answer w/ history."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GEMINI_API_KEY)

    history_aware_retriever = create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_PROMPT)
    qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


class ConversationSession:
    """Holds chat history for one conversation and exposes a simple ask() interface — one instance per chat in the UI."""

    def __init__(self, retriever: BaseRetriever):
        self.chain = build_conversational_chain(retriever)
        self.chat_history: list = []  # alternating HumanMessage/AIMessage, oldest first

    def ask(self, question: str) -> dict:
        result = self.chain.invoke({"input": question, "chat_history": self.chat_history.copy()})

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result["answer"]))

        return {
            "answer": result["answer"],
            "sources": [
                {
                    "text": doc.page_content,
                    "timestamp": doc.metadata.get("start_timestamp"),
                    "video_id": doc.metadata.get("video_id"),
                }
                for doc in result.get("context", [])
            ],
        }