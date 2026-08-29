"""
Tests ConversationSession's history accumulation and source extraction, with the chain mocked out.
No API key or network call needed to run these in CI.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.conversation import ConversationSession


@patch("app.rag.conversation.build_conversational_chain")
def test_ask_appends_to_history(mock_build_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"answer": "First answer.", "context": []}
    mock_build_chain.return_value = mock_chain

    session = ConversationSession(retriever=MagicMock())
    session.ask("First question?")

    assert len(session.chat_history) == 2
    assert isinstance(session.chat_history[0], HumanMessage)
    assert session.chat_history[0].content == "First question?"
    assert isinstance(session.chat_history[1], AIMessage)
    assert session.chat_history[1].content == "First answer."


@patch("app.rag.conversation.build_conversational_chain")
def test_ask_passes_growing_history_to_chain(mock_build_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"answer": "Some answer.", "context": []}
    mock_build_chain.return_value = mock_chain

    session = ConversationSession(retriever=MagicMock())
    session.ask("First question?")
    session.ask("Follow-up question?")

    # second call should receive the first Q&A as history
    second_call_args = mock_chain.invoke.call_args_list[1][0][0]
    assert len(second_call_args["chat_history"]) == 2
    assert second_call_args["input"] == "Follow-up question?"


@patch("app.rag.conversation.build_conversational_chain")
def test_ask_extracts_sources_from_context(mock_build_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Answer with sources.",
        "context": [
            Document(page_content="relevant chunk", metadata={"start_timestamp": 45.0, "video_id": "vid1"}),
        ],
    }
    mock_build_chain.return_value = mock_chain

    session = ConversationSession(retriever=MagicMock())
    result = session.ask("A question?")

    assert len(result["sources"]) == 1
    assert result["sources"][0]["timestamp"] == 45.0
    assert result["sources"][0]["video_id"] == "vid1"