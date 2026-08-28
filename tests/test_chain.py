"""
Tests context formatting and the ask() orchestration, with Gemini mocked out entirely.
No API key or network call needed to run these in CI.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.rag.chains import _format_docs, ask


def test_format_docs_includes_timestamps():
    docs = [
        Document(page_content="first chunk", metadata={"start_timestamp": 12.0}),
        Document(page_content="second chunk", metadata={"start_timestamp": 90.0}),
    ]
    result = _format_docs(docs)
    assert "[12s] first chunk" in result
    assert "[90s] second chunk" in result


def test_format_docs_handles_missing_timestamp():
    docs = [Document(page_content="no timestamp", metadata={})]
    result = _format_docs(docs)
    assert "[0s] no timestamp" in result


@patch("app.rag.chains.build_rag_chain")
def test_ask_returns_answer_and_sources(mock_build_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "This is the answer."
    mock_build_chain.return_value = mock_chain

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="relevant chunk", metadata={"start_timestamp": 30.0, "video_id": "vid1"}),
    ]

    result = ask(mock_retriever, "What is this about?")

    assert result["answer"] == "This is the answer."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["video_id"] == "vid1"
    assert result["sources"][0]["timestamp"] == 30.0