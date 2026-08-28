"""
Tests FAISS index build/save/load/merge using tiny real embeddings (not mocked) —
embedding a few short strings is fast enough to run in CI without a network call.
"""
import pytest
from langchain_core.documents import Document

from app.rag import vectorstore as vs_module


@pytest.fixture(autouse=True)
def temp_vectorstore_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vs_module, "VECTORSTORE_DIR", str(tmp_path))
    yield


def _sample_docs(video_id="vid1"):
    return [
        Document(page_content="cats are great pets", metadata={"video_id": video_id, "start_timestamp": 0.0}),
        Document(page_content="dogs are loyal animals", metadata={"video_id": video_id, "start_timestamp": 5.0}),
    ]


def test_build_vectorstore_rejects_empty_documents():
    with pytest.raises(ValueError):
        vs_module.build_vectorstore([], "vid1")


def test_build_and_load_vectorstore_roundtrip():
    docs = _sample_docs("vid1")
    vs_module.build_vectorstore(docs, "vid1")

    loaded = vs_module.load_vectorstore("vid1")
    results = loaded.similarity_search("pets", k=1)

    assert len(results) == 1
    assert results[0].metadata["video_id"] == "vid1"


def test_load_vectorstore_missing_raises():
    with pytest.raises(FileNotFoundError):
        vs_module.load_vectorstore("nonexistent")


def test_load_multiple_vectorstores_rejects_empty_list():
    with pytest.raises(ValueError):
        vs_module.load_multiple_vectorstores([])


def test_load_multiple_vectorstores_merges_results():
    vs_module.build_vectorstore(_sample_docs("vid1"), "vid1")
    vs_module.build_vectorstore(_sample_docs("vid2"), "vid2")

    merged = vs_module.load_multiple_vectorstores(["vid1", "vid2"])
    results = merged.similarity_search("pets", k=4)

    video_ids = {r.metadata["video_id"] for r in results}
    assert video_ids == {"vid1", "vid2"}