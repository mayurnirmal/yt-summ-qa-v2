"""
Manual end-to-end smoke test: URL -> download -> segment -> transcribe -> embed -> ask.
Not part of the pipeline itself — run directly to sanity-check the chain before building the UI.
"""
import logging

from app.database.sqlite import init_db, create_video_record, update_video_status
from app.ingestion.youtube import get_video_metadata, download_audio, extract_video_id
from app.ingestion.audio import segment_audio
from app.ingestion.transcription import transcribe_video
from app.rag.documents import build_documents
from app.rag.vectorstore import build_vectorstore
from app.rag.retriever import get_retriever
from app.rag.chains import ask

logging.basicConfig(level=logging.INFO)

TEST_URL = "https://www.youtube.com/watch?v=JX56WlESFaU"  # keep this under ~5 min for a fast first run
TEST_QUESTION = "What is this video about?"

if __name__ == "__main__":
    init_db()

    video_id = extract_video_id(TEST_URL)
    metadata = get_video_metadata(TEST_URL)
    create_video_record(video_id, metadata["title"], metadata["channel"], metadata["duration"])
    print(f"Video: {metadata['title']} ({metadata['duration']}s)")

    audio_path = download_audio(TEST_URL, video_id)
    update_video_status(video_id, "downloading")

    segments = segment_audio(audio_path, video_id)
    print(f"Split into {len(segments)} segments")

    transcript = transcribe_video(segments, video_id)
    print(f"Transcribed {len(transcript)} chunks")

    documents = build_documents(video_id, metadata)
    print(f"Built {len(documents)} document chunks")

    store = build_vectorstore(documents, video_id)
    retriever = get_retriever(store)

    result = ask(retriever, TEST_QUESTION)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    for s in result["sources"]:
        print(f"[{s['timestamp']:.0f}s] {s['text'][:80]}...")