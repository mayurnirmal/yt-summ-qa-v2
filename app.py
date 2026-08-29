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
from app.pipeline import process_video
from app.rag.chains import ask  # or ConversationSession for multi-turn

logging.basicConfig(level=logging.INFO)

TEST_URL = "https://www.youtube.com/watch?v=JX56WlESFaU"  # keep this under ~5 min for a fast first run
TEST_QUESTION = "What is this video about?"
retriever = process_video(TEST_URL)
result = ask(retriever, TEST_QUESTION)
print(result)