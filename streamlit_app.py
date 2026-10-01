"""
Streamlit UI: paste a YouTube URL, process it, chat about the video with cited timestamps.
Wires together pipeline.process_video() and rag/conversation.py — no business logic lives here.
"""
import logging
from pathlib import Path
import pickle
import streamlit as st

from app.config import VECTORSTORE_DIR
from app.database.sqlite import init_db, get_completed_videos, get_video_record
from app.pipeline import process_video
from app.rag.conversation import ConversationSession
from app.rag.vectorstore import load_vectorstore
from app.rag.retriever import get_retriever
from app.ingestion.youtube import extract_video_id, InvalidYouTubeURLError, VideoUnavailableError

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="YouTube RAG Q&A", page_icon="🎥", layout="wide")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # video_id -> ConversationSession
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None
if "video_titles" not in st.session_state:
    st.session_state.video_titles = {}  # video_id -> title, for display

init_db()


def format_duration(seconds: int | float | None) -> str:
    """Formats seconds into readable string (e.g., 14m 20s or 1h 05m 12s)."""
    if not seconds:
        return ""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def fetch_processed_videos() -> list[dict]:
    """
    Returns metadata for all videos that have already been processed and indexed.
    Combines SQLite records and on-disk vector stores.
    """
    videos_dict = {}

    # 1. Fetch completed videos from SQLite
    try:
        completed = get_completed_videos()
        for v in completed:
            vid = v["video_id"]
            videos_dict[vid] = {
                "video_id": vid,
                "title": v.get("title") or vid,
                "channel": v.get("channel") or "",
                "duration": v.get("duration"),
                "updated_at": v.get("updated_at"),
            }
    except Exception as e:
        logging.warning(f"Could not read completed videos from DB: {e}")

    # 2. Check on-disk vectorstores in case any exist outside DB records
    vs_dir = Path(VECTORSTORE_DIR)
    if vs_dir.exists():
        for vid_folder in vs_dir.iterdir():
            if vid_folder.is_dir() and (vid_folder / "index.faiss").exists():
                vid = vid_folder.name
                if vid not in videos_dict:
                    videos_dict[vid] = {
                        "video_id": vid,
                        "title": vid,
                        "channel": "",
                        "duration": None,
                        "updated_at": None,
                    }

    # 3. If title is still just the video ID, attempt to read real title from index.pkl metadata
    for vid, item in videos_dict.items():
        if item["title"] == vid:
            pkl_path = vs_dir / vid / "index.pkl"
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        docstore, _ = pickle.load(f)
                        for doc in docstore._dict.values():
                            doc_title = getattr(doc, "metadata", {}).get("title")
                            doc_channel = getattr(doc, "metadata", {}).get("channel")
                            if doc_title:
                                item["title"] = doc_title
                            if doc_channel and not item["channel"]:
                                item["channel"] = doc_channel
                            break
                except Exception as ex:
                    logging.debug(f"Could not extract title from {pkl_path}: {ex}")

    # Keep session titles in sync
    for vid, item in videos_dict.items():
        if item["title"] and item["title"] != vid:
            st.session_state.video_titles[vid] = item["title"]

    return list(videos_dict.values())


def activate_video(video_id: str, title: str | None = None):
    """Activates a video session, loading its FAISS index into a ConversationSession if needed."""
    try:
        if video_id not in st.session_state.sessions:
            store = load_vectorstore(video_id)
            retriever = get_retriever(store)
            st.session_state.sessions[video_id] = ConversationSession(retriever)

        if title:
            st.session_state.video_titles[video_id] = title
        elif video_id not in st.session_state.video_titles:
            rec = get_video_record(video_id)
            st.session_state.video_titles[video_id] = rec.get("title") if rec and rec.get("title") else video_id

        st.session_state.active_video_id = video_id
    except Exception as e:
        st.error(f"Error loading video index for {video_id}: {e}")
        logging.exception(f"Failed to load vectorstore for {video_id}")


st.title("🎥 YouTube RAG Q&A")

# Preload processed videos list
processed_videos = fetch_processed_videos()
processed_map = {v["video_id"]: v for v in processed_videos}

with st.sidebar:
    st.header("Add a video")
    url = st.text_input("YouTube URL", key="url_input")
    process_clicked = st.button("Process Video", type="primary")

    if process_clicked and url:
        try:
            video_id = extract_video_id(url)
        except InvalidYouTubeURLError:
            st.error("That doesn't look like a valid YouTube URL.")
        else:
            if video_id in st.session_state.sessions:
                st.info("Already loaded — switching to it.")
                st.session_state.active_video_id = video_id
            elif video_id in processed_map:
                st.info("Already processed — loading from cache...")
                activate_video(video_id, processed_map[video_id].get("title"))
                st.rerun()
            else:
                with st.spinner("Processing video — this can take a few minutes for long videos..."):
                    try:
                        retriever = process_video(url)
                        st.session_state.sessions[video_id] = ConversationSession(retriever)
                        rec = get_video_record(video_id)
                        title = rec.get("title") if rec and rec.get("title") else video_id
                        st.session_state.video_titles[video_id] = title
                        st.session_state.active_video_id = video_id
                        st.success("Video processed — ask away!")
                        st.rerun()
                    except VideoUnavailableError as e:
                        st.error(f"Couldn't process this video: {e}")
                    except Exception as e:
                        logging.exception("Unexpected pipeline failure")
                        st.error(f"Something went wrong: {e}")

    if processed_videos:
        st.divider()
        st.header("Processed videos")
        for v in processed_videos:
            vid = v["video_id"]
            title = v.get("title") or vid
            short_title = (title[:30] + "...") if len(title) > 33 else title
            is_active = (vid == st.session_state.active_video_id)
            btn_label = f"▶ {short_title}" if is_active else short_title
            if st.button(btn_label, key=f"sidebar_select_{vid}", use_container_width=True, type="primary" if is_active else "secondary"):
                activate_video(vid, title)
                st.rerun()

active_id = st.session_state.active_video_id

if active_id is None:
    if processed_videos:
        st.markdown(
            "### 📺 Ready-to-Use Processed Videos\n"
            "The videos below have already been downloaded, transcribed, and indexed. "
            "**You don't need to process them again** — simply select any video to begin asking questions immediately!"
        )
        st.write("")

        for v in processed_videos:
            vid = v["video_id"]
            title = v.get("title") or vid
            channel = v.get("channel")
            duration_str = format_duration(v.get("duration"))
            yt_link = f"https://www.youtube.com/watch?v={vid}"

            with st.container(border=True):
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(f"#### 🎬 [{title}]({yt_link})")
                    details = []
                    if channel:
                        details.append(f"**Channel:** {channel}")
                    if duration_str:
                        details.append(f"**Duration:** {duration_str}")
                    details.append(f"**Video ID:** `{vid}`")
                    st.markdown(" • ".join(details))
                with col_btn:
                    st.write("")
                    if st.button("💬 Chat", key=f"main_select_{vid}", use_container_width=True, type="primary"):
                        activate_video(vid, title)
                        st.rerun()

        st.divider()
        st.info("💡 Want to ask questions about a different video? Enter any YouTube URL in the sidebar to process a new one.")
    else:
        st.info("👋 Welcome! No videos have been processed yet. Add a YouTube URL in the sidebar to get started.")
else:
    current_title = st.session_state.video_titles.get(active_id, active_id)
    yt_link = f"https://www.youtube.com/watch?v={active_id}"

    col_title, col_switch = st.columns([5, 1])
    with col_title:
        st.subheader(f"💬 Chatting about: [{current_title}]({yt_link})")
    with col_switch:
        if st.button("⬅️ Switch Video", use_container_width=True):
            st.session_state.active_video_id = None
            st.rerun()

    session: ConversationSession = st.session_state.sessions[active_id]

    for msg in session.chat_history:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    question = st.chat_input("Ask something about this video...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = session.ask(question)
            st.write(result["answer"])

            if result["sources"]:
                with st.expander("Sources"):
                    for src in result["sources"]:
                        ts = src["timestamp"] or 0
                        minutes, seconds = int(ts // 60), int(ts % 60)
                        video_link = f"https://youtube.com/watch?v={src['video_id']}&t={int(ts)}s"
                        st.markdown(f"**[{minutes}:{seconds:02d}]({video_link})** {src['text'][:150]}...")