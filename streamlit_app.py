"""
Streamlit UI: paste a YouTube URL, process it, chat about the video with cited timestamps.
Wires together pipeline.process_video() and rag/conversation.py — no business logic lives here.
"""
import logging
import streamlit as st

from app.database.sqlite import init_db
from app.pipeline import process_video
from app.rag.conversation import ConversationSession
from app.ingestion.youtube import extract_video_id, InvalidYouTubeURLError
from app.ingestion.youtube import VideoUnavailableError

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="YouTube RAG Q&A", page_icon="🎥", layout="wide")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # video_id -> ConversationSession
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None
if "video_titles" not in st.session_state:
    st.session_state.video_titles = {}  # video_id -> title, for display

init_db()

st.title("🎥 YouTube RAG Q&A")

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
                st.info("Already processed — switching to it.")
                st.session_state.active_video_id = video_id
            else:
                with st.spinner("Processing video — this can take a few minutes for long videos..."):
                    try:
                        retriever = process_video(url)
                        st.session_state.sessions[video_id] = ConversationSession(retriever)
                        st.session_state.video_titles[video_id] = video_id  # placeholder, see note below
                        st.session_state.active_video_id = video_id
                        st.success("Video processed — ask away!")
                    except VideoUnavailableError as e:
                        st.error(f"Couldn't process this video: {e}")
                    except Exception as e:
                        logging.exception("Unexpected pipeline failure")
                        st.error(f"Something went wrong: {e}")

    if st.session_state.sessions:
        st.divider()
        st.header("Processed videos")
        for vid in st.session_state.sessions:
            label = st.session_state.video_titles.get(vid, vid)
            if st.button(label, key=f"select_{vid}"):
                st.session_state.active_video_id = vid

active_id = st.session_state.active_video_id

if active_id is None:
    st.info("Add a YouTube URL in the sidebar to get started.")
else:
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