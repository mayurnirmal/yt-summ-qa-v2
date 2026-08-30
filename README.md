# YouTube RAG Q&A

Ask questions about any YouTube video and get answers grounded in the video's actual transcript — with clickable timestamps back to the source.

Paste a URL → the video is downloaded, transcribed, chunked, and embedded → ask questions in a chat interface → answers cite the exact moment in the video they came from.

---

## 1. Architecture

```
YouTube URL
    │
    ▼
┌─────────────────────┐
│ 1. Ingestion         │  extract video ID → fetch metadata → download audio (yt-dlp → mp3)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 2. Audio Segmentation│  split into 10-min chunks (audio.py)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 3. Transcription     │  faster-whisper, per-segment, timestamp-offset & cached (transcription.py)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 4. Chunking           │  transcript → LangChain Documents, each tagged with a start timestamp (documents.py)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 5. Embedding + Index  │  HuggingFace sentence-transformer → FAISS index per video (embeddings.py, vectorstore.py)
└─────────┬─────────────┘
          ▼
┌─────────────────────┐
│ 6. Retrieval + RAG    │  similarity search → Gemini 3.6 Flash → grounded answer + cited sources (retriever.py, chains.py, conversation.py)
└─────────┬─────────────┘
          ▼
      Streamlit Chat UI
```

Every stage checks SQLite before doing work, so an interrupted run (network drop, crash, closed laptop) resumes instead of restarting from scratch. See [`app/database/sqlite.py`](app/database/sqlite.py) and the orchestrator in [`app/pipeline.py`](app/pipeline.py).

### Chunking strategy

- **Splitter:** `RecursiveCharacterTextSplitter` (LangChain)
- **Chunk size:** 1000 characters, **overlap:** 150 characters
- **Why:** recursive splitting respects natural text boundaries (paragraphs → sentences → words) instead of cutting mid-sentence, which matters for transcript text that has no paragraph structure of its own. 1000/150 is a starting point sized for typical spoken-language chunk density (roughly 150–200 words) — enough context per chunk for Gemini to answer without hitting too many chunks per query. These are exposed as `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env` specifically so they can be tuned per video length without a code change.
- **Timestamp mapping:** each chunk is mapped back to the transcript segment it starts in, so every chunk carries a `start_timestamp` in its metadata — this is what makes clickable source citations possible.

### Embedding model & vector store

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers, via `langchain-huggingface`) — small (~80MB), fast on CPU, no API cost, and accurate enough for single-video semantic search. Runs locally, so embedding cost is zero and there's no network dependency at this stage.
- **Vector store:** FAISS, one index per video, saved to `data/vectorstores/<video_id>/`. Chosen for zero setup (no server/service to run) and because per-video isolation means re-processing one video never touches another's data. Multi-video queries are supported by merging indexes in memory (`load_multiple_vectorstores`), not by a single shared index.

### Retrieval & prompting

- **Retrieval:** top-k similarity search (`k=5` by default, configurable) over the video's FAISS index.
- **Single-turn Q&A** (`chains.py`): retrieved chunks are formatted with their timestamps and stuffed into a prompt instructing Gemini to answer *only* from that context, and to say "I don't know" rather than fabricate an answer.
- **Conversational Q&A** (`conversation.py`): follow-up questions are first rewritten into standalone questions using chat history (so "what about that?" resolves correctly), then go through the same retrieve → answer flow, with chat history threaded into the final answer prompt too.
- **Source attribution:** every answer returns the raw chunks used to generate it, each with its `start_timestamp` and `video_id` — the UI renders these as clickable `youtube.com/watch?v=...&t=Ns` links so you can jump straight to the moment in the video.
- **LLM:** Gemini 3.6 Flash (`google-generativeai` via `langchain-google-genai`). Chosen for low cost and speed on a laptop-scale project; note Gemini 3.x deprecated `temperature`/`top_p`/`top_k`, so this project doesn't set them.

---

## 2. Setup & Run (< 10 minutes)

**Prerequisites:** Python 3.10+, `ffmpeg` on PATH, a [Gemini API key](https://aistudio.google.com/apikey).

```powershell
git clone <your-repo-url>
cd yt-summ-qa-v2

python -m venv venv
venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

copy .env.example .env             # macOS/Linux: cp .env.example .env
# then open .env and paste in your GEMINI_API_KEY

streamlit run streamlit_app.py
```

Paste a YouTube URL into the sidebar, click **Process Video**, and start asking questions once processing finishes. First run on a video downloads + transcribes + embeds it (a few minutes depending on length); re-processing the same video is near-instant since audio, transcripts, and the vector index are all cached on disk.

### Running tests

```powershell
pytest -v
```

Most tests mock external calls (yt-dlp, Whisper, Gemini) so the suite runs without network access or API keys, except the vectorstore tests, which use the real (local, free) embedding model.

---

## 3. A note on API keys

**Never commit a real API key.** This repo ships a `.env.example` with placeholder values only — copy it to `.env` (which is git-ignored) and put your real `GEMINI_API_KEY` there. If a key is ever accidentally committed, treat it as compromised and regenerate it in [Google AI Studio](https://aistudio.google.com/apikey) immediately.

---

## 4. Advantages

- **Laptop-first, zero infra cost** — no Redis, Kafka, Postgres, Kubernetes, or hosted vector DB required; everything runs on local disk and a single free/cheap LLM API.
- **Resumable by design** — every stage (download, per-segment transcription, embedding) checks existing state before redoing work, so interrupted runs don't waste time or API calls.
- **Grounded, cited answers** — every answer traces back to a specific timestamp in a specific video, not just a plausible-sounding LLM response.
- **Modular** — ingestion, transcription, chunking, embedding, and retrieval are independent modules behind clear function boundaries, so any one piece (e.g. swapping FAISS for another vector store) can change without touching the rest.
- **Tested** — each module has unit tests with external calls (network, Whisper, Gemini) mocked out, so the suite is fast and CI-friendly.

## 5. Known limitations

- **One video's context per chat session** — the UI keeps a separate `ConversationSession` per video; asking a question only searches the active video, even though multi-video retrieval is supported at the vectorstore layer (`load_multiple_vectorstores`). Combining that with per-video chat history wasn't a small extension, so it's out of scope for now.
- **Unbounded chat history** — `ConversationSession` keeps every turn for the life of the session. Fine for a short demo conversation; a long session will eventually push prompts to an uncomfortable size and cost.
- **Single-turn `ask()` retrieves twice** — once inside the LangChain chain, once again to surface sources to the caller, since LangChain's default runnable composition doesn't expose intermediate retriever output. Doubles embedding/search latency per single-turn question (the conversational chain doesn't have this issue).
- **Fixed-length audio segmentation** — segments are split by a fixed 10-minute duration, not by silence/sentence boundaries, so a segment (and therefore a transcription pass) can technically start or end mid-sentence. Doesn't affect answer quality much in practice since text chunking happens after transcription, but it's a rough edge.
- **Vectorstore "completed" check is folder-presence, not integrity-checked** — if `FAISS.save_local()` ever fails partway through, the pipeline could treat a corrupt index as valid. Hasn't been observed in practice.
- **CPU-only Whisper** — transcription runs on CPU with the `base` model for portability; larger models (or GPU) would improve transcription accuracy on unclear audio, background music, or heavy accents, at the cost of speed.
- **No chapter-aware retrieval yet** — video chapter metadata is fetched and stored but not yet used to scope or label retrieval, despite chunks tracking only a start timestamp (not an end timestamp).

## 6. What I'd improve with more time

- **Multi-video conversational retrieval** — let one chat session query across several processed videos, with per-source video attribution in the citations.
- **Chapter-aware retrieval** — use the chapter metadata already being captured to let users scope questions to a specific chapter, and to give each chunk a proper start/end range instead of a single timestamp.
- **Single-pass retrieval** for `ask()` using `RunnableParallel` to avoid the current double-retrieval cost.
- **Hierarchical summarization** for very long videos (2hr+), so "summarize this video" doesn't require stuffing the entire transcript into one prompt.
- **RAG quality evaluation** — a small eval set with a framework like RAGAS to measure answer faithfulness/relevance instead of relying on manual spot-checks.
- **GitHub Actions CI** — run the test suite automatically on push/PR (scaffolded in `.github/workflows/` but not yet wired up).
- **Retry/backoff for the `"failed"` pipeline state** — currently a failed video just gets fully reprocessed (relying on caches to skip completed steps) rather than resuming precisely from its failure point.
- **Bounded/summarized chat history** instead of an ever-growing message list, once conversations get long.

---

## 7. Project structure

```
app/
├── config.py                # centralized settings, loaded from .env
├── pipeline.py               # orchestrates the full pipeline with resume-from-DB-state logic
├── ingestion/
│   ├── youtube.py            # URL parsing, metadata, audio download
│   ├── audio.py               # splits long audio into fixed-length segments
│   └── transcription.py      # faster-whisper transcription, per-segment cached
├── database/
│   └── sqlite.py              # video/segment status tracking for resumability
├── rag/
│   ├── documents.py           # transcript → chunked LangChain Documents w/ timestamps
│   ├── embeddings.py          # shared HuggingFace embedding model instance
│   ├── vectorstore.py         # FAISS build/save/load/merge, per video
│   ├── retriever.py            # top-k retriever wrapper
│   ├── chains.py               # single-turn Gemini RAG chain
│   └── conversation.py        # multi-turn, history-aware RAG chain
└── models/
    └── schemas.py

streamlit_app.py              # chat UI entry point
tests/                        # unit tests, external calls mocked
```