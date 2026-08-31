# Technical Design Document — YouTube RAG Q&A

Deep-dive companion to `README.md`. Where the README tells you *what* the system does and how to run it, this document explains *how each module works internally* — function-by-function, including the design decisions, edge cases, and known trade-offs behind each one.

---

## Table of Contents

1. [Config Layer](#1-config-layer)
2. [Ingestion — YouTube](#2-ingestion--youtube-appingestionyoutubepy)
3. [Ingestion — Audio Segmentation](#3-ingestion--audio-segmentation-appingestionaudiopy)
4. [Ingestion — Transcription](#4-ingestion--transcription-appingestiontranscriptionpy)
5. [State Tracking — SQLite](#5-state-tracking--sqlite-appdatabasesqlitepy)
6. [RAG — Document Chunking](#6-rag--document-chunking-approgdocumentspy)
7. [RAG — Embeddings](#7-rag--embeddings-approgembeddingspy)
8. [RAG — Vector Store](#8-rag--vector-store-approgvectorstorepy)
9. [RAG — Retriever](#9-rag--retriever-approgretrieverpy)
10. [RAG — Single-Turn Chain](#10-rag--single-turn-chain-approgchainspy)
11. [RAG — Conversational Chain](#11-rag--conversational-chain-approgconversationpy)
12. [Pipeline Orchestrator](#12-pipeline-orchestrator-apppipelinepy)
13. [Streamlit UI](#13-streamlit-ui-streamlit_apppy)
14. [Cross-Cutting Concerns](#14-cross-cutting-concerns)

---

## 1. Config Layer

**File:** `app/config.py`

Every other module imports settings from here rather than reading `os.environ` directly. This is a single source of truth so a value like `CHUNK_SIZE` is guaranteed identical wherever it's used — `documents.py` when splitting, and anywhere else that might need to reason about chunk boundaries.

`load_dotenv()` runs at import time, so `.env` must exist (or the real env vars must be set) *before* any other app module is imported — importing `app.config` late in a script after other imports that indirectly need it will still work because Python caches modules, but it's worth knowing the load happens once, at first import.

Paths (`AUDIO_DIR`, `TRANSCRIPT_DIR`, etc.) are relative strings, not `Path` objects — each consumer wraps them in `Path(...)` as needed. This keeps `config.py` dependency-free (no need to import `pathlib` just to store strings) at the cost of every consumer doing that wrapping itself.

---

## 2. Ingestion — YouTube (`app/ingestion/youtube.py`)

### `extract_video_id(url)`

Regex-based, not URL-parsing-library-based. Two patterns cover `watch?v=`, `/VIDEO_ID`-style, and `youtu.be/` shortlinks. This is intentionally permissive — it matches an 11-character ID pattern rather than validating the full URL structure, which means it'll extract an ID from malformed-but-recognizable URLs too. Trade-off: simpler code, but it won't catch every invalid URL — some clearly bad input falls through to `get_video_metadata()`/`download_audio()` instead, where yt-dlp itself will reject it.

### `get_video_metadata(url)`

A metadata-only call — `skip_download: True` in `ydl_opts` guarantees no network transfer of audio/video happens here, only the info JSON. This is deliberately called *before* `download_audio()` in the pipeline so a bad/private/deleted video fails fast, before spending time on a download.

`chapters` is extracted here even though nothing currently consumes it for retrieval — it's essentially free (same API call, no extra cost) and stashing it early means a future chapter-aware retrieval feature doesn't need to re-fetch metadata.

### `download_audio(url, video_id)`

As of the performance pass, this **no longer transcodes to mp3**. Originally it used an `FFmpegExtractAudio` postprocessor to force mp3 output; now it keeps whatever container yt-dlp downloads natively (typically `webm` or `m4a`), because:
- `faster-whisper` resamples all input to 16kHz mono internally regardless of source format, so a 192kbps mp3 re-encode added CPU time for zero downstream benefit.
- ffmpeg (used internally by both yt-dlp's demuxing and pydub/faster-whisper's decoding) reads webm/m4a natively — no compatibility loss.

`concurrent_fragment_downloads: 4` is set to speed up videos delivered as fragmented/DASH streams (common for longer videos) — it has little effect on progressively-downloaded short videos, but costs nothing to leave on.

**Error handling:** all yt-dlp `DownloadError`s are caught and re-raised as `VideoUnavailableError` — this is what lets `pipeline.py` catch one exception type regardless of *why* yt-dlp failed (private video, deleted, geo-blocked, network issue), and is also what lets `main`/the UI show a friendly message instead of a yt-dlp stack trace.

---

## 3. Ingestion — Audio Segmentation (`app/ingestion/audio.py`)

### Why segmentation exists at all

`faster-whisper` can technically transcribe a full 1+ hour file in one call. Segmentation exists for **resumability**, not because Whisper needs smaller inputs: if the process crashes or is interrupted at minute 40 of a 57-minute video, only the remaining ~2 segments need re-transcribing, not the whole thing. This is the same reasoning behind SQLite's per-segment status tracking (see §5).

### `segment_audio(audio_path, video_id)`

Post-performance-pass, this uses **ffmpeg stream-copy (`-c copy`)** via `subprocess`, not `pydub`. The distinction matters:
- `pydub.AudioSegment.export()` fully **decodes** the source audio into raw PCM in memory, then **re-encodes** each 10-minute slice back to a compressed format. For a 57-minute video split into 6 segments, that's 6 full decode+encode cycles.
- `ffmpeg -ss <start> -i <src> -t <duration> -c copy <out>` just re-packages the existing compressed bitstream into a new container without touching the audio samples — dramatically faster, since no codec work happens at all.

**Trade-off introduced by stream-copy:** `-c copy` can only cut on **keyframes** in the underlying codec, so a requested cut at exactly 600.0s might actually land a few hundred milliseconds earlier or later depending on where the nearest keyframe is. This is irrelevant for feeding into transcription (Whisper doesn't care about a few hundred ms of segment boundary drift, and the timestamp-offset math in `transcription.py` uses the *nominal* segment length, not the actual cut point, so there's a theoretical small timestamp drift across segments — see the note in §4 below).

`SEGMENT_LENGTH_SEC = 10 * 60` is a module-level constant, not derived from anything — it's a fixed choice balancing resumability granularity (smaller = more resumable, more files) against overhead (smaller = more ffmpeg subprocess calls). 10 minutes was chosen as a reasonable middle ground, not benchmarked against alternatives.

### `_get_duration_sec(audio_path)`

Uses `ffprobe` (ships with ffmpeg) to get exact duration before deciding how many segments to cut. This avoids loading the whole file into memory just to check its length, which `pydub`'s approach would have required.

---

## 4. Ingestion — Transcription (`app/ingestion/transcription.py`)

### Model loading — `_get_model()`

Lazy-loaded module-level singleton (`_model`/`_batched_model` globals). Loading a Whisper model involves reading weight files from disk and initializing the inference backend — expensive enough that doing it once per process (not once per segment) matters a lot. The same singleton pattern is used for embeddings (§7) for the same reason.

Post-performance-pass, this wraps the base `WhisperModel` in a `BatchedInferencePipeline` and sets `cpu_threads=os.cpu_count()` — by default `faster-whisper` doesn't use all available CPU cores, so this was leaving performance on the table on any multi-core machine.

### `transcribe_segment(segment_path)`

Calls `model.transcribe(..., batch_size=16, vad_filter=True)`. `vad_filter=True` runs a lightweight voice-activity-detection pass first and skips silent stretches — on a typical tutorial-style video with intro silence, pauses, or background-music-only sections, this avoids running the (expensive) transcription model over audio that has nothing to transcribe.

Returns a flat list of `{text, start, end}` dicts — timestamps here are **relative to the start of that segment**, not the video. Offsetting to video-relative time is `transcribe_video()`'s job, not this function's — keeping this function segment-local makes it independently testable (see the existing unit tests, which mock at this boundary).

### `transcribe_video(segment_paths, video_id)` — the resumability core

This is the most complex function in the ingestion layer, and it went through a real bug fix worth understanding:

**Original design flaw:** the function checked `get_transcribed_segment_indices()` and skipped segments already marked `"transcribed"` in SQLite — but never persisted *what was transcribed*, only *that* it was done. On a resumed run, skipped segments contributed nothing to `full_transcript`, silently producing an incomplete (or, as happened in testing, a completely empty) transcript.

**Fix — per-segment caching:** every segment's transcribed-and-offset chunks are now saved to their own JSON file (`_save_segment_cache` / `_load_segment_cache`, under `data/transcripts/<video_id>/segment_NNN.json`) immediately after transcription, *before* the SQLite status is marked `"transcribed"`. On a resume, a skipped segment's chunks are loaded from this cache instead of being silently omitted. The full-video `_save_full_transcript()` JSON (`data/transcripts/<video_id>.json`) is now a convenience artifact assembled from all segment caches — the per-segment files are the actual source of truth for resumability.

**Self-healing fallback:** if SQLite says a segment is `"transcribed"` but its cache file is missing (an inconsistent state — e.g. from data manually deleted, or a bug), the function catches the `FileNotFoundError`, logs a warning, and re-transcribes that one segment rather than raising. This makes the function robust to manually-edited or corrupted cache state without needing a separate repair script.

**Timestamp offset math:** `time_offset` accumulates by a fixed `segment_length_sec = 10 * 60` per segment, *not* by each segment's actual measured duration. This works because `audio.py` produces fixed 10-minute segments (except the last, shorter one) — but it's a **coupling that isn't enforced in code**, only by convention between two files. If `audio.py`'s segmenting logic ever changes (e.g., variable-length silence-aware cuts), this offset math would silently produce wrong timestamps without erroring. Combined with the keyframe-drift noted in §3, video-relative timestamps have a theoretical small accumulating error across many segments — negligible in practice for jumping to roughly the right moment in a video, but not frame-accurate.

---

## 5. State Tracking — SQLite (`app/database/sqlite.py`)

### Why SQLite, not a JSON file or in-memory dict

- Survives process crashes (unlike in-memory state).
- Handles concurrent-ish access more safely than a hand-rolled JSON read-modify-write (though this project doesn't currently run concurrent pipelines).
- Zero setup — no server process, just a file — consistent with the project's laptop-first philosophy.

### Schema

Two tables: `videos` (one row per video, tracks overall `status`) and `segments` (one row per `(video_id, segment_index)` pair, tracks per-segment status). The composite primary key on `segments` is what makes `mark_segment_status`'s `ON CONFLICT ... DO UPDATE` (an upsert) work — re-marking an already-tracked segment updates in place rather than erroring or duplicating.

### `get_connection()` context manager

Opens a new SQLite connection per call rather than holding one open for the app's lifetime. This is simpler and avoids cross-thread SQLite connection issues (SQLite connections aren't safe to share across threads by default, and Streamlit's execution model can involve multiple threads) — at the cost of connection-open overhead on every call, which is negligible for SQLite's file-based access pattern.

`conn.commit()` happens automatically on successful exit of the `with` block; if the block raises, the exception propagates *before* commit, so a failed operation doesn't get accidentally persisted. There's no explicit rollback call — SQLite's default behavior on a connection close without commit is an implicit rollback of the uncommitted transaction.

### `VIDEO_STATUSES` and validation

`update_video_status()` validates against a fixed tuple of allowed statuses and raises `ValueError` on anything else. This is a small but deliberate guard: without it, a typo like `"transcribeing"` would silently succeed and corrupt the state machine in a way that's hard to debug later, since nothing downstream would recognize that status and the video would appear stuck.

### Status flow

```
pending → downloading → segmenting → transcribing → transcribed → embedding → completed
                                                                              ↘ failed (from any stage)
```

Note `segmenting` is defined in `VIDEO_STATUSES` but not currently set anywhere in the code — segmentation is fast enough (post-performance-fix) that it wasn't judged worth its own tracked status. This is a minor inconsistency between the schema and actual usage, harmless but worth knowing if you're debugging by reading raw DB state.

---

## 6. RAG — Document Chunking (`app/rag/documents.py`)

### `load_transcript` → `transcript_to_text_with_timestamps` → `build_documents`

Three separate functions instead of one monolithic one, specifically so each step is independently unit-testable (see `tests/test_documents.py`) without needing to mock file I/O for logic that has nothing to do with file I/O.

### The timestamp-mapping problem

`RecursiveCharacterTextSplitter.split_text()` returns plain strings — it does **not** tell you where in the original text each chunk came from. But every chunk needs a `start_timestamp` for citation purposes. `build_documents()` solves this by:

1. Joining the transcript into one flat string (`full_text`).
2. For each chunk returned by the splitter, calling `full_text.find(chunk_text, cursor)` to locate it — searching *from* a moving `cursor` position rather than from the start each time, both for efficiency and to correctly handle cases where the same short phrase might appear more than once in the transcript (an unqualified `.find()` from position 0 every time could match an earlier, wrong occurrence).
3. Mapping that character offset to a timestamp via `_find_timestamp_for_offset`, which walks the original transcript segments accumulating character length until it finds which original segment the offset falls in, returning *that* segment's start time.

**Known imprecision:** a chunk's `start_timestamp` is where its *first character* falls in the original transcript — but because of `CHUNK_OVERLAP`, a chunk's actual content might start slightly before that if the splitter included overlapping text from the previous chunk. In practice this is a sub-second-to-few-second imprecision, acceptable for "jump to roughly this part of the video" but not frame-accurate.

**Fallback for `find()` returning -1:** if overlap logic ever causes the exact chunk substring to not be found from the current cursor forward (rare, but possible depending on splitter internals), the code falls back to using `cursor` itself as the offset rather than crashing. This trades perfect accuracy for robustness — a slightly-wrong timestamp is preferable to the whole pipeline failing on one video.

---

## 7. RAG — Embeddings (`app/rag/embeddings.py`)

Minimal by design — a single `get_embeddings()` function wrapping `HuggingFaceEmbeddings` in the same lazy-singleton pattern as the Whisper model (§4). `all-MiniLM-L6-v2` was chosen specifically because it runs entirely on CPU at reasonable speed and requires no API key or network call at inference time, keeping the embedding step free and fast — consistent with the project's zero-infra-cost philosophy from the README.

This module has no error handling of its own — if the model fails to download/load (e.g., no internet on first run, since HuggingFace models are fetched from the Hub on first use and cached locally after), the exception propagates unmodified. This is intentional: there's no meaningful fallback if the embedding model itself can't load, so wrapping it in a custom exception type would add complexity without adding value.

---

## 8. RAG — Vector Store (`app/rag/vectorstore.py`)

### Per-video index isolation

Every video gets its own FAISS index directory (`data/vectorstores/<video_id>/`) rather than one shared index for all videos. This means:
- Re-processing or deleting one video's data never risks corrupting another's.
- `build_vectorstore()` for video A can run without knowing anything about video B's existing index.
- The cost: multi-video queries require loading and merging multiple indexes at query time (`load_multiple_vectorstores`), rather than a single index supporting a `WHERE video_id IN (...)`-style filter.

### `load_vectorstore()` and `allow_dangerous_deserialization=True`

FAISS's `load_local()` uses Python's `pickle` internally, which is unsafe to run on untrusted data (arbitrary code execution risk) — hence LangChain requires this flag to be explicitly set to acknowledge that risk. It's safe *here* specifically because every index loaded was also generated by this same codebase (`build_vectorstore()`), never downloaded or received from an external source. This assumption would break if the project ever added a feature to import someone else's pre-built index.

### `load_multiple_vectorstores()`

Loads the first video's index, then calls `.merge_from()` for each subsequent one. FAISS's merge is an in-memory operation — every index being merged is fully loaded into RAM simultaneously. Fine for a handful of videos; the README's limitations section already flags this as the first thing to reconsider if the project scaled to dozens of long videos (lazy loading, or restructuring to a single shared index with metadata filtering instead of per-video files).

---

## 9. RAG — Retriever (`app/rag/retriever.py`)

Deliberately the thinnest module in the codebase — `get_retriever()` is a one-line wrapper around `vectorstore.as_retriever()`. This exists as its own file (rather than inlining the call wherever a retriever is needed) purely so retrieval *strategy* can change in one place later — e.g., switching from plain similarity search to MMR (maximal marginal relevance, which reduces redundancy among retrieved chunks) or adding metadata filters — without touching every call site.

---

## 10. RAG — Single-Turn Chain (`app/rag/chains.py`)

### The LCEL (LangChain Expression Language) pipe chain

```python
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

Reading this right-to-left-then-left-to-right: the input (a question string) is fanned into a dict with two keys — `context` (the question piped through the retriever, then through `_format_docs` to turn Documents into one string) and `question` (the original input, passed through unchanged via `RunnablePassthrough`). That dict fills the `{context}`/`{question}` placeholders in `RAG_PROMPT`, which becomes the LLM input, whose output is parsed to a plain string.

### `_format_docs()` — why timestamps are embedded in the context string

Each retrieved chunk is prefixed with `[Ns]` before being joined into the context block sent to Gemini. This isn't just for the citation feature (which uses the *metadata*, not this string) — it also lets the LLM itself reference approximate timing in its answer if relevant, and gives it a signal about chunk ordering/spacing that plain concatenated text would lose.

### The double-retrieval problem, explained precisely

`ask()` calls `chain.invoke(question)` (which internally retrieves once) and *then* separately calls `retriever.invoke(question)` again to get the source documents for the return value. This happens because the LCEL chain above discards the intermediate retriever output once it's been formatted into the context string — nothing in the chain's final output (`StrOutputParser()` → plain string) carries the source Documents forward. The straightforward fix noted in the README (`RunnableParallel` with explicit branches that preserve both the formatted context *and* the raw retrieved docs) wasn't implemented here to keep the single-turn chain's code simple, given the conversational chain (§11) doesn't have this problem at all — `create_retrieval_chain` natively returns both `answer` and `context` in one call.

### Why `temperature` isn't set

Originally `temperature=0` (for deterministic, less "creative" answers — standard practice for RAG). Removed after switching to Gemini 3.6, which deprecated `temperature`/`top_p`/`top_k` — currently ignored by the API but flagged to start erroring in future model generations, so proactively removing it avoids a future breaking change rather than waiting for it to happen.

---

## 11. RAG — Conversational Chain (`app/rag/conversation.py`)

### Two-stage design: contextualize, then answer

`create_history_aware_retriever` doesn't retrieve based on the raw latest question — it first runs the question *plus* chat history through `CONTEXTUALIZE_PROMPT`, asking the LLM to rewrite it as a standalone question (e.g., "what about the second one?" + history about "chunking strategies" → "what is the second chunking strategy discussed?"). *That* rewritten question is what actually gets embedded and searched against FAISS. Without this step, follow-up questions with pronouns/implicit references would retrieve poorly, since the embedding model has no concept of conversational context — it only sees the literal question text.

The answer stage (`create_stuff_documents_chain` + `ANSWER_PROMPT`) then "stuffs" all retrieved documents directly into the prompt alongside the *original* chat history (not the rewritten question) — the LLM sees the real conversation flow when composing its final answer, even though retrieval used a reformulated version internally.

### `ConversationSession` as a stateful wrapper

Unlike every other module in this project (which are stateless functions), `ConversationSession` is a class holding mutable state (`self.chat_history`) — necessary because conversational memory is inherently per-conversation state, and the Streamlit UI creates one instance per video (`st.session_state.sessions[video_id]`) to keep each video's chat independent.

`self.chat_history` is a flat list of alternating `HumanMessage`/`AIMessage` objects, appended to after every `ask()` call — this is the unbounded-growth limitation flagged in the README. There's no summarization or windowing; every past turn is resent to the LLM on every subsequent turn, which is simple but doesn't scale to very long conversations.

---

## 12. Pipeline Orchestrator (`app/pipeline.py`)

### `process_video()` as the single entry point

Before this module existed, using the system meant manually calling `get_video_metadata` → `download_audio` → `segment_audio` → `transcribe_video` → `build_documents` → `build_vectorstore` → `get_retriever` in sequence (as the original `app.py` smoke-test script did). `process_video()` consolidates that into one call that also adds resume-awareness at two points that the individual functions don't handle themselves:

1. **Audio download skip** (`_get_or_download_audio`) — checks whether `full_audio.<ext>` already exists on disk before calling `download_audio()` again. Note this check is purely presence-based — it doesn't verify the existing file is complete/uncorrupted, so a partial download from an interrupted run *would* be treated as valid and fed into segmentation, likely causing a downstream failure rather than a silent bad result.
2. **Vectorstore skip** (`_vectorstore_exists`) — checks whether a FAISS index folder already exists for the video before re-embedding. As noted in the README, this is folder-presence, not integrity-checked.

Everything *between* those two checks (segmentation, transcription) relies on the resumability already built into `transcribe_video()` itself (§4) — `process_video()` doesn't duplicate that logic, it just calls through.

### Exception handling — status transitions on failure

The `try`/`except` wraps the entire pipeline body. On any exception, `update_video_status(video_id, "failed", error_message=str(e))` is called *before* re-raising — so the DB always reflects the failure with a message, but the caller (UI or script) still sees the actual exception and can react to it (the Streamlit UI catches `VideoUnavailableError` specifically for a friendlier message, and a generic `Exception` as a fallback).

**No dedicated retry logic** — a video with `status="failed"` isn't treated specially. Calling `process_video()` on it again just re-runs the pipeline from the top, relying on the existing per-stage caches (audio file presence, segment transcription cache, vectorstore presence) to skip whatever had already completed before the failure. This was a deliberate simplification: implementing precise "resume from the exact point of failure" would require tracking *which* stage failed and dispatching accordingly, whereas the current approach gets most of the same benefit for free by leaning on caching that already exists for other reasons.

---

## 13. Streamlit UI (`streamlit_app.py`)

### Session state model

Three keys in `st.session_state`:
- `sessions`: `dict[video_id, ConversationSession]` — one conversational chain per processed video, persisted across Streamlit reruns (Streamlit reruns the whole script top-to-bottom on every interaction, so anything not in `session_state` would be lost on the next click).
- `active_video_id`: which video's chat is currently displayed.
- `video_titles`: intended to map `video_id → display title` for the sidebar video list — currently a **known placeholder** (stores `video_id` as its own "title") because `process_video()` returns only a retriever, not the metadata dict that has the real title. Fixing this requires changing `process_video()`'s return signature to `(retriever, metadata)`, which hasn't been done yet to avoid a breaking change to the function's existing callers (the `app.py` smoke-test script, and this file) without the user's confirmation first.

### Why processing happens inside `st.spinner()` synchronously

Streamlit's execution model is single-threaded per session by default — there's no built-in async/background job support without extra infrastructure (Streamlit's newer `st.status`/fragments features could improve the UX but don't change the fundamental synchronous nature). This means the UI **blocks** for the full multi-minute processing duration on a video's first run, with only the spinner as feedback. Acceptable for a portfolio demo; a production version would need a background task queue and polling/websocket-based progress updates.

### Source rendering and timestamp links

`f"https://youtube.com/watch?v={src['video_id']}&t={int(ts)}s"` — the `&t=Ns` URL parameter is YouTube's standard way to deep-link to a timestamp; clicking it opens the video already seeked to that point. This is why `start_timestamp` being carried through every layer (transcription → documents → retrieval → chain output) matters — it's not just internal bookkeeping, it's what makes the UI's core "click to jump to the source" feature possible at all.

---

## 14. Cross-Cutting Concerns

### Lazy-singleton pattern (Whisper, Embeddings)

Both `_get_model()` in `transcription.py` and `get_embeddings()` in `embeddings.py` use the same pattern: a module-level `None` variable, populated on first call, reused thereafter. This is deliberately *not* thread-safe (no lock around the check-and-set) — acceptable because Streamlit's default execution model doesn't call these concurrently within a single user session in a way that would race, but worth knowing if this code were ever adapted to a genuinely concurrent server context (e.g., handling multiple simultaneous users' first requests at once).

### Custom exception types as the error-handling backbone

`InvalidYouTubeURLError` and `VideoUnavailableError` (both in `youtube.py`) are the only custom exception types in the project. Every other module either lets exceptions propagate unmodified (embeddings, vectorstore's `FileNotFoundError`/`ValueError`) or catches broadly at the orchestration layer (`pipeline.py`'s bare `except Exception`). This is a fairly minimal error-typing strategy — sufficient for the two failure modes that actually need distinct handling in the UI (bad URL vs. unavailable video), but wouldn't scale cleanly if more distinct error-handling paths were needed later (e.g., distinguishing a transcription failure from an embedding failure at the UI level).

### Testing philosophy

Every test file mocks the actual external dependency (yt-dlp's network calls, `WhisperModel`, `ChatGoogleGenerativeAI`) rather than hitting real services — **except** `test_vectorstore.py`, which uses the real `all-MiniLM-L6-v2` embedding model on short test strings. This one exception was a deliberate trade-off: mocking FAISS's actual similarity search meaningfully would require either a fake embedding function (which risks not testing what real embeddings would actually retrieve) or significant mock complexity, whereas the real model is small/fast/free enough that using it directly in tests was simpler and more genuinely validating, at the cost of that one test file being slower and requiring the model to be downloaded on first CI run.

---

## Related documents

- [`README.md`](README.md) — setup instructions, high-level architecture, advantages/limitations for a first-time reader or grader
