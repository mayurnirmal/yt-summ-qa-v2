"""
app/ingestion/youtube.py

Handles everything related to a raw YouTube URL before any audio processing
starts: validating/parsing the URL, fetching metadata, and downloading audio.

Why this exists:
    This is the entry point of the whole pipeline (see spec section 9,
    "Component Responsibilities"). Nothing downstream — segmentation,
    transcription, LangChain documents — can run until a video_id is
    resolved and audio is on disk, so this module owns that boundary.

What it does:
    - extract_video_id(url): parses a video ID out of common YouTube URL
      formats (watch?v=, youtu.be/, with extra query params)
    - get_video_metadata(url): fetches title/duration/channel/chapters
      WITHOUT downloading — chapters are grabbed now (cheap, same call)
      even though chapter-aware retrieval is a later milestone
    - download_audio(url): downloads audio-only and converts to mp3 under
      data/audio/<video_id>/, ready for audio.py to segment

Error handling:
    Raises typed exceptions (InvalidYouTubeURLError, VideoUnavailableError)
    instead of letting yt-dlp's raw errors/stack traces bubble up — per
    spec section 20, raw stack traces should never reach the user, so
    main.py can catch these and show a friendly message.
"""
import re
import logging
from pathlib import Path
import yt_dlp
from urllib.parse import urlparse, parse_qs

from app.config import AUDIO_DIR

logger = logging.getLogger(__name__)


class InvalidYouTubeURLError(Exception):
    """Raised when a video ID can't be parsed from the given URL."""
    pass


class VideoUnavailableError(Exception):
    """Raised when a video is private, deleted, geo-blocked, or the
    download otherwise fails."""
    pass


def extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video ID from common URL formats."""

    parsed_url = urlparse(url)
    hostname = (parsed_url.hostname or "").lower()

    # Check that the URL belongs to YouTube
    valid_hosts = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }

    if hostname not in valid_hosts:
        raise InvalidYouTubeURLError(
            f"Not a valid YouTube URL: {url}"
        )

    # Handle:
    # https://www.youtube.com/watch?v=VIDEO_ID
    # https://www.youtube.com/watch?v=VIDEO_ID&t=30s
    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v", [None])[0]

    # Handle:
    # https://youtu.be/VIDEO_ID
    elif hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]

    else:
        video_id = None

    # Validate the extracted video ID
    if video_id and re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id):
        return video_id

    raise InvalidYouTubeURLError(
        f"Could not extract video ID from URL: {url}"
    )


def get_video_metadata(url: str) -> dict:
    """Fetch metadata without downloading. Raises VideoUnavailableError on private/deleted/geo-blocked videos."""
    ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Metadata fetch failed for {url}: {e}")
        raise VideoUnavailableError(f"Video unavailable or private: {url}") from e

    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "duration": info.get("duration"),  # seconds
        "channel": info.get("uploader"),
        "chapters": info.get("chapters") or [],  # used later for chapter-aware retrieval
    }


def download_audio(url: str, video_id: str | None = None) -> str:
    """Download audio-only stream to its native format (no re-encoding) under data/audio/<video_id>/."""
    video_id = video_id or extract_video_id(url)
    out_dir = Path(AUDIO_DIR) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "full_audio.%(ext)s"),
        "concurrent_fragment_downloads": 4,  # speeds up fragmented/DASH downloads (longer videos)
        # no postprocessors — keep native container, skip the mp3 transcode
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Audio download failed for {video_id}: {e}")
        raise VideoUnavailableError(f"Could not download audio for: {url}") from e

    downloaded_ext = info.get("ext", "m4a")
    final_path = out_dir / f"full_audio.{downloaded_ext}"
    if not final_path.exists():
        raise VideoUnavailableError(f"Download reported success but file missing: {final_path}")

    logger.info(f"Downloaded audio for {video_id} -> {final_path}")
    return str(final_path)