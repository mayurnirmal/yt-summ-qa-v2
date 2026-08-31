"""
Splits long audio files into fixed-length chunks for transcription.
Whisper struggles with very long audio in one pass — this keeps segments manageable and resumable.
"""
# app/ingestion/audio.py — replace segment_audio()
import subprocess
from pathlib import Path
from app.config import AUDIO_DIR

SEGMENT_LENGTH_SEC = 10 * 60
SEGMENT_LENGTH_MS = 10 * 60 * 1000

def segment_audio(audio_path: str, video_id: str) -> list[str]:
    """Split audio into 10-min segments via ffmpeg stream-copy (no re-encoding)."""
    src = Path(audio_path)
    ext = src.suffix.lstrip(".")
    segments_dir = Path(AUDIO_DIR) / video_id / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    duration = _get_duration_sec(audio_path)
    segment_paths = []

    for i, start in enumerate(range(0, int(duration) + 1, SEGMENT_LENGTH_SEC)):
        out_path = segments_dir / f"segment_{i:03d}.{ext}"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start), "-i", str(src),
            "-t", str(SEGMENT_LENGTH_SEC), "-c", "copy", str(out_path),
        ], check=True, capture_output=True)
        segment_paths.append(str(out_path))

    return segment_paths


def _get_duration_sec(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())