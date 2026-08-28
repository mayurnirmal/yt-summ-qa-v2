"""
Splits long audio files into fixed-length chunks for transcription.
Whisper struggles with very long audio in one pass — this keeps segments manageable and resumable.
"""
import logging
from pathlib import Path
from pydub import AudioSegment

from app.config import AUDIO_DIR

logger = logging.getLogger(__name__)

SEGMENT_LENGTH_MS = 10 * 60 * 1000  # 10 minutes


def segment_audio(audio_path: str, video_id: str) -> list[str]:
    """Split audio into 10-min chunks under data/audio/<video_id>/segments/. Returns ordered list of paths."""
    audio = AudioSegment.from_file(audio_path)
    segments_dir = Path(AUDIO_DIR) / video_id / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths = []
    total_len = len(audio)

    for i, start in enumerate(range(0, total_len, SEGMENT_LENGTH_MS)):
        end = min(start + SEGMENT_LENGTH_MS, total_len)
        chunk = audio[start:end]
        chunk_path = segments_dir / f"segment_{i:03d}.mp3"
        chunk.export(chunk_path, format="mp3")
        segment_paths.append(str(chunk_path))

    logger.info(f"Split {video_id} into {len(segment_paths)} segments")
    return segment_paths