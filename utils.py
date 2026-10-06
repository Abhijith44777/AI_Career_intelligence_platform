"""
utils.py
--------
Core logic for Milestone 1 - Audio Processing & Transcription.

Covers:
    Task 1 - Whisper Transcription workflow
    Task 2 - File Upload Validation
    Task 3 - Transcript Validation & Saving

This module has NO Streamlit code in it on purpose - it is kept UI-agnostic
so it can be:
    (a) imported by app.py (the Streamlit interface, Task 4)
    (b) imported by accuracy_test.py (Task 5)
    (c) unit-tested on its own
"""

import os
import json
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Task 2 - File Upload Validation
# ---------------------------------------------------------------------------

# Formats we officially support. Whisper itself (via ffmpeg) can decode
# almost any container, but we deliberately whitelist so users get a clean
# error instead of a cryptic ffmpeg crash for something like a .zip or .exe.
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Guardrails so a huge file doesn't hang the app / blow up memory.
MAX_FILE_SIZE_MB = 500
MIN_FILE_SIZE_KB = 1  # reject 0-byte / empty uploads


class ValidationError(Exception):
    """Raised when an uploaded file fails validation (Task 2)."""
    pass


def validate_file(filename: str, file_size_bytes: int) -> dict:
    """
    Validate an uploaded file BEFORE we try to process it.

    Returns a dict: {"valid": bool, "message": str, "extension": str}
    Never raises - callers decide what to do with a failed validation,
    which makes it easy to show a proper error message in the UI (Task 2
    requires "Proper error messages are shown").
    """
    if not filename:
        return {"valid": False, "message": "No file was provided.", "extension": None}

    _, ext = os.path.splitext(filename.lower())

    if ext not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return {
            "valid": False,
            "message": (
                f"Unsupported file type '{ext}'. "
                f"Supported formats are: {supported}"
            ),
            "extension": ext,
        }

    size_kb = file_size_bytes / 1024
    size_mb = file_size_bytes / (1024 * 1024)

    if size_kb < MIN_FILE_SIZE_KB:
        return {
            "valid": False,
            "message": "The uploaded file is empty (0 KB). Please upload a valid recording.",
            "extension": ext,
        }

    if size_mb > MAX_FILE_SIZE_MB:
        return {
            "valid": False,
            "message": f"File is too large ({size_mb:.1f} MB). Max allowed size is {MAX_FILE_SIZE_MB} MB.",
            "extension": ext,
        }

    media_type = "audio" if ext in ALLOWED_AUDIO_EXTENSIONS else "video"
    return {
        "valid": True,
        "message": f"Valid {media_type} file ({size_mb:.2f} MB).",
        "extension": ext,
        "media_type": media_type,
    }


# ---------------------------------------------------------------------------
# Task 1 - Process Audio (extract/normalize audio track before Whisper)
# ---------------------------------------------------------------------------

def process_audio(input_path: str, work_dir: str) -> str:
    """
    Prepare a raw upload for Whisper.

    - If the upload is a video file, extract the audio track with ffmpeg.
    - Normalize everything to 16kHz mono WAV, which is what Whisper expects
      internally anyway - doing it explicitly makes failures easier to debug
      and speeds up transcription of already-long recordings.

    Returns the path to the processed .wav file.
    Raises RuntimeError with a readable message if ffmpeg fails.
    """
    os.makedirs(work_dir, exist_ok=True)
    output_path = os.path.join(
        work_dir, os.path.splitext(os.path.basename(input_path))[0] + "_processed.wav"
    )

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", input_path,
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz sample rate (Whisper's native rate)
        "-vn",             # drop video stream if present
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(
            "Audio processing (ffmpeg) failed. This usually means the file "
            "is corrupted or not a real media file.\n"
            f"ffmpeg stderr: {result.stderr[-500:]}"
        )

    return output_path


# ---------------------------------------------------------------------------
# Task 1 - Run Whisper / Generate Transcript
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def load_whisper_model(model_size: str = "base"):
    """
    Load (and cache) a Whisper model.

    model_size options (speed vs accuracy trade-off):
        tiny   - fastest, lowest accuracy
        base   - good default for meeting recordings
        small
        medium
        large  - best accuracy, slowest / most memory

    Cached in-process so repeated transcriptions in the same Streamlit
    session don't reload the model every time (this matters a lot -
    model load is the slowest part after the transcription itself).
    """
    import whisper  # imported lazily so utils.py can be unit-tested without

    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def run_whisper(audio_path: str, model_size: str = "base", language: str = None) -> dict:
    """
    Run Whisper on a processed audio file.

    Returns:
        {
            "text": full transcript string,
            "segments": list of {start, end, text} dicts (for timestamps),
            "language": detected/used language code,
        }
    """
    model = load_whisper_model(model_size)

    transcribe_kwargs = {"verbose": False}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_path, **transcribe_kwargs)

    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "language": result.get("language", "unknown"),
    }


# ---------------------------------------------------------------------------
# Task 3 - Transcript Validation & Saving
# ---------------------------------------------------------------------------

def validate_transcript(transcript_text: str, min_words: int = 1) -> dict:
    """
    Sanity-check a generated transcript.

    Checks:
        - not empty / not just whitespace
        - has at least `min_words` words (catches Whisper returning near-
          nothing on silent/corrupted audio)
        - no obvious repeated-token failure loop (a known Whisper failure
          mode on noisy audio, e.g. "the the the the the the...")
    """
    if transcript_text is None:
        return {"valid": False, "message": "Transcript is None."}

    stripped = transcript_text.strip()
    if not stripped:
        return {"valid": False, "message": "Transcript is empty."}

    words = stripped.split()
    if len(words) < min_words:
        return {"valid": False, "message": f"Transcript too short ({len(words)} word(s))."}

    # Detect degenerate repetition loops.
    if len(words) >= 10:
        most_common = max(set(words), key=words.count)
        ratio = words.count(most_common) / len(words)
        if ratio > 0.5:
            return {
                "valid": False,
                "message": f"Transcript looks degenerate - word '{most_common}' repeats "
                           f"{ratio*100:.0f}% of the transcript. Likely a Whisper failure on this audio.",
            }

    return {"valid": True, "message": f"Transcript looks valid ({len(words)} words)."}


def save_transcript(transcript_result: dict, source_filename: str, output_dir: str) -> str:
    """
    Persist a transcript to disk as JSON (full data incl. timestamps) and
    a plain .txt sidecar (easy to open/read).

    Returns the path to the saved .json file.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(source_filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
    txt_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")

    record = {
        "source_file": source_filename,
        "generated_at": datetime.now().isoformat(),
        "language": transcript_result.get("language"),
        "text": transcript_result.get("text"),
        "segments": transcript_result.get("segments", []),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_result.get("text", ""))

    return json_path


def transcript_matches_recording(transcript_text: str, audio_duration_seconds: float) -> dict:
    """
    Rough heuristic check that transcript LENGTH is plausible for the audio
    DURATION - catches cases where Whisper silently truncated (e.g. only
    transcribed the first 30s of a 20-minute meeting).

    Average spoken English is ~130-170 words/minute. We use a generous
    band (40-220 wpm) to avoid false positives across accents/pauses.
    """
    words = len(transcript_text.split())
    minutes = audio_duration_seconds / 60 if audio_duration_seconds else 0

    if minutes <= 0:
        return {"plausible": False, "message": "Unknown audio duration - cannot check."}

    wpm = words / minutes

    if wpm < 40:
        return {
            "plausible": False,
            "message": f"Transcript seems too short for a {minutes:.1f} min recording "
                       f"(~{wpm:.0f} words/min). Possible truncation.",
        }
    if wpm > 220:
        return {
            "plausible": False,
            "message": f"Transcript seems unusually dense (~{wpm:.0f} words/min). Review for repetition errors.",
        }

    return {"plausible": True, "message": f"Transcript length is plausible (~{wpm:.0f} words/min)."}


def get_audio_duration(audio_path: str) -> float:
    """Get duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0
