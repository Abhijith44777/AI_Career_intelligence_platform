"""
pipeline.py
------------
Milestone 2 - Task 6: Processing API & Service Integration

Wires Milestone 1 and Milestone 2 together into the single end-to-end
flow from the spec:

    Upload Meeting
         |
      Whisper
         |
     Transcript
         |
    LLM Processing
         |
      Summary
         |
   Action Extraction
         |
  Participant Mapping
         |
       Database

Every stage's failure is caught and reported with which stage failed,
rather than one bare traceback - both the Streamlit UI (app.py) and the
API (api.py) call this same function so the two interfaces can never
drift apart in behavior.
"""

from dataclasses import dataclass, field
import logging
import os
import re
import tempfile

# Milestone 1
from utils import (
    validate_file,
    process_audio,
    run_whisper,
    validate_transcript,
    save_transcript,
    get_audio_duration,
)

# Milestone 2
from llm_service import LLMService, LLMServiceError
from summarization import summarize_meeting, MeetingSummary
from action_items import extract_and_validate_action_items
from participants import map_participants_and_responsibilities
from database import Database

# Milestone 3
from embedding_service import EmbeddingService
from vector_store import VectorStore
from knowledge_repository import index_meeting
from config import settings

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Final path component only, with control characters removed
    ('../../etc/passwd' -> 'passwd'). Falls back to 'recording' if nothing is left."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    return name[:200] or "recording"


class PipelineError(Exception):
    """Raised with a `.stage` attribute identifying which pipeline stage failed,
    so callers can show a precise error instead of a generic failure."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


@dataclass
class PipelineResult:
    meeting_id: int
    transcript: str
    language: str
    audio_duration_seconds: float
    summary: MeetingSummary
    action_items: list = field(default_factory=list)   # list of {"action_item":.., "participant":..}
    participants: list = field(default_factory=list)


def process_meeting_recording(
    raw_file_path: str,
    original_filename: str,
    file_size_bytes: int,
    db: Database = None,
    llm_service: LLMService = None,
    whisper_model_size: str = "base",
    embedding_service: EmbeddingService = None,
    vector_store: VectorStore = None,
    index_for_search: bool = True,
    user_id: int = None,
    source: str = "upload",
) -> PipelineResult:
    """
    Full Milestone 1 + Milestone 2 pipeline, matching the Task 6 flow
    diagram exactly. Each stage below is labeled with the stage name it
    corresponds to in that diagram.
    """
    db = db or Database()
    llm_service = llm_service or LLMService()
    # Never trust a caller-supplied name for anything path-like (uploads,
    # Zoom/Meet recording titles): keep only the final path component.
    original_filename = safe_filename(original_filename)

    # --- Upload Meeting: validate the file (Milestone 1, Task 2) -----------
    validation = validate_file(original_filename, file_size_bytes)
    if not validation["valid"]:
        raise PipelineError("Upload Validation", validation["message"])

    # --- Whisper: process audio + transcribe (Milestone 1, Task 1) ---------
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_audio_path = process_audio(raw_file_path, tmp_dir)
            duration = get_audio_duration(processed_audio_path)
            whisper_result = run_whisper(processed_audio_path, model_size=whisper_model_size)
    except RuntimeError as e:
        raise PipelineError("Whisper", str(e))

    transcript_text = whisper_result["text"]

    # --- Transcript: validate before spending LLM calls on bad input -------
    transcript_check = validate_transcript(transcript_text)
    if not transcript_check["valid"]:
        raise PipelineError("Transcript Validation", transcript_check["message"])

    if settings.save_transcript_files:
        # One sub-folder per owner so transcript files are never mixed between users.
        out_dir = os.path.join(settings.transcripts_dir, str(user_id) if user_id is not None else "shared")
        try:
            save_transcript(whisper_result, original_filename, output_dir=out_dir)
        except OSError as e:  # a full/read-only disk must not lose the meeting - the DB copy is authoritative
            logger.warning("Could not write transcript file: %s", e)

    # Persist the meeting/transcript row now so every later stage has a
    # meeting_id to attach to, even if a later stage fails.
    meeting_id = db.create_meeting(
        filename=original_filename,
        transcript=transcript_text,
        language=whisper_result.get("language"),
        audio_duration_seconds=duration,
        user_id=user_id,
        source=source,
        status="processing",
    )

    # --- LLM Processing -> Summary: structured extraction (Milestone 2, Task 1+2) ---
    try:
        meeting_summary = summarize_meeting(transcript_text, llm_service)
    except LLMServiceError as e:
        db.set_meeting_status(meeting_id, "failed", str(e)[:500])
        raise PipelineError("LLM Processing", str(e))

    db.save_summary(
        meeting_id=meeting_id,
        summary_text=meeting_summary.summary,
        key_points=meeting_summary.key_points,
        decisions=meeting_summary.decisions,
        priorities=meeting_summary.priorities,
    )

    # --- Action Extraction (Milestone 2, Task 3) ----------------------------
    validated_action_items = extract_and_validate_action_items(meeting_summary.action_items)

    # --- Participant Mapping (Milestone 2, Task 4) ---------------------------
    mapping = map_participants_and_responsibilities(meeting_summary.participants, validated_action_items)

    # --- Database: persist participants + linked action items (Task 5) -------
    for entry in mapping["action_items"]:
        item = entry["action_item"]
        participant = entry["participant"]
        participant_id = db.get_or_create_participant(participant.name)
        db.save_action_item(
            meeting_id=meeting_id,
            participant_id=participant_id,
            task=item.task,
            deadline=item.deadline,
            priority=item.priority,
            status=item.status,
        )

    # Also persist participants who were mentioned but have no action item,
    # so they still show up as meeting attendees.
    for participant in mapping["participants"]:
        db.link_participant_to_meeting(meeting_id, db.get_or_create_participant(participant.name))
    for entry in mapping["action_items"]:
        db.link_participant_to_meeting(meeting_id, db.get_or_create_participant(entry["participant"].name))

    db.set_meeting_status(meeting_id, "completed")

    # --- Knowledge Repository indexing (Milestone 3, Tasks 1-3) ------------
    # Embeddings are generated dynamically as each new meeting is processed,
    # per Task 2's requirement. A failure here is LOGGED (not silently
    # swallowed) but never fails the overall pipeline - a meeting that
    # couldn't be embedded is still fully saved and usable; it just won't
    # show up in semantic search until POST /embeddings/generate (or
    # /embeddings/sync) is run for it.
    if index_for_search:
        try:
            embedder = embedding_service or EmbeddingService()
            store = vector_store or VectorStore()
            chunks_indexed = index_meeting(db, meeting_id, embedder, store)
            logger.info("Meeting %s indexed for search: %d chunks", meeting_id, chunks_indexed)
        except Exception as e:  # noqa: BLE001 - indexing is best-effort, never blocks the core pipeline
            logger.error("Indexing failed for meeting_id=%s: %s", meeting_id, e)

    return PipelineResult(
        meeting_id=meeting_id,
        transcript=transcript_text,
        language=whisper_result.get("language"),
        audio_duration_seconds=duration,
        summary=meeting_summary,
        action_items=mapping["action_items"],
        participants=mapping["participants"],
    )
