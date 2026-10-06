"""
knowledge_repository.py
-------------------------
Milestone 3 - Task 1: Meeting Knowledge Repository

Uses the EXISTING database and API (database.py / api.py from Milestone 2 -
unchanged) to organize historical meeting information for AI search:
    Meeting metadata
    Transcript
    Summary
    Decisions
    Action items
    Participants
    Deadlines

and verifies that existing database records can be retrieved correctly and
linked to the corresponding meeting (`verify_meeting_retrieval`).

This module is also where Task 2 (Embedding Generation) and Task 3 (Vector
Database Integration) get wired together for a single meeting via
`index_meeting`: it builds the meeting's embeddable chunks, embeds them, and
upserts them into the vector store, tagged with the meeting_id so every
vector can always be traced back to its meeting.
"""

import logging
from dataclasses import dataclass, field

from database import Database
from embedding_service import EmbeddingService, EmbeddingServiceError, build_meeting_chunks
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class MeetingKnowledge:
    """Task 1's required shape: metadata + transcript + summary + decisions +
    action items + participants + deadlines, for ONE meeting."""
    meeting_id: int
    filename: str
    created_at: str
    language: str
    transcript: str
    summary_text: str
    key_points: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    action_items: list = field(default_factory=list)
    participants: list = field(default_factory=list)   # distinct participant names
    deadlines: list = field(default_factory=list)       # distinct deadlines, derived from action items


def get_meeting_knowledge(db: Database, meeting_id: int) -> MeetingKnowledge:
    """
    Task 1: organize ONE meeting's full record for AI search. Returns None
    if the meeting doesn't exist (mirrors Database.get_meeting_full).
    """
    full = db.get_meeting_full(meeting_id)
    if not full:
        return None

    meeting = full["meeting"]
    summary = full["summary"] or {}
    action_items = full["action_items"]

    participants = []
    seen_participants = set()
    for item in action_items:
        name = item.get("participant_name")
        if name and name not in seen_participants:
            seen_participants.add(name)
            participants.append(name)

    deadlines = []
    seen_deadlines = set()
    for item in action_items:
        deadline = (item.get("deadline") or "").strip()
        if deadline and deadline not in seen_deadlines:
            seen_deadlines.add(deadline)
            deadlines.append(deadline)

    return MeetingKnowledge(
        meeting_id=meeting["id"],
        filename=meeting["filename"],
        created_at=meeting["created_at"],
        language=meeting.get("language"),
        transcript=meeting["transcript"],
        summary_text=summary.get("summary_text", ""),
        key_points=summary.get("key_points", []),
        decisions=summary.get("decisions", []),
        priorities=summary.get("priorities", []),
        action_items=action_items,
        participants=participants,
        deadlines=deadlines,
    )


def verify_meeting_retrieval(db: Database, meeting_id: int) -> dict:
    """
    Task 1 verification - explicitly checks every item the spec calls out:
      1. Existing meeting record can be retrieved
      2. Transcript is linked to the correct meeting
      3. Summary is linked to the correct meeting
      4. Decisions are linked to the correct meeting
      5. Action items are linked to the correct meeting
      6. Participants are linked to the correct meeting
      7. Deadlines are linked to the correct meeting
      8. No meeting data is accidentally associated with another meeting
         (checked by cross-referencing every action item's own meeting_id)

    Returns a report dict rather than True/False so a failure is actionable.
    """
    knowledge = get_meeting_knowledge(db, meeting_id)
    if knowledge is None:
        return {"ok": False, "reason": f"No meeting found with id {meeting_id}"}

    # Every check below reads the database independently, so each one can
    # genuinely fail (the earlier version hardcoded several to True).
    summary_row = db.get_summary(meeting_id)
    attendee_ids = {p["id"] for p in db.get_participants_for_meeting(meeting_id)}
    action_participant_ids = {i["participant_id"] for i in knowledge.action_items if i.get("participant_id")}
    action_deadlines = {(i.get("deadline") or "").strip() for i in knowledge.action_items} - {""}
    checks = {
        "meeting_record_retrieved": knowledge.meeting_id == meeting_id,
        "transcript_linked": bool(knowledge.transcript) and db.get_transcript(meeting_id) == knowledge.transcript,
        # A meeting may legitimately have no summary yet; if one exists it must point at THIS meeting.
        "summary_linked": summary_row is None or summary_row["meeting_id"] == meeting_id,
        "decisions_linked": summary_row is None or list(summary_row["decisions"]) == list(knowledge.decisions),
        "action_items_linked_to_meeting": all(
            item.get("meeting_id") == meeting_id for item in knowledge.action_items),
        "participants_linked": action_participant_ids <= attendee_ids,
        "deadlines_linked": action_deadlines == set(db.get_deadlines_for_meeting(meeting_id)),
        "no_cross_meeting_contamination": all(
            item.get("meeting_id") == meeting_id for item in db.get_action_items(meeting_id)),
    }
    return {"ok": all(checks.values()), "checks": checks, "meeting_id": meeting_id}


# ---------------------------------------------------------------------------
# Indexing - ties Task 1 (repository) + Task 2 (embeddings) + Task 3 (vector
# store) together for one meeting. Called by pipeline.py right after a new
# meeting is persisted, so embeddings are generated dynamically as new
# meetings are processed (Task 2's requirement), and by the reindexing
# helper below to backfill Milestone 2 meetings that predate Milestone 3.
# ---------------------------------------------------------------------------

def _vector_id(meeting_id: int, content_type: str, index: int) -> str:
    # Matches the spec's example format exactly: "123_transcript_001"
    return f"{meeting_id}_{content_type}_{index:03d}"


def index_meeting(db: Database, meeting_id: int,
                   embedding_service: EmbeddingService = None,
                   vector_store: VectorStore = None) -> int:
    """
    Embeds and upserts every chunk (transcript sections, summary, decisions,
    action items) for one meeting into the vector store. Deletes any
    previously-stored vectors for this meeting first, so re-indexing after
    an update never leaves stale/duplicate vectors behind - this is what
    makes "Update embeddings" (Task 3) safe to call repeatedly, and is the
    data-synchronization mechanism: any change to a meeting's transcript,
    summary, decisions or action items is picked up by simply re-running
    this function for that meeting_id.

    Returns the number of chunks indexed.
    """
    embedding_service = embedding_service or EmbeddingService()
    vector_store = vector_store or VectorStore()

    full = db.get_meeting_full(meeting_id)
    if not full:
        logger.warning("index_meeting called for nonexistent meeting_id=%s", meeting_id)
        return 0

    chunks = build_meeting_chunks(meeting_id, full)

    # Embed every chunk FIRST (one batched request for remote providers). If
    # that fails we have not touched the store, so the meeting's existing
    # vectors survive; and we refuse to store a mix of vector sizes.
    vectors = embedding_service.embed_texts([c.text for c in chunks])
    if len({len(v) for v in vectors}) > 1:
        raise EmbeddingServiceError(
            f"Meeting {meeting_id}: embedding returned mixed vector sizes; not indexing it.")

    owner_id = full["meeting"].get("user_id")
    records = []
    for chunk, vector in zip(chunks, vectors):
        vector_id = _vector_id(meeting_id, chunk.content_type, chunk.index)
        metadata = {
            "meeting_id": meeting_id,
            "content_type": chunk.content_type,
            "source_id": vector_id,
            "index": chunk.index,
            "filename": full["meeting"]["filename"],
            "created_at": full["meeting"]["created_at"],
        }
        if owner_id is not None:  # Chroma rejects None metadata values
            metadata["user_id"] = owner_id  # lets every search be filtered to the owner's vectors
        records.append({
            "id": vector_id,
            "vector": vector,
            "document": chunk.text,
            "metadata": metadata,
        })
    vector_store.delete_by_meeting(meeting_id)
    vector_store.upsert_many(records)
    logger.info("Indexed meeting_id=%s: %d chunks embedded and upserted", meeting_id, len(records))
    return len(records)


def reindex_all_meetings(db: Database, embedding_service: EmbeddingService = None,
                          vector_store: VectorStore = None, user_id: int = None) -> dict:
    """
    Backfills embeddings for every meeting already in the database - needed
    once, the first time Milestone 3 is deployed on top of an existing
    Milestone 2 database whose meetings were never embedded. Also serves as
    the general-purpose sync endpoint's implementation (POST /embeddings/sync).
    """
    embedding_service = embedding_service or EmbeddingService()
    vector_store = vector_store or VectorStore()

    results = {}
    for meeting in db.list_meetings(user_id=user_id):  # user_id=None -> every meeting (admin/CLI use only)
        results[meeting["id"]] = index_meeting(db, meeting["id"], embedding_service, vector_store)
    logger.info("Reindexed %d meeting(s)", len(results))
    return results
