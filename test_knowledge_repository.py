"""
test_knowledge_repository.py
------------------------------
Tests for Milestone 3 - Task 1 (Meeting Knowledge Repository), plus the
indexing glue that ties in Task 2 (embeddings) and Task 3 (vector store).

Uses temp SQLite + temp local vector store so tests never touch real data,
same pattern as test_database.py / test_vector_store.py.
"""

import os
import tempfile

from database import Database
from embedding_service import EmbeddingService, EmbeddingConfig, EmbeddingServiceError
from vector_store import VectorStore
from knowledge_repository import (
    get_meeting_knowledge, verify_meeting_retrieval, index_meeting, reindex_all_meetings,
)


def make_temp_db() -> Database:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return Database(db_path=path)


def make_temp_store() -> VectorStore:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return VectorStore(backend="local", path=path)


def local_embedder() -> EmbeddingService:
    return EmbeddingService(EmbeddingConfig(provider="local"))


def seed_meeting(db: Database) -> int:
    meeting_id = db.create_meeting("standup.wav", "Ravi will handle API integration. Priya will test the UI.",
                                    language="en", audio_duration_seconds=90)
    db.save_summary(
        meeting_id=meeting_id,
        summary_text="The team discussed the mobile app launch.",
        key_points=["Launch timeline confirmed"],
        decisions=["Continue with the planned mobile application launch."],
        priorities=["Mobile launch"],
    )
    ravi_id = db.get_or_create_participant("Ravi")
    priya_id = db.get_or_create_participant("Priya")
    db.save_action_item(meeting_id, ravi_id, "Complete API integration", "Friday", "High", "Not Started")
    db.save_action_item(meeting_id, priya_id, "Prepare UI testing report", "", "Medium", "Not Started")
    return meeting_id


# ---------------------------------------------------------------------------
# Task 1 - organizing & retrieving meeting knowledge
# ---------------------------------------------------------------------------

def test_get_meeting_knowledge_returns_all_required_fields():
    db = make_temp_db()
    meeting_id = seed_meeting(db)

    knowledge = get_meeting_knowledge(db, meeting_id)

    assert knowledge.meeting_id == meeting_id
    assert "API integration" in knowledge.transcript
    assert knowledge.summary_text
    assert knowledge.decisions
    assert len(knowledge.action_items) == 2
    assert set(knowledge.participants) == {"Ravi", "Priya"}
    assert "Friday" in knowledge.deadlines


def test_get_meeting_knowledge_returns_none_for_unknown_meeting():
    db = make_temp_db()
    assert get_meeting_knowledge(db, 9999) is None


def test_verify_meeting_retrieval_passes_for_valid_meeting():
    db = make_temp_db()
    meeting_id = seed_meeting(db)
    report = verify_meeting_retrieval(db, meeting_id)
    assert report["ok"] is True
    assert report["checks"]["action_items_linked_to_meeting"] is True


def test_verify_meeting_retrieval_fails_for_unknown_meeting():
    db = make_temp_db()
    report = verify_meeting_retrieval(db, 9999)
    assert report["ok"] is False


def test_no_meeting_data_accidentally_associated_with_another_meeting():
    """Spec's explicit Task 1 check #8: process three meetings and confirm
    each one's action items/participants/deadlines belong ONLY to it."""
    db = make_temp_db()
    meeting_a = seed_meeting(db)   # Ravi/Friday + Priya
    meeting_b = seed_meeting(db)   # a second, independent seed
    meeting_c = seed_meeting(db)

    knowledge_a = get_meeting_knowledge(db, meeting_a)
    knowledge_b = get_meeting_knowledge(db, meeting_b)
    knowledge_c = get_meeting_knowledge(db, meeting_c)

    for item in knowledge_a.action_items:
        assert item["meeting_id"] == meeting_a
    for item in knowledge_b.action_items:
        assert item["meeting_id"] == meeting_b
    for item in knowledge_c.action_items:
        assert item["meeting_id"] == meeting_c

    report_a = verify_meeting_retrieval(db, meeting_a)
    report_b = verify_meeting_retrieval(db, meeting_b)
    assert report_a["checks"]["no_cross_meeting_contamination"] is True
    assert report_b["checks"]["no_cross_meeting_contamination"] is True


# ---------------------------------------------------------------------------
# Indexing (Tasks 2 + 3 wired together for one meeting)
# ---------------------------------------------------------------------------

def test_index_meeting_populates_vector_store():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_meeting(db)

    count = index_meeting(db, meeting_id, embedder, store)

    assert count > 0
    assert store.count() == count


def test_index_meeting_all_vectors_traceable_to_meeting():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_meeting(db)

    index_meeting(db, meeting_id, embedder, store)

    query_vector = embedder.embed_text("API integration")
    results = store.query(query_vector, top_k=10)
    assert all(r.metadata["meeting_id"] == meeting_id for r in results)


def test_reindexing_a_meeting_does_not_duplicate_vectors():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_meeting(db)

    first_count = index_meeting(db, meeting_id, embedder, store)
    second_count = index_meeting(db, meeting_id, embedder, store)

    assert first_count == second_count
    assert store.count() == second_count  # no leftover duplicates from the first pass


def test_reindex_all_meetings_covers_every_meeting():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    m1 = seed_meeting(db)
    m2 = seed_meeting(db)

    results = reindex_all_meetings(db, embedder, store)

    assert set(results.keys()) == {m1, m2}
    assert all(v > 0 for v in results.values())


def test_database_granular_accessors_match_full_knowledge():
    """The new per-field Database helpers (backing the granular API
    endpoints) must agree with get_meeting_knowledge's aggregated view."""
    db = make_temp_db()
    meeting_id = seed_meeting(db)

    assert db.get_transcript(meeting_id) == db.get_meeting(meeting_id)["transcript"]
    assert db.get_decisions(meeting_id) == ["Continue with the planned mobile application launch."]
    participant_names = {p["name"] for p in db.get_participants_for_meeting(meeting_id)}
    assert participant_names == {"Ravi", "Priya"}
    assert db.get_deadlines_for_meeting(meeting_id) == ["Friday"]


class _StubEmbedder:
    """Embedder whose behaviour the test controls."""
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def embed_texts(self, texts, task="RETRIEVAL_DOCUMENT"):
        return self.behaviour(texts)


def test_failed_embedding_leaves_the_meetings_existing_vectors_untouched():
    """Embedding happens BEFORE anything is deleted, so an API failure during
    re-indexing must not wipe the vectors the meeting already has."""
    db = make_temp_db()
    store = make_temp_store()
    meeting_id = seed_meeting(db)
    count = index_meeting(db, meeting_id, local_embedder(), store)

    def boom(texts):
        raise RuntimeError("embedding API exploded")

    raised = False
    try:
        index_meeting(db, meeting_id, _StubEmbedder(boom), store)
    except RuntimeError:
        raised = True
    assert raised
    assert store.count() == count
    assert len(store.get_by_meeting(meeting_id)) == count


def test_mixed_vector_sizes_are_refused_and_the_store_is_left_alone():
    db = make_temp_db()
    store = make_temp_store()
    meeting_id = seed_meeting(db)
    count = index_meeting(db, meeting_id, local_embedder(), store)

    def mixed(texts):
        return [[0.1] * (384 if i % 2 == 0 else 768) for i, _ in enumerate(texts)]

    raised = False
    try:
        index_meeting(db, meeting_id, _StubEmbedder(mixed), store)
    except EmbeddingServiceError:
        raised = True
    assert raised
    assert store.count() == count


def test_indexing_uses_one_batched_embedding_call_per_meeting():
    db = make_temp_db()
    store = make_temp_store()
    meeting_id = seed_meeting(db)
    calls = []
    inner = local_embedder()

    def record(texts):
        calls.append(len(texts))
        return inner.embed_texts(texts)

    count = index_meeting(db, meeting_id, _StubEmbedder(record), store)
    assert calls == [count]


if __name__ == "__main__":
    import sys, inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"✅ PASS - {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL - {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
