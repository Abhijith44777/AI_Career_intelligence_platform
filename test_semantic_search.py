"""
test_semantic_search.py
-------------------------
Tests for Milestone 3 - Task 4 (Semantic Search).

Runs entirely offline (local embedder + local vector store) so latency
measurements reflect this project's own code path, not network variance -
that's the fair way to verify the "within 3 seconds" requirement in a unit
test; production latency with a real embedding API is a separate,
environment-dependent concern.
"""

import os
import tempfile

from database import Database
from embedding_service import EmbeddingService, EmbeddingConfig
from vector_store import VectorStore
from knowledge_repository import index_meeting
from semantic_search import semantic_search, LATENCY_TARGET_SECONDS


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


def seed_indexed_meeting(db, embedder, store, filename, transcript, summary_text, decisions):
    meeting_id = db.create_meeting(filename, transcript, language="en", audio_duration_seconds=60)
    db.save_summary(meeting_id, summary_text, [], decisions, [])
    index_meeting(db, meeting_id, embedder, store)
    return meeting_id


def test_semantic_search_finds_the_relevant_meeting():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()

    db_migration_meeting = seed_indexed_meeting(
        db, embedder, store, "infra-sync.wav",
        "We completed the database migration to the new cluster this week without downtime.",
        "The team completed the database migration.",
        ["Migrate the production database to the new cluster."],
    )
    seed_indexed_meeting(
        db, embedder, store, "hr-standup.wav",
        "We reviewed the holiday schedule and updated the onboarding checklist for new hires.",
        "The team reviewed HR onboarding processes.",
        ["Update the onboarding checklist."],
    )

    response = semantic_search("Which meeting discussed the database migration?",
                                db=db, embedding_service=embedder, vector_store=store, top_k=5)

    assert response.results
    assert response.results[0].meeting_id == db_migration_meeting


def test_semantic_search_returns_within_latency_target():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    seed_indexed_meeting(db, embedder, store, "m.wav", "Some meeting transcript content.",
                          "A summary.", ["A decision."])

    response = semantic_search("summary", db=db, embedding_service=embedder, vector_store=store)

    assert response.elapsed_seconds < LATENCY_TARGET_SECONDS
    assert response.within_latency_target is True


def test_semantic_search_empty_query_returns_no_results():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    response = semantic_search("", db=db, embedding_service=embedder, vector_store=store)
    assert response.results == []


def test_semantic_search_results_are_generated_dynamically_not_hardcoded():
    """Two different queries against the same data return different,
    query-appropriate top results - proving results come from the actual
    vector search rather than a fixed/hardcoded response."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    migration_meeting = seed_indexed_meeting(
        db, embedder, store, "infra.wav", "Database migration completed successfully.",
        "Migration summary.", ["Migrate the database."],
    )
    hiring_meeting = seed_indexed_meeting(
        db, embedder, store, "hr.wav", "We hired two new engineers this quarter.",
        "Hiring summary.", ["Approve two new engineering hires."],
    )

    migration_response = semantic_search("database migration", db=db, embedding_service=embedder, vector_store=store)
    hiring_response = semantic_search("new engineering hires", db=db, embedding_service=embedder, vector_store=store)

    assert migration_response.results[0].meeting_id == migration_meeting
    assert hiring_response.results[0].meeting_id == hiring_meeting


def test_semantic_search_meeting_deleted_after_indexing_is_skipped_not_errored():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_indexed_meeting(db, embedder, store, "m.wav", "Some content here.",
                                       "Summary text.", ["A decision."])

    # Simulate the meeting record disappearing from the DB while its
    # vectors remain (e.g. manual cleanup) - search should skip it, not crash.
    # Child rows are removed first to satisfy the FK constraints that
    # database.py's schema deliberately enforces.
    with db._connect() as conn:
        conn.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,))
        conn.execute("DELETE FROM summaries WHERE meeting_id = ?", (meeting_id,))
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))

    response = semantic_search("summary", db=db, embedding_service=embedder, vector_store=store)
    assert all(r.meeting_id != meeting_id for r in response.results)


def test_semantic_search_rejects_clearly_unrelated_query():
    """Task 5's no-hallucination requirement starts here: a query with no
    topical relevance to anything indexed must not surface a result just
    because it's the 'closest available' vector - it must clear a minimum
    relevance score, or search returns nothing at all."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    seed_indexed_meeting(db, embedder, store, "mobile.wav",
                          "Ravi will handle the API integration by Friday for the mobile app launch.",
                          "The team discussed the mobile application launch.",
                          ["Continue with the planned mobile application launch."])

    response = semantic_search("What is the weather forecast for tomorrow?",
                                db=db, embedding_service=embedder, vector_store=store)
    assert response.results == []


def test_offline_matcher_ignores_filler_words_on_transcript_length_chunks():
    """Regression: with a long, realistic transcript the offline embedder used
    to match on filler words ("what", "is", "the", "for"), so an unrelated
    question scored HIGHER than a genuine one. Content words must decide."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    long_transcript = (
        "So, um, welcome everyone to the regular meeting. Let's start with the agenda. "
        "First thing is the upcoming capture the flag competition. I think we should open "
        "registration by next week, and Rahul, you said you would handle the registration form. "
        "Then the challenge servers. We need the web challenges hosted on the new cloud instance "
        "and Priya is checking the budget for that. We only have four binary exploitation "
        "challenges and we need at least eight, so Karan offered to write three more. "
    ) * 4
    meeting_id = db.create_meeting("club.mp3", long_transcript, language="en")
    db.save_summary(meeting_id, "The club planned the capture the flag competition.", [],
                    ["Hold the competition on the twentieth of next month."], [])
    index_meeting(db, meeting_id, embedder, store)

    related = semantic_search("Who is handling the registration form?", db=db,
                              embedding_service=embedder, vector_store=store)
    unrelated = semantic_search("What is the weather forecast for tomorrow?", db=db,
                                embedding_service=embedder, vector_store=store)
    assert related.results and related.results[0].meeting_id == meeting_id
    assert unrelated.results == []


def test_generic_decision_query_matches_decision_chunk():
    """'what is the key decision' has no word in common with the decision text
    itself - chunk-type labels ("Decision: ...") make it findable."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_indexed_meeting(
        db, embedder, store, "m.wav", "We talked about the launch schedule at length today.",
        "The team planned the launch.", ["Hold the competition on the twentieth of next month."])
    response = semantic_search("what is the key decision", db=db,
                               embedding_service=embedder, vector_store=store)
    assert response.results
    assert response.results[0].content_type == "decision"


def _without_api_keys():
    """Context helper: run a block as if no embedding API key were configured."""
    import contextlib, os

    @contextlib.contextmanager
    def cm():
        saved = {k: os.environ.pop(k, None) for k in
                 ("OPENAI_API_KEY", "EMBEDDING_API_KEY", "GEMINI_API_KEY", "EMBEDDING_PROVIDER")}
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
    return cm()


def _seed_club_meeting(db, embedder, store):
    meeting_id = db.create_meeting(
        "club.mp3",
        ("Welcome to the meeting. Rahul will handle the registration form by Wednesday. "
         "Karan will write three more binary exploitation challenges. ") * 20, language="en")
    db.save_summary(meeting_id, "The club planned the competition.", [],
                    ["Hold the competition on the twentieth of next month."], [])
    rahul = db.get_or_create_participant("Rahul")
    db.save_action_item(meeting_id, rahul, "Prepare the registration form", "Wednesday", "High", "Not Started")
    index_meeting(db, meeting_id, embedder, store)
    return meeting_id


_REAL_QUESTION = ("What are the main action items decided during this meeting, "
                  "and who is responsible for completing each one?")


def test_default_app_configuration_without_any_api_key_still_finds_meetings():
    """The app builds a DEFAULT EmbeddingService(). With no key at all it must
    resolve to the offline embedder AND use the offline relevance floor."""
    with _without_api_keys():
        db, store, embedder = make_temp_db(), make_temp_store(), EmbeddingService()
        assert embedder.config.provider == "local"
        meeting_id = _seed_club_meeting(db, embedder, store)
        found = semantic_search(_REAL_QUESTION, db=db, embedding_service=embedder, vector_store=store)
        assert found.results and found.results[0].meeting_id == meeting_id
        unrelated = semantic_search("What is the weather forecast for tomorrow?", db=db,
                                    embedding_service=embedder, vector_store=store)
        assert unrelated.results == []


def test_configured_remote_provider_without_key_falls_back_with_the_offline_floor():
    """Regression for the original bug: provider says 'openai' but there is no
    key, so the offline embedder actually runs - the floor must follow THAT."""
    with _without_api_keys():
        db, store = make_temp_db(), make_temp_store()
        embedder = EmbeddingService(EmbeddingConfig(provider="openai"))
        assert embedder.min_relevance_score == 0.15
        meeting_id = _seed_club_meeting(db, embedder, store)
        found = semantic_search(_REAL_QUESTION, db=db, embedding_service=embedder, vector_store=store)
        assert found.results and found.results[0].meeting_id == meeting_id


def test_full_search_flow_through_the_gemini_provider_path():
    """Index and search using the Gemini provider code path (stand-in client):
    documents are embedded as RETRIEVAL_DOCUMENT, the question as
    RETRIEVAL_QUERY, and the real-model relevance floor (0.30) is applied."""
    from test_embedding_service import _FakeGeminiClient
    fake = _FakeGeminiClient()
    embedder = EmbeddingService(EmbeddingConfig(provider="gemini"))
    embedder._client = fake
    db, store = make_temp_db(), make_temp_store()
    meeting_id = _seed_club_meeting(db, embedder, store)

    assert store.get_by_meeting(meeting_id)
    found = semantic_search("Who is handling the registration form?", db=db,
                            embedding_service=embedder, vector_store=store)
    assert found.results and found.results[0].meeting_id == meeting_id
    assert embedder.min_relevance_score == 0.30
    tasks = [c["config"]["task_type"] for c in fake.calls]
    assert tasks[0] == "RETRIEVAL_DOCUMENT" and tasks[-1] == "RETRIEVAL_QUERY"
    assert len(fake.calls) == 2       # ONE batched request to index the meeting, one for the question


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
