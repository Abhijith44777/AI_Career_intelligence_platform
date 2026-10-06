"""
test_rag_qa.py
----------------
Tests for Milestone 3 - Task 5 (RAG Question Answering).

Uses the "local" LLM provider (no API key) so these run fully offline: the
local path is purely extractive (built directly from retrieved chunks),
which makes it easy to prove answers are grounded in retrieved meeting
information and never unsupported/hardcoded - exactly what the task asks
to verify.
"""

import os
import tempfile

from database import Database
from embedding_service import EmbeddingService, EmbeddingConfig
from vector_store import VectorStore
from llm_service import LLMService, LLMConfig
from knowledge_repository import index_meeting
from rag_qa import answer_question, NO_CONTEXT_ANSWER


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


def local_llm() -> LLMService:
    return LLMService(LLMConfig(provider="local"))


def seed_indexed_meeting(db, embedder, store):
    meeting_id = db.create_meeting(
        "mobile-launch.wav",
        "Ravi will handle the API integration by Friday. Priya will prepare the UI testing report.",
        language="en", audio_duration_seconds=90,
    )
    db.save_summary(
        meeting_id=meeting_id,
        summary_text="The team discussed the mobile app launch and assigned responsibilities.",
        key_points=["Launch timeline confirmed"],
        decisions=["Continue with the planned mobile application launch."],
        priorities=["Mobile launch"],
    )
    ravi_id = db.get_or_create_participant("Ravi")
    db.save_action_item(meeting_id, ravi_id, "Complete API integration", "Friday", "High", "Not Started")
    index_meeting(db, meeting_id, embedder, store)
    return meeting_id


def test_answer_question_grounds_answer_in_retrieved_content():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    seed_indexed_meeting(db, embedder, store)

    # Note: the offline "local" embedder is a simple bag-of-words hashing
    # vectorizer (no real semantic understanding), so the query is phrased
    # with enough lexical overlap with the source transcript for it to
    # retrieve the right chunk - a real embedding provider (e.g. OpenAI)
    # would not need this. What's under test here is that the retrieved
    # deadline ends up verbatim in the grounded answer, not the retrieval
    # quality of the fallback embedder itself.
    result = answer_question("What is the deadline for the API integration task?",
                              db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())

    assert "Friday" in result.answer
    assert result.grounded is True
    assert result.sources


def test_answer_question_with_no_matching_context_says_it_does_not_know():
    db = make_temp_db()
    store = make_temp_store()  # nothing indexed
    embedder = local_embedder()

    result = answer_question("What was decided about the office relocation?",
                              db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_answer_question_unrelated_question_against_populated_store_does_not_hallucinate():
    """Even with meetings indexed, a question with no real topical overlap
    must not be answered from a merely-closest-but-irrelevant match."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    seed_indexed_meeting(db, embedder, store)

    result = answer_question("What is the weather forecast for tomorrow?",
                              db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_answer_question_empty_question_does_not_crash():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    result = answer_question("", db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())
    assert result.answer == NO_CONTEXT_ANSWER


def test_answer_question_sources_reference_the_correct_meeting():
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    meeting_id = seed_indexed_meeting(db, embedder, store)

    result = answer_question("Who is handling the API integration?",
                              db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())

    assert result.sources
    assert all(s.meeting_id == meeting_id for s in result.sources)


def test_local_fallback_answer_never_contains_text_outside_the_sources():
    """The local extractive path concatenates retrieved snippets verbatim -
    prove the answer is built ONLY from what was retrieved, not generated
    from scratch (i.e. it cannot be an unsupported/hardcoded response)."""
    db = make_temp_db()
    store = make_temp_store()
    embedder = local_embedder()
    seed_indexed_meeting(db, embedder, store)

    result = answer_question("What is the deadline for API integration?",
                              db=db, embedding_service=embedder, vector_store=store,
                              llm_service=local_llm())

    combined_source_text = " ".join(s.snippet for s in result.sources)
    for line in result.answer.split("\n"):
        # Each answer line quotes a source snippet inline, so its core
        # content should appear in the combined retrieved source text.
        snippet_fragment = line.split(": ", 1)[-1]
        assert snippet_fragment in combined_source_text


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
