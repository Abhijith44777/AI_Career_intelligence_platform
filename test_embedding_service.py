"""
test_embedding_service.py
---------------------------
Tests for Milestone 3 - Task 2 (Embedding Generation).

Uses the "local" provider throughout so these tests run fully offline,
deterministically, with no API key - same philosophy as test_llm_service.py
only exercising the parts that don't need a live API call.
"""

import os
import math
import types

import embedding_service as es
from embedding_service import (
    EmbeddingService, EmbeddingConfig, chunk_text, cosine_similarity,
    build_meeting_chunks, MeetingChunk, resolve_auto_provider,
)


def local_service():
    return EmbeddingService(EmbeddingConfig(provider="local"))


def test_embed_text_returns_correct_dimensions():
    service = local_service()
    vector = service.embed_text("The team discussed the mobile app launch.")
    assert len(vector) == service.config.dimensions


def test_embed_text_is_deterministic():
    service = local_service()
    v1 = service.embed_text("Complete API integration by Friday")
    v2 = service.embed_text("Complete API integration by Friday")
    assert v1 == v2


def test_embed_empty_text_returns_zero_vector():
    service = local_service()
    vector = service.embed_text("")
    assert vector == [0.0] * service.config.dimensions


def test_similar_texts_score_higher_than_unrelated_texts():
    service = local_service()
    a = service.embed_text("The database migration was completed successfully this week.")
    b = service.embed_text("We finished migrating the database this week.")
    c = service.embed_text("The office holiday party is scheduled for December.")

    sim_related = cosine_similarity(a, b)
    sim_unrelated = cosine_similarity(a, c)
    assert sim_related > sim_unrelated


def test_embed_texts_batch():
    service = local_service()
    vectors = service.embed_texts(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert all(len(v) == service.config.dimensions for v in vectors)


def test_chunk_text_short_text_not_split():
    chunks = chunk_text("A short sentence.", max_chars=1000)
    assert chunks == ["A short sentence."]


def test_chunk_text_long_text_is_split():
    sentence = "This is one sentence of meeting discussion. "
    long_text = sentence * 100
    chunks = chunk_text(long_text, max_chars=300)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 300 or " " not in c


def test_chunk_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_mismatched_lengths_returns_zero():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# build_meeting_chunks - Task 2's four required inputs
# ---------------------------------------------------------------------------

SAMPLE_MEETING_FULL = {
    "meeting": {
        "id": 1,
        "filename": "standup.wav",
        "transcript": "Ravi will handle the API integration. Priya will test the UI.",
        "created_at": "2026-09-01T10:00:00",
    },
    "summary": {
        "summary_text": "The team discussed the mobile app launch.",
        "decisions": ["Continue with the planned mobile application launch."],
        "priorities": ["Mobile launch"],
    },
    "action_items": [
        {"task": "Complete API integration", "participant_name": "Ravi", "deadline": "Friday"},
        {"task": "Prepare UI testing report", "participant_name": "Priya", "deadline": ""},
    ],
}


def test_build_meeting_chunks_includes_all_four_types():
    chunks = build_meeting_chunks(1, SAMPLE_MEETING_FULL, max_chars_per_chunk=5000)
    types = {c.content_type for c in chunks}
    assert types == {"transcript_section", "summary", "decision", "action_item"}


def test_build_meeting_chunks_every_chunk_has_correct_meeting_id():
    chunks = build_meeting_chunks(42, SAMPLE_MEETING_FULL, max_chars_per_chunk=5000)
    assert all(c.meeting_id == 42 for c in chunks)


def test_build_meeting_chunks_action_item_includes_assignee_and_deadline():
    chunks = build_meeting_chunks(1, SAMPLE_MEETING_FULL, max_chars_per_chunk=5000)
    action_chunks = [c for c in chunks if c.content_type == "action_item"]
    assert any("Ravi" in c.text and "Friday" in c.text for c in action_chunks)


def test_build_meeting_chunks_handles_missing_summary_gracefully():
    minimal = {"meeting": SAMPLE_MEETING_FULL["meeting"], "summary": None, "action_items": []}
    chunks = build_meeting_chunks(1, minimal, max_chars_per_chunk=5000)
    assert all(c.content_type == "transcript_section" for c in chunks)


# ---------------------------------------------------------------------------
# Provider selection (auto) and the Gemini provider
# ---------------------------------------------------------------------------

_ENV_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "EMBEDDING_API_KEY", "EMBEDDING_PROVIDER")


class _env:
    """Temporarily set exactly these embedding-related env vars."""
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v


def test_auto_provider_uses_gemini_when_only_gemini_key_present():
    with _env(GEMINI_API_KEY="g"):
        assert resolve_auto_provider() == "gemini"
        assert EmbeddingService().config.provider == "gemini"
        assert EmbeddingService().config.model == "gemini-embedding-001"


def test_auto_provider_uses_openai_when_only_openai_key_present():
    with _env(OPENAI_API_KEY="o"):
        assert resolve_auto_provider() == "openai"


def test_auto_provider_falls_back_to_local_without_keys():
    with _env():
        assert resolve_auto_provider() == "local"


def test_auto_provider_prefers_gemini_when_both_keys_present_and_explicit_choice_wins():
    with _env(GEMINI_API_KEY="g", OPENAI_API_KEY="o"):
        assert resolve_auto_provider() == "gemini"
    with _env(GEMINI_API_KEY="g", OPENAI_API_KEY="o", EMBEDDING_PROVIDER="openai"):
        assert resolve_auto_provider() == "openai"


class _FakeGeminiClient:
    """Mimics client.models.embed_content(model=, contents=, config=)."""
    def __init__(self, dims=768, transient_failures=0, fatal_error=None):
        self.models = self
        self.calls = []
        self.dims = dims
        self.transient_failures = transient_failures
        self.fatal_error = fatal_error

    def embed_content(self, model, contents, config):
        self.calls.append({"model": model, "n": len(contents), "config": config})
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limit")
        # Deliberately NOT unit length, like Gemini's truncated vectors.
        embeddings = [types.SimpleNamespace(values=[5.0 * x for x in es._local_embed(t, self.dims)])
                      for t in contents]
        return types.SimpleNamespace(embeddings=embeddings)


def gemini_service(fake):
    service = EmbeddingService(EmbeddingConfig(provider="gemini"))
    service._client = fake
    return service


def test_gemini_embeddings_are_normalized_and_have_configured_size():
    fake = _FakeGeminiClient()
    vector = gemini_service(fake).embed_text("The database migration finished.")
    assert len(vector) == 768
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9


def test_gemini_request_uses_model_dimensions_and_task_types():
    fake = _FakeGeminiClient()
    service = gemini_service(fake)
    service.embed_text("a stored chunk")
    service.embed_query("a question")
    assert fake.calls[0]["model"] == "gemini-embedding-001"
    assert fake.calls[0]["config"] == {"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
    assert fake.calls[1]["config"]["task_type"] == "RETRIEVAL_QUERY"


def test_gemini_embed_texts_sends_one_batched_request_and_keeps_order():
    fake = _FakeGeminiClient()
    texts = ["alpha topic", "", "beta topic", "gamma topic"]
    vectors = gemini_service(fake).embed_texts(texts)
    assert len(fake.calls) == 1 and fake.calls[0]["n"] == 3          # empty text not sent
    assert len(vectors) == 4 and vectors[1] == [0.0] * 768           # empty -> zero vector, in place
    assert cosine_similarity(vectors[0], es._local_embed("alpha topic", 768)) > 0.99


def test_gemini_large_input_is_split_into_batches_of_100():
    fake = _FakeGeminiClient()
    vectors = gemini_service(fake).embed_texts([f"chunk number {i}" for i in range(250)])
    assert [c["n"] for c in fake.calls] == [100, 100, 50]
    assert len(vectors) == 250


def test_gemini_retries_rate_limit_errors_then_succeeds():
    fake = _FakeGeminiClient(transient_failures=2)
    slept = []
    real_sleep = es.time.sleep
    es.time.sleep = lambda seconds: slept.append(seconds)
    try:
        service = gemini_service(fake)
        vector = service.embed_text("hello there")
    finally:
        es.time.sleep = real_sleep
    assert len(fake.calls) == 3 and slept == [2.0, 4.0]
    assert len(vector) == 768 and service.using_local_embedder is False
    assert service.min_relevance_score == 0.30


def test_gemini_failure_falls_back_to_local_consistently_and_lowers_floor():
    fake = _FakeGeminiClient(fatal_error="403 API key not valid")
    service = gemini_service(fake)
    vectors = service.embed_texts(["one thing", "another thing"])
    assert len(fake.calls) == 1                                       # not retried: not transient
    assert {len(v) for v in vectors} == {384}                         # whole call fell back, one size
    assert service.using_local_embedder is True and service.min_relevance_score == 0.15


def test_gemini_without_key_reports_helpful_error_and_falls_back():
    with _env():
        service = EmbeddingService(EmbeddingConfig(provider="gemini"))
        assert len(service.embed_text("hello")) == 384
        assert service.using_local_embedder is True


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
