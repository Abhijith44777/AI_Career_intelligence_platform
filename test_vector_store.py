"""
test_vector_store.py
----------------------
Tests for Milestone 3 - Task 3 (Vector Database Integration).

Forces backend="local" so these tests run fully offline without requiring
chromadb to be installed, and use a temp file so they never touch a real
store and can be re-run repeatedly without leftover state - same pattern
as test_database.py's make_temp_db().
"""

import os
import tempfile

from vector_store import VectorStore


def make_temp_store() -> VectorStore:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return VectorStore(backend="local", path=path)


def test_insert_and_query_returns_the_vector():
    store = make_temp_store()
    store.upsert("v1", [1.0, 0.0, 0.0], {"meeting_id": 1, "content_type": "summary"}, "some text")
    results = store.query([1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].id == "v1"
    assert results[0].score > 0.99


def test_similarity_search_ranks_closer_vectors_first():
    store = make_temp_store()
    store.upsert("close", [1.0, 0.0], {"meeting_id": 1}, "close")
    store.upsert("far", [0.0, 1.0], {"meeting_id": 2}, "far")
    results = store.query([0.9, 0.1], top_k=2)
    assert results[0].id == "close"


def test_update_embedding_overwrites_previous_vector():
    store = make_temp_store()
    store.upsert("v1", [1.0, 0.0], {"meeting_id": 1}, "original")
    store.upsert("v1", [0.0, 1.0], {"meeting_id": 1}, "updated")
    assert store.count() == 1
    results = store.query([0.0, 1.0], top_k=1)
    assert results[0].document == "updated"


def test_delete_single_vector():
    store = make_temp_store()
    store.upsert("v1", [1.0, 0.0], {"meeting_id": 1}, "text")
    store.delete("v1")
    assert store.count() == 0


def test_delete_by_meeting_removes_only_that_meetings_vectors():
    store = make_temp_store()
    store.upsert("m1-a", [1.0, 0.0], {"meeting_id": 1}, "a")
    store.upsert("m1-b", [0.0, 1.0], {"meeting_id": 1}, "b")
    store.upsert("m2-a", [1.0, 1.0], {"meeting_id": 2}, "c")

    store.delete_by_meeting(1)

    assert store.count() == 1
    remaining = store.query([1.0, 1.0], top_k=5)
    assert all(r.metadata["meeting_id"] == 2 for r in remaining)


def test_metadata_filtering_restricts_results():
    store = make_temp_store()
    store.upsert("v1", [1.0, 0.0], {"meeting_id": 1, "content_type": "decision"}, "decision text")
    store.upsert("v2", [1.0, 0.0], {"meeting_id": 1, "content_type": "summary"}, "summary text")

    results = store.query([1.0, 0.0], top_k=5, metadata_filter={"content_type": "decision"})
    assert len(results) == 1
    assert results[0].id == "v1"


def test_every_vector_traces_back_to_a_meeting_id():
    store = make_temp_store()
    store.upsert("v1", [1.0, 0.0], {"meeting_id": 7, "content_type": "summary"}, "text")
    results = store.query([1.0, 0.0], top_k=5)
    assert all("meeting_id" in r.metadata for r in results)
    assert results[0].metadata["meeting_id"] == 7


def test_upsert_many_inserts_all_records():
    store = make_temp_store()
    store.upsert_many([
        {"id": "a", "vector": [1.0, 0.0], "metadata": {"meeting_id": 1}, "document": "a"},
        {"id": "b", "vector": [0.0, 1.0], "metadata": {"meeting_id": 2}, "document": "b"},
    ])
    assert store.count() == 2


def test_query_on_empty_store_returns_no_results():
    store = make_temp_store()
    results = store.query([1.0, 0.0], top_k=5)
    assert results == []


def test_falls_back_to_local_when_chroma_backend_unavailable(monkeypatch=None):
    # Force an import failure for chromadb to exercise the graceful
    # fallback path used automatically in production if chromadb isn't
    # installed.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        store = VectorStore(backend="chroma")
        assert store.backend_name == "local"
    finally:
        builtins.__import__ = real_import


def test_multiple_meetings_A_B_C_each_traceable_independently():
    """Spec's explicit multi-meeting scenario: Meeting A -> vectors A,
    Meeting B -> vectors B, Meeting C -> vectors C. Searching content that
    only exists in Meeting B's vectors must identify Meeting B specifically,
    with no cross-contamination between meetings' vectors."""
    store = make_temp_store()

    store.upsert("a1", [1.0, 0.0, 0.0], {"meeting_id": "A", "content_type": "transcript"}, "Meeting A content")
    store.upsert("b1", [0.0, 1.0, 0.0], {"meeting_id": "B", "content_type": "transcript"}, "Meeting B content")
    store.upsert("c1", [0.0, 0.0, 1.0], {"meeting_id": "C", "content_type": "transcript"}, "Meeting C content")

    # get_by_meeting: each meeting's own vectors, and only its own
    assert [r.id for r in store.get_by_meeting("A")] == ["a1"]
    assert [r.id for r in store.get_by_meeting("B")] == ["b1"]
    assert [r.id for r in store.get_by_meeting("C")] == ["c1"]

    # A query that matches Meeting B's vector must resolve to meeting_id "B"
    results = store.query([0.0, 1.0, 0.0], top_k=1)
    assert results[0].metadata["meeting_id"] == "B"

    # Deleting Meeting A's vectors must not touch B or C
    store.delete_by_meeting("A")
    assert store.count() == 2
    assert store.get_by_meeting("B")
    assert store.get_by_meeting("C")


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
