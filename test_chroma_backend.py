"""
test_chroma_backend.py
------------------------
Tests for the ChromaDB backend of vector_store.py.

chromadb itself is not installed in the offline test environment, so these
tests install a small stand-in `chromadb` module that reproduces the real
behaviours this project's code has to cope with:

  * default distance is SQUARED L2 (so "1 - distance" is NOT cosine),
  * a collection is locked to the vector size of its first insert,
  * a `where` with more than one key is rejected (must use $and),
  * embeddings come back as numpy arrays,
  * querying an empty collection raises,
  * get_collection raises for a collection that doesn't exist.

This verifies OUR logic against those behaviours. Running the real chromadb
package is covered by `python diagnose_search.py` on your own machine.
"""

import os
import sys
import tempfile
import types

import numpy as np

from embedding_service import _local_embed, cosine_similarity
from vector_store import VectorStore, _LocalVectorStore, _to_where


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.dim = None
        self.rows = {}   # id -> (embedding, metadata, document)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _check_where(where):
        if where is None:
            return
        if "$and" in where:
            for cond in where["$and"]:
                _FakeCollection._check_where(cond)
            return
        if len(where) != 1:
            raise ValueError("Expected where to have exactly one operator, got: %r" % (where,))

    @staticmethod
    def _matches(metadata, where):
        if where is None:
            return True
        if "$and" in where:
            return all(_FakeCollection._matches(metadata, c) for c in where["$and"])
        (key, value), = where.items()
        return metadata.get(key) == value

    # -- API ------------------------------------------------------------------
    def upsert(self, ids, embeddings, metadatas, documents):
        for id_, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
            if self.dim is None:
                self.dim = len(emb)
            if len(emb) != self.dim:
                raise ValueError(f"Collection expecting embedding with dimension of {self.dim}, got {len(emb)}")
            self.rows[id_] = (np.array(emb, dtype=np.float32), dict(meta), doc)

    def delete(self, ids=None, where=None):
        self._check_where(where)
        for id_ in list(self.rows):
            if (ids is not None and id_ in ids) or (where is not None and self._matches(self.rows[id_][1], where)):
                del self.rows[id_]

    def count(self):
        return len(self.rows)

    def get(self, where=None):
        self._check_where(where)
        hits = [(i, r) for i, r in self.rows.items() if self._matches(r[1], where)]
        return {"ids": [i for i, _ in hits], "metadatas": [r[1] for _, r in hits],
                "documents": [r[2] for _, r in hits]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        self._check_where(where)
        if not self.rows:
            raise ValueError("Cannot query an empty collection")
        q = np.array(query_embeddings[0], dtype=np.float32)
        if len(q) != self.dim:
            raise ValueError(f"Collection expecting embedding with dimension of {self.dim}, got {len(q)}")
        scored = []
        for id_, (emb, meta, doc) in self.rows.items():
            if self._matches(meta, where):
                scored.append((float(np.sum((emb - q) ** 2)), id_, emb, meta, doc))   # squared L2
        scored.sort(key=lambda t: t[0])
        scored = scored[:n_results]
        include = include or ["metadatas", "documents", "distances"]
        out = {"ids": [[s[1] for s in scored]]}
        if "distances" in include:
            out["distances"] = [[s[0] for s in scored]]
        if "metadatas" in include:
            out["metadatas"] = [[s[3] for s in scored]]
        if "documents" in include:
            out["documents"] = [[s[4] for s in scored]]
        out["embeddings"] = [np.array([s[2] for s in scored])] if "embeddings" in include else None
        return out


class _FakeClient:
    def __init__(self, path=None):
        self.collections = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, _FakeCollection(name))

    def get_collection(self, name):
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]


def make_chroma_store():
    """A VectorStore whose chroma backend runs on the stand-in module."""
    fake = types.ModuleType("chromadb")
    fake.PersistentClient = _FakeClient
    real = sys.modules.get("chromadb")
    sys.modules["chromadb"] = fake
    try:
        store = VectorStore(backend="chroma", path=tempfile.mkdtemp())
    finally:
        if real is None:
            del sys.modules["chromadb"]
        else:
            sys.modules["chromadb"] = real
    assert store.backend_name == "chroma"
    return store


def vec(text, dims=384):
    return _local_embed(text, dims)


def seed(store, dims=384):
    texts = {
        "1_summary_000": (1, "summary", "database migration completed on the production cluster"),
        "1_decision_000": (1, "decision", "migrate the production database this friday"),
        "2_summary_000": (2, "summary", "quarterly hiring plan for engineering recruiters"),
    }
    for id_, (meeting_id, ctype, text) in texts.items():
        store.upsert(id_, vec(text, dims), {"meeting_id": meeting_id, "content_type": ctype}, text)


def test_scores_are_true_cosine_not_squared_l2_transform():
    """The regression that made every Chroma search come back empty: with
    unit vectors, '1 - squared_L2' equals 2*cosine-1 and goes NEGATIVE for
    ordinary matches. Scores must be real cosine similarities."""
    chroma = make_chroma_store()
    seed(chroma)
    query = vec("database migration production cluster")

    local = VectorStore(backend="local", path=os.path.join(tempfile.mkdtemp(), "v.json"))
    seed(local)

    chroma_scores = {r.id: r.score for r in chroma.query(query, top_k=3)}
    local_scores = {r.id: r.score for r in local.query(query, top_k=3)}
    assert chroma_scores.keys() == local_scores.keys()
    for id_, score in local_scores.items():
        assert abs(chroma_scores[id_] - score) < 1e-4, (id_, chroma_scores[id_], score)
    assert max(chroma_scores.values()) > 0.2          # a genuine match is clearly positive


def test_results_are_sorted_best_first_and_map_to_meetings():
    chroma = make_chroma_store()
    seed(chroma)
    results = chroma.query(vec("database migration production cluster"), top_k=3)
    assert results[0].metadata["meeting_id"] == 1
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_two_condition_metadata_filter_is_accepted():
    chroma = make_chroma_store()
    seed(chroma)
    results = chroma.query(vec("database migration"), top_k=5,
                           metadata_filter={"content_type": "decision", "meeting_id": 1})
    assert [r.id for r in results] == ["1_decision_000"]
    assert _to_where({"a": 1}) == {"a": 1}
    assert _to_where({"a": 1, "b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}
    assert _to_where(None) is None


def test_switching_vector_size_uses_a_separate_collection_instead_of_crashing():
    chroma = make_chroma_store()
    seed(chroma, dims=384)                     # offline embedder
    seed(chroma, dims=768)                     # later: Gemini-sized vectors, same ids
    assert chroma.count() == 6
    r384 = chroma.query(vec("database migration", 384), top_k=5)
    r768 = chroma.query(vec("database migration", 768), top_k=5)
    assert len(r384) == 3 and len(r768) == 3


def test_query_with_a_vector_size_nothing_was_stored_in_returns_empty_not_error():
    chroma = make_chroma_store()
    seed(chroma, dims=384)
    assert chroma.query(vec("anything", 768), top_k=5) == []


def test_query_on_empty_store_returns_empty_not_error():
    chroma = make_chroma_store()
    assert chroma.query(vec("anything"), top_k=5) == []


def test_delete_by_meeting_and_get_by_meeting_work_across_vector_sizes():
    chroma = make_chroma_store()
    seed(chroma, dims=384)
    seed(chroma, dims=768)
    assert len(chroma.get_by_meeting(1)) == 4 and len(chroma.get_by_meeting(2)) == 2
    chroma.delete_by_meeting(1)
    assert chroma.get_by_meeting(1) == []
    assert chroma.count() == 2 and all(r.metadata["meeting_id"] == 2 for r in chroma.get_by_meeting(2))


def test_delete_single_vector_and_upsert_replaces_in_place():
    chroma = make_chroma_store()
    seed(chroma)
    chroma.upsert("2_summary_000", vec("brand new text about budgets"), {"meeting_id": 2, "content_type": "summary"}, "brand new text about budgets")
    assert chroma.count() == 3
    chroma.delete("1_decision_000")
    assert chroma.count() == 2


def test_top_k_larger_than_stored_count_is_clamped():
    chroma = make_chroma_store()
    seed(chroma)
    assert len(chroma.query(vec("database"), top_k=50)) == 3


def test_falls_back_to_local_store_if_chroma_cannot_start():
    class Broken:
        def PersistentClient(self, path=None):
            raise RuntimeError("cannot open database")
    fake = types.ModuleType("chromadb")
    fake.PersistentClient = Broken().PersistentClient
    real = sys.modules.get("chromadb")
    sys.modules["chromadb"] = fake
    try:
        store = VectorStore(backend="chroma", path=os.path.join(tempfile.mkdtemp(), "v.json"))
    finally:
        if real is None:
            del sys.modules["chromadb"]
        else:
            sys.modules["chromadb"] = real
    assert store.backend_name == "local"


if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"✅ PASS - {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"❌ FAIL - {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
