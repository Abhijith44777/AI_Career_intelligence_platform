"""
vector_store.py
-----------------
Milestone 3 - Task 3: Vector Database Integration

Integrates a vector database with the existing application, supporting:
    Insert embeddings
    Update embeddings
    Delete embeddings
    Similarity search
    Metadata filtering
    Meeting-to-vector mapping

Default backend is ChromaDB (persisted locally under ./chroma_db - no
server to run, easiest fit for this stack). If chromadb isn't installed
or fails to initialize, this transparently falls back to a zero-dependency
pure-Python store (JSON-persisted, cosine similarity in-process) - the
same graceful-degradation philosophy used throughout this project
(llm_service.py / embedding_service.py), so vector search NEVER hard-fails
the app just because an optional dependency is missing, and the whole
Milestone 3 pipeline stays testable offline without installing chromadb.

Every vector is stored with metadata that always includes `meeting_id`,
so any stored vector can be traced back to the meeting it came from
(the "meeting-to-vector mapping" requirement) regardless of backend.
"""

import os
import json
import logging
import threading
from dataclasses import dataclass

from embedding_service import cosine_similarity

logger = logging.getLogger(__name__)

# VECTOR_DATABASE_URL / VECTOR_COLLECTION_NAME - reuse existing project .env
# conventions (see .env.example). VECTOR_DATABASE_URL is treated as a local
# filesystem path here (ChromaDB's PersistentClient / the local JSON store
# both persist to disk - no server process is needed for this project).
DEFAULT_LOCAL_STORE_PATH = os.environ.get(
    "VECTOR_DATABASE_URL", os.path.join(os.path.dirname(__file__), "vector_store_local.json")
)
DEFAULT_CHROMA_PATH = os.environ.get(
    "VECTOR_DATABASE_URL", os.path.join(os.path.dirname(__file__), "chroma_db")
)
DEFAULT_COLLECTION_NAME = os.environ.get("VECTOR_COLLECTION_NAME", "meeting_intelligence")


@dataclass
class VectorRecord:
    id: str
    score: float
    metadata: dict
    document: str = ""


# ---------------------------------------------------------------------------
# Pure-Python local backend (zero dependency, always available)
# ---------------------------------------------------------------------------

class _LocalVectorStore:
    """
    JSON-file-persisted vector store with in-process cosine similarity.
    Not meant to scale to millions of vectors, but implements the exact
    same interface as the ChromaDB-backed store and is perfect for
    development, offline tests, and small meeting archives.
    """

    def __init__(self, path: str = DEFAULT_LOCAL_STORE_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def upsert(self, id: str, vector: list, metadata: dict, document: str = ""):
        with self._lock:
            self._data[id] = {"vector": vector, "metadata": metadata, "document": document}
            self._save()

    def delete(self, id: str):
        with self._lock:
            self._data.pop(id, None)
            self._save()

    def delete_by_meeting(self, meeting_id: int):
        with self._lock:
            to_delete = [k for k, v in self._data.items()
                         if v.get("metadata", {}).get("meeting_id") == meeting_id]
            for k in to_delete:
                del self._data[k]
            self._save()

    def count(self) -> int:
        return len(self._data)

    def get_by_meeting(self, meeting_id: int) -> list:
        return [
            VectorRecord(id=id_, score=1.0, metadata=v.get("metadata", {}), document=v.get("document", ""))
            for id_, v in self._data.items()
            if v.get("metadata", {}).get("meeting_id") == meeting_id
        ]

    def query(self, vector: list, top_k: int = 5, metadata_filter: dict = None) -> list:
        results = []
        for id_, entry in self._data.items():
            metadata = entry.get("metadata", {})
            if metadata_filter and not all(metadata.get(k) == v for k, v in metadata_filter.items()):
                continue
            score = cosine_similarity(vector, entry.get("vector", []))
            results.append(VectorRecord(id=id_, score=score, metadata=metadata,
                                         document=entry.get("document", "")))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# ChromaDB-backed store (preferred, used automatically if available)
# ---------------------------------------------------------------------------

# Vector sizes this project can produce (local, Gemini, OpenAI, Gemini-full).
# A ChromaDB collection is locked to the vector size of its first insert, so
# each size gets its own collection ("<name>_d384", "<name>_d768", ...).
# Switching embedding provider then just works - no crash, no manual reset.
_KNOWN_DIMS = (384, 768, 1536, 3072)


def _to_where(metadata_filter: dict):
    """ChromaDB rejects a where-dict with more than one key; several
    conditions must be wrapped in $and."""
    if not metadata_filter:
        return None
    items = list(metadata_filter.items())
    if len(items) == 1:
        return {items[0][0]: items[0][1]}
    return {"$and": [{k: v} for k, v in items]}


def _rows(result, key):
    """First row of a Chroma result field. Handles missing fields and the
    numpy arrays newer Chroma versions return (which can't be used with `or`)."""
    value = result.get(key)
    if value is None:
        return []
    first = value[0]
    return [] if first is None else first


class _ChromaVectorStore:
    def __init__(self, path: str = DEFAULT_CHROMA_PATH, collection_name: str = DEFAULT_COLLECTION_NAME):
        import chromadb
        self._client = chromadb.PersistentClient(path=path)
        self._base = collection_name
        self._collections = {}

    def _name(self, dim: int) -> str:
        return f"{self._base}_d{dim}"

    def _collection(self, dim: int, create: bool = True):
        if dim in self._collections:
            return self._collections[dim]
        try:
            if create:
                col = self._client.get_or_create_collection(name=self._name(dim))
            else:
                col = self._client.get_collection(name=self._name(dim))
        except Exception:  # noqa: BLE001 - collection simply doesn't exist yet
            return None
        self._collections[dim] = col
        return col

    def _existing_collections(self) -> list:
        found = []
        for dim in sorted(set(_KNOWN_DIMS) | set(self._collections)):
            col = self._collection(dim, create=False)
            if col is not None:
                found.append(col)
        return found

    def upsert(self, id: str, vector: list, metadata: dict, document: str = ""):
        # Chroma metadata values must be str/int/float/bool - stringify anything else.
        clean_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                           for k, v in metadata.items() if v is not None}
        self._collection(len(vector)).upsert(ids=[id], embeddings=[vector], metadatas=[clean_metadata],
                                              documents=[document or ""])

    def delete(self, id: str):
        for col in self._existing_collections():
            col.delete(ids=[id])

    def delete_by_meeting(self, meeting_id: int):
        for col in self._existing_collections():
            col.delete(where={"meeting_id": meeting_id})

    def count(self) -> int:
        return sum(col.count() for col in self._existing_collections())

    def get_by_meeting(self, meeting_id: int) -> list:
        records = []
        for col in self._existing_collections():
            result = col.get(where={"meeting_id": meeting_id})
            ids = result.get("ids")
            metadatas = result.get("metadatas")
            documents = result.get("documents")
            ids = [] if ids is None else list(ids)
            for i, id_ in enumerate(ids):
                records.append(VectorRecord(
                    id=id_, score=1.0,
                    metadata=metadatas[i] if metadatas is not None and i < len(metadatas) else {},
                    document=documents[i] if documents is not None and i < len(documents) else "",
                ))
        return records

    def query(self, vector: list, top_k: int = 5, metadata_filter: dict = None) -> list:
        col = self._collection(len(vector), create=False)
        if col is None:
            logger.warning("No vectors of size %d are stored - re-index your meetings "
                           "(the embedding provider may have changed).", len(vector))
            return []
        total = col.count()
        if total == 0:
            return []

        kwargs = {"query_embeddings": [vector], "n_results": min(max(top_k, 1), total),
                  "include": ["documents", "metadatas", "embeddings", "distances"]}
        where = _to_where(metadata_filter)
        if where:
            kwargs["where"] = where
        result = col.query(**kwargs)

        ids = list(_rows(result, "ids"))
        metadatas = _rows(result, "metadatas")
        documents = _rows(result, "documents")
        embeddings = _rows(result, "embeddings")
        distances = _rows(result, "distances")

        records = []
        for i, id_ in enumerate(ids):
            if i < len(embeddings):
                # True cosine similarity, computed from the stored vector, so
                # scores mean the same thing whatever distance metric the
                # collection uses (Chroma's default is squared-L2, for which
                # "1 - distance" would be 2*cosine - 1 and go negative).
                score = cosine_similarity(vector, [float(x) for x in embeddings[i]])
            else:
                score = 1.0 - (float(distances[i]) / 2.0 if i < len(distances) else 1.0)
            records.append(VectorRecord(
                id=id_, score=score,
                metadata=metadatas[i] if i < len(metadatas) else {},
                document=documents[i] if i < len(documents) else "",
            ))
        records.sort(key=lambda r: r.score, reverse=True)
        return records


# ---------------------------------------------------------------------------
# Public facade - selects backend, exposes a single stable interface
# ---------------------------------------------------------------------------

class VectorStoreError(Exception):
    pass


class VectorStore:
    """
    Facade over the vector database backend. Defaults to ChromaDB;
    automatically falls back to the local pure-Python store if chromadb
    isn't installed / fails to initialize (e.g. offline dev environment,
    CI, or the test suite). Pass backend="local" to force the fallback
    explicitly (used by the offline test suite for determinism/speed).
    """

    def __init__(self, backend: str = "chroma", path: str = None,
                 collection_name: str = DEFAULT_COLLECTION_NAME):
        self.backend_name = backend
        if backend == "local":
            self._impl = _LocalVectorStore(path or DEFAULT_LOCAL_STORE_PATH)
        elif backend == "chroma":
            try:
                self._impl = _ChromaVectorStore(path or DEFAULT_CHROMA_PATH, collection_name)
            except Exception as e:  # noqa: BLE001 - graceful degradation, same pattern as llm_service.py
                logger.warning("ChromaDB backend unavailable (%s); falling back to local vector store.", e)
                self.backend_name = "local"
                self._impl = _LocalVectorStore(path or DEFAULT_LOCAL_STORE_PATH)
        else:
            raise VectorStoreError(f"Unknown vector store backend: {backend}")
        logger.info("VectorStore initialized with backend=%s", self.backend_name)

    # -- Insert / Update (same call - upsert covers both) -------------------

    def upsert(self, id: str, vector: list, metadata: dict, document: str = ""):
        """Insert a new vector, or update it in place if `id` already exists."""
        metadata = dict(metadata or {})
        self._impl.upsert(id, vector, metadata, document)
        logger.info("Vector upserted: id=%s meeting_id=%s content_type=%s",
                    id, metadata.get("meeting_id"), metadata.get("content_type"))

    def upsert_many(self, records: list):
        """records: list of dicts with keys id, vector, metadata, document."""
        for r in records:
            self.upsert(r["id"], r["vector"], r.get("metadata", {}), r.get("document", ""))

    # -- Delete ---------------------------------------------------------------

    def delete(self, id: str):
        self._impl.delete(id)
        logger.info("Vector deleted: id=%s", id)

    def delete_by_meeting(self, meeting_id: int):
        """Remove every vector belonging to a given meeting (e.g. before
        re-indexing it, or if a meeting is removed)."""
        self._impl.delete_by_meeting(meeting_id)
        logger.info("Vectors deleted for meeting_id=%s", meeting_id)

    # -- Similarity search + metadata filtering --------------------------------

    def query(self, vector: list, top_k: int = 5, metadata_filter: dict = None) -> list:
        """Returns a list of VectorRecord, best match first. `metadata_filter`
        is an exact-match dict, e.g. {"content_type": "decision"}."""
        results = self._impl.query(vector, top_k=top_k, metadata_filter=metadata_filter)
        logger.info("Vector search returned %d result(s) (filter=%s)", len(results), metadata_filter)
        return results

    def get_by_meeting(self, meeting_id: int) -> list:
        """List every vector stored for one meeting - the direct
        vector-ID -> meeting_id -> database-record traceability check
        (Task 3's meeting-to-vector mapping requirement), and what backs
        the GET /meetings/{meeting_id}/vectors endpoint."""
        return self._impl.get_by_meeting(meeting_id)

    def count(self) -> int:
        return self._impl.count()
