"""
semantic_search.py
--------------------
Milestone 3 - Task 4: Semantic Search

Implements natural-language search across historical meetings, matching
the spec's flow exactly:

    User Query
        |
    Query Embedding
        |
    Vector Search
        |
    Relevant Meetings
        |
    Search Results

The project requires relevant meetings to be retrieved within 3 seconds -
`semantic_search` measures and returns `elapsed_seconds` on every call so
this can be verified/monitored (see test_semantic_search.py and the
LATENCY_TARGET_SECONDS constant below).
"""

import time
import logging
from dataclasses import dataclass, field

from database import Database
from embedding_service import EmbeddingService
from vector_store import VectorStore

logger = logging.getLogger(__name__)

LATENCY_TARGET_SECONDS = 3.0

# Below this cosine-similarity score, a match is treated as "not actually
# relevant" rather than forced into the results - this is what lets Task 5
# correctly say "I don't have information about that" instead of always
# returning whatever the closest (but irrelevant) chunk happens to be.
# NOTE: this threshold is calibrated for the "local" fallback embedder's
# bag-of-words hashing scores. A real embedding provider (e.g. OpenAI)
# produces a much wider, more semantically meaningful score spread, so this
# same threshold works there too - but if you swap in a different provider
# and see too many/too few "not found" results, this is the constant to
# retune.
MIN_RELEVANCE_SCORE = 0.30


@dataclass
class SearchResult:
    meeting_id: int
    meeting_title: str      # == filename; the existing schema has no separate title field
    meeting_date: str       # == created_at
    content_type: str
    relevance_score: float
    matched_text: str          # short snippet for display (<= 300 chars)
    full_text: str = ""        # the complete matched chunk - used as RAG context


@dataclass
class SearchResponse:
    query: str
    results: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    within_latency_target: bool = True


def semantic_search(query: str, db: Database = None,
                     embedding_service: EmbeddingService = None,
                     vector_store: VectorStore = None,
                     top_k: int = 5, content_type_filter: str = None,
                     meeting_id_filter: int = None,
                     max_chunks_per_meeting: int = 1,
                     user_id: int = None) -> SearchResponse:
    """
    "Which meeting discussed the database migration?" -> ranked meetings.

    Retrieves the top matching chunks from the vector store, then
    aggregates to one result per meeting (keeping each meeting's
    best-scoring chunk) so a meeting with several matching chunks doesn't
    crowd out other relevant meetings in the results. Supports metadata
    filtering by content_type and/or meeting_id. RAG passes
    max_chunks_per_meeting > 1 so the LLM sees several supporting chunks
    (e.g. every action item) from each relevant meeting, not just one.
    """
    start = time.perf_counter()

    db = db or Database()
    embedding_service = embedding_service or EmbeddingService()
    vector_store = vector_store or VectorStore()

    query = (query or "").strip()
    if not query:
        elapsed = time.perf_counter() - start
        logger.info("Empty search query - returning no results (%.3fs)", elapsed)
        return SearchResponse(query=query, results=[], elapsed_seconds=elapsed,
                               within_latency_target=elapsed <= LATENCY_TARGET_SECONDS)

    embed = getattr(embedding_service, "embed_query", None) or embedding_service.embed_text
    query_vector = embed(query)

    metadata_filter = {}
    if content_type_filter:
        metadata_filter["content_type"] = content_type_filter
    if meeting_id_filter is not None:
        metadata_filter["meeting_id"] = meeting_id_filter
    if user_id is not None:
        # Access control: only this user's vectors are even considered.
        metadata_filter["user_id"] = user_id
    metadata_filter = metadata_filter or None

    # Over-fetch chunk-level matches since several chunks can belong to the
    # same meeting; we then collapse to top_k distinct meetings below.
    raw_matches = vector_store.query(
        query_vector, top_k=max(top_k * max(4, max_chunks_per_meeting * 2), 10),
        metadata_filter=metadata_filter)

    by_meeting = {}
    for match in raw_matches:
        meeting_id = match.metadata.get("meeting_id")
        if meeting_id is None:
            continue
        by_meeting.setdefault(meeting_id, []).append(match)

    best_meetings = sorted(by_meeting.values(),
                           key=lambda ms: max(m.score for m in ms), reverse=True)[:top_k]
    ranked = []
    for matches in best_meetings:
        matches.sort(key=lambda m: m.score, reverse=True)
        ranked.extend(matches[:max(1, max_chunks_per_meeting)])
    # Drop matches that don't clear the relevance floor - an irrelevant
    # top-1 is still the "closest" vector, but "closest" isn't the same as
    # "actually relevant" (see MIN_RELEVANCE_SCORE above).
    # The floor depends on the embedding provider (see
    # EmbeddingService.min_relevance_score); MIN_RELEVANCE_SCORE is only the
    # default for embedders that don't declare their own.
    min_score = getattr(embedding_service, "min_relevance_score", MIN_RELEVANCE_SCORE)
    ranked = [m for m in ranked if m.score >= min_score]

    results = []
    for match in ranked:
        meeting = db.get_meeting(match.metadata.get("meeting_id"))
        if meeting and user_id is not None and meeting.get("user_id") != user_id:
            # Defence in depth: even if a vector's metadata were wrong/stale,
            # the database is the source of truth for who owns the meeting.
            logger.warning("Search dropped a vector for meeting_id=%s not owned by user_id=%s",
                           meeting["id"], user_id)
            continue
        if not meeting:
            logger.warning("Vector metadata referenced meeting_id=%s which no longer exists in the "
                            "database - skipping from results.", match.metadata.get("meeting_id"))
            continue  # vector outlived its meeting record - skip rather than error
        results.append(SearchResult(
            meeting_id=meeting["id"],
            meeting_title=meeting["filename"],
            meeting_date=meeting["created_at"],
            content_type=match.metadata.get("content_type", ""),
            relevance_score=round(match.score, 4),
            matched_text=match.document[:300],
            full_text=match.document,
        ))

    elapsed = time.perf_counter() - start
    within_target = elapsed <= LATENCY_TARGET_SECONDS
    logger.info("Semantic search: query=%r results=%d elapsed=%.3fs within_target=%s",
                query, len(results), elapsed, within_target)
    return SearchResponse(query=query, results=results, elapsed_seconds=elapsed,
                           within_latency_target=within_target)
