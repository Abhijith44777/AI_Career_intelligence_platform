"""
embedding_service.py
----------------------
Milestone 3 - Task 2: Embedding Generation

Generates vector embeddings for meeting information so it can be
indexed and semantically searched (Tasks 3-5):
    Transcript sections
    Summaries
    Decisions
    Action items

Follows the same provider/fallback philosophy already established by
llm_service.py: a real embedding API is used when a key is available
(Gemini via GEMINI_API_KEY - the same key the LLM already uses - or OpenAI),
and if no key is configured / the call fails, this transparently degrades to
a zero-dependency, deterministic "local" embedder (hashing vectorizer)
so embeddings are ALWAYS generated - the pipeline never hard-fails just
because no embedding API key is set, matching the project's established
"local rule-based" degradation pattern (see local_extraction.py).

This module has no knowledge of the database or vector store - it only
turns "text" into "a fixed-length vector of floats". That keeps it
independently testable and swappable, same as llm_service.py.
"""

import os
import re
import math
import hashlib
import time
import logging
from dataclasses import dataclass, field

# Load .env here too so scripts/tests that import this module directly (not
# only app.py / api.py) still see GEMINI_API_KEY etc.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is in requirements.txt, but never hard-fail on it
    pass

logger = logging.getLogger(__name__)

# Chunk size for embeddings is smaller than the LLM's chunk size
# (llm_service.py uses 12000 chars) because embedding granularity should
# be fine enough for useful semantic search results, not just to fit a
# context window.
DEFAULT_EMBED_CHUNK_CHARS = 1200

PROVIDER_API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",   # same key the LLM layer already uses
    "openai": "OPENAI_API_KEY",
    "local": "",  # no key needed - deterministic hashing embedder, always available
}

PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "local": "local-hashing-v1",
}

PROVIDER_DIMENSIONS = {
    "gemini": 768,    # gemini-embedding-001 supports 768 / 1536 / 3072
    "openai": 1536,
    "local": 384,
}

AUTO_PROVIDER = "auto"
_BATCH_SIZE = 100  # Gemini accepts up to 100 texts per embed request


def resolve_auto_provider() -> str:
    """Pick an embedding provider from what is configured:
       1. EMBEDDING_PROVIDER (gemini | openai | local) if set explicitly
       2. GEMINI_API_KEY  -> gemini   (the project's primary provider)
       3. OPENAI_API_KEY / EMBEDDING_API_KEY -> openai
       4. otherwise       -> local (offline fallback)"""
    explicit = (os.environ.get("EMBEDDING_PROVIDER") or "").strip().lower()
    if explicit in PROVIDER_API_KEY_ENV_VARS:
        return explicit
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY"):
        return "openai"
    return "local"


def _l2_normalize(vec: list) -> list:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class EmbeddingServiceError(Exception):
    """Base error for anything going wrong in the embedding layer."""
    pass


@dataclass
class EmbeddingConfig:
    # "auto" (default) chooses the provider from the keys in your environment
    # - see resolve_auto_provider(). Any provider silently falls back to the
    # offline "local" embedder if its key is missing or a call fails.
    provider: str = AUTO_PROVIDER     # "auto" | "gemini" | "openai" | "local"
    model: str = ""                   # auto-derived from provider if blank
    dimensions: int = 0               # auto-derived from provider if blank
    max_chars_per_chunk: int = DEFAULT_EMBED_CHUNK_CHARS
    api_key_env_var: str = field(default="")

    def __post_init__(self):
        if self.provider == AUTO_PROVIDER:
            self.provider = resolve_auto_provider()
        if self.provider not in PROVIDER_API_KEY_ENV_VARS:
            raise EmbeddingServiceError(
                f"Unknown embedding provider '{self.provider}'. Must be one of: "
                f"{sorted(PROVIDER_API_KEY_ENV_VARS)}"
            )
        if not self.api_key_env_var:
            self.api_key_env_var = PROVIDER_API_KEY_ENV_VARS[self.provider]
        if not self.model:
            # EMBEDDING_MODEL lets the model be swapped via .env without a
            # code change (set EMBEDDING_PROVIDER alongside it).
            self.model = os.environ.get("EMBEDDING_MODEL") or PROVIDER_DEFAULT_MODELS[self.provider]
        if not self.dimensions:
            self.dimensions = PROVIDER_DIMENSIONS[self.provider]


# ---------------------------------------------------------------------------
# Chunking - transcripts/summaries can be long; split on sentence boundaries
# so we don't cut a thought in half, same approach as llm_service.chunk_transcript.
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = DEFAULT_EMBED_CHUNK_CHARS) -> list:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current = ""
            else:
                current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Local (zero-dependency, deterministic) hashing embedder
# ---------------------------------------------------------------------------

# Filler words carry almost no topical meaning, but in a bag-of-words
# embedding they dominate the overlap between any two English sentences -
# which made unrelated questions ("what is the weather forecast for
# tomorrow?") look as similar to a transcript as genuine ones. Dropping them
# lets the offline fallback match on content words instead.
_STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its
just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those
through to too under until up us very was we were what when where which while who whom
why will with would you your yours yeah okay ok um uh like right well let lets go going
get got know think said say says thing things really actually maybe kind sort
""".split())


def _normalize_token(token: str) -> str:
    # Minimal plural folding so "decisions"/"decision" and "items"/"item"
    # count as the same word (no full stemmer needed for a fallback).
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list:
    return [_normalize_token(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS]


def _local_embed(text: str, dimensions: int = 384) -> list:
    """
    Deterministic hashing-vectorizer embedding: every content token is hashed
    into one of `dimensions` buckets (sign determined by a second hash, the
    classic "hashing trick" / feature-hashing approach), weighted by
    log(1 + term frequency) so a word repeated 20 times in a long transcript
    chunk doesn't drown out everything else, then L2-normalized. No external
    dependency, no network call, same vector for the same text every time -
    good enough to power cosine similarity search when no real embedding API
    key is configured.
    """
    vec = [0.0] * dimensions
    counts = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return vec

    for token, count in counts.items():
        h = hashlib.sha256(token.encode("utf-8")).hexdigest()
        weight = math.log1p(count) / math.sqrt(2)
        # Two independent hash positions per token, each at 1/sqrt(2)
        # weight: a genuine shared word still matches in both positions, but
        # an accidental bucket collision between two unrelated words now
        # only matches in one, so it counts for half as much.
        for bucket_hex, sign_hex in ((h[:8], h[8:10]), (h[10:18], h[18:20])):
            bucket = int(bucket_hex, 16) % dimensions
            sign = 1.0 if int(sign_hex, 16) % 2 == 0 else -1.0
            vec[bucket] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = ("429", "resource_exhausted", "rate limit", "503", "unavailable",
                      "timeout", "timed out", "deadline", "temporarily")


class EmbeddingService:
    """
    Provider-agnostic embedding service. Swap providers by changing
    EmbeddingConfig.provider - callers never need to know which vendor
    (or the local fallback) is behind it.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._client = None
        # True/False once an embedding has been produced: was it the offline
        # local embedder (even if a remote provider is configured)? None until
        # the first embedding.
        self._last_embed_was_local = None

    # -- key / client ---------------------------------------------------------

    def _api_key(self):
        # The provider's own variable first (GEMINI_API_KEY / OPENAI_API_KEY),
        # then the generic EMBEDDING_API_KEY.
        return os.environ.get(self.config.api_key_env_var) or os.environ.get("EMBEDDING_API_KEY")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.config.provider == "local":
            self._client = "local"
            return self._client

        api_key = self._api_key()
        if not api_key:
            raise EmbeddingServiceError(
                f"Missing API key. Set {self.config.api_key_env_var} (or EMBEDDING_API_KEY) in your .env file."
            )
        if self.config.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)
        elif self.config.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise EmbeddingServiceError(f"Unknown provider: {self.config.provider}")
        return self._client

    # -- remote calls -----------------------------------------------------------

    def _embed_batch_remote(self, texts: list, task: str) -> list:
        """One API request for a whole list of texts. `task` is Gemini's
        retrieval task type (RETRIEVAL_DOCUMENT for stored chunks,
        RETRIEVAL_QUERY for questions) - it measurably improves matching."""
        client = self._get_client()
        if self.config.provider == "gemini":
            response = client.models.embed_content(
                model=self.config.model, contents=texts,
                config={"task_type": task, "output_dimensionality": self.config.dimensions},
            )
            vectors = [_l2_normalize([float(x) for x in e.values]) for e in response.embeddings]
        elif self.config.provider == "openai":
            response = client.embeddings.create(model=self.config.model, input=texts)
            vectors = [list(d.embedding) for d in sorted(response.data, key=lambda d: d.index)]
        else:
            raise EmbeddingServiceError(f"Unknown provider: {self.config.provider}")
        if len(vectors) != len(texts):
            raise EmbeddingServiceError(
                f"Provider returned {len(vectors)} embeddings for {len(texts)} texts")
        return vectors

    def _embed_batch_with_retry(self, texts: list, task: str) -> list:
        delay = 2.0
        for attempt in range(3):
            try:
                return self._embed_batch_remote(texts, task)
            except Exception as e:  # noqa: BLE001
                transient = any(m in str(e).lower() for m in _TRANSIENT_MARKERS)
                if attempt == 2 or not transient:
                    raise
                logger.warning("Embedding request hit a transient error (%s); retrying in %.0fs", e, delay)
                time.sleep(delay)
                delay *= 2

    # -- public API ---------------------------------------------------------------

    def embed_texts(self, texts: list, task: str = "RETRIEVAL_DOCUMENT") -> list:
        """
        Embed many texts, returning one vector per input, in order. Remote
        providers get them in batched requests (far fewer API calls, so far
        less rate-limit pressure). If the provider has no key or fails, the
        WHOLE call falls back to the local embedder so every vector in the
        result has the same length.
        """
        texts = [(t or "").strip() for t in texts]
        if not texts:
            return []
        nonempty = [i for i, t in enumerate(texts) if t]

        if self.config.provider == "local":
            self._last_embed_was_local = True
            dims = self.config.dimensions
            return [_local_embed(t, dims) if t else [0.0] * dims for t in texts]

        try:
            vectors_by_index = {}
            for start in range(0, len(nonempty), _BATCH_SIZE):
                idx = nonempty[start:start + _BATCH_SIZE]
                batch = self._embed_batch_with_retry([texts[i] for i in idx], task)
                vectors_by_index.update(zip(idx, batch))
            self._last_embed_was_local = False
            logger.info("Embedded %d text(s) via %s (%s)", len(nonempty),
                        self.config.provider, self.config.model)
            return [vectors_by_index.get(i, [0.0] * self.config.dimensions) for i in range(len(texts))]
        except Exception as e:  # noqa: BLE001 - deliberate graceful degradation boundary
            self._last_embed_was_local = True
            logger.warning("Embedding provider '%s' failed (%s); falling back to local embedder.",
                           self.config.provider, e)
            dims = PROVIDER_DIMENSIONS["local"]
            return [_local_embed(t, dims) if t else [0.0] * dims for t in texts]

    def embed_text(self, text: str, task: str = "RETRIEVAL_DOCUMENT") -> list:
        """Single string -> single embedding vector (same fallback rules)."""
        return self.embed_texts([text], task=task)[0]

    def embed_query(self, text: str) -> list:
        """Embed a user's question. Same as embed_text but tells providers
        that support it (Gemini) that this is a search query."""
        return self.embed_text(text, task="RETRIEVAL_QUERY")

    @property
    def using_local_embedder(self) -> bool:
        """True if embeddings are (or would be) produced by the offline local
        embedder - because it is the configured provider, because no API key
        is set, or because the last remote call failed and fell back."""
        if self._last_embed_was_local is not None:
            return self._last_embed_was_local
        if self.config.provider == "local":
            return True
        return not self._api_key()

    @property
    def min_relevance_score(self) -> float:
        """Minimum cosine similarity for a match to count as relevant.
        Decided by what ACTUALLY produced the embedding, not by the configured
        provider name: the offline embedder's scores are much smaller in
        absolute terms than a real model's, so it needs a lower floor. Used
        by semantic_search / RAG to decide 'no relevant info'."""
        return 0.15 if self.using_local_embedder else 0.30

    def embed_long_text(self, text: str) -> list:
        """
        Chunk a long text (e.g. a full transcript) and embed each chunk.
        Returns a list of (chunk_text, vector) tuples.
        """
        chunks = chunk_text(text, self.config.max_chars_per_chunk)
        return list(zip(chunks, self.embed_texts(chunks)))


# ---------------------------------------------------------------------------
# Meeting-level chunk builder - Task 2's four required inputs:
#   Transcript sections / Summaries / Decisions / Action items
# ---------------------------------------------------------------------------

@dataclass
class MeetingChunk:
    meeting_id: int
    content_type: str     # "transcript_section" | "summary" | "decision" | "action_item"
    index: int
    text: str


def build_meeting_chunks(meeting_id: int, meeting_full: dict,
                          max_chars_per_chunk: int = DEFAULT_EMBED_CHUNK_CHARS) -> list:
    """
    Turns one meeting's full record (as returned by Database.get_meeting_full)
    into the list of MeetingChunk objects that Task 2 requires embeddings
    for: transcript sections, the summary, each decision, and each action
    item - every chunk carries its meeting_id so it can always be traced
    back to the correct meeting (Task 3's "meeting-to-vector mapping").
    """
    chunks = []
    meeting = meeting_full.get("meeting") or {}
    summary = meeting_full.get("summary") or {}
    action_items = meeting_full.get("action_items") or []

    transcript = meeting.get("transcript", "")
    for i, section in enumerate(chunk_text(transcript, max_chars_per_chunk)):
        chunks.append(MeetingChunk(meeting_id, "transcript_section", i, section))

    summary_text = summary.get("summary_text", "")
    if summary_text:
        chunks.append(MeetingChunk(meeting_id, "summary", 0, f"Meeting summary: {summary_text}"))

    for i, decision in enumerate(summary.get("decisions") or []):
        if decision:
            chunks.append(MeetingChunk(meeting_id, "decision", i, f"Decision: {decision}"))

    for i, item in enumerate(action_items):
        task = item.get("task", "")
        if not task:
            continue
        parts = [task]
        if item.get("participant_name"):
            parts.append(f"Assigned to {item['participant_name']}")
        if item.get("deadline"):
            parts.append(f"Deadline: {item['deadline']}")
        chunks.append(MeetingChunk(meeting_id, "action_item", i, "Action item: " + " - ".join(parts)))

    return chunks
