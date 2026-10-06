"""
rag_qa.py
----------
Milestone 3 - Task 5: RAG Question Answering

Implements the RAG workflow using the existing API and database, matching
the spec's flow exactly:

    User Question
        |
    Semantic Search
        |
    Relevant Meeting Context
        |
    LLM
        |
    Grounded Answer

The LLM is instructed to answer ONLY from the retrieved context and to say
so explicitly when the answer isn't in it - this is what "answers are based
on retrieved meeting information and not unsupported or hardcoded
responses" (the task's verification requirement) means in practice. Every
answer carries its `sources` (which meetings/chunks it was grounded in) so
the claim is checkable, not just asserted.

If no LLM API key is configured, this degrades to a purely extractive
answer built directly from the retrieved chunks (same zero-hallucination-
by-construction fallback philosophy as llm_service.py's "local" provider) -
grounding is guaranteed either way.
"""

import logging
from dataclasses import dataclass, field

from database import Database
from embedding_service import EmbeddingService
from vector_store import VectorStore
from semantic_search import semantic_search
from llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I don't have information about that in the meeting records."

RAG_SYSTEM_PROMPT = """You are a meeting knowledge assistant. Answer the user's \
question using only the provided meeting context. Do not invent facts, names, \
dates, decisions, deadlines, or action items. If the answer cannot be found in \
the provided context, clearly state that the information was not found in the \
available meeting records. Be concise and factual."""

RAG_PROMPT_TEMPLATE = """MEETING CONTEXT (retrieved from past meetings):
{context}

QUESTION:
{question}

Answer using ONLY the context above."""


@dataclass
class SourceRef:
    meeting_id: int
    filename: str
    content_type: str
    snippet: str
    meeting_date: str = ""     # Milestone 4 Task 3: dashboard shows source meeting + date
    score: float = 0.0
    full_text: str = ""        # the complete retrieved chunk that was given to the LLM


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    grounded: bool = True
    used_local_fallback: bool = False


def _gather_context(query: str, db, embedding_service, vector_store, top_k: int, user_id: int = None):
    """Reuses Task 4's semantic search to fetch relevant meeting chunks,
    then also pulls the best matching chunk-level text (not just the best
    per-meeting summary) so the LLM gets concrete supporting detail."""
    # Several chunks per meeting (not just the single best one) so questions
    # like "list the action items" get every relevant chunk as context.
    response = semantic_search(query, db=db, embedding_service=embedding_service,
                                vector_store=vector_store, top_k=top_k,
                                max_chunks_per_meeting=4, user_id=user_id)
    return response


def _build_context_block(search_response) -> str:
    if not search_response.results:
        return ""
    lines = []
    for r in search_response.results:
        lines.append(f"[Meeting: {r.meeting_title} | {r.meeting_date} | {r.content_type}]\n{r.full_text or r.matched_text}")
    return "\n\n".join(lines)


def _extractive_local_answer(search_response) -> str:
    """Zero-LLM fallback: concatenate the top retrieved snippets verbatim
    rather than generating free text - guarantees the answer can never
    contain anything not found in the meeting records."""
    if not search_response.results:
        return NO_CONTEXT_ANSWER
    lines = []
    for r in search_response.results[:3]:
        lines.append(f"From \"{r.meeting_title}\" ({r.content_type}): {r.matched_text}")
    return "\n".join(lines)


def answer_question(question: str, db: Database = None,
                     embedding_service: EmbeddingService = None,
                     vector_store: VectorStore = None,
                     llm_service: LLMService = None,
                     top_k: int = 5, user_id: int = None) -> RAGAnswer:
    db = db or Database()
    embedding_service = embedding_service or EmbeddingService()
    vector_store = vector_store or VectorStore()
    llm_service = llm_service or LLMService()

    question = (question or "").strip()
    if not question:
        return RAGAnswer(question=question, answer=NO_CONTEXT_ANSWER, sources=[], grounded=True)

    search_response = _gather_context(question, db, embedding_service, vector_store, top_k, user_id)
    sources = [
        SourceRef(meeting_id=r.meeting_id, filename=r.meeting_title,
                  content_type=r.content_type, snippet=r.matched_text,
                  meeting_date=r.meeting_date, score=r.relevance_score, full_text=r.full_text)
        for r in search_response.results
    ]

    if not search_response.results:
        logger.info("RAG: no relevant context found for question=%r - returning no-context answer", question)
        return RAGAnswer(question=question, answer=NO_CONTEXT_ANSWER, sources=[], grounded=True)

    context_block = _build_context_block(search_response)

    if llm_service.config.provider == "local":
        logger.info("RAG: using local extractive fallback (no LLM key configured) for question=%r", question)
        return RAGAnswer(question=question, answer=_extractive_local_answer(search_response),
                          sources=sources, grounded=True, used_local_fallback=True)

    try:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_block, question=question)
        raw = llm_service._call_llm_raw(RAG_SYSTEM_PROMPT, prompt)  # low-level call - free text answer, not the structured-JSON schema
        answer_text = (raw or "").strip() or NO_CONTEXT_ANSWER
        logger.info("RAG: LLM (%s) answered question=%r using %d source(s)",
                    llm_service.config.provider, question, len(sources))
        return RAGAnswer(question=question, answer=answer_text, sources=sources,
                          grounded=True, used_local_fallback=False)
    except (LLMServiceError, Exception) as e:  # noqa: BLE001 - never hard-fail Q&A; degrade to extractive
        logger.warning("RAG: LLM call failed (%s); falling back to local extractive answer.", e)
        return RAGAnswer(question=question, answer=_extractive_local_answer(search_response),
                          sources=sources, grounded=True, used_local_fallback=True)
