"""
services.py
------------
Milestone 4: the application service layer.

Every operation the API (and therefore the dashboard) can perform on a
meeting goes through `AppServices`, and every method takes the
authenticated `user_id` first. Ownership is enforced HERE, once, so no
endpoint can forget it:

  * a meeting that doesn't exist and a meeting owned by someone else both
    raise `NotFound` (the caller can't tell them apart - no existence leak);
  * search / RAG / vectors are filtered to the caller's own meetings;
  * reports, analytics and integrations only ever see the caller's data.

api.py is a thin HTTP wrapper over this class; tests exercise it directly.
"""

import logging
from datetime import date, datetime, timedelta

from analytics import meeting_analytics, overview_analytics
from database import Database
from embedding_service import EmbeddingService
from integrations.base import IntegrationAuthError, IntegrationError, IntegrationNotConfigured
from integrations.importer import import_recordings
from knowledge_repository import index_meeting, reindex_all_meetings, verify_meeting_retrieval
from llm_service import LLMConfig, LLMService
from pipeline import PipelineError, process_meeting_recording
from rag_qa import answer_question
from reports import report_filename, report_to_csv, report_to_pdf
from semantic_search import semantic_search
from vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 1000
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20
VALID_CONTENT_TYPES = {"transcript_section", "summary", "decision", "action_item"}
VALID_STATUSES = {"processing", "completed", "failed"}
VALID_SOURCES = {"upload", "zoom", "google_meet"}
REPORT_FORMATS = {"pdf": "application/pdf", "csv": "text/csv; charset=utf-8"}


class ServiceError(Exception):
    """Base for errors the API maps to HTTP status codes."""


class NotFound(ServiceError):
    pass


class BadRequest(ServiceError):
    pass


class UpstreamError(ServiceError):
    """A dependency (LLM, embeddings, provider) failed."""


def _parse_date(value, name: str):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise BadRequest(f"{name} must be a date in YYYY-MM-DD format.")


class AppServices:
    def __init__(self, db: Database = None, embedding_service: EmbeddingService = None,
                 vector_store: VectorStore = None, llm_factory=None, processor=None,
                 source_factories: dict = None):
        self.db = db or Database()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.llm_factory = llm_factory or (lambda: LLMService(LLMConfig()))
        self.processor = processor or process_meeting_recording
        self._source_factories = source_factories  # None -> real Zoom/Meet sources, built lazily

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def _owned(self, user_id: int, meeting_id: int) -> dict:
        meeting = self.db.get_meeting_for_user(meeting_id, user_id)
        if meeting is None:
            raise NotFound(f"Meeting {meeting_id} not found")
        return meeting

    # ------------------------------------------------------------------
    # Upload -> full pipeline
    # ------------------------------------------------------------------

    def process_upload(self, user_id: int, raw_path: str, filename: str, size_bytes: int,
                       whisper_model_size: str = None) -> dict:
        from config import settings
        whisper_model_size = whisper_model_size or settings.whisper_model
        try:
            result = self.processor(
                raw_file_path=raw_path, original_filename=filename, file_size_bytes=size_bytes,
                db=self.db, llm_service=self.llm_factory(), whisper_model_size=whisper_model_size,
                embedding_service=self.embedding_service, vector_store=self.vector_store,
                user_id=user_id, source="upload")
        except PipelineError:
            raise
        return self._pipeline_result_to_dict(result)

    @staticmethod
    def _pipeline_result_to_dict(result) -> dict:
        return {
            "meeting_id": result.meeting_id,
            "language": result.language or "unknown",
            "audio_duration_seconds": result.audio_duration_seconds or 0.0,
            "summary": result.summary.summary,
            "key_points": result.summary.key_points,
            "decisions": result.summary.decisions,
            "priorities": result.summary.priorities,
            "action_items": [
                {"task": e["action_item"].task, "assignee": e["participant"].name,
                 "deadline": e["action_item"].deadline, "priority": e["action_item"].priority,
                 "status": e["action_item"].status}
                for e in result.action_items],
            "participants": [p.name for p in result.participants],
        }

    # ------------------------------------------------------------------
    # Reading meetings
    # ------------------------------------------------------------------

    def list_meetings(self, user_id: int, text: str = None, language: str = None, status: str = None,
                      source: str = None, date_from: str = None, date_to: str = None,
                      participant: str = None) -> list:
        if status and status not in VALID_STATUSES:
            raise BadRequest(f"status must be one of {sorted(VALID_STATUSES)}")
        if source and source not in VALID_SOURCES:
            raise BadRequest(f"source must be one of {sorted(VALID_SOURCES)}")
        return self.db.search_meetings(
            user_id, text=(text or "").strip()[:200] or None, language=language or None,
            status=status or None, source=source or None,
            date_from=_parse_date(date_from, "date_from"), date_to=_parse_date(date_to, "date_to"),
            participant=(participant or "").strip() or None)

    def get_meeting(self, user_id: int, meeting_id: int) -> dict:
        """Legacy shape (meeting + summary + action_items) used by GET /meetings/{id}."""
        self._owned(user_id, meeting_id)
        return self.db.get_meeting_full(meeting_id)

    def get_transcript(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return {"meeting_id": meeting_id, "transcript": self.db.get_transcript(meeting_id)}

    def get_summary(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        summary = self.db.get_summary(meeting_id)
        if not summary:
            return {"meeting_id": meeting_id, "summary_text": None, "key_points": [], "priorities": []}
        return {"meeting_id": meeting_id, "summary_text": summary["summary_text"],
                "key_points": summary["key_points"], "priorities": summary["priorities"]}

    def get_decisions(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return {"meeting_id": meeting_id, "decisions": self.db.get_decisions(meeting_id)}

    def get_action_items(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return {"meeting_id": meeting_id, "action_items": self.db.get_action_items(meeting_id)}

    def get_participants(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return {"meeting_id": meeting_id, "participants": self.db.get_participants_for_meeting(meeting_id)}

    def get_deadlines(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return {"meeting_id": meeting_id, "deadlines": self.db.get_deadlines_for_meeting(meeting_id)}

    def verify_meeting(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        return verify_meeting_retrieval(self.db, meeting_id)

    def get_details(self, user_id: int, meeting_id: int) -> dict:
        """Milestone 4 Task 2: the complete meeting-details payload, built from
        one ownership-checked meeting id so nothing from another meeting can appear."""
        meeting = self._owned(user_id, meeting_id)
        summary = self.db.get_summary(meeting_id) or {}
        action_items = self.db.get_action_items(meeting_id)
        participants = self.db.get_participants_for_meeting(meeting_id)
        deadlines = self.db.get_deadlines_for_meeting(meeting_id)
        return {
            "meeting": {k: meeting.get(k) for k in (
                "id", "filename", "language", "audio_duration_seconds", "created_at", "status",
                "error_message", "source")},
            "transcript": meeting["transcript"],
            "summary_text": summary.get("summary_text"),
            "key_points": summary.get("key_points", []),
            "decisions": summary.get("decisions", []),
            "priorities": summary.get("priorities", []),
            "action_items": action_items,
            "participants": participants,
            "deadlines": deadlines,
            "analytics": meeting_analytics(meeting, summary, action_items, participants, deadlines),
        }

    def get_overview(self, user_id: int) -> dict:
        meetings = self.db.search_meetings(user_id)
        return overview_analytics(meetings, self.db.get_action_items_for_user(user_id))

    def list_participants(self, user_id: int) -> list:
        return self.db.list_participants(user_id=user_id)

    def delete_meeting(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        self.vector_store.delete_by_meeting(meeting_id)
        self.db.delete_meeting(meeting_id, user_id)
        return {"meeting_id": meeting_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # Vectors / embeddings
    # ------------------------------------------------------------------

    def get_vectors(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        records = self.vector_store.get_by_meeting(meeting_id)
        return {
            "meeting_id": meeting_id, "vector_count": len(records),
            "vectors": [{"vector_id": r.id, "content_type": r.metadata.get("content_type"),
                         "source_id": r.metadata.get("source_id"), "text": r.document[:200]} for r in records]}

    def generate_embeddings(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        try:
            count = index_meeting(self.db, meeting_id, self.embedding_service, self.vector_store)
        except Exception as e:  # noqa: BLE001
            logger.exception("Embedding generation failed for meeting_id=%s", meeting_id)
            raise UpstreamError("Embedding generation failed. See server logs for details.") from e
        return {"meeting_id": meeting_id, "chunks_indexed": count}

    def sync_embeddings(self, user_id: int) -> dict:
        try:
            results = reindex_all_meetings(self.db, self.embedding_service, self.vector_store, user_id=user_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("Embedding sync failed")
            raise UpstreamError("Embedding sync failed. See server logs for details.") from e
        return {"meetings_synced": len(results), "chunks_per_meeting": results}

    def delete_embeddings(self, user_id: int, meeting_id: int) -> dict:
        self._owned(user_id, meeting_id)
        self.vector_store.delete_by_meeting(meeting_id)
        return {"meeting_id": meeting_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # Search & RAG
    # ------------------------------------------------------------------

    def search(self, user_id: int, query: str, top_k: int = 5, content_type: str = None,
               meeting_id: int = None) -> dict:
        query = (query or "").strip()
        if not query:
            raise BadRequest("query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise BadRequest(f"query must be at most {MAX_QUERY_CHARS} characters")
        if content_type and content_type not in VALID_CONTENT_TYPES:
            raise BadRequest(f"content_type must be one of {sorted(VALID_CONTENT_TYPES)}")
        top_k = max(1, min(int(top_k or 5), MAX_TOP_K))
        response = semantic_search(query, db=self.db, embedding_service=self.embedding_service,
                                   vector_store=self.vector_store, top_k=top_k,
                                   content_type_filter=content_type, meeting_id_filter=meeting_id,
                                   user_id=user_id)
        return {
            "query": response.query,
            "results": [{"meeting_id": r.meeting_id, "meeting_title": r.meeting_title,
                         "meeting_date": r.meeting_date, "content_type": r.content_type,
                         "relevance_score": r.relevance_score, "matched_text": r.matched_text}
                        for r in response.results],
            "search_completed_in_seconds": round(response.elapsed_seconds, 3),
            "within_latency_target": response.within_latency_target,
        }

    def ask(self, user_id: int, question: str, top_k: int = 5) -> dict:
        question = (question or "").strip()
        if not question:
            raise BadRequest("question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise BadRequest(f"question must be at most {MAX_QUESTION_CHARS} characters")
        top_k = max(1, min(int(top_k or 5), MAX_TOP_K))
        result = answer_question(question, db=self.db, embedding_service=self.embedding_service,
                                 vector_store=self.vector_store, llm_service=self.llm_factory(),
                                 top_k=top_k, user_id=user_id)
        return {
            "question": result.question, "answer": result.answer,
            "sources": [{"meeting_id": s.meeting_id, "meeting_title": s.filename, "meeting_date": s.meeting_date,
                         "content_type": s.content_type, "relevance_score": s.score,
                         "relevant_content": s.snippet, "retrieved_context": s.full_text or s.snippet}
                        for s in result.sources],
            "grounded": result.grounded, "used_local_fallback": result.used_local_fallback,
        }

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def build_report(self, user_id: int, meeting_id: int) -> dict:
        details = self.get_details(user_id, meeting_id)
        return {
            "meeting": details["meeting"], "summary_text": details["summary_text"],
            "key_points": details["key_points"], "decisions": details["decisions"],
            "action_items": details["action_items"],
            "participants": [p["name"] for p in details["participants"]],
            "deadlines": details["deadlines"],
        }

    def export_report(self, user_id: int, meeting_id: int, fmt: str) -> tuple:
        """Returns (bytes, media_type, filename)."""
        fmt = (fmt or "").lower()
        if fmt not in REPORT_FORMATS:
            raise BadRequest("format must be 'pdf' or 'csv'")
        report = self.build_report(user_id, meeting_id)
        data = report_to_pdf(report) if fmt == "pdf" else report_to_csv(report)
        return data, REPORT_FORMATS[fmt], report_filename(report, fmt)

    # ------------------------------------------------------------------
    # Integrations (Zoom / Google Meet)
    # ------------------------------------------------------------------

    def _make_source(self, provider: str):
        if self._source_factories is not None:
            factory = self._source_factories.get(provider)
        else:
            from integrations.google_meet import GoogleMeetSource
            from integrations.zoom import ZoomSource
            factory = {"zoom": ZoomSource, "google_meet": GoogleMeetSource}.get(provider)
        if factory is None:
            raise BadRequest("provider must be 'zoom' or 'google_meet'")
        return factory()

    def integration_status(self, user_id: int) -> dict:
        from config import settings
        return {
            "zoom": {"configured": settings.zoom_configured},
            "google_meet": {"configured": settings.google_configured},
            "recent_recordings": self.db.list_recordings(user_id)[:50],
        }

    def sync_recordings(self, user_id: int, provider: str, days: int = None) -> dict:
        source = self._make_source(provider)
        since = date.today() - timedelta(days=max(1, min(int(days), 90))) if days else None

        def process(raw_path, filename, size, user_id, source):
            from config import settings
            return self.processor(
                raw_file_path=raw_path, original_filename=filename, file_size_bytes=size,
                db=self.db, llm_service=self.llm_factory(), whisper_model_size=settings.whisper_model,
                embedding_service=self.embedding_service, vector_store=self.vector_store,
                user_id=user_id, source=source)

        try:
            report = import_recordings(source, self.db, user_id, process, since=since)
        except IntegrationNotConfigured as e:
            raise BadRequest(str(e)) from e
        except IntegrationAuthError as e:
            raise UpstreamError(str(e)) from e
        except IntegrationError as e:
            raise UpstreamError(str(e)) from e
        return report.to_dict()
