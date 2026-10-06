"""
api.py
-------
FastAPI backend for the AI-Powered Career Intelligence Platform.

    Milestone 1  audio processing & transcription
    Milestone 2  summarization, action items, participants, persistence
    Milestone 3  knowledge repository, embeddings, vector search, RAG
    Milestone 4  authentication & per-user access control, meeting details &
                 analytics, PDF/CSV reports, Zoom / Google Meet import,
                 production hardening

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Every endpoint except /health*, /auth/register and /auth/login requires
    Authorization: Bearer <token>        (token from POST /auth/login)
and only ever sees the calling user's own meetings. This module is a thin
HTTP wrapper: all business rules live in services.py / auth.py.

Endpoints
---------
Health / auth
    GET  /health                       liveness
    GET  /health/ready                 readiness (database + vector store)
    POST /auth/register                create an account
    POST /auth/login                   -> bearer token
    POST /auth/logout                  revoke the current token
    GET  /auth/me                      current user
    POST /auth/change-password
Meetings
    POST /process-meeting              upload -> Whisper -> LLM -> DB -> embeddings
    GET  /meetings                     list (filters: text, language, status, source,
                                       date_from, date_to, participant)
    GET  /meetings/{id}                meeting + summary + action items
    GET  /meetings/{id}/details        everything for the details page + analytics
    GET  /meetings/{id}/transcript | summary | decisions | action-items |
                        participants | deadlines | verify | vectors
    GET  /meetings/{id}/report?format=pdf|csv
    DELETE /meetings/{id}
    GET  /participants                 participants across the caller's meetings
    GET  /analytics/overview           dashboard totals
Knowledge / AI
    POST /embeddings/generate | /embeddings/sync      DELETE /embeddings/{id}
    POST /search/meetings              semantic search
    POST /ask/meetings                 RAG question answering
Integrations
    GET  /integrations/status
    POST /integrations/{zoom|google_meet}/sync
"""

import logging
import os
import tempfile
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from auth import AccountLocked, AuthError, AuthService, RegistrationError
from config import settings
from database import Database
from logging_config import capture_exception, request_id_var, setup_logging
from pipeline import PipelineError, safe_filename
from services import AppServices, BadRequest, NotFound, ServiceError, UpstreamError
from upload_security import UploadTooLarge, looks_like_media, save_upload_limited
from utils import validate_file

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request bodies (length limits reject oversized/abusive payloads early)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(max_length=64)
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(max_length=254, description="username or email")
    password: str = Field(max_length=256)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class EmbedRequest(BaseModel):
    meeting_id: int


class SearchRequest(BaseModel):
    query: str = Field(max_length=2000)
    top_k: int = 5
    content_type: str = None
    meeting_id: int = None


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    top_k: int = 5


class SyncRequest(BaseModel):
    days: int = None


def _bearer_token(authorization: str) -> str:
    if not authorization:
        return ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def create_app(db: Database = None, services: AppServices = None, auth: AuthService = None) -> FastAPI:
    problems = settings.validate_for_production()
    if problems:
        raise RuntimeError("Refusing to start with insecure production configuration:\n  - "
                           + "\n  - ".join(problems))

    db = db or (services.db if services else Database())
    services = services or AppServices(db=db)
    auth = auth or AuthService(db)
    db.delete_expired_sessions()

    app = FastAPI(
        title="AI-Powered Career Intelligence Platform - API",
        description="Meeting transcription, summarization, action items, semantic search & RAG, "
                    "with per-user access control.",
        version="2.0.0",
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.services, app.state.auth, app.state.db = services, auth, db

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"],
    )

    # -- Cross-cutting: request id, security headers, error handling -----------

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        status = 404 if isinstance(exc, NotFound) else 400 if isinstance(exc, BadRequest) \
            else 502 if isinstance(exc, UpstreamError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, exc: AuthError):
        if isinstance(exc, AccountLocked):
            return JSONResponse(status_code=429, content={"detail": str(exc)})
        if isinstance(exc, RegistrationError):
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return JSONResponse(status_code=401, content={"detail": str(exc)},
                            headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError):
        logger.error("Pipeline failed at stage=%s: %s", exc.stage, exc)
        return JSONResponse(status_code=422, content={"detail": {"stage": exc.stage, "message": str(exc)}})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # Full detail goes to the log / error monitor; the client only gets a generic message.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        capture_exception(exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    # -- Dependencies ------------------------------------------------------------

    def current_user(authorization: str = Header(default=None)) -> dict:
        return auth.authenticate(_bearer_token(authorization))

    # -- Health -----------------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        checks = {}
        try:
            db.list_meetings(user_id=-1)
            checks["database"] = "ok"
        except Exception as e:  # noqa: BLE001
            logger.error("Readiness: database check failed: %s", e)
            checks["database"] = "error"
        try:
            services.vector_store.count()
            checks["vector_store"] = "ok"
        except Exception as e:  # noqa: BLE001
            logger.error("Readiness: vector store check failed: %s", e)
            checks["vector_store"] = "error"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "degraded", **checks})

    # -- Auth ---------------------------------------------------------------------------

    @app.post("/auth/register", status_code=201)
    def register(body: RegisterRequest):
        return auth.register(body.username, body.email, body.password)

    @app.post("/auth/login")
    def login(body: LoginRequest):
        return auth.login(body.username, body.password)

    @app.post("/auth/logout")
    def logout(authorization: str = Header(default=None), user: dict = Depends(current_user)):
        auth.logout(_bearer_token(authorization))
        return {"status": "logged_out"}

    @app.get("/auth/me")
    def me(user: dict = Depends(current_user)):
        return user

    @app.post("/auth/change-password")
    def change_password(body: ChangePasswordRequest, user: dict = Depends(current_user)):
        auth.change_password(user["id"], body.old_password, body.new_password)
        return {"status": "password_changed", "detail": "All sessions were signed out. Please log in again."}

    # -- Upload -> pipeline ------------------------------------------------------------------

    @app.post("/process-meeting")
    def process_meeting(file: UploadFile = File(...), user: dict = Depends(current_user)):
        """Plain `def` (not async): the Whisper/LLM work is blocking and FastAPI runs
        sync endpoints in a worker thread, so one long upload never freezes other users."""
        filename = safe_filename(file.filename or "")
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = os.path.join(tmp_dir, filename)
            try:
                size = save_upload_limited(file.file, raw_path, settings.max_upload_mb * 1024 * 1024)
            except UploadTooLarge as e:
                raise HTTPException(status_code=413, detail=str(e))

            check = validate_file(filename, size)
            if not check["valid"]:
                raise HTTPException(status_code=422, detail={"stage": "Upload Validation", "message": check["message"]})
            if not looks_like_media(raw_path):
                raise HTTPException(status_code=422, detail={
                    "stage": "Upload Validation",
                    "message": "The file content does not match its extension - is it really an audio/video file?"})
            return services.process_upload(user["id"], raw_path, filename, size)

    # -- Meetings ----------------------------------------------------------------------------

    @app.get("/meetings")
    def list_meetings(text: str = None, language: str = None, status: str = None, source: str = None,
                      date_from: str = None, date_to: str = None, participant: str = None,
                      user: dict = Depends(current_user)):
        return services.list_meetings(user["id"], text, language, status, source, date_from, date_to, participant)

    @app.get("/meetings/{meeting_id}")
    def get_meeting(meeting_id: int, user: dict = Depends(current_user)):
        return JSONResponse(content=services.get_meeting(user["id"], meeting_id))

    @app.get("/meetings/{meeting_id}/details")
    def get_details(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_details(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/transcript")
    def get_transcript(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_transcript(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/summary")
    def get_summary(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_summary(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/decisions")
    def get_decisions(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_decisions(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/action-items")
    def get_action_items(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_action_items(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/participants")
    def get_participants(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_participants(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/deadlines")
    def get_deadlines(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_deadlines(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/verify")
    def verify_meeting(meeting_id: int, user: dict = Depends(current_user)):
        return services.verify_meeting(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/vectors")
    def get_vectors(meeting_id: int, user: dict = Depends(current_user)):
        return services.get_vectors(user["id"], meeting_id)

    @app.get("/meetings/{meeting_id}/report")
    def get_report(meeting_id: int, format: str = Query("pdf"), user: dict = Depends(current_user)):
        data, media_type, filename = services.export_report(user["id"], meeting_id, format)
        return Response(content=data, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.delete("/meetings/{meeting_id}")
    def delete_meeting(meeting_id: int, user: dict = Depends(current_user)):
        return services.delete_meeting(user["id"], meeting_id)

    @app.get("/participants")
    def list_participants(user: dict = Depends(current_user)):
        return services.list_participants(user["id"])

    @app.get("/analytics/overview")
    def analytics_overview(user: dict = Depends(current_user)):
        return services.get_overview(user["id"])

    # -- Embeddings / search / RAG ------------------------------------------------------------------

    @app.post("/embeddings/generate")
    def generate_embeddings(body: EmbedRequest, user: dict = Depends(current_user)):
        return services.generate_embeddings(user["id"], body.meeting_id)

    @app.post("/embeddings/sync")
    def sync_embeddings(user: dict = Depends(current_user)):
        return services.sync_embeddings(user["id"])

    @app.delete("/embeddings/{meeting_id}")
    def delete_embeddings(meeting_id: int, user: dict = Depends(current_user)):
        return services.delete_embeddings(user["id"], meeting_id)

    @app.post("/search/meetings")
    def search_meetings(body: SearchRequest, user: dict = Depends(current_user)):
        return services.search(user["id"], body.query, body.top_k, body.content_type, body.meeting_id)

    @app.post("/ask/meetings")
    def ask_meetings(body: AskRequest, user: dict = Depends(current_user)):
        return services.ask(user["id"], body.question, body.top_k)

    # -- Integrations ----------------------------------------------------------------------------------

    @app.get("/integrations/status")
    def integrations_status(user: dict = Depends(current_user)):
        return services.integration_status(user["id"])

    @app.post("/integrations/{provider}/sync")
    def integrations_sync(provider: str, body: SyncRequest = None, user: dict = Depends(current_user)):
        return services.sync_recordings(user["id"], provider, days=body.days if body else None)

    return app


app = create_app()
