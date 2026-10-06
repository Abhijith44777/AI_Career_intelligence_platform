"""
testutil.py
------------
Shared helpers for the Milestone 4 tests. Everything runs OFFLINE: temporary
SQLite + local vector store, the local (keyword) embedder, the local rule-based
LLM engine, and a fake Whisper - no API keys, ffmpeg or network needed.
"""

import contextlib
import os
import tempfile

import pipeline
from auth import AuthService
from database import Database
from embedding_service import EmbeddingConfig, EmbeddingService
from llm_service import LLMConfig, LLMService
from services import AppServices
from vector_store import VectorStore

SAMPLE_TRANSCRIPT = (
    "Hi everyone, thanks for joining the mobile application launch meeting. "
    "Ravi will complete the API integration by Friday. "
    "Priya will prepare the UI testing report by next Wednesday. "
    "We decided to continue with the planned mobile application launch. "
    "Karan should review the database migration plan as soon as possible. "
    "The team agreed that the launch date stays on the fifteenth."
)
OTHER_TRANSCRIPT = (
    "Welcome to the finance planning call. "
    "Meera will send the quarterly budget forecast by Monday. "
    "We decided to freeze hiring until the audit finishes. "
    "Dev is going to negotiate the vendor contract next Tuesday."
)

# 2 KB of bytes that pass the magic-number check for .wav
FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 4096


@contextlib.contextmanager
def fake_whisper(transcript: str = SAMPLE_TRANSCRIPT, duration: float = 300.0, language: str = "en"):
    """Replaces the ffmpeg/Whisper stages of pipeline.py so the REST of the real
    pipeline (validation, LLM stage, action items, participants, database,
    embeddings, vector store) runs unchanged."""
    saved = (pipeline.process_audio, pipeline.get_audio_duration, pipeline.run_whisper)
    pipeline.process_audio = lambda raw, tmp: raw
    pipeline.get_audio_duration = lambda path: duration
    pipeline.run_whisper = lambda path, model_size="base", language=None: {
        "text": transcript, "language": language_default(language), "segments": []}
    try:
        yield
    finally:
        pipeline.process_audio, pipeline.get_audio_duration, pipeline.run_whisper = saved


def language_default(language):
    return language or "en"


class Env:
    """A complete, isolated application stack."""

    def __init__(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(db_path)
        fd, vec_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(vec_path)
        self.db = Database(db_path=db_path)
        self.vectors = VectorStore(backend="local", path=vec_path)
        self.embedder = EmbeddingService(EmbeddingConfig(provider="local"))
        self.services = AppServices(
            db=self.db, embedding_service=self.embedder, vector_store=self.vectors,
            llm_factory=lambda: LLMService(LLMConfig(provider="local")))
        self.auth = AuthService(self.db, max_failed_logins=3, lockout_minutes=10)
        self.transcripts_dir = tempfile.mkdtemp(prefix="transcripts_")

    def user(self, name: str, password: str = "correct-horse-9") -> dict:
        created = self.auth.register(name, f"{name.lower()}@example.com", password)
        return created

    def upload(self, user_id: int, transcript: str = SAMPLE_TRANSCRIPT, filename: str = "sync.wav",
               duration: float = 300.0) -> dict:
        from config import settings
        settings.transcripts_dir = self.transcripts_dir
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, filename)
        with open(path, "wb") as f:
            f.write(FAKE_WAV)
        with fake_whisper(transcript, duration):
            return self.services.process_upload(user_id, path, filename, os.path.getsize(path))


def make_client(env: Env):
    """A requests-style test client for the real FastAPI app when FastAPI is
    installed, else None (the HTTP tests then skip themselves)."""
    try:
        from fastapi.testclient import TestClient
        import api
    except ImportError:
        return None
    return TestClient(api.create_app(db=env.db, services=env.services, auth=env.auth),
                      raise_server_exceptions=False)


class raises:
    """`with raises(SomeError) as info:` - pytest-free replacement for pytest.raises."""

    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but nothing was raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc
        return True
