"""
integrations/base.py
----------------------
Shared types for recording sources (Zoom, Google Meet).

A source only knows how to (1) authenticate, (2) list recordings, (3) download
one. Everything after the download - Whisper, LLM, storage, embeddings,
duplicate handling, failure handling - lives in integrations/importer.py and
is identical for every provider, which is what guarantees "Google Meet
recordings follow the same application workflow as uploaded meetings".
"""

from dataclasses import dataclass
from datetime import date


class IntegrationError(Exception):
    """A provider call failed. `retryable` tells the importer whether trying again may help."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class IntegrationAuthError(IntegrationError):
    """Credentials missing/invalid/expired - the whole sync cannot proceed."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class IntegrationNotConfigured(IntegrationAuthError):
    pass


@dataclass
class RemoteRecording:
    provider: str
    external_id: str        # provider-unique id - the duplicate-detection key
    name: str               # human title (meeting topic / Drive file name)
    start_time: str
    size_bytes: int
    download_url: str       # provider-specific locator passed back to download()
    extension: str          # ".mp4" / ".m4a"


class RecordingSource:
    """Interface implemented by ZoomSource and GoogleMeetSource."""
    provider = "unknown"

    def authenticate(self) -> None:
        raise NotImplementedError

    def list_recordings(self, since: date) -> list:
        raise NotImplementedError

    def download(self, recording: RemoteRecording, dest_path: str) -> int:
        """Streams the recording to `dest_path`; returns bytes written."""
        raise NotImplementedError


def check_response(response, what: str):
    """Maps an HTTP response to the right exception type (shared by both providers)."""
    status = response.status_code
    if status < 400:
        return
    if status in (401, 403):
        raise IntegrationAuthError(f"{what}: authentication failed (HTTP {status}). Check the credentials/scopes.")
    if status == 404:
        raise IntegrationError(f"{what}: not found (HTTP 404).", retryable=False)
    if status == 429 or status >= 500:
        raise IntegrationError(f"{what}: provider temporarily unavailable (HTTP {status}).", retryable=True)
    raise IntegrationError(f"{what}: request rejected (HTTP {status}).", retryable=False)


def stream_to_file(response, dest_path: str, max_bytes: int, chunk_size: int = 1024 * 1024) -> int:
    """Writes a streamed HTTP response to disk, aborting (and deleting the
    partial file) if it exceeds `max_bytes` - a hostile/buggy provider can't fill the disk."""
    import os
    written = 0
    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise IntegrationError(f"Recording exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                                           retryable=False)
                f.write(chunk)
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    return written
