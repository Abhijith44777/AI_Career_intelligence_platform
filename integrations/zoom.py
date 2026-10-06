"""
integrations/zoom.py
----------------------
Milestone 4, Task 4: Zoom recording retrieval.

Uses a Zoom **Server-to-Server OAuth** app (Zoom Marketplace -> Develop ->
Build App -> Server-to-Server OAuth). Required scope:
`cloud_recording:read:list_user_recordings:admin` (or the classic
`recording:read:admin`). Set ZOOM_ACCOUNT_ID / ZOOM_CLIENT_ID /
ZOOM_CLIENT_SECRET (see .env.example).

The HTTP session is injectable so the whole class is tested offline.
"""

import logging
from datetime import date, timedelta

from config import settings
from integrations.base import (IntegrationAuthError, IntegrationError, IntegrationNotConfigured,
                               RecordingSource, RemoteRecording, check_response, stream_to_file)

logger = logging.getLogger(__name__)

TOKEN_URL = "https://zoom.us/oauth/token"
API_BASE = "https://api.zoom.us/v2"
HTTP_TIMEOUT = 30
_AUDIO_TYPES = {"M4A"}
_VIDEO_TYPES = {"MP4"}


class ZoomSource(RecordingSource):
    provider = "zoom"

    def __init__(self, account_id: str = None, client_id: str = None, client_secret: str = None,
                 user_id: str = None, session=None):
        self.account_id = settings.zoom_account_id if account_id is None else account_id
        self.client_id = settings.zoom_client_id if client_id is None else client_id
        self.client_secret = settings.zoom_client_secret if client_secret is None else client_secret
        self.zoom_user = user_id or settings.zoom_user_id or "me"
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self._token = None

    # -- Authentication ----------------------------------------------------------

    def authenticate(self) -> None:
        if not (self.account_id and self.client_id and self.client_secret):
            raise IntegrationNotConfigured(
                "Zoom is not configured. Set ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET.")
        try:
            response = self.session.post(
                TOKEN_URL, params={"grant_type": "account_credentials", "account_id": self.account_id},
                auth=(self.client_id, self.client_secret), timeout=HTTP_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - network errors of any kind
            raise IntegrationError(f"Zoom authentication request failed: {type(e).__name__}", retryable=True)
        if response.status_code in (400, 401):
            raise IntegrationAuthError("Zoom rejected the credentials (invalid account id, client id or secret).")
        check_response(response, "Zoom authentication")
        token = (response.json() or {}).get("access_token")
        if not token:
            raise IntegrationAuthError("Zoom returned no access token.")
        self._token = token
        logger.info("Zoom authentication succeeded")

    def _headers(self) -> dict:
        if not self._token:
            raise IntegrationAuthError("Not authenticated with Zoom - call authenticate() first.")
        return {"Authorization": f"Bearer {self._token}"}

    # -- Listing ----------------------------------------------------------------------

    def list_recordings(self, since: date) -> list:
        today = date.today()
        found, seen = [], set()
        # Zoom accepts at most a one-month window per request, so walk back in 30-day slices.
        window_end = today
        while window_end >= since:
            window_start = max(since, window_end - timedelta(days=29))
            for recording in self._list_window(window_start, window_end):
                if recording.external_id not in seen:
                    seen.add(recording.external_id)
                    found.append(recording)
            window_end = window_start - timedelta(days=1)
        logger.info("Zoom: %d downloadable recording(s) since %s", len(found), since)
        return found

    def _list_window(self, start: date, end: date) -> list:
        recordings, next_token = [], ""
        while True:
            params = {"from": start.isoformat(), "to": end.isoformat(), "page_size": 300}
            if next_token:
                params["next_page_token"] = next_token
            headers = self._headers()   # outside the try: "not authenticated" must not be masked as a network error
            try:
                response = self.session.get(f"{API_BASE}/users/{self.zoom_user}/recordings",
                                            headers=headers, params=params, timeout=HTTP_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                raise IntegrationError(f"Zoom recording list failed: {type(e).__name__}", retryable=True)
            check_response(response, "Zoom recording list")
            payload = response.json() or {}
            for meeting in payload.get("meetings") or []:
                chosen = self._pick_file(meeting)
                if chosen:
                    recordings.append(chosen)
            next_token = payload.get("next_page_token") or ""
            if not next_token:
                return recordings

    @staticmethod
    def _pick_file(meeting: dict):
        """One file per Zoom meeting: prefer audio-only (smallest, all Whisper needs),
        else the MP4. Files still being processed by Zoom are ignored."""
        files = [f for f in (meeting.get("recording_files") or [])
                 if (f.get("status") or "completed").lower() == "completed" and f.get("download_url")]
        audio = [f for f in files if (f.get("file_type") or "").upper() in _AUDIO_TYPES]
        video = [f for f in files if (f.get("file_type") or "").upper() in _VIDEO_TYPES]
        chosen = (audio or video or [None])[0]
        if chosen is None:
            return None
        ext = "." + (chosen.get("file_extension") or chosen.get("file_type") or "mp4").lower().lstrip(".")
        external_id = chosen.get("id") or f"{meeting.get('uuid', '')}:{chosen.get('file_type', '')}"
        return RemoteRecording(
            provider="zoom", external_id=str(external_id), name=meeting.get("topic") or "Zoom meeting",
            start_time=meeting.get("start_time") or "", size_bytes=int(chosen.get("file_size") or 0),
            download_url=chosen["download_url"], extension=ext)

    # -- Download -----------------------------------------------------------------------

    def download(self, recording: RemoteRecording, dest_path: str) -> int:
        headers = self._headers()
        try:
            # Token goes in the Authorization header, never in the URL (URLs end up in logs).
            response = self.session.get(recording.download_url, headers=headers, stream=True,
                                        timeout=HTTP_TIMEOUT, allow_redirects=True)
        except Exception as e:  # noqa: BLE001
            raise IntegrationError(f"Zoom download failed: {type(e).__name__}", retryable=True)
        check_response(response, "Zoom download")
        return stream_to_file(response, dest_path, settings.max_upload_mb * 1024 * 1024)
