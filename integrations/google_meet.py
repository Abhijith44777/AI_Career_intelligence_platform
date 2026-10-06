"""
integrations/google_meet.py
-----------------------------
Milestone 4, Task 5: Google Meet recording retrieval.

Google Meet saves recordings to the organiser's Google Drive, in a folder
called "Meet Recordings". This source authenticates with an OAuth **refresh
token** (scope `https://www.googleapis.com/auth/drive.readonly`), finds that
folder, and lists/downloads the video files in it. Set GOOGLE_CLIENT_ID /
GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN (see .env.example and README).
"""

import logging
from datetime import date

from config import settings
from integrations.base import (IntegrationAuthError, IntegrationError, IntegrationNotConfigured,
                               RecordingSource, RemoteRecording, check_response, stream_to_file)

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
MEET_FOLDER_NAME = "Meet Recordings"
HTTP_TIMEOUT = 30
_EXT_BY_MIME = {"video/mp4": ".mp4", "audio/mp4": ".m4a", "video/webm": ".webm", "audio/mpeg": ".mp3"}


class GoogleMeetSource(RecordingSource):
    provider = "google_meet"

    def __init__(self, client_id: str = None, client_secret: str = None, refresh_token: str = None,
                 session=None):
        self.client_id = settings.google_client_id if client_id is None else client_id
        self.client_secret = settings.google_client_secret if client_secret is None else client_secret
        self.refresh_token = settings.google_refresh_token if refresh_token is None else refresh_token
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self._token = None

    def authenticate(self) -> None:
        if not (self.client_id and self.client_secret and self.refresh_token):
            raise IntegrationNotConfigured(
                "Google Meet is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET and GOOGLE_REFRESH_TOKEN.")
        try:
            response = self.session.post(TOKEN_URL, data={
                "client_id": self.client_id, "client_secret": self.client_secret,
                "refresh_token": self.refresh_token, "grant_type": "refresh_token"}, timeout=HTTP_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            raise IntegrationError(f"Google authentication request failed: {type(e).__name__}", retryable=True)
        if response.status_code in (400, 401):
            raise IntegrationAuthError("Google rejected the credentials (refresh token revoked or client details wrong).")
        check_response(response, "Google authentication")
        token = (response.json() or {}).get("access_token")
        if not token:
            raise IntegrationAuthError("Google returned no access token.")
        self._token = token
        logger.info("Google authentication succeeded")

    def _headers(self) -> dict:
        if not self._token:
            raise IntegrationAuthError("Not authenticated with Google - call authenticate() first.")
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, url: str, params: dict, what: str, stream: bool = False):
        headers = self._headers()
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT,
                                        stream=stream)
        except Exception as e:  # noqa: BLE001
            raise IntegrationError(f"{what} failed: {type(e).__name__}", retryable=True)
        check_response(response, what)
        return response

    def _find_folder_ids(self) -> list:
        response = self._get(DRIVE_FILES_URL, {
            "q": f"mimeType='application/vnd.google-apps.folder' and name='{MEET_FOLDER_NAME}' and trashed=false",
            "fields": "files(id,name)", "pageSize": 10}, "Google Drive folder lookup")
        return [f["id"] for f in (response.json() or {}).get("files", [])]

    def list_recordings(self, since: date) -> list:
        found = []
        for folder_id in self._find_folder_ids():
            page_token = None
            while True:
                params = {
                    "q": (f"'{folder_id}' in parents and trashed=false and "
                          f"(mimeType contains 'video/' or mimeType contains 'audio/') and "
                          f"createdTime >= '{since.isoformat()}T00:00:00'"),
                    "fields": "nextPageToken,files(id,name,mimeType,size,createdTime)",
                    "pageSize": 100, "orderBy": "createdTime desc"}
                if page_token:
                    params["pageToken"] = page_token
                payload = self._get(DRIVE_FILES_URL, params, "Google Drive recording list").json() or {}
                for f in payload.get("files", []):
                    ext = _EXT_BY_MIME.get(f.get("mimeType", ""), ".mp4")
                    found.append(RemoteRecording(
                        provider="google_meet", external_id=f["id"], name=f.get("name") or "Google Meet recording",
                        start_time=f.get("createdTime", ""), size_bytes=int(f.get("size") or 0),
                        download_url=f["id"], extension=ext))
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
        logger.info("Google Meet: %d recording(s) since %s", len(found), since)
        return found

    def download(self, recording: RemoteRecording, dest_path: str) -> int:
        response = self._get(f"{DRIVE_FILES_URL}/{recording.download_url}", {"alt": "media"},
                             "Google Drive download", stream=True)
        return stream_to_file(response, dest_path, settings.max_upload_mb * 1024 * 1024)
