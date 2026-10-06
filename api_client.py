"""
api_client.py
--------------
Milestone 4, Task 1: the Streamlit dashboard's connection to the backend.

The dashboard never touches the database directly - every screen reads and
writes through the FastAPI backend with the logged-in user's bearer token, so
the API is the single place where authentication and access control are
enforced. This module is UI-free (no Streamlit import) so it is unit-tested.
"""

import logging

from config import settings

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """A backend call failed. `status` is the HTTP status (0 = could not connect)."""

    def __init__(self, status: int, message: str, stage: str = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.stage = stage


class AuthExpired(ApiError):
    """HTTP 401 - the token is missing, expired or revoked; the user must log in again."""


def _extract_message(response) -> tuple:
    """(message, stage) from an error response, whatever shape the backend used."""
    try:
        detail = response.json().get("detail")
    except Exception:  # noqa: BLE001 - non-JSON error body
        return (f"Request failed (HTTP {response.status_code}).", None)
    if isinstance(detail, dict):
        return (detail.get("message") or str(detail), detail.get("stage"))
    if isinstance(detail, list):  # FastAPI validation errors
        return ("; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail), None)
    return (str(detail) if detail else f"Request failed (HTTP {response.status_code}).", None)


class ApiClient:
    def __init__(self, base_url: str = None, token: str = None, session=None, timeout: int = None):
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self.token = token
        self.timeout = timeout or settings.request_timeout_seconds
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    # -- plumbing ---------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, raw: bool = False, **kwargs):
        try:
            response = self.session.request(method, self.base_url + path, headers=self._headers(),
                                            timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except Exception as e:  # noqa: BLE001 - connection refused, DNS, timeout...
            logger.warning("API call %s %s failed: %s", method, path, type(e).__name__)
            raise ApiError(0, f"Cannot reach the backend at {self.base_url}. Is the API running?") from e
        if response.status_code >= 400:
            message, stage = _extract_message(response)
            if response.status_code == 401:
                raise AuthExpired(401, message, stage)
            raise ApiError(response.status_code, message, stage)
        return response if raw else response.json()

    # -- auth ---------------------------------------------------------------------

    def register(self, username: str, email: str, password: str) -> dict:
        return self._request("POST", "/auth/register",
                             json={"username": username, "email": email, "password": password})

    def login(self, username: str, password: str) -> dict:
        data = self._request("POST", "/auth/login", json={"username": username, "password": password})
        self.token = data["token"]
        return data

    def logout(self) -> None:
        try:
            self._request("POST", "/auth/logout")
        finally:
            self.token = None

    def me(self) -> dict:
        return self._request("GET", "/auth/me")

    def change_password(self, old_password: str, new_password: str) -> dict:
        return self._request("POST", "/auth/change-password",
                             json={"old_password": old_password, "new_password": new_password})

    # -- meetings ---------------------------------------------------------------------

    def upload_meeting(self, filename: str, data: bytes) -> dict:
        return self._request("POST", "/process-meeting", files={"file": (filename, data)})

    def list_meetings(self, **filters) -> list:
        params = {k: v for k, v in filters.items() if v not in (None, "")}
        return self._request("GET", "/meetings", params=params)

    def meeting_details(self, meeting_id: int) -> dict:
        return self._request("GET", f"/meetings/{int(meeting_id)}/details")

    def delete_meeting(self, meeting_id: int) -> dict:
        return self._request("DELETE", f"/meetings/{int(meeting_id)}")

    def participants(self) -> list:
        return self._request("GET", "/participants")

    def overview(self) -> dict:
        return self._request("GET", "/analytics/overview")

    def download_report(self, meeting_id: int, fmt: str) -> tuple:
        """-> (bytes, filename)"""
        response = self._request("GET", f"/meetings/{int(meeting_id)}/report", raw=True, params={"format": fmt})
        disposition = response.headers.get("Content-Disposition", "")
        filename = disposition.split('filename="')[-1].rstrip('"') if 'filename="' in disposition \
            else f"meeting_{meeting_id}.{fmt}"
        return response.content, filename

    # -- AI --------------------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, content_type: str = None) -> dict:
        body = {"query": query, "top_k": top_k}
        if content_type:
            body["content_type"] = content_type
        return self._request("POST", "/search/meetings", json=body)

    def ask(self, question: str, top_k: int = 5) -> dict:
        return self._request("POST", "/ask/meetings", json={"question": question, "top_k": top_k})

    def sync_embeddings(self) -> dict:
        return self._request("POST", "/embeddings/sync")

    # -- integrations -----------------------------------------------------------------------

    def integration_status(self) -> dict:
        return self._request("GET", "/integrations/status")

    def sync_recordings(self, provider: str, days: int = None) -> dict:
        return self._request("POST", f"/integrations/{provider}/sync", json={"days": days} if days else {})
