"""
test_api_client.py
-------------------
Milestone 4 Task 1: the dashboard talks to the backend ONLY through ApiClient.
Verified with a fake HTTP session (no server, no Streamlit).
"""

from api_client import ApiClient, ApiError, AuthExpired
from testutil import raises


class Resp:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code, self._payload, self.content, self.headers = status, payload, content, headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class Sess:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def test_login_stores_token_and_sends_it_on_later_calls():
    s = Sess(Resp(200, {"token": "TKN", "user": {"id": 1}}), Resp(200, []))
    c = ApiClient("http://api:8000/", session=s)
    c.login("alice", "pw")
    c.list_meetings()
    assert s.calls[0][1] == "http://api:8000/auth/login" and "Authorization" not in s.calls[0][2]["headers"]
    assert s.calls[1][2]["headers"] == {"Authorization": "Bearer TKN"}


def test_filters_drop_empty_values():
    s = Sess(Resp(200, []))
    ApiClient("http://x", token="t", session=s).list_meetings(text="launch", language="", status=None, source="zoom")
    assert s.calls[0][2]["params"] == {"text": "launch", "source": "zoom"}


def test_401_becomes_auth_expired():
    c = ApiClient("http://x", token="t", session=Sess(Resp(401, {"detail": "Session expired. Please log in again."})))
    with raises(AuthExpired) as info:
        c.me()
    assert "expired" in str(info.value)


def test_error_shapes_are_all_understood():
    cases = [(Resp(422, {"detail": {"stage": "Whisper", "message": "bad audio"}}), "bad audio", "Whisper"),
             (Resp(400, {"detail": "query must not be empty"}), "query must not be empty", None),
             (Resp(422, {"detail": [{"msg": "field required"}]}), "field required", None),
             (Resp(500, None), "HTTP 500", None)]
    for resp, text, stage in cases:
        with raises(ApiError) as info:
            ApiClient("http://x", token="t", session=Sess(resp)).overview()
        assert text in str(info.value) and info.value.stage == stage


def test_connection_failure_gives_a_friendly_error():
    c = ApiClient("http://x", token="t", session=Sess(ConnectionError("refused")))
    with raises(ApiError) as info:
        c.overview()
    assert info.value.status == 0 and "Cannot reach the backend" in str(info.value)


def test_report_download_reads_filename_from_header():
    s = Sess(Resp(200, content=b"%PDF-1.4", headers={"Content-Disposition": 'attachment; filename="meeting_3_x.pdf"'}))
    data, name = ApiClient("http://x", token="t", session=s).download_report(3, "pdf")
    assert data == b"%PDF-1.4" and name == "meeting_3_x.pdf" and s.calls[0][2]["params"] == {"format": "pdf"}


def test_upload_sends_multipart_file_and_search_ask_bodies():
    s = Sess(Resp(200, {"meeting_id": 1}))
    c = ApiClient("http://x", token="t", session=s)
    c.upload_meeting("a.wav", b"data")
    c.search("db migration", top_k=3, content_type="decision")
    c.ask("what deadline?")
    assert s.calls[0][2]["files"] == {"file": ("a.wav", b"data")}
    assert s.calls[1][2]["json"] == {"query": "db migration", "top_k": 3, "content_type": "decision"}
    assert s.calls[2][2]["json"] == {"question": "what deadline?", "top_k": 5}


def test_logout_clears_token_even_if_call_fails():
    c = ApiClient("http://x", token="t", session=Sess(ConnectionError("down")))
    try:
        c.logout()
    except ApiError:
        pass
    assert c.token is None


def test_meeting_id_is_coerced_to_int_in_paths():
    s = Sess(Resp(200, {}))
    ApiClient("http://x", token="t", session=s).meeting_details("7")
    assert s.calls[0][1].endswith("/meetings/7/details")
    with raises(ValueError):
        ApiClient("http://x", token="t", session=s).meeting_details("7/../../auth/me")
