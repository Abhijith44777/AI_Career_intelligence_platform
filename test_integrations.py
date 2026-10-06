"""
test_integrations.py
---------------------
Milestone 4, Tasks 4 & 5: Zoom and Google Meet integrations - authentication,
recording retrieval, file processing, duplicate recordings, processing failures.
All HTTP is faked; no credentials or network are used.
"""

import os
from datetime import date, timedelta

from integrations.base import (IntegrationAuthError, IntegrationError, IntegrationNotConfigured,
                               RecordingSource, RemoteRecording)
from integrations.google_meet import GoogleMeetSource
from integrations.importer import import_recordings
from integrations.zoom import ZoomSource
from services import BadRequest, UpstreamError
from testutil import FAKE_WAV, OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env, fake_whisper, raises


# ---------------------------------------------------------------------------
# A tiny fake `requests` session
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self._content = status, payload, content
        self.headers = {}

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for (m, prefix), handler in self.routes.items():
            if m == method and url.startswith(prefix):
                out = handler(url, kw) if callable(handler) else handler
                if isinstance(out, Exception):
                    raise out
                return out
        return FakeResponse(404, {})

    def get(self, url, **kw):
        return self._dispatch("GET", url, **kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, **kw)


def zoom_meeting(mid, topic, files, start="2026-09-28T10:00:00Z"):
    return {"uuid": f"uuid-{mid}", "id": mid, "topic": topic, "start_time": start, "recording_files": files}


def zfile(fid, ftype, status="completed", size=2048, ext=None):
    return {"id": fid, "file_type": ftype, "file_extension": ext or ftype, "status": status,
            "file_size": size, "download_url": f"https://zoom.us/rec/download/{fid}"}


def zoom_source(routes=None, **creds):
    base = {("POST", "https://zoom.us/oauth/token"): FakeResponse(200, {"access_token": "ZT-secret-token"})}
    base.update(routes or {})
    return ZoomSource(account_id="acc", client_id="cid", client_secret="csec", session=FakeSession(base), **creds)


# ---------------------------------------------------------------------------
# Zoom: authentication
# ---------------------------------------------------------------------------

def test_zoom_authentication_success_uses_account_credentials_grant():
    src = zoom_source()
    src.authenticate()
    method, url, kw = src.session.calls[0]
    assert method == "POST" and kw["params"] == {"grant_type": "account_credentials", "account_id": "acc"}
    assert kw["auth"] == ("cid", "csec")


def test_zoom_not_configured():
    with raises(IntegrationNotConfigured):
        ZoomSource(account_id="", client_id="", client_secret="", session=FakeSession({})).authenticate()


def test_zoom_bad_credentials_and_network_failure():
    src = zoom_source({("POST", "https://zoom.us/oauth/token"): FakeResponse(401, {})})
    with raises(IntegrationAuthError):
        src.authenticate()
    src = zoom_source({("POST", "https://zoom.us/oauth/token"): ConnectionError("dns")})
    with raises(IntegrationError) as info:
        src.authenticate()
    assert info.value.retryable and not isinstance(info.value, IntegrationAuthError)


def test_zoom_listing_requires_authentication_first():
    with raises(IntegrationAuthError):
        zoom_source().list_recordings(date.today())


# ---------------------------------------------------------------------------
# Zoom: recording retrieval
# ---------------------------------------------------------------------------

def test_zoom_lists_recordings_prefers_audio_and_skips_unfinished_files():
    payload = {"meetings": [
        zoom_meeting(1, "Weekly sync", [zfile("v1", "MP4"), zfile("a1", "M4A"), zfile("t1", "TRANSCRIPT", ext="vtt")]),
        zoom_meeting(2, "Still processing", [zfile("v2", "MP4", status="processing")]),
        zoom_meeting(3, "Only chat", [zfile("c3", "CHAT", ext="txt")]),
        zoom_meeting(4, "Video only", [zfile("v4", "MP4")]),
    ]}
    src = zoom_source({("GET", "https://api.zoom.us/v2/users/me/recordings"): FakeResponse(200, payload)})
    src.authenticate()
    recs = src.list_recordings(date.today() - timedelta(days=3))
    assert [(r.external_id, r.extension) for r in recs] == [("a1", ".m4a"), ("v4", ".mp4")]
    assert recs[0].name == "Weekly sync" and recs[0].provider == "zoom"


def test_zoom_follows_pagination_and_walks_30_day_windows():
    pages = {"": {"meetings": [zoom_meeting(1, "A", [zfile("f1", "M4A")])], "next_page_token": "P2"},
             "P2": {"meetings": [zoom_meeting(2, "B", [zfile("f2", "M4A")])]}}
    seen_params = []

    def handler(url, kw):
        seen_params.append(dict(kw["params"]))
        page = pages[kw["params"].get("next_page_token", "")]
        return FakeResponse(200, page if len(seen_params) <= 2 else {"meetings": []})

    src = zoom_source({("GET", "https://api.zoom.us/v2/users/me/recordings"): handler})
    src.authenticate()
    recs = src.list_recordings(date.today() - timedelta(days=45))
    assert {r.external_id for r in recs} == {"f1", "f2"}
    windows = {(p["from"], p["to"]) for p in seen_params}
    assert len(windows) == 2, "45 days needs two <=30-day windows"
    assert all((date.fromisoformat(t) - date.fromisoformat(f)).days <= 29 for f, t in windows)


def test_zoom_download_streams_file_with_header_auth_only(tmp_dir=None):
    body = b"x" * 5000
    src = zoom_source({("GET", "https://zoom.us/rec/download/a1"): FakeResponse(200, content=body)})
    src.authenticate()
    rec = RemoteRecording("zoom", "a1", "n", "", 5000, "https://zoom.us/rec/download/a1", ".m4a")
    import tempfile
    dest = os.path.join(tempfile.mkdtemp(), "a.m4a")
    assert src.download(rec, dest) == 5000 and open(dest, "rb").read() == body
    _, url, kw = src.session.calls[-1]
    assert kw["headers"] == {"Authorization": "Bearer ZT-secret-token"}
    assert "ZT-secret-token" not in url, "token must never be placed in a URL"


def test_zoom_http_errors_are_classified():
    for status, retryable, auth in ((401, False, True), (404, False, False), (429, True, False), (503, True, False)):
        src = zoom_source({("GET", "https://api.zoom.us/v2/users/me/recordings"): FakeResponse(status, {})})
        src.authenticate()
        try:
            src.list_recordings(date.today())
        except IntegrationError as e:
            assert e.retryable is retryable and isinstance(e, IntegrationAuthError) is auth, status
        else:
            raise AssertionError(f"status {status} should raise")


def test_download_aborts_and_cleans_up_when_file_is_too_large():
    from config import settings
    old = settings.max_upload_mb
    settings.max_upload_mb = 1
    try:
        src = zoom_source({("GET", "https://zoom.us/rec/download/big"): FakeResponse(200, content=b"x" * (3 * 1024 * 1024))})
        src.authenticate()
        rec = RemoteRecording("zoom", "big", "n", "", 0, "https://zoom.us/rec/download/big", ".mp4")
        import tempfile
        dest = os.path.join(tempfile.mkdtemp(), "big.mp4")
        with raises(IntegrationError):
            src.download(rec, dest)
        assert not os.path.exists(dest), "partial download must be deleted"
    finally:
        settings.max_upload_mb = old


# ---------------------------------------------------------------------------
# Google Meet
# ---------------------------------------------------------------------------

def google_source(files=None, folders=None, token_status=200):
    folders = [{"id": "F1"}] if folders is None else folders
    routes = {
        ("POST", "https://oauth2.googleapis.com/token"): FakeResponse(token_status, {"access_token": "GT-secret"}),
        ("GET", "https://www.googleapis.com/drive/v3/files/REC1"): FakeResponse(200, content=FAKE_WAV),
        ("GET", "https://www.googleapis.com/drive/v3/files"): lambda url, kw: FakeResponse(
            200, {"files": folders} if "folder" in kw["params"]["q"] else {"files": files or []}),
    }
    return GoogleMeetSource(client_id="c", client_secret="s", refresh_token="r", session=FakeSession(routes))


def test_google_authentication():
    src = google_source()
    src.authenticate()
    _, _, kw = src.session.calls[0]
    assert kw["data"]["grant_type"] == "refresh_token" and kw["data"]["refresh_token"] == "r"
    with raises(IntegrationAuthError):
        google_source(token_status=400).authenticate()
    with raises(IntegrationNotConfigured):
        GoogleMeetSource(client_id="", client_secret="", refresh_token="", session=FakeSession({})).authenticate()


def test_google_lists_and_downloads_meet_recordings():
    files = [{"id": "REC1", "name": "Design review (2026-09-28)", "mimeType": "video/mp4", "size": "4096",
              "createdTime": "2026-09-28T10:00:00Z"}]
    src = google_source(files)
    src.authenticate()
    recs = src.list_recordings(date.today() - timedelta(days=7))
    assert len(recs) == 1 and recs[0].external_id == "REC1" and recs[0].extension == ".mp4"
    assert recs[0].provider == "google_meet" and recs[0].size_bytes == 4096
    import tempfile
    dest = os.path.join(tempfile.mkdtemp(), "r.mp4")
    assert src.download(recs[0], dest) == len(FAKE_WAV)
    assert src.session.calls[-1][2]["params"] == {"alt": "media"}


def test_google_without_meet_folder_returns_nothing():
    src = google_source(files=[{"id": "x"}], folders=[])
    src.authenticate()
    assert src.list_recordings(date.today()) == []


# ---------------------------------------------------------------------------
# Importer (shared by both providers)
# ---------------------------------------------------------------------------

class FakeSource(RecordingSource):
    provider = "zoom"

    def __init__(self, recordings, fail_download=(), auth_error=None, payload=FAKE_WAV):
        self.recordings, self.fail_download, self.auth_error, self.payload = recordings, set(fail_download), auth_error, payload
        self.downloaded, self.paths = [], []

    def authenticate(self):
        if self.auth_error:
            raise self.auth_error

    def list_recordings(self, since):
        return list(self.recordings)

    def download(self, recording, dest_path):
        if recording.external_id in self.fail_download:
            raise IntegrationError("provider hiccup", retryable=True)
        with open(dest_path, "wb") as f:
            f.write(self.payload)
        self.downloaded.append(recording.external_id)
        self.paths.append(dest_path)
        return len(self.payload)


def rec(eid, name="Weekly sync", ext=".wav", size=len(FAKE_WAV)):
    return RemoteRecording("zoom", eid, name, "2026-09-28T10:00:00Z", size, f"https://x/{eid}", ext)


class FakeProcessor:
    def __init__(self, env, fail_for=()):
        self.env, self.fail_for, self.calls = env, set(fail_for), []

    def __call__(self, raw_path, filename, size, user_id, source):
        self.calls.append((filename, user_id, source))
        if filename in self.fail_for:
            raise RuntimeError("Whisper crashed token=abc123456789")
        mid = self.env.db.create_meeting(filename, "transcript text", user_id=user_id, source=source)
        return type("R", (), {"meeting_id": mid})()


def test_importer_imports_new_recordings_for_the_right_user():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env)
    report = import_recordings(FakeSource([rec("a", "Sync A"), rec("b", "Sync B")]), env.db, u["id"], proc)
    assert (report.found, len(report.imported), len(report.failed)) == (2, 2, 0)
    assert [c[1:] for c in proc.calls] == [(u["id"], "zoom")] * 2
    rows = {r["external_id"]: r for r in env.db.list_recordings(u["id"])}
    assert all(r["status"] == "completed" and r["meeting_id"] for r in rows.values())
    assert {m["source"] for m in env.db.list_meetings(u["id"])} == {"zoom"}


def test_importer_skips_duplicates_on_repeat_sync():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env)
    src = FakeSource([rec("a"), rec("b")])
    import_recordings(src, env.db, u["id"], proc)
    second = import_recordings(src, env.db, u["id"], proc)
    assert second.skipped_duplicates == ["a", "b"] and second.imported == []
    assert len(proc.calls) == 2 and src.downloaded == ["a", "b"], "nothing downloaded or processed twice"
    assert len(env.db.list_meetings(u["id"])) == 2


def test_recording_in_progress_elsewhere_is_not_processed_again():
    env = Env()
    u = env.user("alice")
    row, _ = env.db.register_recording(u["id"], "zoom", "busy")
    env.db.claim_recording(row["id"], 3)                         # another worker is on it
    proc = FakeProcessor(env)
    report = import_recordings(FakeSource([rec("busy")]), env.db, u["id"], proc)
    assert report.skipped_duplicates == ["busy"] and proc.calls == []


def test_one_failure_does_not_abort_the_batch_and_is_retried_next_sync():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env, fail_for={"Bad.wav"})
    src = FakeSource([rec("a", "Good1"), rec("b", "Bad"), rec("c", "Good2")])
    first = import_recordings(src, env.db, u["id"], proc)
    assert [i["external_id"] for i in first.imported] == ["a", "c"]
    assert [f["external_id"] for f in first.failed] == ["b"]
    row = next(r for r in env.db.list_recordings(u["id"]) if r["external_id"] == "b")
    assert row["status"] == "failed" and row["attempts"] == 1 and row["meeting_id"] is None
    assert "abc123456789" not in row["error"], "secrets are redacted from stored errors"

    proc.fail_for.clear()                                        # transient problem fixed
    second = import_recordings(src, env.db, u["id"], proc)
    assert [i["external_id"] for i in second.imported] == ["b"]
    assert next(r for r in env.db.list_recordings(u["id"]) if r["external_id"] == "b")["status"] == "completed"


def test_retries_stop_after_max_attempts():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env, fail_for={"Bad.wav"})
    src = FakeSource([rec("b", "Bad")])
    for _ in range(3):
        import_recordings(src, env.db, u["id"], proc, max_attempts=3)
    assert len(proc.calls) == 3
    fourth = import_recordings(src, env.db, u["id"], proc, max_attempts=3)
    assert len(proc.calls) == 3 and fourth.skipped_duplicates == ["b"], "gave up after 3 attempts"


def test_download_failure_is_recorded_as_failed():
    env = Env()
    u = env.user("alice")
    report = import_recordings(FakeSource([rec("a")], fail_download={"a"}), env.db, u["id"], FakeProcessor(env))
    assert report.failed[0]["error"].startswith("IntegrationError") and report.imported == []


def test_unsupported_or_fake_media_is_a_permanent_failure_never_retried():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env)
    bad_ext = import_recordings(FakeSource([rec("x", "Doc", ext=".pdf")]), env.db, u["id"], proc)
    assert bad_ext.failed and proc.calls == []
    fake = import_recordings(FakeSource([rec("y", "Fake")], payload=b"MZ\x90\x00 not audio at all" * 10),
                             env.db, u["id"], proc)
    assert fake.failed and proc.calls == []
    for row in env.db.list_recordings(u["id"]):
        assert row["attempts"] >= 3, "permanent failures spend all retries"


def test_temp_downloads_are_always_deleted():
    env = Env()
    u = env.user("alice")
    src = FakeSource([rec("a"), rec("b", "Bad")])
    import_recordings(src, env.db, u["id"], FakeProcessor(env, fail_for={"Bad.wav"}))
    assert src.paths and not any(os.path.exists(p) for p in src.paths)


def test_authentication_failure_stops_the_sync_with_a_clear_error():
    env = Env()
    u = env.user("alice")
    with raises(IntegrationAuthError):
        import_recordings(FakeSource([rec("a")], auth_error=IntegrationAuthError("bad creds")),
                          env.db, u["id"], FakeProcessor(env))
    assert env.db.list_recordings(u["id"]) == []


def test_dangerous_recording_titles_become_safe_filenames():
    env = Env()
    u = env.user("alice")
    proc = FakeProcessor(env)
    import_recordings(FakeSource([rec("a", "../../etc/cron.d/evil")]), env.db, u["id"], proc)
    assert proc.calls[0][0] == "evil.wav"


# ---------------------------------------------------------------------------
# Whole workflow through AppServices: recording -> app -> transcription ->
# summary -> action items -> knowledge repository -> search
# ---------------------------------------------------------------------------

class ServiceSource(FakeSource):
    provider = "google_meet"

    def __init__(self):
        super().__init__([])

    def list_recordings(self, since):
        return [RemoteRecording("google_meet", "drive-1", "Launch call", "2026-09-28", len(FAKE_WAV), "drive-1", ".mp4")]

    def download(self, recording, dest_path):
        with open(dest_path, "wb") as f:  # a valid MP4 header
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096)
        return 4096 + 12


def test_recordings_follow_exactly_the_same_workflow_as_uploads():
    env = Env()
    u = env.user("alice")
    env.services._source_factories = {"google_meet": ServiceSource}
    with fake_whisper(SAMPLE_TRANSCRIPT):
        report = env.services.sync_recordings(u["id"], "google_meet")
        upload = env.upload(u["id"], SAMPLE_TRANSCRIPT, "manual.wav")
    assert report["imported_count"] == 1 and report["failed_count"] == 0
    imported_id = report["imported"][0]["meeting_id"]

    imported, uploaded = env.services.get_details(u["id"], imported_id), env.services.get_details(u["id"], upload["meeting_id"])
    assert imported["meeting"]["source"] == "google_meet" and uploaded["meeting"]["source"] == "upload"
    assert imported["transcript"] == uploaded["transcript"]
    assert imported["summary_text"] == uploaded["summary_text"]
    assert [i["task"] for i in imported["action_items"]] == [i["task"] for i in uploaded["action_items"]]
    assert imported["participants"] == uploaded["participants"] and imported["decisions"] == uploaded["decisions"]
    # knowledge repository + embeddings + vector DB + search all work for the imported meeting
    assert env.services.get_vectors(u["id"], imported_id)["vector_count"] > 0
    assert env.services.verify_meeting(u["id"], imported_id)["ok"]
    hits = env.services.search(u["id"], "API integration Ravi Friday")
    assert imported_id in {r["meeting_id"] for r in hits["results"]}

    again = env.services.sync_recordings(u["id"], "google_meet")
    assert again["imported_count"] == 0 and again["duplicate_count"] == 1


def test_service_errors_for_unknown_provider_and_missing_configuration():
    env = Env()
    u = env.user("alice")
    with raises(BadRequest):
        env.services.sync_recordings(u["id"], "skype")
    env.services._source_factories = {"zoom": lambda: ZoomSource(account_id="", client_id="", client_secret="",
                                                                  session=FakeSession({}))}
    with raises(BadRequest):
        env.services.sync_recordings(u["id"], "zoom")
    env.services._source_factories = {"zoom": lambda: ZoomSource(
        account_id="a", client_id="b", client_secret="c",
        session=FakeSession({("POST", "https://zoom.us/oauth/token"): FakeResponse(401, {})}))}
    with raises(UpstreamError):
        env.services.sync_recordings(u["id"], "zoom")
