"""
test_api.py
------------
Milestone 4 Tasks 7 & 9 at the HTTP layer: every protected endpoint refuses
anonymous callers, cross-user requests return 404, invalid requests get clean
4xx errors, unexpected failures return a generic 500 (no stack traces / secrets),
uploads are validated, and reports download correctly.

Needs FastAPI + httpx installed (see requirements.txt); the tests skip
themselves if FastAPI is not importable.
"""

import io

from testutil import FAKE_WAV, OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env, fake_whisper, make_client

PROTECTED_GET = ["/meetings", "/meetings/1", "/meetings/1/details", "/meetings/1/transcript",
                 "/meetings/1/summary", "/meetings/1/decisions", "/meetings/1/action-items",
                 "/meetings/1/participants", "/meetings/1/deadlines", "/meetings/1/verify",
                 "/meetings/1/vectors", "/meetings/1/report?format=pdf", "/participants",
                 "/analytics/overview", "/integrations/status", "/auth/me"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def setup():
    env = Env()
    client = make_client(env)
    if client is None:
        return None, None
    return env, client


def login(client, name="alice", password="correct-horse-9"):
    client.post("/auth/register", json={"username": name, "email": f"{name}@example.com", "password": password})
    r = client.post("/auth/login", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_health_is_public():
    env, client = setup()
    if not client:
        return
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["database"] == "ok"


def test_anonymous_requests_are_rejected_everywhere():
    env, client = setup()
    if not client:
        return
    for path in PROTECTED_GET:
        assert client.get(path).status_code == 401, path
    for path in ("/embeddings/sync", "/integrations/zoom/sync"):
        assert client.post(path).status_code == 401, path
    assert client.post("/search/meetings", json={"query": "x"}).status_code == 401
    assert client.post("/ask/meetings", json={"question": "x"}).status_code == 401
    assert client.delete("/meetings/1").status_code == 401
    assert client.post("/process-meeting", files={"file": ("a.wav", FAKE_WAV)}).status_code == 401


def test_bad_tokens_are_rejected():
    env, client = setup()
    if not client:
        return
    for header in ({"Authorization": "Bearer nonsense"}, {"Authorization": "Basic abc"},
                   {"Authorization": "Bearer "}, {"Authorization": "nonsense"}):
        assert client.get("/meetings", headers=header).status_code == 401


def test_register_login_me_logout_flow():
    env, client = setup()
    if not client:
        return
    r = client.post("/auth/register", json={"username": "kim", "email": "kim@example.com", "password": "goodpass1"})
    assert r.status_code == 201 and "password" not in r.text and "hash" not in r.text
    assert client.post("/auth/register", json={"username": "kim", "email": "k2@example.com",
                                               "password": "goodpass1"}).status_code == 400
    assert client.post("/auth/login", json={"username": "kim", "password": "wrong-pass"}).status_code == 401
    token = client.post("/auth/login", json={"username": "kim", "password": "goodpass1"}).json()["token"]
    assert client.get("/auth/me", headers=bearer(token)).json()["username"] == "kim"
    assert client.post("/auth/logout", headers=bearer(token)).status_code == 200
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401, "token is dead after logout"


def test_lockout_returns_429():
    env, client = setup()
    if not client:
        return
    client.post("/auth/register", json={"username": "lee", "email": "lee@example.com", "password": "goodpass1"})
    codes = [client.post("/auth/login", json={"username": "lee", "password": "nope-nope1"}).status_code
             for _ in range(3)]
    assert codes == [401, 401, 401]
    assert client.post("/auth/login", json={"username": "lee", "password": "goodpass1"}).status_code == 429


def test_cross_user_access_over_http_is_404_everywhere():
    env, client = setup()
    if not client:
        return
    ta, tb = login(client, "alice"), login(client, "bob")
    ua, ub = env.db.get_user_by_username("alice")["id"], env.db.get_user_by_username("bob")["id"]
    ma = env.upload(ua, SAMPLE_TRANSCRIPT, "alice.wav")["meeting_id"]
    env.upload(ub, OTHER_TRANSCRIPT, "bob.wav")
    for path in (f"/meetings/{ma}", f"/meetings/{ma}/details", f"/meetings/{ma}/transcript",
                 f"/meetings/{ma}/summary", f"/meetings/{ma}/decisions", f"/meetings/{ma}/action-items",
                 f"/meetings/{ma}/participants", f"/meetings/{ma}/deadlines", f"/meetings/{ma}/verify",
                 f"/meetings/{ma}/vectors", f"/meetings/{ma}/report?format=pdf",
                 f"/meetings/{ma}/report?format=csv"):
        assert client.get(path, headers=bearer(tb)).status_code == 404, path
        assert client.get(path, headers=bearer(ta)).status_code == 200, path
    assert client.delete(f"/meetings/{ma}", headers=bearer(tb)).status_code == 404
    assert client.post("/embeddings/generate", json={"meeting_id": ma}, headers=bearer(tb)).status_code == 404
    assert client.delete(f"/embeddings/{ma}", headers=bearer(tb)).status_code == 404
    assert env.db.get_meeting(ma) is not None
    listed = client.get("/meetings", headers=bearer(tb)).json()
    assert [m["filename"] for m in listed] == ["bob.wav"]
    hits = client.post("/search/meetings", json={"query": "API integration launch Ravi"}, headers=bearer(tb)).json()
    assert all(h["meeting_id"] != ma for h in hits["results"])
    ask = client.post("/ask/meetings", json={"question": "Who does the API integration?"}, headers=bearer(tb)).json()
    assert "Ravi" not in ask["answer"] and all(s["meeting_id"] != ma for s in ask["sources"])


def test_upload_pipeline_over_http_and_report_download():
    env, client = setup()
    if not client:
        return
    token = login(client)
    with fake_whisper(SAMPLE_TRANSCRIPT):
        r = client.post("/process-meeting", files={"file": ("sync.wav", FAKE_WAV)}, headers=bearer(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["participants"] and body["action_items"] and body["decisions"]
    mid = body["meeting_id"]

    pdf = client.get(f"/meetings/{mid}/report?format=pdf", headers=bearer(token))
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    assert "attachment" in pdf.headers["Content-Disposition"] and ".pdf" in pdf.headers["Content-Disposition"]
    csv_r = client.get(f"/meetings/{mid}/report?format=csv", headers=bearer(token))
    assert csv_r.status_code == 200 and "Action Items" in csv_r.text
    assert client.get(f"/meetings/{mid}/report?format=docx", headers=bearer(token)).status_code == 400
    assert client.get(f"/meetings/{mid}/details", headers=bearer(token)).json()["analytics"]["action_item_count"] >= 1


def test_upload_validation_errors():
    env, client = setup()
    if not client:
        return
    token = login(client)
    for name, data in (("malware.exe", FAKE_WAV), ("notes.txt", b"hello" * 1000)):
        r = client.post("/process-meeting", files={"file": (name, data)}, headers=bearer(token))
        assert r.status_code == 422 and r.json()["detail"]["stage"] == "Upload Validation", name
    r = client.post("/process-meeting", files={"file": ("renamed.wav", b"MZ\x90\x00" + b"\x00" * 4096)},
                    headers=bearer(token))
    assert r.status_code == 422 and "does not match" in r.json()["detail"]["message"]
    r = client.post("/process-meeting", files={"file": ("empty.wav", b"")}, headers=bearer(token))
    assert r.status_code == 422
    assert env.db.list_meetings() == [], "rejected uploads leave no meeting behind"


def test_oversized_upload_is_rejected_with_413():
    env, client = setup()
    if not client:
        return
    from config import settings
    token, old = login(client), settings.max_upload_mb
    settings.max_upload_mb = 1
    try:
        r = client.post("/process-meeting", files={"file": ("big.wav", FAKE_WAV + b"\x00" * (2 * 1024 * 1024))},
                        headers=bearer(token))
    finally:
        settings.max_upload_mb = old
    assert r.status_code == 413


def test_invalid_requests_get_clean_client_errors():
    env, client = setup()
    if not client:
        return
    token = login(client)
    h = bearer(token)
    assert client.post("/search/meetings", json={"query": "   "}, headers=h).status_code == 400
    assert client.post("/search/meetings", json={"query": "x", "content_type": "bogus"}, headers=h).status_code == 400
    assert client.post("/ask/meetings", json={"question": ""}, headers=h).status_code == 400
    assert client.post("/search/meetings", json={"query": "x" * 5000}, headers=h).status_code in (400, 422)
    assert client.get("/meetings?status=hacked", headers=h).status_code == 400
    assert client.get("/meetings?date_from=notadate", headers=h).status_code == 400
    assert client.get("/meetings/999999/details", headers=h).status_code == 404
    assert client.post("/integrations/skype/sync", headers=h).status_code == 400
    assert client.get("/no-such-route", headers=h).status_code == 404


def test_unexpected_errors_return_generic_500_without_leaking_details():
    env, client = setup()
    if not client:
        return
    token = login(client)

    def explode(*a, **k):
        raise RuntimeError("secret internals: key=AIzaSyABCDEFGHIJKLMNOPQRSTUV")

    env.services.get_overview = explode
    r = client.get("/analytics/overview", headers=bearer(token))
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}
    assert "AIza" not in r.text and "secret" not in r.text and "Traceback" not in r.text


def test_security_headers_present():
    env, client = setup()
    if not client:
        return
    r = client.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff" and r.headers.get("X-Request-ID")
    assert r.headers.get("Cache-Control") == "no-store"


def test_production_refuses_to_start_with_insecure_config():
    try:
        import api
    except ImportError:      # FastAPI not installed - the other API tests skip the same way
        return
    from config import settings
    saved = (settings.app_env, settings.password_pepper, settings.cors_origins)
    try:
        settings.app_env, settings.password_pepper, settings.cors_origins = "production", "", ["*"]
        env = Env()
        try:
            api.create_app(db=env.db, services=env.services, auth=env.auth)
        except RuntimeError as e:
            assert "PASSWORD_PEPPER" in str(e) and "CORS_ORIGINS" in str(e)
        else:
            raise AssertionError("should refuse to start")
    finally:
        settings.app_env, settings.password_pepper, settings.cors_origins = saved
