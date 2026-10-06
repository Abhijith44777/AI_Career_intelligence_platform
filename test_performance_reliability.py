"""
test_performance_reliability.py
--------------------------------
Milestone 4, Task 9: API response time, large recordings, multiple users,
search / database / vector-search performance, invalid input, sensitive-data
protection, and "failures are handled and the application does not crash".

Time budgets are deliberately generous (CI machines vary); they exist to catch
order-of-magnitude regressions, not to benchmark.
"""

import concurrent.futures
import logging
import time

from config import Settings, settings
from logging_config import RedactingFilter, redact
from pipeline import safe_filename
from services import BadRequest
from testutil import OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env, raises
from upload_security import UploadTooLarge, looks_like_media, save_upload_limited

SEARCH_TARGET_SECONDS = 3.0   # the project requirement from Milestone 3, Task 4


def timed(fn, *a, **k):
    start = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - start


# -- performance ---------------------------------------------------------------------------

def test_search_and_rag_stay_within_3_seconds_with_many_meetings():
    env = Env()
    u = env.user("alice")
    topics = ["database migration", "mobile launch", "budget review", "hiring plan", "security audit"]
    for i in range(30):
        env.db.create_meeting(f"meeting-{i}.wav", f"Discussion {i} about the {topics[i % 5]} and next steps.",
                              user_id=u["id"])
    env.services.sync_embeddings(u["id"])
    assert env.vectors.count() >= 30
    result, elapsed = timed(env.services.search, u["id"], "Which meeting discussed the database migration?")
    assert elapsed < SEARCH_TARGET_SECONDS and result["within_latency_target"]
    assert result["results"] and "migration" in result["results"][0]["matched_text"].lower()
    _, elapsed = timed(env.services.ask, u["id"], "What was discussed about the security audit?")
    assert elapsed < SEARCH_TARGET_SECONDS * 2


def test_database_queries_are_fast_with_thousands_of_rows():
    env = Env()
    u = env.user("alice")
    with env.db._connect() as conn:
        conn.executemany(
            "INSERT INTO meetings (filename, transcript, language, created_at, user_id, status, source) "
            "VALUES (?, ?, 'en', '2026-09-01T10:00:00', ?, 'completed', 'upload')",
            [(f"m{i}.wav", f"transcript number {i} " * 20, u["id"]) for i in range(3000)])
    _, t_list = timed(env.services.list_meetings, u["id"])
    _, t_filter = timed(env.services.list_meetings, u["id"], text="number 2999", status="completed")
    _, t_overview = timed(env.services.get_overview, u["id"])
    assert t_list < 2.0 and t_filter < 2.0 and t_overview < 2.0


def test_large_transcript_is_processed_without_error():
    env = Env()
    u = env.user("alice")
    long_text = " ".join(f"Sentence {i} about the quarterly plan." for i in range(6000))
    long_text += " Ravi will finalise the roadmap by Friday."
    result, elapsed = timed(env.upload, u["id"], long_text, "long-meeting.wav", 3 * 3600.0)
    assert elapsed < 60
    vectors = env.services.get_vectors(u["id"], result["meeting_id"])
    assert vectors["vector_count"] > 20, "long transcripts are split into many embedded sections"
    assert env.services.get_details(u["id"], result["meeting_id"])["analytics"]["duration_minutes"] == 180.0


def test_multiple_users_processing_concurrently_stay_isolated():
    env = Env()
    users = [env.user(f"user{i}") for i in range(6)]
    texts = {u["id"]: f"Person{u['id']} will deliver report number {u['id']} by Friday." for u in users}

    def work(u):
        return u["id"], env.upload(u["id"], texts[u["id"]], f"u{u['id']}.wav")["meeting_id"]

    # Whisper is patched globally by fake_whisper, so upload sequentially but query concurrently.
    ids = dict(work(u) for u in users)

    def read(u):
        mid = ids[u["id"]]
        details = env.services.get_details(u["id"], mid)
        hits = env.services.search(u["id"], f"report number {u['id']} Friday")
        return u["id"], details["transcript"], {h["meeting_id"] for h in hits["results"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for uid, transcript, hit_ids in pool.map(read, users):
            assert transcript == texts[uid]
            assert hit_ids <= {ids[uid]}, "search results never cross users under concurrency"


def test_concurrent_logins_and_writes_do_not_lock_the_database():
    env = Env()
    for i in range(5):
        env.user(f"user{i}")

    def login(i):
        return env.auth.login(f"user{i % 5}", "correct-horse-9")["token"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        tokens = list(pool.map(login, range(20)))
    assert len(set(tokens)) == 20
    assert all(env.auth.authenticate(t) for t in tokens)


# -- invalid input -----------------------------------------------------------------------------

def test_invalid_search_and_question_inputs():
    env = Env()
    u = env.user("alice")
    for bad in ("", "   ", None, "x" * 1001):
        with raises(BadRequest):
            env.services.search(u["id"], bad)
    with raises(BadRequest):
        env.services.ask(u["id"], "q" * 2001)
    assert env.services.search(u["id"], "anything at all")["results"] == [], "no meetings -> empty, not an error"
    assert "don't have information" in env.services.ask(u["id"], "What was decided?")["answer"]
    assert len(env.services.search(u["id"], "x", top_k=10 ** 6)["results"]) <= 20      # top_k is clamped
    assert env.services.search(u["id"], "x", top_k=-5) is not None


def test_weird_but_valid_text_never_crashes_the_pipeline():
    env = Env()
    u = env.user("alice")
    nasty = ["Ravi will fix it by Friday 🚀 <script>alert(1)</script>", "'; DROP TABLE meetings; --",
             "बैठक में राजेश शुक्रवार तक रिपोर्ट भेजेगा।", "a " * 5000, "\u202e\x00 null bytes and RTL override"]
    for i, text in enumerate(nasty):
        try:
            env.upload(u["id"], text, f"w{i}.wav")
        except Exception as e:  # noqa: BLE001 - only the pipeline's own validation may reject
            from pipeline import PipelineError
            assert isinstance(e, PipelineError), type(e)
    assert env.db.list_meetings(user_id=u["id"]) is not None       # tables intact
    assert env.services.list_meetings(u["id"])


def test_filename_sanitising_blocks_path_traversal():
    for evil in ("../../etc/passwd", "..\\..\\windows\\system.ini", "/abs/path.wav", "a\x00b.wav", "", "....//"):
        cleaned = safe_filename(evil)
        assert "/" not in cleaned and "\\" not in cleaned and ".." not in cleaned and "\x00" not in cleaned
        assert cleaned


def test_upload_limiter_streams_and_cleans_up():
    import io
    import os
    import tempfile
    dest = os.path.join(tempfile.mkdtemp(), "f.wav")
    assert save_upload_limited(io.BytesIO(b"a" * 1000), dest, 2000) == 1000
    with raises(UploadTooLarge):
        save_upload_limited(io.BytesIO(b"a" * 5000), dest, 2000)
    assert not os.path.exists(dest)


def test_magic_number_check_rejects_disguised_files():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    samples = {"ok.wav": b"RIFF\x00\x00\x00\x00WAVEfmt ", "ok.mp3": b"ID3\x03\x00\x00", "ok.flac": b"fLaC\x00\x00",
               "ok.mp4": b"\x00\x00\x00\x20ftypisom", "ok.ogg": b"OggS\x00\x02", "ok.mkv": bytes.fromhex("1A45DFA3") + b"\x00"}
    for name, head in samples.items():
        p = os.path.join(d, name)
        open(p, "wb").write(head + b"\x00" * 64)
        assert looks_like_media(p), name
    for name, head in {"exe.mp3": b"MZ\x90\x00", "zip.mp4": b"PK\x03\x04", "text.wav": b"just some text", "pdf.flac": b"%PDF-1.7",
                       "empty.wav": b"", "script.m4a": b"#!/bin/sh\n"}.items():
        p = os.path.join(d, name)
        open(p, "wb").write(head)
        assert not looks_like_media(p), name
    assert not looks_like_media(os.path.join(d, "missing.wav"))


# -- sensitive data --------------------------------------------------------------------------------

def test_secrets_are_redacted_from_logs():
    samples = {
        "Authorization: Bearer abcdef1234567890abcdef": "abcdef1234567890",
        "GEMINI_API_KEY=AIzaSyA1234567890abcdefghijkl": "A1234567890abcdef",
        "password=hunter2hunter2": "hunter2",
        "refresh_token: 1//0gABCDEFGHIJ": "0gABCDEFGHIJ",
        "key sk-abcdefghijklmnopqrstuvwx used": "abcdefghijklmnopqrstuvwx",
        "groq gsk_abcdefghijklmnopqrstuv": "abcdefghijklmnopqrstuv",
        "api_key='topsecretvalue'": "topsecretvalue",
    }
    for text, secret in samples.items():
        assert secret not in redact(text), text


def test_logging_filter_redacts_formatted_arguments():
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "call failed for %s", ("token=abc123456789xyz",), None)
    RedactingFilter().filter(record)
    assert "abc123456789xyz" not in record.getMessage() and record.request_id


def test_settings_redaction_hides_every_secret():
    s = Settings()
    s.password_pepper, s.zoom_client_secret, s.google_refresh_token = "pepper-value-123456", "zsecret", "1//refresh"
    dump = str(s.redacted())
    for secret in ("pepper-value-123456", "zsecret", "1//refresh"):
        assert secret not in dump
    assert "***set***" in dump


def test_production_config_validation():
    s = Settings()
    s.app_env, s.password_pepper, s.cors_origins, s.api_base_url = "production", "short", ["*"], "http://api.example.com"
    problems = " ".join(s.validate_for_production())
    assert "PASSWORD_PEPPER" in problems and "CORS_ORIGINS" in problems and "https" in problems
    s.password_pepper, s.cors_origins, s.api_base_url = "x" * 32, ["https://app.example.com"], "https://api.example.com"
    assert s.validate_for_production() == []
    s.app_env = "development"
    s.password_pepper = ""
    assert s.validate_for_production() == []


def test_stored_secrets_are_not_recoverable_from_the_database():
    env = Env()
    env.auth.register("alice", "a@example.com", "My-Very-Secret-Pass9")
    env.auth.login("alice", "My-Very-Secret-Pass9")
    with env.db._connect() as conn:
        dump = " ".join(str(tuple(r)) for t in ("users", "sessions") for r in conn.execute(f"SELECT * FROM {t}"))
    assert "My-Very-Secret-Pass9" not in dump


# -- reliability --------------------------------------------------------------------------------------

def test_embedding_failure_does_not_lose_the_meeting():
    env = Env()
    u = env.user("alice")

    class BrokenEmbedder:
        min_relevance_score = 0.3

        def embed_texts(self, texts, task=None):
            raise RuntimeError("embedding provider down")

    env.services.embedding_service = BrokenEmbedder()
    result = env.upload(u["id"], SAMPLE_TRANSCRIPT, "resilient.wav")
    details = env.services.get_details(u["id"], result["meeting_id"])
    assert details["action_items"] and details["meeting"]["status"] == "completed"
    assert env.services.get_vectors(u["id"], result["meeting_id"])["vector_count"] == 0
    from services import UpstreamError
    with raises(UpstreamError):
        env.services.generate_embeddings(u["id"], result["meeting_id"])


def test_transcript_file_write_failure_does_not_lose_the_meeting():
    env = Env()
    u = env.user("alice")
    import os
    import tempfile
    blocker = tempfile.mktemp()
    open(blocker, "w").write("a file, so a directory can't be created beneath it")
    env.transcripts_dir = os.path.join(blocker, "sub")
    from config import settings as s
    original = env.upload
    tmp = tempfile.mkdtemp()
    s.transcripts_dir = env.transcripts_dir
    from testutil import FAKE_WAV, fake_whisper
    path = os.path.join(tmp, "x.wav")
    open(path, "wb").write(FAKE_WAV)
    with fake_whisper(SAMPLE_TRANSCRIPT):
        result = env.services.process_upload(u["id"], path, "x.wav", len(FAKE_WAV))
    assert env.db.get_meeting(result["meeting_id"]) is not None


def test_search_survives_orphaned_vectors():
    env = Env()
    u = env.user("alice")
    mid = env.upload(u["id"], SAMPLE_TRANSCRIPT, "a.wav")["meeting_id"]
    with env.db._connect() as conn:                       # simulate a meeting deleted behind the vector store's back
        conn.execute("DELETE FROM action_items WHERE meeting_id = ?", (mid,))
        conn.execute("DELETE FROM summaries WHERE meeting_id = ?", (mid,))
        conn.execute("DELETE FROM meeting_participants WHERE meeting_id = ?", (mid,))
        conn.execute("DELETE FROM meetings WHERE id = ?", (mid,))
    assert env.services.search(u["id"], "API integration Ravi")["results"] == []
    assert "don't have information" in env.services.ask(u["id"], "Who does the API integration?")["answer"]
