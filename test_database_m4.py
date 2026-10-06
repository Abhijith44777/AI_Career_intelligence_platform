"""
test_database_m4.py
--------------------
Milestone 4 persistence: in-place upgrade of a Milestone 2/3 database, the
meeting_participants table, race-free participant creation, list filters and
atomic recording claims.
"""

import os
import sqlite3
import tempfile
import threading

from database import Database
from testutil import Env

# The exact Milestone 3 schema (no user_id / status / source, no M4 tables).
M3_SCHEMA = """
CREATE TABLE meetings (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, transcript TEXT NOT NULL,
    language TEXT, audio_duration_seconds REAL, created_at TEXT NOT NULL);
CREATE TABLE participants (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);
CREATE TABLE summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, summary_text TEXT,
    key_points TEXT, decisions TEXT, priorities TEXT, created_at TEXT NOT NULL);
CREATE TABLE action_items (id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, participant_id INTEGER,
    task TEXT NOT NULL, deadline TEXT, priority TEXT, status TEXT, created_at TEXT NOT NULL);
"""


def test_m3_database_upgrades_in_place_without_data_loss():
    path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.executescript(M3_SCHEMA)
    conn.execute("INSERT INTO meetings VALUES (1,'old.wav','Ravi will do it.','en',60,'2026-09-01T10:00:00')")
    conn.execute("INSERT INTO participants VALUES (1,'Ravi')")
    conn.execute("INSERT INTO summaries VALUES (1,1,'S','[]','[\"D\"]','[]','2026-09-01T10:01:00')")
    conn.execute("INSERT INTO action_items VALUES (1,1,1,'Do it','Friday','High','Not Started','2026-09-01T10:01:00')")
    conn.commit()
    conn.close()

    db = Database(db_path=path)               # runs the migration
    meeting = db.get_meeting(1)
    assert meeting["transcript"] == "Ravi will do it."
    assert meeting["user_id"] is None and meeting["status"] == "completed" and meeting["source"] == "upload"
    assert db.get_summary(1)["decisions"] == ["D"]
    assert [p["name"] for p in db.get_participants_for_meeting(1)] == ["Ravi"]
    # attendance was back-filled from the action items
    with db._connect() as c:
        assert c.execute("SELECT COUNT(*) FROM meeting_participants").fetchone()[0] == 1
    Database(db_path=path)                    # second start is a no-op (idempotent)
    assert db.get_meeting(1)["filename"] == "old.wav"


def test_attendee_without_action_item_is_listed_for_the_meeting():
    env = Env()
    a = env.user("alice")
    mid = env.db.create_meeting("m.wav", "t", user_id=a["id"])
    env.db.link_participant_to_meeting(mid, env.db.get_or_create_participant("Silent Sam"))
    assert [p["name"] for p in env.db.get_participants_for_meeting(mid)] == ["Silent Sam"]


def test_get_or_create_participant_is_race_free():
    env = Env()
    errors, ids = [], []

    def worker():
        try:
            ids.append(env.db.get_or_create_participant("Concurrent Carol"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(set(ids)) == 1 and len(env.db.list_participants()) == 1


def test_meeting_filters():
    env = Env()
    u = env.user("alice")
    m1 = env.db.create_meeting("launch.wav", "we discuss the mobile launch", language="en", user_id=u["id"])
    m2 = env.db.create_meeting("budget.zoom.mp4", "budget review", language="hi", user_id=u["id"], source="zoom")
    env.db.set_meeting_status(m2, "failed", "LLM down")
    env.db.link_participant_to_meeting(m1, env.db.get_or_create_participant("Ravi"))
    ids = lambda **kw: [m["id"] for m in env.db.search_meetings(u["id"], **kw)]
    assert ids() == [m2, m1]
    assert ids(text="mobile") == [m1]
    assert ids(text="LAUNCH") == [m1]                       # matches filename, case-insensitive
    assert ids(language="hi") == [m2]
    assert ids(status="failed") == [m2]
    assert ids(source="zoom") == [m2]
    assert ids(participant="Ravi") == [m1]
    assert ids(date_from="2999-01-01") == []
    assert ids(date_to="2000-01-01") == []
    assert ids(text="mobile", language="hi") == []           # filters are ANDed
    row = env.db.search_meetings(u["id"], participant="Ravi")[0]
    assert row["participant_count"] == 1 and row["action_item_count"] == 0


def test_search_text_treats_like_wildcards_literally():
    env = Env()
    u = env.user("alice")
    env.db.create_meeting("a.wav", "growth was 50% this quarter", user_id=u["id"])
    env.db.create_meeting("b.wav", "nothing relevant", user_id=u["id"])
    assert len(env.db.search_meetings(u["id"], text="50%")) == 1
    assert len(env.db.search_meetings(u["id"], text="%")) == 1
    assert len(env.db.search_meetings(u["id"], text="_")) == 0


def test_recording_registration_is_unique_and_claims_are_atomic():
    env = Env()
    u = env.user("alice")
    row, created = env.db.register_recording(u["id"], "zoom", "abc", "Weekly")
    again, created2 = env.db.register_recording(u["id"], "zoom", "abc", "Weekly")
    assert created and not created2 and row["id"] == again["id"]

    wins = []

    def claim():
        wins.append(env.db.claim_recording(row["id"], max_attempts=3))

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1, "exactly one worker may claim a recording"


def test_failed_recording_retries_are_capped():
    env = Env()
    u = env.user("alice")
    row, _ = env.db.register_recording(u["id"], "zoom", "r1")
    for attempt in range(3):
        assert env.db.claim_recording(row["id"], 3)
        env.db.mark_recording_failed(row["id"], "boom", max_attempts=3)
    assert not env.db.claim_recording(row["id"], 3), "out of retries"


def test_permanent_failure_spends_all_retries():
    env = Env()
    u = env.user("alice")
    row, _ = env.db.register_recording(u["id"], "zoom", "r2")
    assert env.db.claim_recording(row["id"], 3)
    env.db.mark_recording_failed(row["id"], "unsupported", permanent=True, max_attempts=3)
    assert not env.db.claim_recording(row["id"], 3)


def test_delete_meeting_requires_ownership():
    env = Env()
    a, b = env.user("alice"), env.user("bob")
    mid = env.db.create_meeting("m.wav", "t", user_id=a["id"])
    assert env.db.delete_meeting(mid, b["id"]) is False
    assert env.db.get_meeting(mid) is not None
    assert env.db.delete_meeting(mid, a["id"]) is True


def test_unique_user_constraints_are_case_insensitive():
    env = Env()
    env.db.create_user("Zed", "zed@example.com", "hash")
    for username, email in (("zed", "other@example.com"), ("other", "ZED@example.com")):
        try:
            env.db.create_user(username, email, "hash")
        except sqlite3.IntegrityError:
            continue
        raise AssertionError("duplicate should have been rejected")
