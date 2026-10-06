"""
database.py
------------
Milestone 2 - Task 5: Meeting Data Model & Database Persistence

Implements the flow:
    Transcript -> AI Processing -> Structured Output -> Validation -> Database

Uses SQLite via the standard library (no extra dependency, zero-config,
works identically on Windows/Mac/Linux) - a good fit for this stage of
the project. The schema is deliberately normalized (participants are a
separate table, not duplicated per action item) so the "duplicate
participant records are avoided" requirement from Task 4 holds at the
persistence layer too, across meetings, not just within one.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime

from config import settings

DEFAULT_DB_PATH = settings.database_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    transcript TEXT NOT NULL,
    language TEXT,
    audio_duration_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    summary_text TEXT,
    key_points TEXT,     -- JSON-encoded list
    decisions TEXT,       -- JSON-encoded list
    priorities TEXT,      -- JSON-encoded list
    created_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings (id)
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    participant_id INTEGER,
    task TEXT NOT NULL,
    deadline TEXT,
    priority TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings (id),
    FOREIGN KEY (participant_id) REFERENCES participants (id)
);

CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items (meeting_id);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting ON summaries (meeting_id);

-- Milestone 4 ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL
);

-- Only a SHA-256 of the session token is stored, so a leaked database
-- cannot be used to impersonate a logged-in user.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Which participants attended which meeting (independent of action items, so
-- someone who was mentioned but has no task still shows up as an attendee).
CREATE TABLE IF NOT EXISTS meeting_participants (
    meeting_id INTEGER NOT NULL,
    participant_id INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, participant_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE CASCADE,
    FOREIGN KEY (participant_id) REFERENCES participants (id)
);

-- Zoom / Google Meet recordings pulled in by the integrations. The UNIQUE
-- constraint is the hard guarantee against processing a recording twice.
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | completed | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    meeting_id INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (user_id, provider, external_id),
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (meeting_id) REFERENCES meetings (id) ON DELETE SET NULL
);
"""

# Columns added to pre-existing tables. Applied by Database._migrate() so a
# Milestone 2/3 meetings.db upgrades in place with no data loss.
MIGRATIONS = {
    "meetings": [
        ("user_id", "INTEGER"),                                  # owner; NULL == legacy (pre-Milestone-4) meeting
        ("status", "TEXT NOT NULL DEFAULT 'completed'"),         # processing | completed | failed
        ("error_message", "TEXT"),
        ("source", "TEXT NOT NULL DEFAULT 'upload'"),            # upload | zoom | google_meet
    ],
}

POST_MIGRATION_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_recordings_user ON recordings (user_id, provider);
"""


class Database:
    """Thin, explicit persistence layer - no ORM, so the schema and every
    query are easy to read and audit."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        # timeout: wait (instead of failing with "database is locked") when
        # several users write at once - Task 9 multi-user reliability.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            try:
                conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
            except sqlite3.DatabaseError:
                pass
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.executescript(POST_MIGRATION_SCHEMA)

    @staticmethod
    def _migrate(conn):
        """Idempotent in-place upgrade of Milestone 2/3 databases."""
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, ddl in columns:
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        # Backfill attendance from action items for meetings that predate the
        # meeting_participants table (INSERT OR IGNORE keeps this re-runnable).
        conn.execute(
            """INSERT OR IGNORE INTO meeting_participants (meeting_id, participant_id)
               SELECT DISTINCT meeting_id, participant_id FROM action_items
               WHERE participant_id IS NOT NULL"""
        )

    # -- Meetings -------------------------------------------------------------

    def create_meeting(self, filename: str, transcript: str, language: str = None,
                        audio_duration_seconds: float = None, user_id: int = None,
                        source: str = "upload", status: str = "completed") -> int:
        """Persist the transcript stage of the pipeline. Returns the new meeting_id.
        `user_id` is the owner (Milestone 4); None is only used by legacy callers/tests."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO meetings (filename, transcript, language, audio_duration_seconds, created_at, "
                "user_id, source, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (filename, transcript, language, audio_duration_seconds, datetime.now().isoformat(),
                 user_id, source, status),
            )
            return cur.lastrowid

    def set_meeting_status(self, meeting_id: int, status: str, error_message: str = None):
        with self._connect() as conn:
            conn.execute("UPDATE meetings SET status = ?, error_message = ? WHERE id = ?",
                         (status, error_message, meeting_id))

    def get_meeting_for_user(self, meeting_id: int, user_id: int) -> dict:
        """Ownership-checked fetch: returns None both when the meeting doesn't
        exist AND when it belongs to someone else, so callers can't tell the
        two apart (no existence leak)."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ? AND user_id = ?",
                               (meeting_id, user_id)).fetchone()
            return dict(row) if row else None

    def delete_meeting(self, meeting_id: int, user_id: int) -> bool:
        """Deletes a meeting owned by `user_id` plus everything hanging off it."""
        with self._connect() as conn:
            owned = conn.execute("SELECT 1 FROM meetings WHERE id = ? AND user_id = ?",
                                 (meeting_id, user_id)).fetchone()
            if not owned:
                return False
            conn.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,))
            conn.execute("DELETE FROM summaries WHERE meeting_id = ?", (meeting_id,))
            conn.execute("DELETE FROM meeting_participants WHERE meeting_id = ?", (meeting_id,))
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            return True

    def assign_legacy_meetings(self, user_id: int) -> int:
        """One-off: give every pre-Milestone-4 (owner-less) meeting to `user_id`."""
        with self._connect() as conn:
            cur = conn.execute("UPDATE meetings SET user_id = ? WHERE user_id IS NULL", (user_id,))
            return cur.rowcount

    def get_meeting(self, meeting_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
            return dict(row) if row else None

    def list_meetings(self, user_id: int = None) -> list:
        """All meetings, or - when `user_id` is given - only that user's."""
        with self._connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    "SELECT id, filename, language, created_at, user_id, status, source "
                    "FROM meetings ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, filename, language, created_at, user_id, status, source "
                    "FROM meetings WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
            return [dict(r) for r in rows]

    def search_meetings(self, user_id: int, text: str = None, language: str = None,
                        status: str = None, source: str = None, date_from: str = None,
                        date_to: str = None, participant: str = None) -> list:
        """Dashboard list + filters (Milestone 4, Task 1). Always scoped to
        `user_id`. Each row carries counts so the list needs no follow-up
        queries. Filters are ANDed; every value is a bound parameter."""
        where, params = ["m.user_id = ?"], [user_id]
        if text:
            where.append("(m.filename LIKE ? ESCAPE '\\' OR m.transcript LIKE ? ESCAPE '\\')")
            like = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params += [like, like]
        if language:
            where.append("m.language = ?")
            params.append(language)
        if status:
            where.append("m.status = ?")
            params.append(status)
        if source:
            where.append("m.source = ?")
            params.append(source)
        if date_from:
            where.append("substr(m.created_at, 1, 10) >= ?")
            params.append(date_from)
        if date_to:
            where.append("substr(m.created_at, 1, 10) <= ?")
            params.append(date_to)
        if participant:
            where.append("""EXISTS (SELECT 1 FROM meeting_participants mp
                             JOIN participants p ON p.id = mp.participant_id
                             WHERE mp.meeting_id = m.id AND p.name = ?)""")
            params.append(participant)
        sql = f"""SELECT m.id, m.filename, m.language, m.created_at, m.status, m.source,
                         m.audio_duration_seconds,
                         (SELECT COUNT(*) FROM action_items a WHERE a.meeting_id = m.id) AS action_item_count,
                         (SELECT COUNT(*) FROM meeting_participants mp WHERE mp.meeting_id = m.id) AS participant_count
                  FROM meetings m WHERE {' AND '.join(where)} ORDER BY m.id DESC"""
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # -- Participants (deduplicated by UNIQUE name constraint) ----------------

    def get_or_create_participant(self, name: str) -> int:
        """
        Insert-or-fetch a participant by name. The UNIQUE constraint on
        `name` is the hard guarantee against duplicate participant records
        at the database level - even if two different meetings' processing
        both try to create "Ravi Kumar", only one row will ever exist.
        """
        with self._connect() as conn:
            # INSERT OR IGNORE + SELECT is race-free: two requests creating the
            # same name at once can no longer hit a UNIQUE-constraint error.
            conn.execute("INSERT OR IGNORE INTO participants (name) VALUES (?)", (name,))
            row = conn.execute("SELECT id FROM participants WHERE name = ?", (name,)).fetchone()
            return row["id"]

    def link_participant_to_meeting(self, meeting_id: int, participant_id: int):
        """Record that a participant attended a meeting (idempotent)."""
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO meeting_participants (meeting_id, participant_id) VALUES (?, ?)",
                         (meeting_id, participant_id))

    def list_participants(self, user_id: int = None) -> list:
        """All participants, or - when `user_id` is given - only those who
        attended that user's own meetings (participant names are shared rows,
        but never shared *visibility*)."""
        with self._connect() as conn:
            if user_id is None:
                rows = conn.execute("SELECT * FROM participants ORDER BY name").fetchall()
            else:
                rows = conn.execute(
                    """SELECT DISTINCT p.* FROM participants p
                       JOIN meeting_participants mp ON mp.participant_id = p.id
                       JOIN meetings m ON m.id = mp.meeting_id
                       WHERE m.user_id = ? ORDER BY p.name""", (user_id,)).fetchall()
            return [dict(r) for r in rows]

    # -- Summaries --------------------------------------------------------------

    def save_summary(self, meeting_id: int, summary_text: str, key_points: list,
                      decisions: list, priorities: list) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO summaries (meeting_id, summary_text, key_points, decisions, priorities, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (meeting_id, summary_text, json.dumps(key_points), json.dumps(decisions),
                 json.dumps(priorities), datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_summary(self, meeting_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE meeting_id = ? ORDER BY id DESC LIMIT 1", (meeting_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["key_points"] = json.loads(result["key_points"] or "[]")
            result["decisions"] = json.loads(result["decisions"] or "[]")
            result["priorities"] = json.loads(result["priorities"] or "[]")
            return result

    # -- Action items (linked to meeting + participant) --------------------------

    def save_action_item(self, meeting_id: int, participant_id: int, task: str,
                          deadline: str, priority: str, status: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO action_items (meeting_id, participant_id, task, deadline, priority, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (meeting_id, participant_id, task, deadline, priority, status, datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_action_items(self, meeting_id: int) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT action_items.*, participants.name AS participant_name
                   FROM action_items
                   LEFT JOIN participants ON action_items.participant_id = participants.id
                   WHERE action_items.meeting_id = ?
                   ORDER BY action_items.id""",
                (meeting_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Milestone 3 read helpers (Task 1: granular per-field retrieval) --------
    # Thin wrappers over the existing tables/queries above - no schema change,
    # no duplicate storage. They exist so api.py can expose one focused
    # endpoint per data type (GET /meetings/{id}/transcript, /decisions, etc.)
    # without every caller having to know the join logic themselves.

    def get_transcript(self, meeting_id: int) -> str:
        meeting = self.get_meeting(meeting_id)
        return meeting["transcript"] if meeting else None

    def get_decisions(self, meeting_id: int) -> list:
        summary = self.get_summary(meeting_id)
        return summary["decisions"] if summary else []

    def get_participants_for_meeting(self, meeting_id: int) -> list:
        """Distinct participants of this meeting: everyone in the
        meeting_participants table PLUS anyone holding one of its action items
        (the latter keeps Milestone 2/3 data correct even if never backfilled)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT participants.id, participants.name FROM participants
                   WHERE participants.id IN (
                       SELECT participant_id FROM meeting_participants WHERE meeting_id = ?
                       UNION
                       SELECT participant_id FROM action_items
                       WHERE meeting_id = ? AND participant_id IS NOT NULL)
                   ORDER BY participants.name""",
                (meeting_id, meeting_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_deadlines_for_meeting(self, meeting_id: int) -> list:
        """Distinct, non-empty deadlines linked to this meeting via its
        action items."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT deadline FROM action_items
                   WHERE meeting_id = ? AND deadline IS NOT NULL AND TRIM(deadline) != ''
                   ORDER BY deadline""",
                (meeting_id,),
            ).fetchall()
            return [r["deadline"] for r in rows]

    # -- Full meeting read (used by API / UI) --------------------------------

    def get_meeting_full(self, meeting_id: int) -> dict:
        """Returns the meeting plus its summary and action items in one call -
        exactly the shape the Streamlit UI and API responses need."""
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            return None
        return {
            "meeting": meeting,
            "summary": self.get_summary(meeting_id),
            "action_items": self.get_action_items(meeting_id),
        }


    # -- Users & sessions (Milestone 4, Task 7) --------------------------------

    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Raises sqlite3.IntegrityError if the username or email is taken."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, datetime.now().isoformat()))
            return cur.lastrowid

    def get_user_by_username(self, username: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            return dict(row) if row else None

    def get_user(self, user_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def update_login_state(self, user_id: int, failed_logins: int, locked_until: str = None):
        with self._connect() as conn:
            conn.execute("UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                         (failed_logins, locked_until, user_id))

    def update_password_hash(self, user_id: int, password_hash: str):
        with self._connect() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))

    def create_session(self, token_hash: str, user_id: int, expires_at: str):
        with self._connect() as conn:
            conn.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                         (token_hash, user_id, datetime.now().isoformat(), expires_at))

    def get_session(self, token_hash: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
            return dict(row) if row else None

    def delete_session(self, token_hash: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def delete_user_sessions(self, user_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def delete_expired_sessions(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (datetime.now().isoformat(),))
            return cur.rowcount

    # -- Imported recordings (Milestone 4, Tasks 4-5) ---------------------------

    def register_recording(self, user_id: int, provider: str, external_id: str, name: str = None):
        """Claims a recording for processing. Returns (row, created): created is
        False when this user already has that recording (duplicate)."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO recordings (user_id, provider, external_id, name, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)", (user_id, provider, external_id, name, now, now))
            created = cur.rowcount == 1
            row = conn.execute(
                "SELECT * FROM recordings WHERE user_id = ? AND provider = ? AND external_id = ?",
                (user_id, provider, external_id)).fetchone()
            return dict(row), created

    def update_recording(self, recording_id: int, status: str, meeting_id: int = None,
                         error: str = None, increment_attempts: bool = False):
        with self._connect() as conn:
            conn.execute(
                "UPDATE recordings SET status = ?, meeting_id = COALESCE(?, meeting_id), error = ?, "
                "attempts = attempts + ?, updated_at = ? WHERE id = ?",
                (status, meeting_id, error, 1 if increment_attempts else 0,
                 datetime.now().isoformat(), recording_id))

    def list_recordings(self, user_id: int, provider: str = None) -> list:
        with self._connect() as conn:
            if provider:
                rows = conn.execute("SELECT * FROM recordings WHERE user_id = ? AND provider = ? ORDER BY id DESC",
                                    (user_id, provider)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM recordings WHERE user_id = ? ORDER BY id DESC",
                                    (user_id,)).fetchall()
            return [dict(r) for r in rows]

    # -- Cross-meeting reads for analytics (Milestone 4, Task 2) ------------------

    def get_action_items_for_user(self, user_id: int) -> list:
        """Every action item of every meeting owned by `user_id`, one query."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT a.*, p.name AS participant_name
                   FROM action_items a
                   JOIN meetings m ON m.id = a.meeting_id
                   LEFT JOIN participants p ON p.id = a.participant_id
                   WHERE m.user_id = ? ORDER BY a.id""", (user_id,)).fetchall()
            return [dict(r) for r in rows]

    def claim_recording(self, recording_id: int, max_attempts: int) -> bool:
        """Atomically moves a recording to 'processing'. Returns False if another
        worker already claimed it, it is complete, or it has used up its retries -
        so two concurrent syncs can never process the same recording twice."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE recordings SET status = 'processing', attempts = attempts + 1, error = NULL, updated_at = ? "
                "WHERE id = ? AND status IN ('pending', 'failed') AND attempts < ?",
                (datetime.now().isoformat(), recording_id, max_attempts))
            return cur.rowcount == 1

    def mark_recording_failed(self, recording_id: int, error: str, permanent: bool = False,
                              max_attempts: int = 3):
        """Records a failure; `permanent` also spends all remaining retries
        (an unsupported/oversize file can never succeed on retry)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE recordings SET status = 'failed', error = ?, updated_at = ?, "
                "attempts = CASE WHEN ? THEN MAX(attempts, ?) ELSE attempts END WHERE id = ?",
                (error, datetime.now().isoformat(), 1 if permanent else 0, max_attempts, recording_id))
