"""
test_database.py
------------------
Tests for Milestone 2 - Task 5 (Meeting Data Model & Database Persistence).

Uses a temporary SQLite file per test run so tests never touch the real
meetings.db and can be re-run repeatedly without leftover state.
"""

import os
import tempfile

from database import Database


def make_temp_db() -> Database:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # Database() creates the schema fresh
    return Database(db_path=path)


def test_create_and_fetch_meeting():
    db = make_temp_db()
    meeting_id = db.create_meeting("meeting1.wav", "This is the transcript.", language="en", audio_duration_seconds=120.5)
    meeting = db.get_meeting(meeting_id)
    assert meeting["filename"] == "meeting1.wav"
    assert meeting["transcript"] == "This is the transcript."
    assert meeting["language"] == "en"
    assert meeting["audio_duration_seconds"] == 120.5


def test_participant_deduplication_at_db_level():
    db = make_temp_db()
    id1 = db.get_or_create_participant("Ravi Kumar")
    id2 = db.get_or_create_participant("Ravi Kumar")  # exact same name again
    assert id1 == id2
    assert len(db.list_participants()) == 1


def test_multiple_distinct_participants_persisted():
    db = make_temp_db()
    db.get_or_create_participant("Ravi")
    db.get_or_create_participant("Priya")
    db.get_or_create_participant("Amit")
    assert len(db.list_participants()) == 3


def test_save_and_fetch_summary():
    db = make_temp_db()
    meeting_id = db.create_meeting("m.wav", "transcript text")
    db.save_summary(
        meeting_id=meeting_id,
        summary_text="The team discussed the launch.",
        key_points=["Point A", "Point B"],
        decisions=["Continue with launch"],
        priorities=["Mobile launch"],
    )
    summary = db.get_summary(meeting_id)
    assert summary["summary_text"] == "The team discussed the launch."
    assert summary["key_points"] == ["Point A", "Point B"]
    assert summary["decisions"] == ["Continue with launch"]


def test_save_and_fetch_action_items_linked_to_participant():
    db = make_temp_db()
    meeting_id = db.create_meeting("m.wav", "transcript text")
    participant_id = db.get_or_create_participant("Ravi")
    db.save_action_item(
        meeting_id=meeting_id, participant_id=participant_id,
        task="Complete API integration", deadline="Friday", priority="High", status="Not Started",
    )
    items = db.get_action_items(meeting_id)
    assert len(items) == 1
    assert items[0]["task"] == "Complete API integration"
    assert items[0]["participant_name"] == "Ravi"


def test_action_items_scoped_to_correct_meeting():
    """Responsibilities are linked to the CORRECT meeting - critical check
    since multiple meetings can share the same participant."""
    db = make_temp_db()
    meeting_1 = db.create_meeting("m1.wav", "transcript 1")
    meeting_2 = db.create_meeting("m2.wav", "transcript 2")
    ravi_id = db.get_or_create_participant("Ravi")  # same participant, two meetings

    db.save_action_item(meeting_id=meeting_1, participant_id=ravi_id, task="Task from meeting 1", deadline="", priority="Medium", status="Not Started")
    db.save_action_item(meeting_id=meeting_2, participant_id=ravi_id, task="Task from meeting 2", deadline="", priority="Medium", status="Not Started")

    items_1 = db.get_action_items(meeting_1)
    items_2 = db.get_action_items(meeting_2)

    assert len(items_1) == 1 and items_1[0]["task"] == "Task from meeting 1"
    assert len(items_2) == 1 and items_2[0]["task"] == "Task from meeting 2"


def test_get_meeting_full_returns_combined_shape():
    db = make_temp_db()
    meeting_id = db.create_meeting("m.wav", "transcript text", language="en", audio_duration_seconds=60)
    db.save_summary(meeting_id, "Summary text", ["kp1"], ["decision1"], ["priority1"])
    participant_id = db.get_or_create_participant("Priya")
    db.save_action_item(meeting_id, participant_id, "Prepare UI testing report", "", "Medium", "Not Started")

    full = db.get_meeting_full(meeting_id)
    assert full["meeting"]["filename"] == "m.wav"
    assert full["summary"]["summary_text"] == "Summary text"
    assert len(full["action_items"]) == 1
    assert full["action_items"][0]["participant_name"] == "Priya"


def test_get_meeting_full_returns_none_for_unknown_id():
    db = make_temp_db()
    assert db.get_meeting_full(9999) is None


def test_list_meetings_returns_all():
    db = make_temp_db()
    db.create_meeting("m1.wav", "t1")
    db.create_meeting("m2.wav", "t2")
    meetings = db.list_meetings()
    assert len(meetings) == 2


if __name__ == "__main__":
    import sys, inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and inspect.isfunction(obj)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"✅ PASS - {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL - {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
