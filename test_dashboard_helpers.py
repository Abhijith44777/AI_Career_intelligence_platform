"""test_dashboard_helpers.py - Milestone 4 Task 1: dashboard data preparation."""

from datetime import date

from dashboard_helpers import (action_item_rows, build_filters, chart_series, fmt_date, fmt_duration,
                               meeting_label, meeting_rows, password_problems, source_row)


def test_fmt_duration():
    assert fmt_duration(None) == "-" and fmt_duration(0) == "-"
    assert fmt_duration(45) == "45s" and fmt_duration(125) == "2m 05s" and fmt_duration(3 * 3600 + 120) == "3h 02m"


def test_fmt_date_handles_bad_input():
    assert fmt_date("2026-09-28T10:30:00") == "28 Sep 2026, 10:30"
    assert fmt_date("") == "-" and fmt_date("garbage-value-here!!") == "garbage-value-he"


def test_build_filters_drops_unset_values():
    assert build_filters() == {}
    f = build_filters(text="  launch ", language="en", status="All", source="zoom",
                      date_from=date(2026, 9, 1), date_to=None, participant="Ravi")
    assert f == {"text": "launch", "language": "en", "source": "zoom", "date_from": "2026-09-01",
                 "participant": "Ravi"}


def test_meeting_rows_and_label():
    rows = meeting_rows([{"id": 4, "filename": "a.wav", "created_at": "2026-09-28T10:30:00", "source": "google_meet",
                          "language": "en", "audio_duration_seconds": 600, "participant_count": 3,
                          "action_item_count": 5, "status": "completed"}])
    assert rows[0]["Source"] == "Google Meet" and rows[0]["Duration"] == "10m 00s" and rows[0]["Status"] == "Completed"
    assert meeting_label({"id": 4, "filename": "a.wav", "created_at": "2026-09-28T10:30:00"}).startswith("#4 · a.wav")
    assert meeting_rows([{"id": 1, "filename": "x"}])[0]["Participants"] == 0


def test_action_items_sorted_by_priority_then_deadline():
    rows = action_item_rows([
        {"task": "low", "priority": "Low", "deadline": "Friday"},
        {"task": "high-nodate", "priority": "High", "deadline": ""},
        {"task": "high-date", "priority": "High", "deadline": "Monday", "participant_name": "Ravi"},
        {"task": "odd", "priority": None}])
    assert [r["Task"] for r in rows] == ["high-date", "high-nodate", "low", "odd"]
    assert rows[0]["Assignee"] == "Ravi" and rows[1]["Assignee"] == "Unassigned" and rows[3]["Priority"] == "-"


def test_chart_series_and_source_row():
    assert chart_series({"High": 2}, ("High", "Medium", "Low")) == {"High": 2, "Medium": 0, "Low": 0}
    assert chart_series({"a": 1}) == {"a": 1}
    row = source_row({"meeting_id": 3, "meeting_title": "x.wav", "meeting_date": "2026-09-28T10:30:00",
                      "content_type": "action_item", "relevance_score": 0.876})
    assert row == {"Meeting": "#3 · x.wav", "Date": "28 Sep 2026, 10:30", "Type": "Action Item", "Relevance": "88%"}


def test_password_problems():
    assert password_problems("goodpass1", "goodpass1") == []
    assert len(password_problems("abc", "abd")) == 2
