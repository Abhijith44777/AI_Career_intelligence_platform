"""
test_analytics_reports.py
--------------------------
Milestone 4 Task 2 (analytics + details) and Task 6 (PDF / CSV export).
"""

import csv
import io

from analytics import meeting_analytics, overview_analytics, top_keywords
from reports import _csv_safe, report_filename, report_to_csv, report_to_pdf
from testutil import OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env


def seeded():
    env = Env()
    a = env.user("alice")
    ma = env.upload(a["id"], SAMPLE_TRANSCRIPT, "launch.wav", duration=600.0)["meeting_id"]
    b = env.user("bob")
    mb = env.upload(b["id"], OTHER_TRANSCRIPT, "finance.wav", duration=120.0)["meeting_id"]
    return env, a, b, ma, mb


def read_csv(data: bytes) -> list:
    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))


# -- Details page payload --------------------------------------------------------

def test_details_payload_has_every_required_section_for_the_selected_meeting():
    env, a, b, ma, mb = seeded()
    d = env.services.get_details(a["id"], ma)
    for key in ("meeting", "transcript", "summary_text", "key_points", "decisions", "action_items",
                "participants", "deadlines", "analytics"):
        assert key in d
    assert d["meeting"]["id"] == ma and d["transcript"] == SAMPLE_TRANSCRIPT
    assert {i["meeting_id"] for i in d["action_items"]} == {ma}
    assert {p["name"] for p in d["participants"]} >= {"Ravi", "Priya", "Karan"}
    assert "Friday" in d["deadlines"]
    assert d["decisions"], "the decisions in the transcript were extracted"


def test_details_never_mix_in_another_meetings_content():
    env, a, b, ma, mb = seeded()
    text = repr(env.services.get_details(a["id"], ma))
    assert "Meera" not in text and "budget" not in text
    text_b = repr(env.services.get_details(b["id"], mb))
    assert "Ravi" not in text_b and "mobile" not in text_b


# -- Analytics ---------------------------------------------------------------------------

def test_meeting_analytics_numbers():
    meeting = {"id": 1, "transcript": "word " * 300, "audio_duration_seconds": 120.0}
    items = [
        {"participant_name": "Ravi", "priority": "High", "status": "Completed", "deadline": "Friday"},
        {"participant_name": "Ravi", "priority": "Low", "status": "Not Started", "deadline": ""},
        {"participant_name": "Priya", "priority": "High", "status": "In Progress", "deadline": "Monday"},
        {"participant_name": "Unknown", "priority": "Medium", "status": "Not Started", "deadline": None},
    ]
    a = meeting_analytics(meeting, {"decisions": ["d1", "d2"], "key_points": ["k"]}, items,
                          [{"name": "Ravi"}, {"name": "Priya"}], ["Friday", "Monday"])
    assert a["word_count"] == 300 and a["duration_minutes"] == 2.0 and a["words_per_minute"] == 150.0
    assert a["action_item_count"] == 4 and a["decision_count"] == 2 and a["key_point_count"] == 1
    assert a["by_priority"] == {"High": 2, "Medium": 1, "Low": 1}
    assert a["by_assignee"]["Ravi"] == 2 and a["unassigned_action_items"] == 1
    assert a["action_items_with_deadline"] == 2 and a["action_items_without_deadline"] == 2
    assert a["completion_rate"] == 25.0 and a["participant_count"] == 2


def test_analytics_handles_empty_and_zero_duration_meetings():
    a = meeting_analytics({"id": 1, "transcript": "", "audio_duration_seconds": None}, None, [], [], [])
    assert a["words_per_minute"] is None and a["completion_rate"] == 0.0 and a["action_item_count"] == 0


def test_overview_analytics_is_scoped_to_the_callers_data():
    env, a, b, ma, mb = seeded()
    o = env.services.get_overview(a["id"])
    assert o["meeting_count"] == 1 and o["total_hours"] == round(600 / 3600, 2)
    assert o["action_item_count"] == len(env.db.get_action_items(ma))
    assert "Meera" not in {x["name"] for x in o["top_assignees"]}
    assert overview_analytics([], []) ["meeting_count"] == 0


def test_top_keywords_skips_stopwords():
    kws = [k["keyword"] for k in top_keywords("launch launch launch the the the migration migration once")]
    assert kws[:2] == ["launch", "migration"] and "the" not in kws


# -- CSV -------------------------------------------------------------------------------------

def test_csv_contains_exactly_the_selected_meetings_information():
    env, a, b, ma, mb = seeded()
    data, media, name = env.services.export_report(a["id"], ma, "csv")
    rows = read_csv(data)
    assert rows[0] == ["Section", "Item", "Assignee", "Deadline", "Priority", "Status", "Details"]
    sections = {r[0] for r in rows[1:]}
    assert {"Meeting Details", "Summary", "Decisions", "Action Items", "Participants", "Deadlines"} <= sections
    assert media.startswith("text/csv") and name == f"meeting_{ma}_launch.csv"
    flat = " ".join(" ".join(r) for r in rows)
    assert "launch.wav" in flat and "Ravi" in flat and "Friday" in flat
    assert "Meera" not in flat and "finance" not in flat
    actions = [r for r in rows if r[0] == "Action Items"]
    assert len(actions) == len(env.db.get_action_items(ma))
    ravi = next(r for r in actions if r[2] == "Ravi")
    assert ravi[3] == "Friday" and "API integration" in ravi[1]


def test_csv_neutralises_spreadsheet_formulas():
    assert _csv_safe("=SUM(A1:A9)") == "'=SUM(A1:A9)"
    for evil in ("+1", "-2", "@cmd", "\tx"):
        assert _csv_safe(evil).startswith("'")
    assert _csv_safe("normal") == "normal" and _csv_safe(None) == ""
    report = {"meeting": {"id": 1, "filename": "=evil.wav", "created_at": "2026-01-01T00:00:00"},
              "summary_text": "=1+1", "key_points": [], "decisions": ["@x"], "participants": [], "deadlines": [],
              "action_items": [{"task": '=HYPERLINK("http://evil")', "participant_name": "-A", "deadline": "",
                                "priority": "", "status": ""}]}
    for row in read_csv(report_to_csv(report))[1:]:
        assert not any(cell.startswith(("=", "+", "-", "@")) for cell in row), row


def test_csv_preserves_unicode_and_commas():
    report = {"meeting": {"id": 2, "filename": "बैठक.wav", "created_at": "2026-01-01T00:00:00"},
              "summary_text": "Budget, \"final\", approved", "key_points": [], "decisions": ["Ship – café"],
              "participants": ["Zoë"], "deadlines": [], "action_items": []}
    rows = read_csv(report_to_csv(report))
    assert ["Summary", "Summary", "", "", "", "", 'Budget, "final", approved'] in rows
    assert any("बैठक.wav" in r[6] for r in rows) and any(r[1] == "Zoë" for r in rows)


# -- PDF ---------------------------------------------------------------------------------------

def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)


def test_pdf_is_valid_and_contains_the_selected_meetings_information():
    env, a, b, ma, mb = seeded()
    data, media, name = env.services.export_report(a["id"], ma, "pdf")
    assert data[:5] == b"%PDF-" and media == "application/pdf" and name.endswith(".pdf")
    text = pdf_text(data)
    for expected in ("Meeting Report", "launch.wav", "Summary", "Decisions", "Action Items", "Participants",
                     "Deadlines", "Ravi", "Priya", "Friday"):
        assert expected in text, expected
    assert "Meera" not in text and "finance" not in text


def test_pdf_escapes_markup_and_handles_empty_meetings():
    report = {"meeting": {"id": 3, "filename": "<b>x</b>.wav", "created_at": "2026-01-01T00:00:00"},
              "summary_text": "A <i>tag</i> & an ampersand", "key_points": [], "decisions": [],
              "participants": [], "deadlines": [], "action_items": []}
    text = pdf_text(report_to_pdf(report))
    assert "<b>x</b>.wav" in text and "<i>tag</i>" in text and "No action items." in text


def test_report_filenames_are_safe():
    r = {"meeting": {"id": 9, "filename": "../../etc/passwd; rm -rf.mp4"}}
    name = report_filename(r, "pdf")
    assert "/" not in name and ".." not in name and " " not in name and name.startswith("meeting_9_")


def test_invalid_report_format_is_rejected():
    from services import BadRequest
    from testutil import raises
    env, a, b, ma, mb = seeded()
    with raises(BadRequest):
        env.services.export_report(a["id"], ma, "docx")
