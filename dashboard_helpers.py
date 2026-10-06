"""
dashboard_helpers.py
---------------------
Milestone 4, Task 1: the pure (Streamlit-free) logic behind the dashboard, so
formatting, filter building and table preparation are unit-tested.
"""

from datetime import datetime

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
STATUS_LABELS = {"completed": "Completed", "processing": "Processing", "failed": "Failed"}
SOURCE_LABELS = {"upload": "Upload", "zoom": "Zoom", "google_meet": "Google Meet"}


def fmt_duration(seconds) -> str:
    if not seconds:
        return "-"
    seconds = int(round(float(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def fmt_date(iso: str) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso[:16]


def build_filters(text="", language="All", status="All", source="All", date_from=None, date_to=None,
                  participant="All") -> dict:
    """UI widget values -> /meetings query parameters ('All' / empty / None are dropped)."""
    def pick(value):
        return None if value in (None, "", "All") else value

    def day(value):
        return value.isoformat() if hasattr(value, "isoformat") else (value or None)

    filters = {"text": (text or "").strip() or None, "language": pick(language), "status": pick(status),
               "source": pick(source), "date_from": day(date_from), "date_to": day(date_to),
               "participant": pick(participant)}
    return {k: v for k, v in filters.items() if v}


def meeting_rows(meetings: list) -> list:
    """Rows for the meeting-list table."""
    return [{
        "ID": m["id"], "Meeting": m["filename"], "Date": fmt_date(m.get("created_at")),
        "Source": SOURCE_LABELS.get(m.get("source"), m.get("source") or "Upload"),
        "Language": m.get("language") or "-", "Duration": fmt_duration(m.get("audio_duration_seconds")),
        "Participants": m.get("participant_count", 0), "Action items": m.get("action_item_count", 0),
        "Status": STATUS_LABELS.get(m.get("status"), m.get("status") or "-"),
    } for m in meetings]


def meeting_label(m: dict) -> str:
    return f"#{m['id']} · {m['filename']} · {fmt_date(m.get('created_at'))}"


def action_item_rows(items: list) -> list:
    """Sorted by priority (High first) then deadline presence."""
    ordered = sorted(items, key=lambda i: (PRIORITY_ORDER.get(i.get("priority"), 3), not (i.get("deadline") or "")))
    return [{"Task": i.get("task", ""), "Assignee": i.get("participant_name") or "Unassigned",
             "Deadline": i.get("deadline") or "-", "Priority": i.get("priority") or "-",
             "Status": i.get("status") or "-"} for i in ordered]


def chart_series(mapping: dict, order: tuple = None) -> dict:
    """{label: count} -> chart-ready dict (optionally in a fixed order, zeros kept)."""
    if order:
        return {k: mapping.get(k, 0) for k in order}
    return dict(mapping)


def source_row(s: dict) -> dict:
    """RAG/search source -> display dict (meeting, date, type, relevance, text)."""
    return {"Meeting": f"#{s['meeting_id']} · {s.get('meeting_title', '')}",
            "Date": fmt_date(s.get("meeting_date")),
            "Type": (s.get("content_type") or "").replace("_", " ").title(),
            "Relevance": f"{float(s.get('relevance_score', 0)) * 100:.0f}%"}


def password_problems(password: str, confirm: str) -> list:
    problems = []
    if password != confirm:
        problems.append("Passwords do not match.")
    if len(password) < 8:
        problems.append("Password must be at least 8 characters.")
    return problems
