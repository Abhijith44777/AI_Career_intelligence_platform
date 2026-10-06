"""
analytics.py
-------------
Milestone 4, Task 2: Meeting analytics (per meeting) plus the cross-meeting
overview shown on the dashboard home page.

Pure functions over already-fetched data - no I/O - so they are trivially
testable and can never leak another meeting's data by themselves.
"""

import re
from collections import Counter

from embedding_service import _STOPWORDS

PRIORITIES = ("High", "Medium", "Low")
STATUSES = ("Not Started", "In Progress", "Completed")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def _count_words(text: str) -> int:
    return len((text or "").split())


def top_keywords(text: str, limit: int = 8) -> list:
    counts = Counter(w.lower() for w in _WORD_RE.findall(text or ""))
    words = [(w, c) for w, c in counts.most_common() if w not in _STOPWORDS and c > 1]
    return [{"keyword": w, "count": c} for w, c in words[:limit]]


def meeting_analytics(meeting: dict, summary: dict, action_items: list, participants: list,
                      deadlines: list) -> dict:
    """Everything here is derived from the ONE meeting passed in."""
    summary = summary or {}
    transcript = meeting.get("transcript") or ""
    words = _count_words(transcript)
    duration = meeting.get("audio_duration_seconds") or 0.0
    total = len(action_items)

    by_priority = Counter((i.get("priority") or "Unspecified") for i in action_items)
    by_status = Counter((i.get("status") or "Unspecified") for i in action_items)
    by_assignee = Counter((i.get("participant_name") or "Unassigned") for i in action_items)
    completed = by_status.get("Completed", 0) + by_status.get("Done", 0)

    return {
        "meeting_id": meeting.get("id"),
        "word_count": words,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60.0, 1),
        "words_per_minute": round(words / (duration / 60.0), 1) if duration > 0 else None,
        "participant_count": len(participants),
        "decision_count": len(summary.get("decisions") or []),
        "key_point_count": len(summary.get("key_points") or []),
        "action_item_count": total,
        "deadline_count": len(deadlines),
        "action_items_with_deadline": sum(1 for i in action_items if (i.get("deadline") or "").strip()),
        "action_items_without_deadline": sum(1 for i in action_items if not (i.get("deadline") or "").strip()),
        "unassigned_action_items": by_assignee.get("Unassigned", 0) + by_assignee.get("Unknown", 0),
        "completion_rate": round(100.0 * completed / total, 1) if total else 0.0,
        "by_priority": {p: by_priority.get(p, 0) for p in PRIORITIES} | (
            {"Unspecified": by_priority["Unspecified"]} if by_priority.get("Unspecified") else {}),
        "by_status": dict(by_status),
        "by_assignee": dict(by_assignee.most_common()),
        "top_keywords": top_keywords(transcript),
    }


def overview_analytics(meetings: list, action_items: list) -> dict:
    """Cross-meeting totals for the dashboard home page. `meetings` come from
    Database.search_meetings (already scoped to one user)."""
    by_priority = Counter((i.get("priority") or "Unspecified") for i in action_items)
    by_status = Counter((i.get("status") or "Unspecified") for i in action_items)
    by_assignee = Counter((i.get("participant_name") or "Unassigned") for i in action_items)
    per_day = Counter((m.get("created_at") or "")[:10] for m in meetings if m.get("created_at"))
    completed = by_status.get("Completed", 0) + by_status.get("Done", 0)
    total_seconds = sum((m.get("audio_duration_seconds") or 0.0) for m in meetings)

    return {
        "meeting_count": len(meetings),
        "total_hours": round(total_seconds / 3600.0, 2),
        "action_item_count": len(action_items),
        "open_action_items": len(action_items) - completed,
        "completed_action_items": completed,
        "completion_rate": round(100.0 * completed / len(action_items), 1) if action_items else 0.0,
        "by_priority": dict(by_priority),
        "by_status": dict(by_status),
        "top_assignees": [{"name": n, "count": c} for n, c in by_assignee.most_common(5)],
        "meetings_per_day": dict(sorted(per_day.items())),
        "by_source": dict(Counter(m.get("source") or "upload" for m in meetings)),
        "by_language": dict(Counter(m.get("language") or "unknown" for m in meetings)),
        "failed_meetings": sum(1 for m in meetings if m.get("status") == "failed"),
    }
