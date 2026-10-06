"""
summarization.py
------------------
Milestone 2 - Task 2: Meeting Summarization Module

Turns the raw structured output from the LLM service into a clean,
production-ready "MeetingSummary" and renders it in the human-readable
format specified in the task:

    Summary:
    The team discussed the mobile application launch and assigned
    API integration and UI testing responsibilities.

    Key Decisions:
    - Continue with the planned mobile application launch.

    Action Items:
    - Complete API integration - Ravi - Friday
    - Prepare UI testing report - Priya
"""

from dataclasses import dataclass, field

from llm_service import LLMService, EMPTY_MEETING_INTELLIGENCE


@dataclass
class MeetingSummary:
    summary: str
    key_points: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    action_items: list = field(default_factory=list)   # raw dicts, refined later by action_items.py
    participants: list = field(default_factory=list)
    deadlines: list = field(default_factory=list)
    priorities: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "MeetingSummary":
        merged = dict(EMPTY_MEETING_INTELLIGENCE)
        merged.update(data or {})
        return cls(
            summary=merged["summary"],
            key_points=merged["key_points"],
            decisions=merged["decisions"],
            action_items=merged["action_items"],
            participants=merged["participants"],
            deadlines=merged["deadlines"],
            priorities=merged["priorities"],
        )

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "decisions": self.decisions,
            "action_items": self.action_items,
            "participants": self.participants,
            "deadlines": self.deadlines,
            "priorities": self.priorities,
        }

    def is_empty(self) -> bool:
        """True if the LLM essentially found nothing useful in the transcript."""
        return not self.summary and not self.key_points and not self.decisions and not self.action_items


def summarize_meeting(transcript: str, llm_service: LLMService = None) -> MeetingSummary:
    """
    Task 2 entry point: transcript -> MeetingSummary.

    This is a thin, production-facing wrapper around the LLM service
    (Task 1) - it exists so the rest of the app depends on a stable
    `MeetingSummary` object rather than raw LLM JSON, and so summarization
    concerns (formatting, empty-result handling) live in one place.
    """
    llm_service = llm_service or LLMService()
    raw = llm_service.extract_meeting_intelligence(transcript)
    return MeetingSummary.from_dict(raw)


def format_action_item_line(item: dict) -> str:
    """
    Renders one action item as: "Task - Assignee - Deadline"
    omitting parts that are empty, matching the task's example format:
        "Complete API integration - Ravi - Friday"
        "Prepare UI testing report - Priya"
    """
    parts = [item.get("task", "").strip()]
    if item.get("assignee", "").strip():
        parts.append(item["assignee"].strip())
    if item.get("deadline", "").strip():
        parts.append(item["deadline"].strip())
    return " - ".join(p for p in parts if p)


def render_summary_text(meeting_summary: MeetingSummary) -> str:
    """
    Render a MeetingSummary as the human-readable text block shown in the
    Task 2 spec (Summary / Key Decisions / Action Items sections).
    """
    lines = []

    lines.append("Summary:")
    lines.append(meeting_summary.summary or "(no summary generated)")
    lines.append("")

    lines.append("Key Decisions:")
    if meeting_summary.decisions:
        for d in meeting_summary.decisions:
            lines.append(f"- {d}")
    else:
        lines.append("- (none identified)")
    lines.append("")

    lines.append("Action Items:")
    if meeting_summary.action_items:
        for item in meeting_summary.action_items:
            lines.append(f"- {format_action_item_line(item)}")
    else:
        lines.append("- (none identified)")

    return "\n".join(lines)
