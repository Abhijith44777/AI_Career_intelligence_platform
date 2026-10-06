"""
action_items.py
-----------------
Milestone 2 - Task 3: Action Item Extraction Engine

The LLM (Task 1) already extracts candidate action items as part of the
structured output. This module is the production hardening layer on top
of that: normalizing fields, filling safe defaults, inferring priority
when the LLM left it ambiguous, and validating each item before it's
allowed into the database.

Each action item is guaranteed to have:
    - assigned participant (or explicit "Unassigned")
    - deadline (or explicit "No deadline")
    - priority (High / Medium / Low)
    - status (Not Started / In Progress / Completed)
"""

from dataclasses import dataclass

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_STATUSES = {"Not Started", "In Progress", "Completed"}

# Keyword signals used to infer priority when the LLM didn't set one
# confidently, or to sanity-check/upgrade what it returned.
HIGH_PRIORITY_KEYWORDS = {"urgent", "asap", "immediately", "critical", "blocker", "today", "high priority"}
LOW_PRIORITY_KEYWORDS = {"whenever", "no rush", "low priority", "eventually", "nice to have"}


@dataclass
class ActionItem:
    task: str
    assignee: str
    deadline: str
    priority: str
    status: str

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "assignee": self.assignee,
            "deadline": self.deadline,
            "priority": self.priority,
            "status": self.status,
        }


class ActionItemValidationError(Exception):
    pass


def infer_priority(task_text: str, fallback: str = "Medium") -> str:
    """Infer priority from urgency language when the LLM's value is missing/invalid."""
    text = task_text.lower()
    if any(kw in text for kw in HIGH_PRIORITY_KEYWORDS):
        return "High"
    if any(kw in text for kw in LOW_PRIORITY_KEYWORDS):
        return "Low"
    return fallback


def normalize_action_item(raw: dict) -> ActionItem:
    """
    Convert a raw LLM-produced action item dict into a validated ActionItem
    with safe, consistent defaults. Never raises on missing/messy input -
    extraction pipelines should degrade gracefully, not crash a whole
    meeting's processing because one action item was malformed.
    """
    task = (raw.get("task") or "").strip()
    assignee = (raw.get("assignee") or "").strip() or "Unassigned"
    deadline = (raw.get("deadline") or "").strip() or "No deadline"

    priority = (raw.get("priority") or "").strip().title()
    if priority not in VALID_PRIORITIES:
        priority = infer_priority(task)

    status = (raw.get("status") or "").strip().title()
    # Normalize common LLM phrasing variants to the canonical set.
    status_aliases = {
        "Not Started": "Not Started", "Todo": "Not Started", "To Do": "Not Started",
        "In Progress": "In Progress", "Ongoing": "In Progress", "Doing": "In Progress",
        "Completed": "Completed", "Done": "Completed", "Complete": "Completed",
    }
    status = status_aliases.get(status, "Not Started")

    return ActionItem(task=task, assignee=assignee, deadline=deadline, priority=priority, status=status)


def normalize_action_items(raw_items: list) -> list:
    """Normalize a full list of raw action items, dropping any with no task text."""
    normalized = []
    for raw in raw_items or []:
        item = normalize_action_item(raw)
        if item.task:  # a task with no description isn't actionable - skip it
            normalized.append(item)
    return normalized


def validate_action_item(item: ActionItem) -> dict:
    """Final validation gate before an action item is persisted to the DB."""
    errors = []
    if not item.task:
        errors.append("task cannot be empty")
    if item.priority not in VALID_PRIORITIES:
        errors.append(f"invalid priority: {item.priority}")
    if item.status not in VALID_STATUSES:
        errors.append(f"invalid status: {item.status}")
    return {"valid": len(errors) == 0, "errors": errors}


def extract_and_validate_action_items(raw_items: list) -> list:
    """
    Task 3 entry point: raw LLM action items -> list of validated
    ActionItem objects, ready for participant mapping (Task 4) and
    persistence (Task 5).
    """
    normalized = normalize_action_items(raw_items)
    valid_items = []
    for item in normalized:
        check = validate_action_item(item)
        if check["valid"]:
            valid_items.append(item)
        # Invalid items are silently dropped rather than raised - a single
        # malformed item should never block the rest of the meeting's
        # action items from being saved. (Logged by the caller if needed.)
    return valid_items
