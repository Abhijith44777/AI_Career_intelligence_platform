"""
participants.py
-----------------
Milestone 2 - Task 4: Participant & Responsibility Mapping

Verifies/ensures:
    - Participant names are correctly identified
    - Names are mapped consistently (e.g. "ravi", "Ravi", "RAVI " -> one person)
    - Multiple participants are handled
    - Unknown participants are handled safely (never crash, never silently drop a task)
    - Duplicate participant records are avoided
    - Responsibilities (action items) are linked to the correct meeting
"""

import re
import difflib
from dataclasses import dataclass, field

UNKNOWN_PARTICIPANT = "Unknown"

# How similar two names need to be (0-1) to be treated as the same person
# when they aren't an exact match after normalization - catches minor
# transcription variants like "Ravi" vs "Ravy" without merging unrelated
# people like "Ravi" and "Priya".
FUZZY_MATCH_THRESHOLD = 0.85


def normalize_name(name: str) -> str:
    """
    Consistent name normalization: trim whitespace, collapse internal
    whitespace, title-case. "ravi   kumar" and "RAVI KUMAR" both become
    "Ravi Kumar" so they map to the same participant record.
    """
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name.strip())
    return cleaned.title()


@dataclass
class Participant:
    id: int
    name: str
    aliases: set = field(default_factory=set)  # raw name variants seen, for audit/debug


class ParticipantRegistry:
    """
    In-memory registry used during a single meeting's processing to
    resolve every mentioned name (from `participants` and from each
    action item's `assignee`) to ONE canonical Participant record,
    avoiding duplicates within that meeting.

    Persistence (Task 5) is responsible for merging this registry's
    output against participants that already exist in the database
    across meetings.
    """

    def __init__(self):
        self._participants: list = []
        self._next_id = 1

    def _find_existing(self, normalized_name: str) -> Participant:
        # 1. Exact match on normalized name
        for p in self._participants:
            if p.name == normalized_name:
                return p

        # 2. Fuzzy match - guards against minor Whisper/LLM name-spelling
        #    variance within the same meeting without merging distinct people.
        for p in self._participants:
            similarity = difflib.SequenceMatcher(None, p.name.lower(), normalized_name.lower()).ratio()
            if similarity >= FUZZY_MATCH_THRESHOLD:
                return p

        return None

    def find_or_create(self, raw_name: str) -> Participant:
        """
        Resolve a raw name string to a canonical Participant, creating a
        new record only if no existing match is found. Handles unknown/
        blank names safely by mapping them to a shared "Unknown" participant
        rather than crashing or fabricating a name.
        """
        normalized = normalize_name(raw_name)

        if not normalized or normalized.lower() in {"unassigned", "unknown", "n/a", "none"}:
            normalized = UNKNOWN_PARTICIPANT

        existing = self._find_existing(normalized)
        if existing:
            if raw_name:
                existing.aliases.add(raw_name.strip())
            return existing

        participant = Participant(id=self._next_id, name=normalized, aliases={raw_name.strip()} if raw_name else set())
        self._next_id += 1
        self._participants.append(participant)
        return participant

    def all(self) -> list:
        return list(self._participants)


def map_participants_and_responsibilities(participants_raw: list, action_items: list) -> dict:
    """
    Task 4 entry point.

    Args:
        participants_raw: list of raw name strings from the LLM's
            "participants" field.
        action_items: list of ActionItem objects (from Task 3) whose
            `.assignee` fields need resolving to canonical participants.

    Returns:
        {
            "participants": [Participant, ...],   # deduplicated
            "action_items": [                       # each paired with its resolved participant
                {"action_item": ActionItem, "participant": Participant}
            ]
        }
    """
    registry = ParticipantRegistry()

    # Register everyone explicitly listed as a participant first, so
    # "Ravi" from the participants list and "Ravi" as an assignee resolve
    # to the SAME record instead of two.
    for raw_name in participants_raw or []:
        registry.find_or_create(raw_name)

    linked_action_items = []
    for item in action_items:
        participant = registry.find_or_create(item.assignee)
        linked_action_items.append({"action_item": item, "participant": participant})

    return {
        "participants": registry.all(),
        "action_items": linked_action_items,
    }
