"""
test_participants.py
----------------------
Tests for Milestone 2 - Task 4 (Participant & Responsibility Mapping).
"""

from participants import (
    normalize_name,
    ParticipantRegistry,
    map_participants_and_responsibilities,
    UNKNOWN_PARTICIPANT,
)
from action_items import extract_and_validate_action_items


def test_normalize_name_handles_case_and_whitespace():
    assert normalize_name("ravi   kumar") == "Ravi Kumar"
    assert normalize_name("RAVI") == "Ravi"
    assert normalize_name("  priya  ") == "Priya"


def test_normalize_name_empty_returns_empty():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


def test_registry_dedupes_exact_case_variants():
    registry = ParticipantRegistry()
    p1 = registry.find_or_create("Ravi")
    p2 = registry.find_or_create("ravi")
    p3 = registry.find_or_create("RAVI")
    assert p1.id == p2.id == p3.id
    assert len(registry.all()) == 1


def test_registry_handles_multiple_distinct_participants():
    registry = ParticipantRegistry()
    registry.find_or_create("Ravi")
    registry.find_or_create("Priya")
    registry.find_or_create("Amit")
    assert len(registry.all()) == 3


def test_registry_handles_unknown_participant_safely():
    registry = ParticipantRegistry()
    p1 = registry.find_or_create("")
    p2 = registry.find_or_create("Unassigned")
    p3 = registry.find_or_create(None)
    # All blank/unassigned variants map to the SAME shared Unknown record
    assert p1.name == UNKNOWN_PARTICIPANT
    assert p1.id == p2.id == p3.id
    assert len(registry.all()) == 1


def test_registry_does_not_merge_unrelated_names():
    registry = ParticipantRegistry()
    p1 = registry.find_or_create("Ravi")
    p2 = registry.find_or_create("Priya")
    assert p1.id != p2.id
    assert p1.name != p2.name


def test_full_mapping_links_action_items_to_correct_participant():
    raw_action_items = [
        {"task": "Complete API integration", "assignee": "Ravi", "deadline": "Friday", "priority": "High", "status": "Not Started"},
        {"task": "Prepare UI testing report", "assignee": "Priya", "deadline": "", "priority": "Medium", "status": "Not Started"},
    ]
    action_items = extract_and_validate_action_items(raw_action_items)
    result = map_participants_and_responsibilities(["Ravi", "Priya"], action_items)

    assert len(result["participants"]) == 2
    assert len(result["action_items"]) == 2

    ravi_entry = next(e for e in result["action_items"] if e["action_item"].task == "Complete API integration")
    assert ravi_entry["participant"].name == "Ravi"

    priya_entry = next(e for e in result["action_items"] if e["action_item"].task == "Prepare UI testing report")
    assert priya_entry["participant"].name == "Priya"


def test_participant_from_action_items_list_merges_with_participants_list():
    # "Ravi" appears both in the participants list AND as an assignee -
    # must resolve to ONE record, not two.
    raw_action_items = [{"task": "Do X", "assignee": "ravi", "deadline": "", "priority": "Medium", "status": "Not Started"}]
    action_items = extract_and_validate_action_items(raw_action_items)
    result = map_participants_and_responsibilities(["Ravi"], action_items)

    assert len(result["participants"]) == 1
    assert result["action_items"][0]["participant"].id == result["participants"][0].id


def test_unassigned_action_item_maps_to_unknown_not_crash():
    raw_action_items = [{"task": "Some orphan task", "assignee": "", "deadline": "", "priority": "Medium", "status": "Not Started"}]
    action_items = extract_and_validate_action_items(raw_action_items)
    result = map_participants_and_responsibilities([], action_items)

    assert len(result["action_items"]) == 1
    assert result["action_items"][0]["participant"].name == UNKNOWN_PARTICIPANT


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
