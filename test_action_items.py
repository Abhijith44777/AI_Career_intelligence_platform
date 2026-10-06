"""
test_action_items.py
----------------------
Tests for Milestone 2 - Task 3 (Action Item Extraction Engine).
"""

from action_items import (
    normalize_action_item,
    normalize_action_items,
    validate_action_item,
    extract_and_validate_action_items,
    infer_priority,
)


def test_normalize_fills_defaults_for_missing_fields():
    item = normalize_action_item({"task": "Complete API integration"})
    assert item.assignee == "Unassigned"
    assert item.deadline == "No deadline"
    assert item.status == "Not Started"
    assert item.priority in {"High", "Medium", "Low"}


def test_normalize_preserves_given_values():
    item = normalize_action_item({
        "task": "Complete API integration", "assignee": "Ravi",
        "deadline": "Friday", "priority": "High", "status": "Not Started",
    })
    assert item.assignee == "Ravi"
    assert item.deadline == "Friday"
    assert item.priority == "High"


def test_status_aliases_normalize_to_canonical_set():
    assert normalize_action_item({"task": "x", "status": "done"}).status == "Completed"
    assert normalize_action_item({"task": "x", "status": "ongoing"}).status == "In Progress"
    assert normalize_action_item({"task": "x", "status": "todo"}).status == "Not Started"
    assert normalize_action_item({"task": "x", "status": "garbage value"}).status == "Not Started"


def test_infer_priority_from_urgency_language():
    assert infer_priority("This is urgent, needs to happen ASAP") == "High"
    assert infer_priority("No rush on this one, whenever works") == "Low"
    assert infer_priority("Just a normal follow-up task") == "Medium"


def test_invalid_priority_gets_inferred_instead():
    item = normalize_action_item({"task": "Fix the urgent blocker issue", "priority": "Extremely High"})
    assert item.priority == "High"  # inferred from "urgent"/"blocker" keywords


def test_items_with_no_task_are_dropped():
    items = normalize_action_items([{"task": ""}, {"task": "Real task here"}])
    assert len(items) == 1
    assert items[0].task == "Real task here"


def test_validate_action_item_catches_empty_task():
    item = normalize_action_item({"task": ""})
    # normalize keeps empty task if explicitly passed through directly (not via normalize_action_items)
    result = validate_action_item(item)
    assert result["valid"] is False


def test_full_pipeline_end_to_end():
    raw = [
        {"task": "Complete API integration", "assignee": "Ravi", "deadline": "Friday", "priority": "High", "status": "Not Started"},
        {"task": "Prepare UI testing report", "assignee": "Priya", "deadline": "", "priority": "", "status": ""},
        {"task": ""},  # should be dropped - no task text
    ]
    result = extract_and_validate_action_items(raw)
    assert len(result) == 2
    assert result[0].assignee == "Ravi"
    assert result[1].assignee == "Priya"
    assert result[1].deadline == "No deadline"


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
