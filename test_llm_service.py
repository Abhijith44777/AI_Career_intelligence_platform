"""
test_llm_service.py
---------------------
Tests for Milestone 2 - Task 1 (LLM Service & Prompt Engineering).

These tests exercise everything that does NOT require a live API call:
JSON parsing robustness, schema validation, chunking, and merge logic.
The actual LLM call (_call_llm_raw) needs a real API key and is exercised
manually / in integration testing, not here.
"""

from llm_service import (
    validate_structured_output,
    parse_json_response,
    chunk_transcript,
    merge_lists_unique,
    naive_merge_partials,
    LLMValidationError,
    EMPTY_MEETING_INTELLIGENCE,
)

VALID_RESPONSE = {
    "summary": "The team discussed the mobile app launch.",
    "key_points": ["Launch timeline confirmed"],
    "decisions": ["Continue with planned launch"],
    "action_items": [
        {"task": "Complete API integration", "assignee": "Ravi", "deadline": "Friday", "priority": "High", "status": "Not Started"}
    ],
    "participants": ["Ravi", "Priya"],
    "deadlines": ["Friday"],
    "priorities": ["Mobile launch"],
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_valid_response_passes_validation():
    result = validate_structured_output(VALID_RESPONSE)
    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_field_fails_validation():
    bad = dict(VALID_RESPONSE)
    del bad["action_items"]
    result = validate_structured_output(bad)
    assert result["valid"] is False
    assert any("action_items" in e for e in result["errors"])


def test_wrong_type_fails_validation():
    bad = dict(VALID_RESPONSE)
    bad["summary"] = 12345  # should be a string
    result = validate_structured_output(bad)
    assert result["valid"] is False


def test_non_dict_top_level_fails():
    result = validate_structured_output(["not", "a", "dict"])
    assert result["valid"] is False


def test_malformed_action_item_fails():
    bad = dict(VALID_RESPONSE)
    bad["action_items"] = [{"task": "Do thing"}]  # missing assignee/deadline/priority/status
    result = validate_structured_output(bad)
    assert result["valid"] is False
    assert any("action_items[0]" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# JSON parsing robustness
# ---------------------------------------------------------------------------

def test_parse_clean_json():
    import json
    text = json.dumps(VALID_RESPONSE)
    parsed = parse_json_response(text)
    assert parsed["summary"] == VALID_RESPONSE["summary"]


def test_parse_json_with_markdown_fence():
    import json
    text = f"```json\n{json.dumps(VALID_RESPONSE)}\n```"
    parsed = parse_json_response(text)
    assert parsed["summary"] == VALID_RESPONSE["summary"]


def test_parse_json_with_leading_prose():
    import json
    text = f"Here is the extracted meeting data:\n{json.dumps(VALID_RESPONSE)}\nLet me know if you need anything else."
    parsed = parse_json_response(text)
    assert parsed["summary"] == VALID_RESPONSE["summary"]


def test_parse_empty_response_raises():
    try:
        parse_json_response("")
        assert False, "should have raised"
    except LLMValidationError:
        pass


def test_parse_garbage_raises():
    try:
        parse_json_response("this is not json at all")
        assert False, "should have raised"
    except LLMValidationError:
        pass


# ---------------------------------------------------------------------------
# Chunking - Task 1's "Handle token/context limitations for long transcripts"
# ---------------------------------------------------------------------------

def test_short_transcript_not_chunked():
    text = "This is a short meeting transcript."
    chunks = chunk_transcript(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_transcript_is_chunked():
    sentence = "This is one sentence of meeting discussion. "
    long_text = sentence * 200  # way over any reasonable chunk size
    chunks = chunk_transcript(long_text, max_chars=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500 or " " not in chunk  # allow only for unsplittable single "sentence"


def test_chunking_preserves_all_content_roughly():
    sentence = "Speaker A said something important. "
    long_text = sentence * 100
    chunks = chunk_transcript(long_text, max_chars=300)
    rejoined_word_count = sum(len(c.split()) for c in chunks)
    original_word_count = len(long_text.split())
    assert rejoined_word_count == original_word_count


def test_single_sentence_longer_than_max_chars_hard_split():
    huge_sentence = "word " * 1000  # one giant "sentence" with no punctuation
    chunks = chunk_transcript(huge_sentence, max_chars=200)
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Merge logic (used when a transcript had to be chunked)
# ---------------------------------------------------------------------------

def test_merge_lists_unique_deduplicates_case_insensitive():
    result = merge_lists_unique(["Ravi", "Priya"], ["ravi", "Amit"])
    assert result == ["Ravi", "Priya", "Amit"]


def test_naive_merge_partials_combines_action_items_without_duplicates():
    partial_1 = dict(EMPTY_MEETING_INTELLIGENCE)
    partial_1["action_items"] = [{"task": "Complete API integration", "assignee": "Ravi", "deadline": "Friday", "priority": "High", "status": "Not Started"}]
    partial_1["participants"] = ["Ravi"]

    partial_2 = dict(EMPTY_MEETING_INTELLIGENCE)
    # Same task+assignee repeated across chunks - should be deduplicated
    partial_2["action_items"] = [{"task": "Complete API integration", "assignee": "Ravi", "deadline": "Friday", "priority": "High", "status": "Not Started"}]
    partial_2["action_items"].append({"task": "Prepare UI testing report", "assignee": "Priya", "deadline": "", "priority": "Medium", "status": "Not Started"})
    partial_2["participants"] = ["Priya"]

    merged = naive_merge_partials([partial_1, partial_2])
    assert len(merged["action_items"]) == 2
    assert set(merged["participants"]) == {"Ravi", "Priya"}


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
