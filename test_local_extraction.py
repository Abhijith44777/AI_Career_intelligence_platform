"""
test_local_extraction.py
--------------------------
Tests for the zero-API-key local rule-based extraction engine
(local_extraction.py) - used as the default provider and as the automatic
fallback when a real LLM provider has no valid key. Fully offline, no
network, no dependencies beyond the standard library.
"""

from local_extraction import (
    split_sentences,
    extract_decisions,
    extract_priority_mentions,
    extract_deadline,
    extract_action_items,
    extract_participants,
    extract_meeting_intelligence_local,
)

SAMPLE_TRANSCRIPT = (
    "Good morning everyone, thanks for joining. Today we discussed the mobile "
    "application launch. We decided to continue with the planned mobile "
    "application launch. Ravi will complete the API integration by Friday. "
    "This is a high priority for the team. Priya needs to prepare the UI "
    "testing report. Ravi and Priya both agreed the timeline is realistic. "
    "That's everything for today, thanks all."
)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    sentences = split_sentences("This is one. This is two. This is three.")
    assert len(sentences) == 3


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ---------------------------------------------------------------------------
# Decisions & priorities
# ---------------------------------------------------------------------------

def test_extract_decisions_finds_decision_keyword():
    sentences = split_sentences(SAMPLE_TRANSCRIPT)
    decisions = extract_decisions(sentences)
    assert any("decided" in d.lower() for d in decisions)


def test_extract_decisions_catches_implicit_commitment_phrasing():
    # No explicit "decided"/"agreed" keyword - just a team-oriented,
    # forward-looking commitment, which real meetings often use instead.
    sentences = split_sentences(
        "We're going to switch to the new vendor starting next quarter. "
        "The team will adopt the updated process from Monday."
    )
    decisions = extract_decisions(sentences)
    assert len(decisions) == 2


def test_extract_priority_mentions_finds_priority_language():
    sentences = split_sentences(SAMPLE_TRANSCRIPT)
    priorities = extract_priority_mentions(sentences)
    assert any("priority" in p.lower() for p in priorities)


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------

def test_extract_deadline_finds_weekday():
    assert extract_deadline("Please finish this by Friday.") == "Friday"


def test_extract_deadline_returns_empty_when_none_present():
    assert extract_deadline("This sentence has no date in it.") == ""


def test_extract_deadline_finds_eod():
    assert extract_deadline("Send it over EOD please.").upper() == "EOD"


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

def test_extract_action_items_captures_named_assignee_and_task():
    sentences = split_sentences(SAMPLE_TRANSCRIPT)
    items = extract_action_items(sentences)
    assert len(items) >= 2
    ravi_items = [i for i in items if i["assignee"] == "Ravi"]
    assert ravi_items, f"expected a Ravi action item, got: {items}"
    assert "api integration" in ravi_items[0]["task"].lower()
    assert ravi_items[0]["deadline"] == "Friday"


def test_extract_action_items_every_item_matches_schema_keys():
    sentences = split_sentences(SAMPLE_TRANSCRIPT)
    items = extract_action_items(sentences)
    for item in items:
        assert set(item.keys()) == {"task", "assignee", "deadline", "priority", "status"}
        assert item["status"] == "Not Started"


def test_extract_action_items_no_duplicates():
    sentences = split_sentences(SAMPLE_TRANSCRIPT + " Ravi will complete the API integration by Friday.")
    items = extract_action_items(sentences)
    keys = [(i["assignee"].lower(), i["task"].lower()) for i in items]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

def test_extract_participants_includes_action_item_assignees():
    sentences = split_sentences(SAMPLE_TRANSCRIPT)
    items = extract_action_items(sentences)
    participants = extract_participants(sentences, items)
    assert "Ravi" in participants
    assert "Priya" in participants


def test_extract_participants_single_mention_full_name_included():
    sentences = split_sentences(
        "Ravi Kumar joined a bit late today. He shared some updates on the project."
    )
    participants = extract_participants(sentences, [])
    assert "Ravi Kumar" in participants


def test_extract_participants_single_word_name_needs_repetition():
    sentences = split_sentences(
        "Then Sanjay left the room briefly. Nobody else mentioned him again."
    )
    participants = extract_participants(sentences, [])
    assert "Sanjay" not in participants


def test_extract_participants_speaking_verb_counts_on_single_mention():
    sentences = split_sentences(
        "Priya mentioned that the vendor contract needs review. No one else spoke on it."
    )
    participants = extract_participants(sentences, [])
    assert "Priya" in participants


def test_extract_participants_excludes_common_sentence_starters():
    sentences = split_sentences("This is fine. This works well. This should be okay.")
    participants = extract_participants(sentences, [])
    assert "This" not in participants


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def test_full_extraction_matches_expected_schema_keys():
    result = extract_meeting_intelligence_local(SAMPLE_TRANSCRIPT)
    assert set(result.keys()) == {
        "summary", "key_points", "decisions", "action_items",
        "participants", "deadlines", "priorities",
    }
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["action_items"], list)


def test_full_extraction_on_empty_transcript_returns_empty_schema():
    result = extract_meeting_intelligence_local("")
    assert result["summary"] == ""
    assert result["action_items"] == []
    assert result["participants"] == []


def test_fallback_note_prefixed_when_requested():
    result = extract_meeting_intelligence_local(SAMPLE_TRANSCRIPT, fallback_note=True)
    assert result["summary"].startswith("[Local rule-based engine")


def test_no_fallback_note_by_default():
    result = extract_meeting_intelligence_local(SAMPLE_TRANSCRIPT)
    assert not result["summary"].startswith("[Local rule-based engine")


def test_full_extraction_finds_deadlines_list():
    result = extract_meeting_intelligence_local(SAMPLE_TRANSCRIPT)
    assert "Friday" in result["deadlines"]


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
        except Exception as e:
            print(f"💥 ERROR - {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
