"""
test_validation.py
-------------------
Task 2 - File Upload Validation tests
Task 3 - Transcript Validation tests

Run with:
    python -m pytest test_validation.py -v
or simply:
    python test_validation.py
"""

from utils import validate_file, validate_transcript, transcript_matches_recording


# ---------------------------------------------------------------------------
# Task 2 - File Upload Validation
# ---------------------------------------------------------------------------

def test_valid_audio_formats():
    for ext in [".mp3", ".wav", ".m4a", ".flac"]:
        result = validate_file(f"meeting{ext}", 500_000)
        assert result["valid"] is True, f"{ext} should be accepted"


def test_valid_video_formats():
    for ext in [".mp4", ".mov", ".mkv"]:
        result = validate_file(f"meeting{ext}", 500_000)
        assert result["valid"] is True, f"{ext} should be accepted"


def test_invalid_format_rejected():
    result = validate_file("resume.pdf", 500_000)
    assert result["valid"] is False
    assert "Unsupported file type" in result["message"]


def test_empty_file_rejected():
    result = validate_file("meeting.wav", 0)
    assert result["valid"] is False
    assert "empty" in result["message"].lower()


def test_oversized_file_rejected():
    too_big = 600 * 1024 * 1024  # 600MB > 500MB limit
    result = validate_file("meeting.wav", too_big)
    assert result["valid"] is False
    assert "too large" in result["message"].lower()


def test_no_filename_rejected():
    result = validate_file("", 500_000)
    assert result["valid"] is False


# ---------------------------------------------------------------------------
# Task 3 - Transcript Validation
# ---------------------------------------------------------------------------

def test_empty_transcript_rejected():
    result = validate_transcript("")
    assert result["valid"] is False


def test_whitespace_only_transcript_rejected():
    result = validate_transcript("     \n\t  ")
    assert result["valid"] is False


def test_normal_transcript_accepted():
    result = validate_transcript("This is a normal meeting transcript with real content in it.")
    assert result["valid"] is True


def test_degenerate_repetition_rejected():
    bad = " ".join(["the"] * 20)
    result = validate_transcript(bad)
    assert result["valid"] is False


def test_transcript_length_matches_duration():
    # ~150 words for a 1-minute recording => ~150 wpm, within plausible band
    text = " ".join(["word"] * 150)
    result = transcript_matches_recording(text, audio_duration_seconds=60)
    assert result["plausible"] is True


def test_transcript_too_short_for_duration_flagged():
    text = "just a few words"
    result = transcript_matches_recording(text, audio_duration_seconds=600)  # 10 min
    assert result["plausible"] is False


if __name__ == "__main__":
    import sys
    import inspect

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
