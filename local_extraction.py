"""
local_extraction.py
---------------------
Milestone 2 - Task 1 support: a zero-dependency, zero-API-key fallback
extraction engine ("local" provider).

This is NOT a claim to replace a real LLM's semantic understanding - it's
a deterministic, rule-based extractive engine (word-frequency sentence
scoring + keyword/regex heuristics) that lets the whole Milestone 2
pipeline run end-to-end with ZERO external services and ZERO API keys.
It is used in two places (see llm_service.py):

  1. As the DEFAULT provider, so the app works out of the box with no
     setup at all - no Anthropic key, no Gemini key, nothing.
  2. As an automatic fallback inside LLMService whenever a configured
     real provider (Gemini / Anthropic / OpenAI) has no API key set, an
     invalid key, or otherwise fails after all retries - so a bad/missing
     key degrades the OUTPUT QUALITY instead of breaking the pipeline.

Swap in a real provider any time (see llm_service.LLMConfig) for higher
quality, genuinely semantic extraction - this module is intentionally
simple and fully offline/stdlib-only so it is always available.
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "we", "you", "i", "he", "she", "they", "them",
    "his", "her", "our", "your", "their", "as", "not", "no", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "shall", "may", "might", "must", "there", "here", "then", "than", "just",
    "also", "very", "really", "one", "all", "some", "any", "us",
}

# Sentence-initial capitalized words that are almost never a person's name -
# used to filter false-positive "participants" out of a plain-prose
# transcript with no speaker labels.
NON_NAME_WORDS = {
    "the", "this", "that", "these", "those", "it", "we", "you", "i", "he",
    "she", "they", "so", "and", "but", "if", "when", "then", "there", "here",
    "very", "sometimes", "always", "never", "also", "now", "today",
    "tomorrow", "yesterday", "next", "last", "first", "second", "third",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "ok", "okay", "yes", "no", "well", "actually", "basically", "meeting",
    "team", "everyone", "let", "let's", "please", "thanks", "thank",
}

DECISION_KEYWORDS = [
    "decided", "decide", "decision", "agreed", "agree", "approved",
    "approve", "finalized", "finalize", "confirmed", "confirm",
    "will proceed", "move forward", "moving forward", "go with",
    "going with", "went with", "resolved", "concluded", "we'll go",
    "signed off", "sign off", "settled on", "settled with", "landed on",
    "opted for", "opted to", "consensus", "final call", "final decision",
    "everyone agreed", "team agreed", "we're going with", "sticking with",
    "stick with", "switching to", "switch to", "chose to", "chose the",
    "selected the", "green light", "greenlit", "signed the", "locked in",
]

# Broader pattern for decision-flavored sentences that don't use one of the
# explicit keywords above: a team-oriented subject making a forward-looking
# commitment, e.g. "we'll launch next month", "the team will adopt the new
# process", "we're going to switch to the new vendor".
DECISION_PATTERN = re.compile(
    r"\b(we|we'll|we're|the team|our team|the group|everyone)\b.{0,20}\b"
    r"(will|'re going to|are going to|plan to|planning to)\b.{0,20}\b"
    r"(proceed|launch|adopt|implement|continue|build|use|go with|"
    r"switch to|move to|choose|select|roll out|ship|start|begin|"
    r"stick with|keep|maintain)\b",
    re.IGNORECASE,
)

PRIORITY_KEYWORDS = [
    "priority", "top priority", "high priority", "critical", "important",
    "urgent", "asap", "focus area", "key focus", "must-have",
]

HIGH_URGENCY_WORDS = {"urgent", "asap", "immediately", "critical", "blocker"}

# Real obligation/future-commitment language only. Deliberately excludes a
# bare "to" - that matches far too many ordinary infinitives ("happy to
# help", "want to discuss") and would flood the output with false-positive
# action items. Bare "Name to <task>" meeting-notes shorthand is still
# handled by NAMED_ACTION_PATTERN below, where a preceding capitalized
# name makes it a much more reliable signal.
GENERIC_VERB_PHRASE = (
    r"(will|should|needs to|need to|has to|have to|must|"
    r"is going to|are going to|is responsible for)"
)
NAMED_VERB_PHRASE = GENERIC_VERB_PHRASE[:-1] + r"|to)"  # also allow bare "Name to <task>"

# "<Name> will <task>" / "<Name> needs to <task>" / "<Name> to <task>" -
# captures a leading 1-3 word capitalized name immediately before the verb.
NAMED_ACTION_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})\s+" + NAMED_VERB_PHRASE + r"\s+(.{3,140})",
)

# Fallback with no captured name - "will complete the report by Friday"
GENERIC_ACTION_PATTERN = re.compile(
    r"\b" + GENERIC_VERB_PHRASE + r"\s+(.{3,140})", re.IGNORECASE
)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

DEADLINE_PATTERN = re.compile(
    r"\b(?:by\s+)?("
    r"(?:" + "|".join(WEEKDAYS) + r")"
    r"|(?:" + "|".join(MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|tomorrow|tonight|today"
    r"|end of (?:the\s+)?day|EOD|COB"
    r"|next week|this week|next month"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Sentence splitting & scoring
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list:
    """Simple, dependency-free sentence splitter. Not perfect (abbreviations,
    decimals, etc. can trip it up) but good enough for extractive heuristics."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital,
    # or end of string.
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def _words(sentence: str) -> list:
    return re.findall(r"[A-Za-z']+", sentence.lower())


def score_sentences(sentences: list) -> dict:
    """Classic frequency-based extractive scoring: sentences containing more
    frequent, non-stopword words score higher. Normalized by sentence length
    so long sentences don't win purely on word count."""
    word_freq = Counter()
    for s in sentences:
        for w in _words(s):
            if w not in STOPWORDS and len(w) > 2:
                word_freq[w] += 1

    scores = {}
    for s in sentences:
        words = [w for w in _words(s) if w not in STOPWORDS and len(w) > 2]
        if not words:
            scores[s] = 0.0
            continue
        scores[s] = sum(word_freq[w] for w in words) / len(words)
    return scores


def extractive_summary(sentences: list, scores: dict, max_sentences: int = 3) -> list:
    if not sentences:
        return []
    ranked = sorted(sentences, key=lambda s: scores.get(s, 0.0), reverse=True)
    top = set(ranked[:max_sentences])
    # Return in ORIGINAL order so the summary reads coherently.
    return [s for s in sentences if s in top]


def extract_key_points(sentences: list, scores: dict, exclude: set, max_points: int = 5) -> list:
    remaining = [s for s in sentences if s not in exclude]
    ranked = sorted(remaining, key=lambda s: scores.get(s, 0.0), reverse=True)
    return ranked[:max_points]


# ---------------------------------------------------------------------------
# Decisions / priorities
# ---------------------------------------------------------------------------

def extract_decisions(sentences: list) -> list:
    out = []
    for s in sentences:
        low = s.lower()
        if any(kw in low for kw in DECISION_KEYWORDS) or DECISION_PATTERN.search(s):
            out.append(s)
    return out


def extract_priority_mentions(sentences: list) -> list:
    out = []
    for s in sentences:
        low = s.lower()
        if any(kw in low for kw in PRIORITY_KEYWORDS):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------

def extract_deadline(sentence: str) -> str:
    match = DEADLINE_PATTERN.search(sentence)
    return match.group(1) if match else ""


def extract_deadline_mentions(sentences: list) -> list:
    found = []
    for s in sentences:
        d = extract_deadline(s)
        if d and d not in found:
            found.append(d)
    return found


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

def _looks_like_name(candidate: str) -> bool:
    first_word = candidate.split()[0].lower()
    if first_word in NON_NAME_WORDS:
        return False
    if len(candidate) < 3:
        return False
    return True


def extract_action_items(sentences: list) -> list:
    items = []
    seen = set()
    for s in sentences:
        match = NAMED_ACTION_PATTERN.search(s)
        assignee = ""
        task = ""
        if match and _looks_like_name(match.group(1)):
            assignee = match.group(1).strip()
            task = match.group(3).strip().rstrip(".!? ")
        else:
            generic = GENERIC_ACTION_PATTERN.search(s)
            if generic:
                task = generic.group(2).strip().rstrip(".!? ")

        if not task:
            continue

        dedup_key = (assignee.lower(), task.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        items.append({
            "task": task[:1].upper() + task[1:] if task else task,
            "assignee": assignee,
            "deadline": extract_deadline(s),
            "priority": "High" if any(w in s.lower() for w in HIGH_URGENCY_WORDS) else "",
            "status": "Not Started",
        })
    return items


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

NAME_CANDIDATE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")

# "<Name> said/mentioned/asked/noted/suggested/added/reported/raised/
# proposed/pointed out ..." - a very strong participant signal even on a
# single mention, common in meeting-style transcripts that reference who
# said what without using commitment language.
SPEAKING_VERB_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+"
    r"(said|says|mentioned|asked|noted|suggested|added|reported|raised|"
    r"proposed|pointed out|explained|clarified|confirmed|responded|replied)\b"
)


def extract_participants(sentences: list, action_items: list) -> list:
    text = " ".join(sentences)
    counts = Counter()
    for m in NAME_CANDIDATE_PATTERN.finditer(text):
        candidate = m.group(1)
        if candidate.split()[0].lower() in NON_NAME_WORDS:
            continue
        counts[candidate] += 1

    assignees = {item["assignee"] for item in action_items if item.get("assignee")}
    speakers = {
        m.group(1) for m in SPEAKING_VERB_PATTERN.finditer(text)
        if m.group(1).split()[0].lower() not in NON_NAME_WORDS
    }

    participants = set(assignees) | speakers
    for name, count in counts.items():
        # A two-word Title-Case phrase ("Ravi Kumar") reads as a full name
        # even on a single mention - that pattern is a strong signal by
        # itself. A single capitalized word is ambiguous (could be any
        # proper noun), so it still needs to recur to count as a person.
        if " " in name or count >= 2:
            participants.add(name)

    return sorted(participants)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_meeting_intelligence_local(transcript: str, fallback_note: bool = False) -> dict:
    """
    transcript -> structured meeting intelligence dict matching
    llm_service.MEETING_INTELLIGENCE_SCHEMA, using ONLY deterministic
    rule-based heuristics (no network call, no API key).

    fallback_note: when True, prefixes the summary with a short notice
    that this result came from the local engine rather than a real LLM
    (used when LLMService transparently falls back after a real provider
    failed due to a missing/invalid key).
    """
    text = (transcript or "").strip()
    if not text:
        return {
            "summary": "",
            "key_points": [],
            "decisions": [],
            "action_items": [],
            "participants": [],
            "deadlines": [],
            "priorities": [],
        }

    sentences = split_sentences(text)
    scores = score_sentences(sentences)

    summary_sentences = extractive_summary(sentences, scores, max_sentences=3)
    summary = " ".join(summary_sentences) or "(No summary could be generated from this transcript.)"
    if fallback_note:
        summary = "[Local rule-based engine - no LLM API key configured] " + summary

    decisions = extract_decisions(sentences)
    priorities = extract_priority_mentions(sentences)
    action_items = extract_action_items(sentences)
    participants = extract_participants(sentences, action_items)
    deadlines = sorted(set(
        [item["deadline"] for item in action_items if item.get("deadline")]
        + extract_deadline_mentions(sentences)
    ))
    key_points = extract_key_points(
        sentences, scores, exclude=set(summary_sentences) | set(decisions), max_points=5
    )

    return {
        "summary": summary,
        "key_points": key_points,
        "decisions": decisions,
        "action_items": action_items,
        "participants": participants,
        "deadlines": deadlines,
        "priorities": priorities,
    }
