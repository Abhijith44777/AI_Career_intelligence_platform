"""
llm_service.py
---------------
Milestone 2 - Task 1: LLM Service & Prompt Engineering

Responsibilities:
    - Select/configure the approved LLM. Default provider is Google
      Gemini (free-tier friendly, no Anthropic key required); OpenAI and
      Anthropic are supported as alternate providers by changing
      LLMConfig.provider. A fully offline, zero-API-key "local" rule-based
      engine (local_extraction.py) is also available and is used
      automatically as a graceful fallback whenever the configured
      provider has no key / an invalid key / fails after all retries.
    - Reusable prompt templates for meeting-intelligence extraction
    - Structured output handling (forces JSON, validates against a schema)
    - Input/output validation
    - Token/context-limit handling for long transcripts (chunk + merge)
    - Retry & failure handling (exponential backoff on transient errors)

This module has NO knowledge of Streamlit, the database, or Whisper -
it only knows how to turn "transcript text" into "validated structured
JSON". That keeps it independently testable and swappable (e.g. change
providers later without touching summarization/action-item code).
"""

import os
import json
import time
import re
import inspect
from dataclasses import dataclass, field

from local_extraction import extract_meeting_intelligence_local


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    """Base error for anything going wrong in the LLM layer."""
    pass


class LLMValidationError(LLMServiceError):
    """Raised when the LLM's response doesn't match the required schema,
    even after retries."""
    pass


class LLMRetryExhaustedError(LLMServiceError):
    """Raised when all retry attempts failed (API errors, timeouts, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Config - "Select and configure the approved LLM"
# ---------------------------------------------------------------------------

PROVIDER_API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "local": "",   # no key needed - rule-based engine, always available
}

PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "local": "rule-based-v1",
}


@dataclass
class LLMConfig:
    # Default provider is Gemini (free-tier friendly, no Anthropic key
    # required) with automatic fallback to the zero-key "local" rule-based
    # engine if no valid key is configured - see LLMService below. Swap to
    # "anthropic" / "openai" any time by passing a different LLMConfig.
    provider: str = "gemini"             # "gemini" | "anthropic" | "openai" | "local"
    model: str = ""                      # auto-derived from provider if blank
    max_output_tokens: int = 2000
    temperature: float = 0.2             # low temperature - factual extraction, not creative writing
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0   # doubles each retry: 2s, 4s, 8s
    # Whisper's transcript can be very long. We chunk on CHARACTER count
    # (cheap, no tokenizer dependency) using a conservative estimate of
    # ~4 chars/token, leaving headroom for the prompt + schema + output.
    max_chars_per_chunk: int = 12000
    api_key_env_var: str = field(default="")  # auto-derived from provider if blank

    def __post_init__(self):
        if self.provider not in PROVIDER_API_KEY_ENV_VARS:
            raise LLMServiceError(
                f"Unknown provider '{self.provider}'. Must be one of: {sorted(PROVIDER_API_KEY_ENV_VARS)}"
            )
        if not self.api_key_env_var:
            self.api_key_env_var = PROVIDER_API_KEY_ENV_VARS[self.provider]
        if not self.model:
            self.model = PROVIDER_DEFAULT_MODELS[self.provider]


# ---------------------------------------------------------------------------
# Structured output schema - "Ensure the LLM response follows a predefined
# JSON/schema structure"
# ---------------------------------------------------------------------------

# This is the canonical shape every LLM response must conform to. It maps
# directly onto the flow diagram from the spec:
#   Transcript -> LLM Processing Service -> Structured Meeting Intelligence
#       Summary / Key Points / Decisions / Action Items / Participants /
#       Deadlines / Priorities
MEETING_INTELLIGENCE_SCHEMA = {
    "summary": str,
    "key_points": list,
    "decisions": list,
    "action_items": list,   # each item validated separately, see below
    "participants": list,
    "deadlines": list,
    "priorities": list,
}

ACTION_ITEM_SCHEMA = {
    "task": str,
    "assignee": str,
    "deadline": str,
    "priority": str,
    "status": str,
}

EMPTY_MEETING_INTELLIGENCE = {
    "summary": "",
    "key_points": [],
    "decisions": [],
    "action_items": [],
    "participants": [],
    "deadlines": [],
    "priorities": [],
}


def validate_structured_output(data: dict) -> dict:
    """
    Validate a parsed LLM response against MEETING_INTELLIGENCE_SCHEMA.

    Returns {"valid": bool, "errors": [str, ...]}
    Does NOT raise - callers (with retry loops) decide what to do.
    """
    errors = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": [f"Top-level response must be a JSON object, got {type(data).__name__}"]}

    for field_name, expected_type in MEETING_INTELLIGENCE_SCHEMA.items():
        if field_name not in data:
            errors.append(f"Missing required field: '{field_name}'")
            continue
        if not isinstance(data[field_name], expected_type):
            errors.append(
                f"Field '{field_name}' should be {expected_type.__name__}, "
                f"got {type(data[field_name]).__name__}"
            )

    if "action_items" in data and isinstance(data["action_items"], list):
        for i, item in enumerate(data["action_items"]):
            if not isinstance(item, dict):
                errors.append(f"action_items[{i}] must be an object, got {type(item).__name__}")
                continue
            for field_name, expected_type in ACTION_ITEM_SCHEMA.items():
                if field_name not in item:
                    errors.append(f"action_items[{i}] missing field '{field_name}'")
                elif not isinstance(item[field_name], expected_type):
                    errors.append(f"action_items[{i}].{field_name} should be {expected_type.__name__}")

    return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Prompt templates - "Create reusable prompt templates / Design prompts"
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a meeting-intelligence extraction engine for a career \
intelligence platform. You read meeting transcripts and output ONLY valid JSON \
matching the exact schema you are given. You never include prose, markdown \
fences, or explanation outside the JSON object. You are precise and conservative: \
you never invent facts, names, or deadlines that are not supported by the transcript. \
If information is not present, use an empty string or empty list for that field, \
never a placeholder like "N/A" or "unknown" in a name field."""

EXTRACTION_PROMPT_TEMPLATE = """Extract structured meeting intelligence from the transcript below.

Return ONLY a single JSON object with EXACTLY this shape (no extra keys, no missing keys):

{{
  "summary": "2-4 sentence plain-English summary of the meeting",
  "key_points": ["short bullet of a key point discussed", "..."],
  "decisions": ["short bullet of a decision that was made", "..."],
  "action_items": [
    {{
      "task": "what needs to be done",
      "assignee": "person's name as mentioned in the transcript, or empty string if unclear",
      "deadline": "deadline as mentioned (e.g. 'Friday', '2026-09-10'), or empty string if none given",
      "priority": "High, Medium, or Low - infer from urgency language if not explicit",
      "status": "Not Started"
    }}
  ],
  "participants": ["every distinct person's name mentioned as attending or speaking"],
  "deadlines": ["every distinct deadline/date mentioned anywhere in the meeting"],
  "priorities": ["High-level priorities or focus areas mentioned in the meeting"]
}}

Rules:
- "action_items" status should always start as "Not Started" unless the transcript explicitly says something is already done/in progress.
- Do not fabricate assignees or deadlines - leave the field as "" if the transcript doesn't say.
- Keep "summary" factual and concise.
- Output ONLY the JSON object. No markdown code fences, no commentary.

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"
"""

# Used when a transcript had to be chunked - each chunk is summarized
# individually first, then this template merges the partial results into
# one coherent final result.
MERGE_PROMPT_TEMPLATE = """You are merging partial meeting-intelligence extractions \
from consecutive chunks of ONE long meeting transcript into a single, de-duplicated \
final result.

Return ONLY a single JSON object with this exact shape:

{{
  "summary": "2-4 sentence overall summary of the WHOLE meeting",
  "key_points": ["..."],
  "decisions": ["..."],
  "action_items": [{{"task": "...", "assignee": "...", "deadline": "...", "priority": "...", "status": "Not Started"}}],
  "participants": ["..."],
  "deadlines": ["..."],
  "priorities": ["..."]
}}

Merge rules:
- De-duplicate participants, decisions, and action items that appear in multiple chunks (same task+assignee = duplicate).
- Keep the union of all distinct key points, deadlines, and priorities.
- Write ONE cohesive overall summary, not a list of per-chunk summaries.
- Output ONLY the JSON object, nothing else.

PARTIAL RESULTS (one JSON object per chunk, in order):
{partial_results}
"""


def build_extraction_prompt(transcript: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(transcript=transcript)


def build_merge_prompt(partial_results: list) -> str:
    return MERGE_PROMPT_TEMPLATE.format(partial_results=json.dumps(partial_results, indent=2))


# ---------------------------------------------------------------------------
# JSON parsing - robust to models that wrap output in markdown fences
# ---------------------------------------------------------------------------

def parse_json_response(raw_text: str) -> dict:
    """
    Extract a JSON object from an LLM's raw text response, tolerating
    common wrapping issues (```json fences, leading/trailing prose).
    Raises LLMValidationError if no valid JSON object can be found.
    """
    if not raw_text or not raw_text.strip():
        raise LLMValidationError("Empty response from LLM.")

    text = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # If there's still leading/trailing prose, grab the outermost { ... }
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMValidationError(f"Could not parse LLM response as JSON: {e}\nRaw text: {raw_text[:300]}")


# ---------------------------------------------------------------------------
# Chunking - "Handle token/context limitations for long transcripts"
# ---------------------------------------------------------------------------

def chunk_transcript(transcript: str, max_chars: int) -> list:
    """
    Split a long transcript into chunks that fit within the context budget,
    splitting on sentence boundaries where possible so we don't cut a
    speaker's sentence in half mid-thought.
    """
    transcript = transcript.strip()
    if len(transcript) <= max_chars:
        return [transcript]

    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            # A single sentence longer than max_chars - hard split it.
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current = ""
            else:
                current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


def merge_lists_unique(*lists) -> list:
    """De-duplicate strings across lists while preserving order (case-insensitive)."""
    seen = set()
    result = []
    for lst in lists:
        for item in lst:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
    return result


def naive_merge_partials(partials: list) -> dict:
    """
    Fallback merge used if the LLM merge call itself fails - simple
    union/concatenation instead of an LLM-written cohesive summary.
    Ensures the pipeline degrades gracefully instead of losing all data.
    """
    merged = dict(EMPTY_MEETING_INTELLIGENCE)
    merged["summary"] = " ".join(p.get("summary", "") for p in partials if p.get("summary")).strip()
    merged["key_points"] = merge_lists_unique(*[p.get("key_points", []) for p in partials])
    merged["decisions"] = merge_lists_unique(*[p.get("decisions", []) for p in partials])
    merged["participants"] = merge_lists_unique(*[p.get("participants", []) for p in partials])
    merged["deadlines"] = merge_lists_unique(*[p.get("deadlines", []) for p in partials])
    merged["priorities"] = merge_lists_unique(*[p.get("priorities", []) for p in partials])

    seen_tasks = set()
    action_items = []
    for p in partials:
        for item in p.get("action_items", []):
            key = (item.get("task", "").strip().lower(), item.get("assignee", "").strip().lower())
            if key not in seen_tasks:
                seen_tasks.add(key)
                action_items.append(item)
    merged["action_items"] = action_items

    return merged


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

def _extract_offending_kwarg(type_error: TypeError) -> str:
    """
    Parse an SDK TypeError like "Messages.create() got an unexpected
    keyword argument 'temperature'" to find which kwarg to drop and retry
    without. Returns "" if the message doesn't match that pattern.
    """
    match = re.search(r"unexpected keyword argument '(\w+)'", str(type_error))
    return match.group(1) if match else ""


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    """
    Proactively drop kwargs that the installed SDK's method doesn't
    actually accept, by inspecting its real signature - rather than
    finding out via a crash. This protects against SDK version drift
    (older/newer `anthropic`/`openai` packages that added, removed, or
    renamed parameters like `temperature`) without needing a code change
    every time a dependency updates.

    Falls back to returning kwargs unchanged if inspection itself fails
    (e.g. a C-extension callable with no introspectable signature) - the
    TypeError catch-and-retry in _call_llm_raw is the safety net for that case.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # method accepts **kwargs - nothing to filter

    return {k: v for k, v in kwargs.items() if k in params}


class LLMService:
    """
    Provider-agnostic LLM service. Swap providers by changing LLMConfig.provider
    - the rest of the pipeline never has to know which vendor is behind it.
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._client = None  # lazily created on first real call

    # -- provider setup -----------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.config.provider == "local":
            self._client = "local"  # sentinel - no real network client needed
            return self._client

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise LLMServiceError(
                f"Missing API key. Set the {self.config.api_key_env_var} environment variable."
            )

        if self.config.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif self.config.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        elif self.config.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)
        else:
            raise LLMServiceError(f"Unknown provider: {self.config.provider}")

        return self._client

    # -- low-level call with retry/backoff -----------------------------------

    def _call_llm_raw(self, system_prompt: str, user_prompt: str) -> str:
        """
        Single LLM call. Retries on transient failures (rate limits, timeouts,
        connection errors) with exponential backoff. Raises
        LLMRetryExhaustedError if every attempt fails.
        """
        client = self._get_client()
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                if self.config.provider == "anthropic":
                    kwargs = {
                        "model": self.config.model,
                        "max_tokens": self.config.max_output_tokens,
                        "temperature": self.config.temperature,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    }
                    kwargs = _filter_supported_kwargs(client.messages.create, kwargs)
                    try:
                        response = client.messages.create(**kwargs)
                    except TypeError as te:
                        # Second line of defense if signature inspection
                        # itself couldn't see the real constraint.
                        offending = _extract_offending_kwarg(te)
                        if offending and offending in kwargs:
                            kwargs.pop(offending)
                            response = client.messages.create(**kwargs)
                        else:
                            raise
                    return "".join(block.text for block in response.content if hasattr(block, "text"))

                elif self.config.provider == "openai":
                    kwargs = {
                        "model": self.config.model,
                        "max_tokens": self.config.max_output_tokens,
                        "temperature": self.config.temperature,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                    kwargs = _filter_supported_kwargs(client.chat.completions.create, kwargs)
                    try:
                        response = client.chat.completions.create(**kwargs)
                    except TypeError as te:
                        offending = _extract_offending_kwarg(te)
                        if offending and offending in kwargs:
                            kwargs.pop(offending)
                            response = client.chat.completions.create(**kwargs)
                        else:
                            raise
                    return response.choices[0].message.content

                elif self.config.provider == "gemini":
                    from google.genai import types
                    gen_config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                    )
                    response = client.models.generate_content(
                        model=self.config.model,
                        contents=user_prompt,
                        config=gen_config,
                    )
                    text = getattr(response, "text", None)
                    if not text:
                        # Defensive fallback for SDK versions where .text is
                        # empty (e.g. finish_reason != STOP) but parts exist.
                        try:
                            text = "".join(
                                part.text
                                for cand in (response.candidates or [])
                                for part in (cand.content.parts or [])
                                if hasattr(part, "text") and part.text
                            )
                        except Exception:
                            text = ""
                    return text

            except Exception as e:  # noqa: BLE001 - deliberately broad, this is the retry boundary
                last_error = e
                if attempt < self.config.max_retries:
                    wait = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(wait)
                continue

        hint = ""
        if isinstance(last_error, TypeError) and "unexpected keyword argument" in str(last_error):
            hint = (
                " This looks like an installed-SDK/code version mismatch rather than "
                "a real API problem - run 'pip install --upgrade anthropic' (or 'openai') "
                "and fully restart the app (stop it completely, don't rely on hot-reload)."
            )

        raise LLMRetryExhaustedError(
            f"LLM call failed after {self.config.max_retries} attempts. Last error: {last_error}{hint}"
        )

    # -- validated structured call -------------------------------------------

    def _call_and_validate(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Calls the LLM and validates the JSON response against the schema.
        If validation fails, retries the WHOLE call (a fresh generation
        often fixes transient formatting mistakes) up to max_retries times.
        """
        last_errors = None

        for attempt in range(1, self.config.max_retries + 1):
            raw = self._call_llm_raw(system_prompt, user_prompt)
            try:
                parsed = parse_json_response(raw)
            except LLMValidationError as e:
                last_errors = [str(e)]
                continue

            check = validate_structured_output(parsed)
            if check["valid"]:
                return parsed
            last_errors = check["errors"]

        raise LLMValidationError(
            f"LLM output failed schema validation after {self.config.max_retries} attempts: {last_errors}"
        )

    # -- public API -----------------------------------------------------------

    def _extract_chunk(self, chunk_text: str) -> tuple:
        """
        Runs one chunk through the configured provider. Returns
        (result_dict, used_local_fallback: bool).

        If the configured provider (Gemini/Anthropic/OpenAI) has no API
        key, an invalid key, or fails after all retries, this transparently
        degrades to the local rule-based engine instead of raising - a
        missing/broken key should never hard-crash the whole pipeline, it
        should just mean lower-quality output. The UI surfaces this via the
        "[Local rule-based engine ...]" note prefixed onto the summary.
        """
        try:
            return self._call_and_validate(SYSTEM_PROMPT, build_extraction_prompt(chunk_text)), False
        except (LLMServiceError, LLMValidationError):
            return extract_meeting_intelligence_local(chunk_text, fallback_note=True), True

    def extract_meeting_intelligence(self, transcript: str) -> dict:
        """
        Main entry point: transcript text -> validated structured JSON
        matching MEETING_INTELLIGENCE_SCHEMA.

        Handles long transcripts transparently by chunking, extracting
        each chunk, then merging into one cohesive result. Never raises
        just because an API key is missing/invalid - see _extract_chunk.
        """
        if not transcript or not transcript.strip():
            raise LLMServiceError("Cannot process an empty transcript.")

        if self.config.provider == "local":
            return extract_meeting_intelligence_local(transcript)

        chunks = chunk_transcript(transcript, self.config.max_chars_per_chunk)

        if len(chunks) == 1:
            result, _used_fallback = self._extract_chunk(chunks[0])
            return result

        # Long transcript: extract each chunk, then merge.
        partials = []
        any_fallback = False
        for chunk in chunks:
            result, used_fallback = self._extract_chunk(chunk)
            partials.append(result)
            any_fallback = any_fallback or used_fallback

        if any_fallback:
            # Don't spend another (likely-failing) real LLM call on the
            # merge step if the provider already proved unreachable/unkeyed.
            return naive_merge_partials(partials)

        try:
            merge_prompt = build_merge_prompt(partials)
            return self._call_and_validate(SYSTEM_PROMPT, merge_prompt)
        except LLMServiceError:
            # Graceful degradation: if the merge call itself fails, fall
            # back to a simple deterministic merge rather than losing all
            # the work already done extracting each chunk.
            return naive_merge_partials(partials)
