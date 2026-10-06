"""
test_end_to_end.py
-------------------
Milestone 4, Task 8: the COMPLETE workflow, with no manual step between stages:

  User Login -> Upload/Import -> Audio Validation -> Whisper -> Transcript
  Storage -> LLM Summary -> Action Items -> Participant Assignment ->
  Knowledge Repository -> Embeddings -> Vector DB -> RAG Search -> FastAPI ->
  Dashboard data (via ApiClient) -> PDF / CSV report

Real code runs at every stage except the two that need heavy external
software: ffmpeg/Whisper (replaced by `fake_whisper`, which returns a fixed
transcript) and the network LLM (the built-in offline "local" engine is used).
"""

import csv
import io
import os

from api_client import ApiClient
from testutil import FAKE_WAV, OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env, fake_whisper, make_client


def test_full_workflow_through_the_service_layer():
    env = Env()
    # 1. register + login
    env.auth.register("ravi", "ravi@example.com", "goodpass1")
    session = env.auth.login("ravi", "goodpass1")
    user = env.auth.authenticate(session["token"])

    # 2-5. upload -> validate -> "Whisper" -> transcript stored
    result = env.upload(user["id"], SAMPLE_TRANSCRIPT, "launch-meeting.wav", duration=420.0)
    mid = result["meeting_id"]
    meeting = env.db.get_meeting(mid)
    assert meeting["transcript"] == SAMPLE_TRANSCRIPT and meeting["user_id"] == user["id"]
    assert meeting["status"] == "completed" and meeting["audio_duration_seconds"] == 420.0
    transcript_files = [f for _, _, fs in os.walk(env.transcripts_dir) for f in fs]
    assert any(f.endswith(".txt") for f in transcript_files), "transcript file saved"

    # 6. LLM summary   7. action items   8. participant assignment
    d = env.services.get_details(user["id"], mid)
    assert d["summary_text"] and d["decisions"]
    tasks = {(i["participant_name"], i["deadline"]) for i in d["action_items"]}
    assert ("Ravi", "Friday") in tasks and ("Priya", "Wednesday") in tasks
    assert {p["name"] for p in d["participants"]} >= {"Ravi", "Priya", "Karan"}

    # 9-11. knowledge repository -> embeddings -> vector DB (all traceable to the meeting)
    assert env.services.verify_meeting(user["id"], mid)["ok"]
    vectors = env.services.get_vectors(user["id"], mid)
    types = {v["content_type"] for v in vectors["vectors"]}
    assert {"transcript_section", "summary", "decision", "action_item"} <= types
    assert all(r.metadata["meeting_id"] == mid for r in env.vectors.get_by_meeting(mid))

    # 12. RAG search
    hits = env.services.search(user["id"], "Which meeting discussed the API integration?")
    assert hits["results"][0]["meeting_id"] == mid and hits["within_latency_target"]
    answer = env.services.ask(user["id"], "Who will complete the API integration and by when?")
    assert answer["grounded"] and answer["sources"][0]["meeting_id"] == mid
    assert "Ravi" in answer["answer"] and "Friday" in answer["answer"]
    assert answer["sources"][0]["meeting_date"] and answer["sources"][0]["retrieved_context"]

    # 15. reports
    pdf, _, _ = env.services.export_report(user["id"], mid, "pdf")
    rows = list(csv.reader(io.StringIO(env.services.export_report(user["id"], mid, "csv")[0].decode("utf-8-sig"))))
    assert pdf[:5] == b"%PDF-" and any(r[0] == "Action Items" and r[2] == "Ravi" for r in rows)

    # logout ends the session
    env.auth.logout(session["token"])
    from auth import SessionInvalid
    from testutil import raises
    with raises(SessionInvalid):
        env.auth.authenticate(session["token"])


def test_full_workflow_over_http_as_the_dashboard_would_drive_it():
    """FastAPI -> ApiClient (what the Streamlit pages call) -> report."""
    env = Env()
    http = make_client(env)
    if http is None:
        return

    class Session:  # adapts the Starlette TestClient to requests.Session.request()
        def request(self, method, url, headers=None, params=None, json=None, files=None, timeout=None):
            path = url.split("://", 1)[1].split("/", 1)[1]
            path = "/" + path
            if params:
                from urllib.parse import urlencode
                path += "?" + urlencode(params)
            kw = {"headers": headers}
            if json is not None:
                kw["json"] = json
            if files is not None:
                kw["files"] = files
            return getattr(http, method.lower())(path, **kw)

    api = ApiClient("http://testserver", session=Session())
    api.register("dana", "dana@example.com", "goodpass1")
    api.login("dana", "goodpass1")
    assert api.me()["username"] == "dana"

    with fake_whisper(SAMPLE_TRANSCRIPT):
        processed = api.upload_meeting("weekly.wav", FAKE_WAV)
    mid = processed["meeting_id"]

    listed = api.list_meetings()
    assert [m["id"] for m in listed] == [mid] and listed[0]["action_item_count"] >= 3
    assert api.list_meetings(participant="Ravi")[0]["id"] == mid and api.list_meetings(participant="Nobody") == []
    details = api.meeting_details(mid)
    assert details["transcript"] == SAMPLE_TRANSCRIPT and details["analytics"]["action_item_count"] >= 3
    assert api.overview()["meeting_count"] == 1 and api.participants()

    search = api.search("database migration plan")
    assert search["results"][0]["meeting_id"] == mid
    ask = api.ask("Who should review the database migration plan?")
    assert "Karan" in ask["answer"] and ask["sources"][0]["meeting_title"] == "weekly.wav"

    data, name = api.download_report(mid, "pdf")
    assert data[:5] == b"%PDF-" and name.endswith(".pdf")
    csv_data, csv_name = api.download_report(mid, "csv")
    assert b"Ravi" in csv_data and csv_name.endswith(".csv")

    api.delete_meeting(mid)
    assert api.list_meetings() == []
    assert env.vectors.get_by_meeting(mid) == [], "deleting a meeting also removes its vectors"


def test_failed_pipeline_leaves_a_visible_failed_meeting_not_a_silent_gap():
    env = Env()
    u = env.user("alice")

    class BrokenLLM:
        config = type("C", (), {"provider": "gemini"})()

        def extract_meeting_intelligence(self, transcript):
            from llm_service import LLMServiceError
            raise LLMServiceError("quota exceeded")

    env.services.llm_factory = lambda: BrokenLLM()
    from pipeline import PipelineError
    from testutil import raises
    try:
        env.upload(u["id"], SAMPLE_TRANSCRIPT, "will-fail.wav")
    except PipelineError as e:
        assert e.stage == "LLM Processing"
    else:
        raise AssertionError("expected PipelineError")
    meetings = env.services.list_meetings(u["id"], status="failed")
    assert len(meetings) == 1 and meetings[0]["filename"] == "will-fail.wav"
    detail = env.services.get_details(u["id"], meetings[0]["id"])
    assert "quota" in detail["meeting"]["error_message"]
