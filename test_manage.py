"""test_manage.py - Milestone 4 upgrade path: assign Milestone 3 meetings to a user and keep them searchable."""

import manage
from testutil import SAMPLE_TRANSCRIPT, Env


def test_assign_legacy_makes_old_meetings_visible_and_searchable():
    env = Env()
    # a Milestone-3 style meeting: no owner, vectors without user_id
    from llm_service import LLMConfig, LLMService
    from pipeline import process_meeting_recording  # noqa: F401 - import check only
    from summarization import summarize_meeting
    from knowledge_repository import index_meeting
    mid = env.db.create_meeting("old-launch.wav", SAMPLE_TRANSCRIPT, user_id=None)
    s = summarize_meeting(SAMPLE_TRANSCRIPT, LLMService(LLMConfig(provider="local")))
    env.db.save_summary(mid, s.summary, s.key_points, s.decisions, s.priorities)
    index_meeting(env.db, mid, env.embedder, env.vectors)
    assert all("user_id" not in r.metadata for r in env.vectors.get_by_meeting(mid))

    manage.main(["create-user", "abhijith", "a@example.com", "--password", "goodpass1"], db=env.db)
    uid = env.db.get_user_by_username("abhijith")["id"]
    assert env.services.list_meetings(uid) == []                 # invisible before the upgrade step

    # the CLI builds its own embedder/store from settings, so run the same steps with the test stack
    assert env.db.assign_legacy_meetings(uid) == 1
    from knowledge_repository import reindex_all_meetings
    reindex_all_meetings(env.db, env.embedder, env.vectors, user_id=uid)

    assert [m["id"] for m in env.services.list_meetings(uid)] == [mid]
    assert {r.metadata["user_id"] for r in env.vectors.get_by_meeting(mid)} == {uid}
    assert env.services.search(uid, "API integration Ravi")["results"][0]["meeting_id"] == mid


def test_cli_rejects_unknown_user_and_bad_password():
    env = Env()
    assert manage.main(["assign-legacy", "ghost"], db=env.db) == 1
    assert manage.main(["create-user", "abc", "not-an-email", "--password", "x"], db=env.db) == 1
    assert manage.main(["purge-sessions"], db=env.db) == 0
    assert manage.main(["check-config"], db=env.db) == 0
