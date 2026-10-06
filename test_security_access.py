"""
test_security_access.py
------------------------
Milestone 4, Task 7: "Verify that one user cannot access another user's
private meeting information." Every read, write, search, export and
integration path is checked with two users A and B.
"""

from database import Database
from services import BadRequest, NotFound
from testutil import OTHER_TRANSCRIPT, SAMPLE_TRANSCRIPT, Env, raises


def two_users():
    env = Env()
    a, b = env.user("alice"), env.user("bob")
    ra = env.upload(a["id"], SAMPLE_TRANSCRIPT, "alice-launch.wav")
    rb = env.upload(b["id"], OTHER_TRANSCRIPT, "bob-finance.wav")
    return env, a, b, ra["meeting_id"], rb["meeting_id"]


def test_meeting_is_stored_with_its_owner():
    env, a, b, ma, mb = two_users()
    assert env.db.get_meeting(ma)["user_id"] == a["id"]
    assert env.db.get_meeting(mb)["user_id"] == b["id"]


def test_lists_contain_only_own_meetings():
    env, a, b, ma, mb = two_users()
    assert [m["id"] for m in env.services.list_meetings(a["id"])] == [ma]
    assert [m["id"] for m in env.services.list_meetings(b["id"])] == [mb]


def test_every_read_endpoint_hides_other_users_meeting():
    env, a, b, ma, mb = two_users()
    s = env.services
    for method in (s.get_meeting, s.get_details, s.get_transcript, s.get_summary, s.get_decisions,
                   s.get_action_items, s.get_participants, s.get_deadlines, s.verify_meeting,
                   s.get_vectors, s.build_report):
        with raises(NotFound):
            method(b["id"], ma)       # bob asks for alice's meeting
        method(a["id"], ma)           # ...alice can read it


def test_not_found_is_identical_for_missing_and_foreign_meetings():
    env, a, b, ma, mb = two_users()
    with raises(NotFound) as foreign:
        env.services.get_details(b["id"], ma)
    with raises(NotFound) as missing:
        env.services.get_details(b["id"], 99999)
    assert str(foreign.value).replace(str(ma), "N") == str(missing.value).replace("99999", "N")


def test_reports_cannot_be_exported_for_other_users_meetings():
    env, a, b, ma, mb = two_users()
    for fmt in ("pdf", "csv"):
        with raises(NotFound):
            env.services.export_report(b["id"], ma, fmt)
        data, _, _ = env.services.export_report(a["id"], ma, fmt)
        assert data


def test_delete_and_embedding_operations_are_owner_only():
    env, a, b, ma, mb = two_users()
    for call in (env.services.delete_meeting, env.services.delete_embeddings, env.services.generate_embeddings):
        with raises(NotFound):
            call(b["id"], ma)
    assert env.db.get_meeting(ma) is not None, "foreign delete attempt must not delete anything"
    assert env.vectors.get_by_meeting(ma), "foreign delete attempt must not remove vectors"
    env.services.delete_meeting(a["id"], ma)
    assert env.db.get_meeting(ma) is None
    assert env.vectors.get_by_meeting(ma) == []


def test_delete_removes_all_dependent_rows():
    env, a, b, ma, mb = two_users()
    env.services.delete_meeting(a["id"], ma)
    assert env.db.get_summary(ma) is None and env.db.get_action_items(ma) == []
    assert env.db.get_participants_for_meeting(ma) == []
    assert env.db.get_meeting(mb) is not None


def test_semantic_search_never_returns_other_users_meetings():
    env, a, b, ma, mb = two_users()
    hits_b = env.services.search(b["id"], "mobile application launch API integration")
    assert all(r["meeting_id"] == mb for r in hits_b["results"])
    hits_a = env.services.search(a["id"], "mobile application launch API integration")
    assert hits_a["results"] and all(r["meeting_id"] == ma for r in hits_a["results"])
    # a filter that names the OTHER user's meeting still returns nothing
    assert env.services.search(b["id"], "mobile application launch", meeting_id=ma)["results"] == []


def test_rag_answers_only_from_own_meetings():
    env, a, b, ma, mb = two_users()
    answer = env.services.ask(b["id"], "What deadline was decided for the mobile application API integration?")
    assert all(s["meeting_id"] == mb for s in answer["sources"])
    assert "Ravi" not in answer["answer"] and "Friday" not in answer["answer"]
    own = env.services.ask(a["id"], "Who will complete the API integration?")
    assert own["sources"] and all(s["meeting_id"] == ma for s in own["sources"])


def test_search_drops_vectors_whose_metadata_disagrees_with_the_database_owner():
    """Defence in depth: even a mislabelled vector (metadata says bob, DB says alice) is not returned."""
    env, a, b, ma, mb = two_users()
    for record in env.vectors.get_by_meeting(ma):
        vec = env.embedder.embed_text(record.document)
        meta = dict(record.metadata)
        meta["user_id"] = b["id"]                       # corrupted / forged ownership label
        env.vectors.upsert(record.id, vec, meta, record.document)
    leaked = env.services.search(b["id"], "mobile application launch API integration")
    assert all(r["meeting_id"] != ma for r in leaked["results"])


def test_vectors_carry_the_owner_id():
    env, a, b, ma, mb = two_users()
    assert {r.metadata["user_id"] for r in env.vectors.get_by_meeting(ma)} == {a["id"]}
    assert {r.metadata["user_id"] for r in env.vectors.get_by_meeting(mb)} == {b["id"]}


def test_sync_embeddings_only_touches_own_meetings():
    env, a, b, ma, mb = two_users()
    env.vectors.delete_by_meeting(ma)
    env.vectors.delete_by_meeting(mb)
    result = env.services.sync_embeddings(b["id"])
    assert set(result["chunks_per_meeting"]) == {mb}
    assert env.vectors.get_by_meeting(ma) == [] and env.vectors.get_by_meeting(mb)


def test_participants_and_analytics_are_scoped():
    env, a, b, ma, mb = two_users()
    names_b = {p["name"] for p in env.services.list_participants(b["id"])}
    names_a = {p["name"] for p in env.services.list_participants(a["id"])}
    assert "Meera" in names_b and "Ravi" not in names_b
    assert "Ravi" in names_a and "Meera" not in names_a
    assert env.services.get_overview(a["id"])["meeting_count"] == 1
    assert env.services.get_overview(b["id"])["action_item_count"] == len(env.db.get_action_items(mb))


def test_participant_named_the_same_in_two_accounts_is_not_shared_visibility():
    env = Env()
    a, b = env.user("alice"), env.user("bob")
    env.upload(a["id"], "Ravi will finish the report by Friday.", "a.wav")
    mb = env.upload(b["id"], "Priya will send the invoice by Monday.", "b.wav")["meeting_id"]
    assert "Ravi" not in {p["name"] for p in env.services.list_participants(b["id"])}
    assert [p["name"] for p in env.services.get_participants(b["id"], mb)["participants"]] == ["Priya"]


def test_legacy_meetings_without_owner_are_invisible_until_assigned():
    env = Env()
    a = env.user("alice")
    legacy = env.db.create_meeting("old.wav", "Old M3 meeting about budgets.", user_id=None)
    assert env.services.list_meetings(a["id"]) == []
    with raises(NotFound):
        env.services.get_details(a["id"], legacy)
    assert env.db.assign_legacy_meetings(a["id"]) == 1
    assert [m["id"] for m in env.services.list_meetings(a["id"])] == [legacy]


def test_search_filters_are_validated_and_parameterised():
    env, a, b, ma, mb = two_users()
    with raises(BadRequest):
        env.services.list_meetings(a["id"], status="hacked")
    with raises(BadRequest):
        env.services.list_meetings(a["id"], date_from="yesterday")
    # a classic injection string is just an unmatched search term
    assert env.services.list_meetings(a["id"], text="' OR 1=1 --") == []
    assert env.services.list_meetings(a["id"], participant="x' OR '1'='1") == []
    assert len(env.services.list_meetings(a["id"], participant="Ravi")) == 1


def test_recordings_are_per_user():
    env, a, b, ma, mb = two_users()
    env.db.register_recording(a["id"], "zoom", "rec-1", "Alice call")
    assert env.db.list_recordings(b["id"]) == []
    _, created = env.db.register_recording(b["id"], "zoom", "rec-1", "Bob call")
    assert created, "same provider id under a different account is a different record"
    assert env.services.integration_status(b["id"])["recent_recordings"][0]["name"] == "Bob call"
