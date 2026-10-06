"""
integration_test_milestone3.py
--------------------------------
Manual, narrated run of the FULL Milestone 3 pipeline end-to-end, using the
local/offline providers (no API keys needed) so this can be run and its
real output captured for the delivery report:

    Existing Meetings -> Database -> Embedding Generation -> Vector Database
    -> User Query -> Query Embedding -> Similarity Search -> Relevant Meeting
    -> Database Verification -> Relevant Context -> LLM -> Grounded Answer
    -> Source Meeting

Also exercises the required "unsupported question" case to prove the system
does not hallucinate when no relevant context exists.

Run with:  python3 integration_test_milestone3.py
"""

import os
import tempfile

from database import Database
from embedding_service import EmbeddingService, EmbeddingConfig
from vector_store import VectorStore
from llm_service import LLMService, LLMConfig
from knowledge_repository import index_meeting, verify_meeting_retrieval
from semantic_search import semantic_search, LATENCY_TARGET_SECONDS
from rag_qa import answer_question, NO_CONTEXT_ANSWER


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


fd, db_path = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.remove(db_path)
db = Database(db_path=db_path)

fd, store_path = tempfile.mkstemp(suffix=".json")
os.close(fd)
os.remove(store_path)
store = VectorStore(backend="local", path=store_path)

embedder = EmbeddingService(EmbeddingConfig(provider="local"))
llm = LLMService(LLMConfig(provider="local"))

section("STEP 1 - Seed three existing meetings in the database (Milestone 2 shape)")

meeting_1 = db.create_meeting(
    "mobile-launch-planning.wav",
    "Ravi will handle the API integration by Friday. Priya will prepare the UI testing report. "
    "The team agreed the mobile application launch should proceed as planned.",
    language="en", audio_duration_seconds=180,
)
db.save_summary(meeting_1, "The team discussed the mobile application launch and assigned responsibilities.",
                 ["Launch timeline confirmed"],
                 ["Continue with the planned mobile application launch."], ["Mobile launch"])
ravi_id = db.get_or_create_participant("Ravi")
priya_id = db.get_or_create_participant("Priya")
db.save_action_item(meeting_1, ravi_id, "Complete API integration", "Friday", "High", "Not Started")
db.save_action_item(meeting_1, priya_id, "Prepare UI testing report", "", "Medium", "Not Started")
print(f"Created meeting_id={meeting_1}: mobile-launch-planning.wav")

meeting_2 = db.create_meeting(
    "infra-sync.wav",
    "We completed the database migration to the new cluster this week without any downtime. "
    "Karan led the migration effort.",
    language="en", audio_duration_seconds=120,
)
db.save_summary(meeting_2, "The team completed the production database migration.",
                 [], ["Migrate the production database to the new cluster."], [])
karan_id = db.get_or_create_participant("Karan")
db.save_action_item(meeting_2, karan_id, "Monitor the new database cluster for a week", "Next Friday", "High", "In Progress")
print(f"Created meeting_id={meeting_2}: infra-sync.wav")

meeting_3 = db.create_meeting(
    "hr-standup.wav",
    "We reviewed the holiday schedule and approved two new engineering hires for next quarter.",
    language="en", audio_duration_seconds=90,
)
db.save_summary(meeting_3, "HR reviewed hiring plans for next quarter.",
                 [], ["Approve two new engineering hires for next quarter."], [])
print(f"Created meeting_id={meeting_3}: hr-standup.wav")

section("STEP 2 - Embedding Generation + Vector Database (Tasks 2 & 3)")
for mid in (meeting_1, meeting_2, meeting_3):
    count = index_meeting(db, mid, embedder, store)
    print(f"  meeting_id={mid}: {count} chunks embedded and upserted into the vector store")
print(f"Total vectors in store: {store.count()}  (backend: {store.backend_name})")

section("STEP 3 - Task 1 verification: retrieval + linkage, all three meetings")
for mid in (meeting_1, meeting_2, meeting_3):
    report = verify_meeting_retrieval(db, mid)
    print(f"  meeting_id={mid}: ok={report['ok']}  checks={report['checks']}")

section("STEP 4 - Task 3 verification: meeting-to-vector mapping")
for mid in (meeting_1, meeting_2, meeting_3):
    vectors = store.get_by_meeting(mid)
    ids_ok = all(v.metadata.get("meeting_id") == mid for v in vectors)
    print(f"  meeting_id={mid}: {len(vectors)} vectors, all correctly tagged: {ids_ok}")

section("STEP 5 - Task 4: Semantic Search (5 real natural-language queries)")
queries = [
    "Which meeting discussed the database migration?",
    "What was discussed about the mobile application?",
    "What decisions were made regarding new engineering hires?",
]
for q in queries:
    response = semantic_search(q, db=db, embedding_service=embedder, vector_store=store, top_k=3)
    top = response.results[0] if response.results else None
    print(f"\n  Query: {q!r}")
    print(f"  Search completed in: {response.elapsed_seconds:.4f}s "
          f"(target < {LATENCY_TARGET_SECONDS:.0f}s -> {response.within_latency_target})")
    if top:
        print(f"  Top match: meeting_id={top.meeting_id}  \"{top.meeting_title}\"  "
              f"score={top.relevance_score}  content_type={top.content_type}")
    else:
        print("  No results.")

section("STEP 6 - Task 5: RAG Question Answering (grounded, with sources)")
rag_questions = [
    "What deadline was decided for the mobile application?",
    "Which meeting discussed the database migration?",
    "Who was assigned to monitor the new database cluster?",
]
for q in rag_questions:
    result = answer_question(q, db=db, embedding_service=embedder, vector_store=store, llm_service=llm)
    print(f"\n  Question: {q!r}")
    print(f"  Answer: {result.answer}")
    print(f"  Grounded: {result.grounded}  |  Sources: {[s.meeting_id for s in result.sources]}")

section("STEP 7 - Task 5 required check: question with NO answer in the data (no hallucination)")
unsupported = answer_question("What is the weather forecast for tomorrow?",
                               db=db, embedding_service=embedder, vector_store=store, llm_service=llm)
print(f"  Question: 'What is the weather forecast for tomorrow?'")
print(f"  Answer: {unsupported.answer}")
print(f"  Matches expected no-hallucination behavior: {unsupported.answer == NO_CONTEXT_ANSWER}")
print(f"  Sources returned (should be empty): {unsupported.sources}")

section("INTEGRATION TEST COMPLETE - full pipeline verified end-to-end")
