"""
diagnose_search.py - run from the project folder:  python diagnose_search.py
Optional:  python diagnose_search.py "your question here"

Shows (1) which embedding provider is really in use and whether a LIVE call to
it works, (2) what is stored, and (3) the real scores your question gets.
"""
import sys
import logging

logging.disable(logging.WARNING)  # hide the per-call "falling back" noise; section 1 reports it properly

import embedding_service as es
from embedding_service import EmbeddingService
from database import Database
from vector_store import VectorStore

QUESTION = sys.argv[1] if len(sys.argv) > 1 else (
    "What are the main action items decided during this meeting, "
    "and who is responsible for completing each one?"
)

print("=" * 72)
print("1) EMBEDDING PROVIDER")
print("=" * 72)
if not hasattr(es, "resolve_auto_provider"):
    print(">>> OLD embedding_service.py is still in this folder (no Gemini support).")
    print(">>> Copy the new files over the old ones, then run this again.")
    sys.exit(1)

emb = EmbeddingService()
cfg = emb.config
print(f"provider : {cfg.provider}")
print(f"model    : {cfg.model}")
print(f"vector   : {cfg.dimensions} dimensions")
print(f"API key  : {'found' if emb._api_key() else 'NOT FOUND (' + (cfg.api_key_env_var or 'none needed') + ')'}")
try:
    from importlib.metadata import version
    print(f"google-genai package: {version('google-genai')}")
except Exception:
    print("google-genai package: not installed")

if cfg.provider == "local":
    print("\n>>> Using the OFFLINE matcher (no embedding API key found in .env).")
else:
    print(f"\nLive test call to {cfg.provider} ...")
    try:
        vectors = emb._embed_batch_remote(["connectivity test"], "RETRIEVAL_QUERY")
        print(f"  OK - got a {len(vectors[0])}-dimension vector back. {cfg.provider} embeddings WORK.")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED - {type(e).__name__}: {e}")
        print("  (Searches will silently use the weaker offline matcher until this is fixed.)")

print()
print("=" * 72)
print("2) WHAT IS STORED (for the vector size in use)")
print("=" * 72)
db = Database()
store = VectorStore()
meetings = db.list_meetings()
print(f"meetings in database: {len(meetings)}    vectors in store: {store.count()}    backend: {store.backend_name}")
for m in meetings:
    n = len(store.get_by_meeting(m["id"]))
    print(f"  meeting {m['id']}: {n} vectors" + ("" if n else "   <-- NOT INDEXED: run  python diagnose.py"))

print()
print("=" * 72)
print("3) SCORES FOR YOUR QUESTION")
print("=" * 72)
print("question:", QUESTION)
vec = emb.embed_query(QUESTION)
floor = emb.min_relevance_score
print(f"query vector: {len(vec)} dimensions   |   relevance floor: {floor}   |   "
      f"embedder used: {'OFFLINE matcher' if emb.using_local_embedder else cfg.provider}")
raw = store.query(vec, top_k=8)
if not raw:
    print("nothing stored with this vector size -> run  python diagnose.py  to (re)index every meeting")
for r in raw:
    verdict = "PASS" if r.score >= floor else "below floor"
    print(f"  {r.score:6.3f}  {verdict:11}  meeting {r.metadata.get('meeting_id')}  "
          f"{str(r.metadata.get('content_type')):18} {r.document[:60]!r}")
print(f"\n(chunks scoring below {floor} are treated as 'not relevant')")
