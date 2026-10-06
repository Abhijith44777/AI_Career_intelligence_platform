# AI-Powered Career Intelligence Platform

Meeting recordings → Whisper transcript → LLM summary, decisions & action items → participant
mapping → database → embeddings → semantic search & RAG → Streamlit dashboard, PDF/CSV reports,
Zoom / Google Meet import. Multi-user, with per-user access control.

| Milestone | What it added |
|---|---|
| 1 | Audio validation, Whisper transcription |
| 2 | LLM summarization, action items, participants, SQLite persistence |
| 3 | Knowledge repository, embeddings, vector DB (ChromaDB), semantic search, RAG Q&A |
| **4** | **Login & access control, Streamlit dashboard, meeting details & analytics, PDF/CSV export, Zoom & Google Meet import, hardening, Docker deployment** |

## Architecture

```
Streamlit dashboard (app.py) ──HTTP + bearer token──▶ FastAPI (api.py)
   api_client.py                                        │  auth.py (register/login/sessions)
   dashboard_helpers.py                                 ▼
                                                   services.py   ← every call is scoped to the signed-in user
                          ┌─────────────┬───────────────┼────────────────┬─────────────────┐
                     pipeline.py   knowledge_repository  semantic_search   reports.py     integrations/
                   (Whisper→LLM→DB) (embeddings→vectors)  + rag_qa.py     (PDF, CSV)    (Zoom, Google Meet)
                          └─────────────┴──── database.py (SQLite) · vector_store.py (Chroma / local) ───┘
```

The dashboard never reads the database. Authentication and ownership checks live in one place
(`services.py`), so no endpoint can forget them.

## Quick start (local)

```bash
python -m venv venv && venv\Scripts\activate        # Windows   (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt                      # ffmpeg must also be installed for Whisper
copy .env.example .env                               # then set GEMINI_API_KEY etc.
scripts\run_local.bat                                # API :8000 + dashboard :8501   (sh scripts/run_local.sh on macOS/Linux)
```
Open http://localhost:8501, **Create account**, then **Upload**. API docs: http://localhost:8000/docs (development only).

### Upgrading from Milestone 3
The database upgrades itself in place on first start (nothing is deleted). Existing meetings have no
owner, so they are invisible until you assign them:

```bash
python manage.py create-user yourname you@example.com      # prompts for a password
python manage.py assign-legacy yourname                    # gives old meetings to that account + rebuilds their vectors
```
Other commands: `check-config`, `reindex [user]`, `purge-sessions`.

## Configuration
Everything comes from environment variables / `.env` (see **`.env.example`**, fully commented):
API/dashboard addresses, upload limit, Whisper model, session & lockout policy, storage paths,
logging, Sentry, Zoom and Google credentials. `python manage.py check-config` prints the effective
settings with secrets masked. With `APP_ENV=production` the API **refuses to start** if
`PASSWORD_PEPPER` is missing/weak, `CORS_ORIGINS` contains `*`, or `API_BASE_URL` is plain http to a public host.

## Zoom & Google Meet
Both feed the **same pipeline as an upload**. Duplicates are impossible (unique per user + provider + recording id),
failures are recorded per recording and retried on the next sync (max `INTEGRATION_MAX_ATTEMPTS`), and one bad recording never stops the batch.
* **Zoom** – create a *Server-to-Server OAuth* app; scope `cloud_recording:read:list_user_recordings:admin`. Set `ZOOM_ACCOUNT_ID/CLIENT_ID/CLIENT_SECRET`.
* **Google Meet** – recordings are in Drive → "Meet Recordings". Enable the Drive API, create an OAuth client, obtain a refresh token once for scope `drive.readonly`. Set `GOOGLE_CLIENT_ID/CLIENT_SECRET/REFRESH_TOKEN`.

Then use **Integrations → Sync** in the dashboard.

## Deployment (Docker)
```bash
copy .env.example .env      # set APP_ENV=production, PASSWORD_PEPPER, API keys, CORS_ORIGINS
sh scripts/deploy.sh        # docker compose build + up + wait for /health/ready
```
Data (database, vectors, transcripts, logs) lives in the `app_data` volume (`/data`). Put a TLS-terminating
reverse proxy (nginx/Caddy/cloud LB) in front, serve the API over https, and back up the volume.
Run a **single API worker** (SQLite + the local vector store are single-node); for horizontal scale move to Postgres + a server-mode vector DB.

Health: `GET /health` (liveness), `GET /health/ready` (database + vector store). Logs: console + rotating `logs/app.log`,
request-id tagged, with API keys/tokens/passwords redacted. Optional Sentry via `SENTRY_DSN`.

## Testing
```bash
python run_tests.py          # whole offline suite - no API keys, network, ffmpeg or pytest needed
python run_tests.py -v auth  # one module, verbose
python -m pytest             # also works
```
See **MILESTONE4_REPORT.md** for what each task's tests prove and what still needs a live check.

## Security model (summary)
scrypt-hashed passwords + server pepper · generic login errors & timing equalisation · lockout after repeated failures ·
256-bit session tokens stored only as SHA-256, with expiry and revocation · every meeting read/write/search/RAG/export/delete
filtered by owner (foreign ids return 404, indistinguishable from missing) · vectors carry the owner id and are re-checked against
the database · upload size streaming limit, extension whitelist and magic-number check · filename sanitising · CSV formula-injection
neutralised · parameterised SQL everywhere · generic 500s (details only in logs).

## Project layout
`app.py` dashboard · `api.py` HTTP API · `services.py` business layer · `auth.py` · `config.py` · `logging_config.py` ·
`database.py` · `pipeline.py` · `llm_service.py`, `summarization.py`, `action_items.py`, `participants.py`, `local_extraction.py` (M2) ·
`embedding_service.py`, `vector_store.py`, `knowledge_repository.py`, `semantic_search.py`, `rag_qa.py` (M3) ·
`analytics.py`, `reports.py`, `integrations/` (M4) · `manage.py` CLI · `diagnose_search.py`, `accuracy_test.py` tools · `test_*.py`.
