"""
app.py - Streamlit dashboard (Milestone 4)
--------------------------------------------
Run (the API must be running first):

    uvicorn api:app --port 8000
    streamlit run app.py

The dashboard never reads the database itself: every screen calls the FastAPI
backend through `ApiClient` with the signed-in user's token, so login, per-user
meeting isolation and validation all happen in ONE place (the API).

Pages: Home - Upload - Meetings - Meeting Details - AI Assistant - Integrations - Account
"""

import streamlit as st

from api_client import ApiClient, ApiError, AuthExpired
from config import settings
from dashboard_helpers import (action_item_rows, build_filters, chart_series, fmt_date, fmt_duration,
                               meeting_label, meeting_rows, password_problems, source_row)
from logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="Career Intelligence Platform", page_icon="🎙️", layout="wide")

# ---------------------------------------------------------------------------
# Theme: teal / amber "instrument panel" (Manrope headings, Inter body)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
h1, h2, h3, .hero h1 { font-family: 'Manrope', 'Inter', sans-serif; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
.hero { background: linear-gradient(135deg, #0B3B3A 0%, #0F766E 60%, #14B8A6 100%); color: #fff;
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
        border-bottom: 4px solid #F59E0B; box-shadow: 0 8px 22px rgba(15,118,110,.18); }
.hero h1 { margin: 0; font-size: 1.7rem; font-weight: 800; }
.hero p { margin: .35rem 0 0; opacity: .92; font-size: .92rem; }
div[data-testid="stMetric"] { background: #F0FDFA; border: 1px solid #CCFBF1; border-left: 4px solid #F59E0B;
        border-radius: 10px; padding: .6rem .9rem; }
div[data-testid="stMetricValue"] { color: #0F766E; font-weight: 700; font-family: 'Manrope', sans-serif; }
.chip { display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.75rem; font-weight:600;
        background:#CCFBF1; color:#115E59; margin-right:.3rem; }
.chip.amber { background:#FEF3C7; color:#92400E; }
.chip.red { background:#FEE2E2; color:#991B1B; }
.src-card { border:1px solid #E5E7EB; border-left:4px solid #0F766E; border-radius:10px; padding:.7rem 1rem;
        margin-bottom:.6rem; background:#FAFAFA; }
.footer-note { color:#9CA3AF; font-size:.78rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

PAGES = ["Home", "Upload", "Meetings", "Meeting Details", "AI Assistant", "Integrations", "Account"]


# ---------------------------------------------------------------------------
# Session & API plumbing
# ---------------------------------------------------------------------------

def api() -> ApiClient:
    return ApiClient(base_url=settings.api_base_url, token=st.session_state.get("token"))


def sign_out(message: str = None):
    for key in ("token", "user", "chat", "selected_meeting"):
        st.session_state.pop(key, None)
    if message:
        st.session_state["flash"] = message
    st.rerun()


def call(fn, *args, default=None, **kwargs):
    """Runs an API call; turns backend errors into friendly messages instead of tracebacks."""
    try:
        return fn(*args, **kwargs)
    except AuthExpired:
        sign_out("Your session expired. Please sign in again.")
    except ApiError as e:
        prefix = f"**{e.stage}:** " if e.stage else ""
        st.error(f"{prefix}{e.message}")
        return default


def hero(title: str, subtitle: str = ""):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Login / registration
# ---------------------------------------------------------------------------

def login_screen():
    hero("🎙️ Career Intelligence Platform", "Turn meeting recordings into summaries, action items and searchable knowledge.")
    if st.session_state.get("flash"):
        st.warning(st.session_state.pop("flash"))
    _, centre, _ = st.columns([1, 1.4, 1])
    with centre:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("login"):
                user = st.text_input("Username or email")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    client = api()
                    try:
                        data = client.login(user, pw)
                    except ApiError as e:
                        st.error(e.message)
                    else:
                        st.session_state["token"], st.session_state["user"] = data["token"], data["user"]
                        st.rerun()
        with tab_up:
            with st.form("register"):
                name = st.text_input("Username")
                email = st.text_input("Email")
                pw1 = st.text_input("Password", type="password", help="At least 8 characters, with a letter and a number.")
                pw2 = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", use_container_width=True):
                    problems = password_problems(pw1, pw2)
                    if problems:
                        st.error(" ".join(problems))
                    else:
                        try:
                            api().register(name, email, pw1)
                            st.success("Account created - you can sign in now.")
                        except ApiError as e:
                            st.error(e.message)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home():
    hero("Overview", "Everything across your meetings at a glance.")
    data = call(api().overview)
    if not data:
        return
    c = st.columns(5)
    c[0].metric("Meetings", data["meeting_count"])
    c[1].metric("Hours recorded", data["total_hours"])
    c[2].metric("Action items", data["action_item_count"])
    c[3].metric("Open items", data["open_action_items"])
    c[4].metric("Completion", f"{data['completion_rate']}%")
    if not data["meeting_count"]:
        st.info("No meetings yet - head to **Upload** or **Integrations** to add your first one.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Action items by priority")
        st.bar_chart(chart_series(data["by_priority"], ("High", "Medium", "Low")))
        st.subheader("Meetings per day")
        st.bar_chart(data["meetings_per_day"])
    with right:
        st.subheader("Busiest assignees")
        if data["top_assignees"]:
            st.dataframe([{"Assignee": a["name"], "Action items": a["count"]} for a in data["top_assignees"]],
                         hide_index=True, use_container_width=True)
        st.subheader("Sources")
        st.bar_chart(data["by_source"])
        if data["failed_meetings"]:
            st.warning(f"{data['failed_meetings']} meeting(s) failed processing - see **Meetings** (status: failed).")


def page_upload():
    hero("Upload a meeting", "Audio or video → transcript → summary → action items → searchable knowledge.")
    st.caption("Supported: MP3, WAV, M4A, MP4, MOV, FLAC, OGG, AAC, WEBM, MKV · max "
               f"{settings.max_upload_mb} MB · processing a long recording can take several minutes.")
    uploaded = st.file_uploader("Choose a recording", type=["mp3", "wav", "m4a", "mp4", "mov", "flac", "ogg",
                                                             "aac", "wma", "webm", "mkv", "avi"])
    if uploaded and st.button("Process meeting", type="primary"):
        with st.spinner("Transcribing and analysing… this can take a while for long recordings."):
            result = call(api().upload_meeting, uploaded.name, uploaded.getvalue())
        if result:
            st.success(f"Processed '{uploaded.name}' - {len(result['action_items'])} action items, "
                       f"{len(result['participants'])} participants, {len(result['decisions'])} decisions.")
            st.session_state["selected_meeting"] = result["meeting_id"]
            st.markdown("**Summary**")
            st.write(result["summary"])
            st.info("Open **Meeting Details** to see the full breakdown, analytics and exports.")


def page_meetings():
    hero("Meetings", "Search and filter your meeting history.")
    client = api()
    participants = call(client.participants, default=[]) or []
    with st.expander("Filters", expanded=True):
        a, b, c = st.columns(3)
        text = a.text_input("Contains text", placeholder="filename or transcript")
        status = b.selectbox("Status", ["All", "completed", "processing", "failed"])
        source = c.selectbox("Source", ["All", "upload", "zoom", "google_meet"])
        d, e, f = st.columns(3)
        language = d.text_input("Language code", placeholder="en").strip().lower() or "All"
        date_from = e.date_input("From", value=None)
        date_to = f.date_input("To", value=None)
        person = st.selectbox("Participant", ["All"] + [p["name"] for p in participants])
    filters = build_filters(text, language, status, source, date_from, date_to, person)
    meetings = call(client.list_meetings, default=[], **filters) or []
    st.caption(f"{len(meetings)} meeting(s)")
    if not meetings:
        st.info("No meetings match these filters.")
        return
    st.dataframe(meeting_rows(meetings), hide_index=True, use_container_width=True)
    chosen = st.selectbox("Open a meeting", meetings, format_func=meeting_label)
    if st.button("View details →"):
        st.session_state["selected_meeting"] = chosen["id"]
        st.session_state["goto"] = "Meeting Details"
        st.rerun()


def page_details():
    hero("Meeting details", "Transcript, summary, decisions, action items, participants and analytics.")
    client = api()
    meetings = call(client.list_meetings, default=[]) or []
    if not meetings:
        st.info("No meetings yet.")
        return
    ids = [m["id"] for m in meetings]
    current = st.session_state.get("selected_meeting")
    index = ids.index(current) if current in ids else 0
    meeting = st.selectbox("Meeting", meetings, index=index, format_func=meeting_label)
    st.session_state["selected_meeting"] = meeting["id"]
    d = call(client.meeting_details, meeting["id"])
    if not d:
        return
    m, an = d["meeting"], d["analytics"]

    if m["status"] == "failed":
        st.error(f"This meeting failed processing: {m.get('error_message') or 'unknown error'}")
    st.markdown(f'<span class="chip">{m["source"]}</span><span class="chip amber">{m["language"] or "unknown"}</span>'
                f'<span class="chip">{fmt_duration(m["audio_duration_seconds"])}</span>'
                f'<span class="chip">{fmt_date(m["created_at"])}</span>', unsafe_allow_html=True)
    st.subheader(m["filename"])

    tabs = st.tabs(["Summary", "Decisions", "Action items", "Participants", "Deadlines", "Transcript",
                    "Analytics", "Export"])
    with tabs[0]:
        st.write(d["summary_text"] or "No summary available.")
        if d["key_points"]:
            st.markdown("**Key points**")
            for p in d["key_points"]:
                st.markdown(f"- {p}")
    with tabs[1]:
        for x in d["decisions"] or []:
            st.markdown(f"- ✅ {x}")
        if not d["decisions"]:
            st.info("No decisions were recorded for this meeting.")
    with tabs[2]:
        rows = action_item_rows(d["action_items"])
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No action items identified.")
    with tabs[3]:
        if d["participants"]:
            owners = {}
            for item in d["action_items"]:
                owners.setdefault(item.get("participant_name") or "Unassigned", []).append(item["task"])
            for p in d["participants"]:
                with st.expander(p["name"], expanded=False):
                    tasks = owners.get(p["name"], [])
                    st.write("\n".join(f"- {t}" for t in tasks) if tasks else "Attended - no action items assigned.")
        else:
            st.info("No participants identified.")
    with tabs[4]:
        for dl in d["deadlines"] or []:
            tasks = [i["task"] for i in d["action_items"] if (i.get("deadline") or "") == dl]
            st.markdown(f"**{dl}** - " + "; ".join(tasks))
        if not d["deadlines"]:
            st.info("No deadlines identified.")
    with tabs[5]:
        st.text_area("Transcript", d["transcript"], height=420, label_visibility="collapsed")
    with tabs[6]:
        c = st.columns(5)
        c[0].metric("Words", an["word_count"])
        c[1].metric("Duration", f"{an['duration_minutes']} min")
        c[2].metric("Participants", an["participant_count"])
        c[3].metric("Action items", an["action_item_count"])
        c[4].metric("Completion", f"{an['completion_rate']}%")
        l, r = st.columns(2)
        with l:
            st.markdown("**Action items by priority**")
            st.bar_chart(an["by_priority"])
            st.markdown("**With / without a deadline**")
            st.bar_chart({"With deadline": an["action_items_with_deadline"],
                          "Without": an["action_items_without_deadline"]})
        with r:
            st.markdown("**Workload by person**")
            st.bar_chart(an["by_assignee"])
            if an["top_keywords"]:
                st.markdown("**Top keywords**")
                st.write(", ".join(f"{k['keyword']} ({k['count']})" for k in an["top_keywords"]))
    with tabs[7]:
        st.caption("Reports contain this meeting's details, summary, decisions, action items, participants and deadlines.")
        cache_key = f"reports_{m['id']}"
        if st.button("Prepare PDF & CSV reports"):      # built on demand, not on every rerun
            built = {}
            for fmt in ("pdf", "csv"):
                data = call(client.download_report, m["id"], fmt)
                if data:
                    built[fmt] = data
            st.session_state[cache_key] = built
        built = st.session_state.get(cache_key, {})
        a, b = st.columns(2)
        for column, fmt, mime in ((a, "pdf", "application/pdf"), (b, "csv", "text/csv")):
            if fmt in built:
                column.download_button(f"⬇ Download {fmt.upper()}", data=built[fmt][0], file_name=built[fmt][1],
                                       mime=mime, use_container_width=True)
        st.divider()
        with st.expander("Danger zone"):
            if st.checkbox("I understand this permanently deletes the meeting and its search index"):
                if st.button("Delete this meeting", type="secondary"):
                    if call(client.delete_meeting, m["id"]):
                        st.session_state.pop("selected_meeting", None)
                        st.success("Deleted.")
                        st.rerun()


def page_assistant():
    hero("AI Assistant", "Ask questions across all of your meetings - answers are grounded in what was actually said.")
    client = api()
    mode = st.radio("Mode", ["Ask a question", "Search meetings"], horizontal=True, label_visibility="collapsed")
    top_k = st.slider("Sources to retrieve", 1, 10, 5)

    if mode == "Search meetings":
        query = st.text_input("Search", placeholder="Which meeting discussed the database migration?")
        if st.button("Search", type="primary") and query.strip():
            result = call(client.search, query, top_k)
            if result:
                st.caption(f"{len(result['results'])} result(s) in {result['search_completed_in_seconds']}s")
                if not result["results"]:
                    st.info("No relevant meetings found.")
                for r in result["results"]:
                    st.markdown(f'<div class="src-card"><b>#{r["meeting_id"]} · {r["meeting_title"]}</b> '
                                f'<span class="chip">{source_row(r)["Type"]}</span>'
                                f'<span class="chip amber">{source_row(r)["Relevance"]}</span><br>'
                                f'<small>{fmt_date(r["meeting_date"])}</small></div>', unsafe_allow_html=True)
                    st.caption(r["matched_text"][:400])
        return

    chat = st.session_state.setdefault("chat", [])
    for turn in chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("sources"):
                render_sources(turn["sources"])
    question = st.chat_input("e.g. What deadline was decided for the mobile application?")
    if question:
        chat.append({"role": "user", "content": question})
        with st.spinner("Searching your meetings…"):
            result = call(client.ask, question, top_k)
        if result:
            chat.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        st.rerun()
    if chat and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()


def render_sources(sources: list):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            row = source_row(s)
            st.markdown(f"**{row['Meeting']}** · {row['Date']} · {row['Type']} · relevance {row['Relevance']}")
            st.caption("Retrieved context")
            st.code(s.get("retrieved_context") or s.get("relevant_content", ""), language=None)


def page_integrations():
    hero("Integrations", "Pull recordings from Zoom and Google Meet through the same pipeline as uploads.")
    client = api()
    status = call(client.integration_status)
    if not status:
        return
    cols = st.columns(2)
    for col, key, label, hint in ((cols[0], "zoom", "Zoom", "ZOOM_ACCOUNT_ID / ZOOM_CLIENT_ID / ZOOM_CLIENT_SECRET"),
                                  (cols[1], "google_meet", "Google Meet",
                                   "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN")):
        with col:
            st.subheader(label)
            if status[key]["configured"]:
                st.success("Credentials configured")
                days = st.number_input("Look back (days)", 1, 90, settings.integration_lookback_days, key=f"d_{key}")
                if st.button(f"Sync {label} recordings", key=f"sync_{key}", type="primary"):
                    with st.spinner(f"Importing {label} recordings…"):
                        report = call(client.sync_recordings, key, int(days))
                    if report:
                        st.success(f"Found {report['found']} · imported {report['imported_count']} · "
                                   f"duplicates skipped {report['duplicate_count']} · failed {report['failed_count']}")
                        for f in report["failed"]:
                            st.error(f"{f['name']}: {f['error']}")
            else:
                st.warning(f"Not configured. Set {hint} on the server (see .env.example).")
    st.subheader("Recent imports")
    recordings = status["recent_recordings"]
    if recordings:
        st.dataframe([{"Provider": r["provider"], "Recording": r["name"], "Status": r["status"],
                       "Attempts": r["attempts"], "Meeting": r["meeting_id"] or "-", "Error": r["error"] or ""}
                      for r in recordings], hide_index=True, use_container_width=True)
    else:
        st.caption("No recordings imported yet.")


def page_account():
    hero("Account", "")
    user = st.session_state["user"]
    st.write(f"Signed in as **{user['username']}** ({user['email']})")
    with st.expander("Change password"):
        with st.form("pw"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            again = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Change password"):
                problems = password_problems(new, again)
                if problems:
                    st.error(" ".join(problems))
                elif call(api().change_password, old, new):
                    sign_out("Password changed - please sign in again.")
    with st.expander("Search index"):
        st.caption("Rebuild embeddings for all of your meetings (e.g. after switching embedding provider).")
        if st.button("Re-index my meetings"):
            with st.spinner("Re-indexing…"):
                result = call(api().sync_embeddings)
            if result:
                st.success(f"Re-indexed {result['meetings_synced']} meeting(s).")
    if st.button("Sign out"):
        call(api().logout)
        sign_out()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main():
    if not st.session_state.get("token"):
        login_screen()
        return
    if "goto" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("goto")
    with st.sidebar:
        st.markdown("### 🎙️ Career Intelligence")
        st.caption(f"Signed in as **{st.session_state['user']['username']}**")
        page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    {"Home": page_home, "Upload": page_upload, "Meetings": page_meetings, "Meeting Details": page_details,
     "AI Assistant": page_assistant, "Integrations": page_integrations, "Account": page_account}[page]()
    st.markdown('<div class="footer-note">Career Intelligence Platform · your meetings are private to your account</div>',
                unsafe_allow_html=True)


main()
