#!/usr/bin/env sh
# Local development: API on :8000 and dashboard on :8501.  Usage: sh scripts/run_local.sh
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "No .env found - copy .env.example to .env first."; exit 1; }
python manage.py check-config || true
uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
sleep 3
streamlit run app.py
