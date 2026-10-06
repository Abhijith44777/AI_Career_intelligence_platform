#!/usr/bin/env sh
# Production deploy with Docker.  Usage: sh scripts/deploy.sh
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "Create .env from .env.example first (PASSWORD_PEPPER is required)."; exit 1; }
docker compose build
docker compose up -d
echo "Waiting for the API to become healthy..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health/ready >/dev/null 2>&1; then
    echo "API is ready.   Dashboard: http://localhost:8501   API: http://localhost:8000"
    exit 0
  fi
  sleep 3
done
echo "API did not become healthy in time. Inspect: docker compose logs api"
exit 1
