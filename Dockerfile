# Career Intelligence Platform - one image, two processes (api | dashboard).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# ffmpeg is required by Whisper for audio/video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; /data is the persistent volume (db, vectors, transcripts, logs).
RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser /app /data
USER appuser

ENV APP_ENV=production \
    DATABASE_PATH=/data/meetings.db \
    VECTOR_DATABASE_URL=/data/chroma_db \
    TRANSCRIPTS_DIR=/data/transcripts \
    LOG_DIR=/data/logs \
    API_HOST=0.0.0.0 API_PORT=8000

EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn api:app --host ${API_HOST} --port ${API_PORT}"]
