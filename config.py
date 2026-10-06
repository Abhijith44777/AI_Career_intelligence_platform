"""
config.py
----------
Milestone 4 - Task 10: Production configuration.

One place that reads every deployment setting from environment variables
(or a local .env file), with safe defaults for development and hard
validation for production. Nothing in the codebase hardcodes a path, port,
secret, or credential - modules import `settings` from here instead.

    APP_ENV=production   -> `validate_for_production()` refuses to start with
                            insecure defaults (see below).
"""

import os
from dataclasses import dataclass, field

try:  # python-dotenv is a hard requirement of the project, but stay import-safe
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_INSECURE_SECRET_MARKERS = ("change-me", "changeme", "your-secret", "secret", "")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: str = "") -> list:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # --- Environment ------------------------------------------------------
    app_env: str = field(default_factory=lambda: _env("APP_ENV", "development").lower())
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", os.path.join(BASE_DIR, "logs")))
    log_to_file: bool = field(default_factory=lambda: _env_bool("LOG_TO_FILE", True))
    sentry_dsn: str = field(default_factory=lambda: _env("SENTRY_DSN"))

    # --- Database / storage ------------------------------------------------
    database_path: str = field(default_factory=lambda: _env("DATABASE_PATH", os.path.join(BASE_DIR, "meetings.db")))
    transcripts_dir: str = field(default_factory=lambda: _env("TRANSCRIPTS_DIR", os.path.join(BASE_DIR, "transcripts")))
    save_transcript_files: bool = field(default_factory=lambda: _env_bool("SAVE_TRANSCRIPT_FILES", True))

    # --- API -----------------------------------------------------------------
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    cors_origins: list = field(default_factory=lambda: _env_list("CORS_ORIGINS", "http://localhost:8501"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 500))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))   # tiny|base|small|medium|large

    # --- Frontend (Streamlit talks to the API over HTTP) ----------------------
    api_base_url: str = field(default_factory=lambda: _env("API_BASE_URL", "http://127.0.0.1:8000"))
    request_timeout_seconds: int = field(default_factory=lambda: _env_int("API_REQUEST_TIMEOUT", 900))

    # --- Authentication --------------------------------------------------------
    session_ttl_hours: int = field(default_factory=lambda: _env_int("SESSION_TTL_HOURS", 12))
    max_failed_logins: int = field(default_factory=lambda: _env_int("MAX_FAILED_LOGINS", 5))
    lockout_minutes: int = field(default_factory=lambda: _env_int("LOCKOUT_MINUTES", 15))
    allow_registration: bool = field(default_factory=lambda: _env_bool("ALLOW_REGISTRATION", True))
    # Pepper mixed into password hashes; rotating it invalidates every password.
    password_pepper: str = field(default_factory=lambda: _env("PASSWORD_PEPPER"))

    # --- Zoom (Server-to-Server OAuth app) ----------------------------------------
    zoom_account_id: str = field(default_factory=lambda: _env("ZOOM_ACCOUNT_ID"))
    zoom_client_id: str = field(default_factory=lambda: _env("ZOOM_CLIENT_ID"))
    zoom_client_secret: str = field(default_factory=lambda: _env("ZOOM_CLIENT_SECRET"))
    zoom_user_id: str = field(default_factory=lambda: _env("ZOOM_USER_ID", "me"))

    # --- Google Meet (recordings are saved to Google Drive) -------------------------
    google_client_id: str = field(default_factory=lambda: _env("GOOGLE_CLIENT_ID"))
    google_client_secret: str = field(default_factory=lambda: _env("GOOGLE_CLIENT_SECRET"))
    google_refresh_token: str = field(default_factory=lambda: _env("GOOGLE_REFRESH_TOKEN"))

    # --- Integrations behaviour ---------------------------------------------------------
    integration_max_attempts: int = field(default_factory=lambda: _env_int("INTEGRATION_MAX_ATTEMPTS", 3))
    integration_lookback_days: int = field(default_factory=lambda: _env_int("INTEGRATION_LOOKBACK_DAYS", 7))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def zoom_configured(self) -> bool:
        return bool(self.zoom_account_id and self.zoom_client_id and self.zoom_client_secret)

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret and self.google_refresh_token)

    def validate_for_production(self) -> list:
        """Returns a list of human-readable problems (empty list == OK)."""
        problems = []
        if not self.is_production:
            return problems
        if self.password_pepper.strip().lower() in _INSECURE_SECRET_MARKERS or len(self.password_pepper) < 16:
            problems.append("PASSWORD_PEPPER must be set to a random string of at least 16 characters.")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain '*' in production.")
        if self.api_base_url.startswith("http://"):
            host = self.api_base_url[len("http://"):].split("/")[0].split(":")[0].lower()
            # Plain http is only acceptable for loopback or the internal docker-compose service name.
            if host not in ("localhost", "127.0.0.1", "api", "backend"):
                problems.append("API_BASE_URL should use https:// in production.")
        return problems

    def redacted(self) -> dict:
        """Settings safe to log / show: every secret becomes '***set***' or '(not set)'."""
        secret_fields = {"password_pepper", "zoom_client_secret", "google_client_secret",
                         "google_refresh_token", "sentry_dsn"}
        out = {}
        for key, value in self.__dict__.items():
            if key in secret_fields:
                out[key] = "***set***" if value else "(not set)"
            else:
                out[key] = value
        return out


def load_settings() -> Settings:
    return Settings()


settings = load_settings()
