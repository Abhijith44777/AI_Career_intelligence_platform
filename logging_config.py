"""
logging_config.py
-------------------
Milestone 4 - Task 10: Logging & error monitoring.

* Console + rotating file logs (`logs/app.log`, 5 x 5 MB) with a request-id
  field so one API call can be traced across modules.
* A redaction filter that masks API keys, bearer tokens, passwords and
  session tokens in every log line - sensitive data never reaches the logs
  (Task 9: "Sensitive data protection").
* Optional Sentry error monitoring, enabled only when SENTRY_DSN is set AND
  `sentry_sdk` is installed (no hard dependency).
"""

import contextvars
import logging
import logging.handlers
import os
import re

from config import settings

request_id_var = contextvars.ContextVar("request_id", default="-")

_REDACTIONS = [
    # Authorization headers / bearer tokens
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-~+/=]{8,}"), r"\1***"),
    # key=value or key: value style secrets
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|pepper|refresh[_-]?token|authorization)"
                r"(\s*[=:]\s*)(['\"]?)[^\s'\",;&]{3,}"), r"\1\2\3***"),
    # Well-known key shapes
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "AIza***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
    (re.compile(r"gsk_[A-Za-z0-9]{16,}"), "gsk_***"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging itself crash the app
            return True
        record.msg = redact(message)
        record.args = ()
        record.request_id = request_id_var.get()
        return True


_configured = False


def setup_logging(force: bool = False) -> None:
    """Idempotent - safe to call from both api.py and the dashboard."""
    global _configured
    if _configured and not force:
        return

    fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if settings.log_to_file:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            handlers.append(logging.handlers.RotatingFileHandler(
                os.path.join(settings.log_dir, "app.log"),
                maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"))
        except OSError as e:  # read-only filesystem etc. - console logging still works
            logging.getLogger(__name__).warning("File logging disabled: %s", e)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    redactor = RedactingFilter()
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt))
        handler.addFilter(redactor)
        root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    _init_sentry()
    _configured = True


def _init_sentry() -> None:
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.app_env,
                        traces_sample_rate=0.0, send_default_pii=False)
        logging.getLogger(__name__).info("Sentry error monitoring enabled")
    except ImportError:
        logging.getLogger(__name__).warning("SENTRY_DSN is set but sentry_sdk is not installed - skipping.")


def capture_exception(exc: BaseException) -> None:
    """Forward to Sentry when enabled; otherwise a no-op (the log line already exists)."""
    if not settings.sentry_dsn:
        return
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        pass
