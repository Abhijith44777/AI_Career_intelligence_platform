"""
integrations/importer.py
--------------------------
Milestone 4, Tasks 4-5 (shared by Zoom and Google Meet):

    Recording -> duplicate check -> download -> validate -> the SAME pipeline
    as an uploaded file (Whisper -> LLM -> action items -> participants ->
    database -> embeddings -> vector store)

Guarantees (each covered by test_integrations.py):
  * Authentication problems stop the sync with a clear error.
  * A recording already imported (or currently being imported) is skipped -
    duplicates are impossible, even across concurrent syncs.
  * One bad recording never aborts the batch; its failure is stored on the
    `recordings` row and it is retried on the next sync, up to
    INTEGRATION_MAX_ATTEMPTS times.
  * Temporary downloads are always deleted.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta

from config import settings
from database import Database
from integrations.base import IntegrationAuthError, IntegrationError, RecordingSource
from logging_config import redact
from pipeline import safe_filename
from upload_security import looks_like_media
from utils import validate_file

logger = logging.getLogger(__name__)


@dataclass
class ImportReport:
    provider: str
    found: int = 0
    imported: list = field(default_factory=list)            # [{"external_id", "name", "meeting_id"}]
    skipped_duplicates: list = field(default_factory=list)  # [external_id]
    failed: list = field(default_factory=list)              # [{"external_id", "name", "error"}]

    def to_dict(self) -> dict:
        return {"provider": self.provider, "found": self.found, "imported": self.imported,
                "skipped_duplicates": self.skipped_duplicates, "failed": self.failed,
                "imported_count": len(self.imported), "duplicate_count": len(self.skipped_duplicates),
                "failed_count": len(self.failed)}


def _clean_error(exc: Exception) -> str:
    return redact(f"{type(exc).__name__}: {exc}")[:500]


def import_recordings(source: RecordingSource, db: Database, user_id: int, process_fn,
                      since: date = None, max_attempts: int = None) -> ImportReport:
    """
    `process_fn(raw_path, filename, size_bytes, user_id=..., source=...)` is
    normally pipeline.process_meeting_recording (wrapped by the service layer
    to inject db/LLM/embedding singletons); tests pass a fake.
    """
    max_attempts = max_attempts or settings.integration_max_attempts
    since = since or (date.today() - timedelta(days=settings.integration_lookback_days))
    report = ImportReport(provider=source.provider)

    source.authenticate()                      # raises IntegrationAuthError -> caller reports it
    recordings = source.list_recordings(since)
    report.found = len(recordings)

    for rec in recordings:
        row, created = db.register_recording(user_id, rec.provider, rec.external_id, rec.name)
        if not created and row["status"] in ("completed", "processing"):
            report.skipped_duplicates.append(rec.external_id)
            logger.info("%s: skipping duplicate recording %s", rec.provider, rec.external_id)
            continue
        if not db.claim_recording(row["id"], max_attempts):
            # failed too many times, or another worker just claimed it
            report.skipped_duplicates.append(rec.external_id)
            continue

        filename = safe_filename(rec.name)
        if not filename.lower().endswith(rec.extension.lower()):
            filename += rec.extension

        try:
            check = validate_file(filename, rec.size_bytes if rec.size_bytes else 10 * 1024)
            if not check["valid"]:
                # Permanent: retrying can't fix an unsupported type / oversize file.
                db.mark_recording_failed(row["id"], f"Validation: {check['message']}"[:500],
                                         permanent=True, max_attempts=max_attempts)
                report.failed.append({"external_id": rec.external_id, "name": rec.name, "error": check["message"]})
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                raw_path = os.path.join(tmp_dir, filename)
                size = source.download(rec, raw_path)
                if not looks_like_media(raw_path):
                    raise IntegrationError("Downloaded file is not a valid audio/video recording.", retryable=False)
                result = process_fn(raw_path, filename, size, user_id=user_id, source=rec.provider)

            db.update_recording(row["id"], "completed", meeting_id=result.meeting_id)
            report.imported.append({"external_id": rec.external_id, "name": rec.name,
                                    "meeting_id": result.meeting_id})
            logger.info("%s: imported recording %s as meeting %s", rec.provider, rec.external_id, result.meeting_id)

        except IntegrationAuthError:
            db.mark_recording_failed(row["id"], "Authentication failed during download.", max_attempts=max_attempts)
            raise
        except Exception as e:  # noqa: BLE001 - isolate every failure to its own recording
            message = _clean_error(e)
            permanent = isinstance(e, IntegrationError) and not e.retryable
            db.mark_recording_failed(row["id"], message, permanent=permanent, max_attempts=max_attempts)
            report.failed.append({"external_id": rec.external_id, "name": rec.name, "error": message})
            logger.error("%s: recording %s failed: %s", rec.provider, rec.external_id, message)

    return report

