"""
upload_security.py
--------------------
Milestone 4, Task 9: file validation hardening for uploads.

* `save_upload_limited` streams an upload to disk and aborts as soon as it
  exceeds the size limit - a 5 GB upload never fills the disk or memory.
* `looks_like_media` checks the file's leading bytes ("magic numbers") against
  its claimed extension, so a renamed executable/zip/text file is rejected
  before ffmpeg or Whisper ever sees it.
"""

import os

CHUNK = 1024 * 1024


class UploadTooLarge(Exception):
    pass


def save_upload_limited(fileobj, dest_path: str, max_bytes: int) -> int:
    """Copies `fileobj` to `dest_path`; raises UploadTooLarge (and removes the
    partial file) once more than `max_bytes` have been received."""
    written = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = fileobj.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
                out.write(chunk)
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    return written


def _is_iso_bmff(head: bytes) -> bool:      # mp4 / m4a / mov / 3gp
    return len(head) >= 12 and head[4:8] == b"ftyp"


def _is_mp3(head: bytes) -> bool:
    return head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)


def _is_aac(head: bytes) -> bool:
    return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xF0) == 0xF0


_CHECKS = {
    ".wav": lambda h: h[:4] == b"RIFF" and h[8:12] == b"WAVE",
    ".avi": lambda h: h[:4] == b"RIFF" and h[8:12] == b"AVI ",
    ".mp3": _is_mp3,
    ".m4a": _is_iso_bmff,
    ".mp4": _is_iso_bmff,
    ".mov": lambda h: _is_iso_bmff(h) or h[4:8] in (b"moov", b"mdat", b"wide", b"free"),
    ".flac": lambda h: h[:4] == b"fLaC",
    ".ogg": lambda h: h[:4] == b"OggS",
    ".aac": lambda h: _is_aac(h) or h[:3] == b"ID3",
    ".wma": lambda h: h[:4] == bytes.fromhex("3026B275"),
    ".mkv": lambda h: h[:4] == bytes.fromhex("1A45DFA3"),
    ".webm": lambda h: h[:4] == bytes.fromhex("1A45DFA3"),
}


def looks_like_media(path: str) -> bool:
    """True if the file's first bytes match what its extension claims.
    Unknown extensions return False (the extension whitelist runs first)."""
    ext = os.path.splitext(path)[1].lower()
    check = _CHECKS.get(ext)
    if check is None:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    return bool(head) and bool(check(head))
