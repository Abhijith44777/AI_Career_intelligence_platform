"""
auth.py
--------
Milestone 4 - Task 7: User Access & Security.

Registration, login, session management. Deliberately dependency-free
(stdlib `hashlib.scrypt`, `secrets`, `hmac`) so it runs anywhere the rest of
the project does.

Security properties (each has a test in test_auth.py / test_security_access.py):
  * Passwords are stored as salted scrypt hashes (+ optional server-side
    pepper); the plaintext is never stored or logged.
  * Login errors are identical for "no such user" and "wrong password", and a
    dummy hash is computed for unknown users so response time doesn't reveal
    which usernames exist.
  * Repeated failures lock the account temporarily (brute-force protection).
  * Session tokens are 256-bit random values; only their SHA-256 is stored.
    Sessions expire, and can be revoked individually (logout) or all at once
    (password change).
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
from datetime import datetime, timedelta

from config import settings
from database import Database

logger = logging.getLogger(__name__)

# scrypt cost parameters (~16 MB, tens of ms per hash on commodity hardware).
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
_COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwertyuiop", "qwerty123", "iloveyou1", "admin123", "welcome123", "letmein123",
}


class AuthError(Exception):
    """Base class; `.code` is a stable machine-readable reason for the API."""
    code = "auth_error"


class RegistrationError(AuthError):
    code = "invalid_registration"


class InvalidCredentials(AuthError):
    code = "invalid_credentials"


class AccountLocked(AuthError):
    code = "account_locked"


class SessionInvalid(AuthError):
    code = "invalid_session"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _prehash(password: str) -> bytes:
    """Mixes in the server-side pepper (HMAC) before scrypt. With no pepper
    configured this is just the UTF-8 password."""
    pepper = settings.password_pepper.encode()
    raw = password.encode("utf-8")
    return hmac.new(pepper, raw, hashlib.sha256).digest() if pepper else raw


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(_prehash(password), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        actual = hashlib.scrypt(_prehash(password), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_DUMMY_HASH = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("dummy-password-for-timing-equalisation")
    return _DUMMY_HASH


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_registration(username: str, email: str, password: str) -> list:
    """Returns a list of problems (empty == valid)."""
    problems = []
    username, email, password = username or "", email or "", password or ""
    if not USERNAME_RE.match(username):
        problems.append("Username must be 3-32 characters: letters, numbers, '.', '_' or '-'.")
    if len(email) > 254 or not EMAIL_RE.match(email):
        problems.append("Enter a valid email address.")
    if len(password) < 8:
        problems.append("Password must be at least 8 characters.")
    if len(password) > 128:
        problems.append("Password must be at most 128 characters.")
    if password and not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        problems.append("Password must contain at least one letter and one number.")
    if password.lower() in _COMMON_PASSWORDS:
        problems.append("That password is too common - choose another.")
    if username and password and username.lower() == password.lower():
        problems.append("Password must not be the same as the username.")
    return problems


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def public_user(user: dict) -> dict:
    """The only shape of a user that ever leaves this module - no hash, no lock state."""
    return {"id": user["id"], "username": user["username"], "email": user["email"],
            "created_at": user["created_at"]}


class AuthService:
    def __init__(self, db: Database, session_ttl_hours: int = None, max_failed_logins: int = None,
                 lockout_minutes: int = None, allow_registration: bool = None, now=datetime.now):
        self.db = db
        self.session_ttl = timedelta(hours=session_ttl_hours if session_ttl_hours is not None
                                     else settings.session_ttl_hours)
        self.max_failed = max_failed_logins if max_failed_logins is not None else settings.max_failed_logins
        self.lockout = timedelta(minutes=lockout_minutes if lockout_minutes is not None
                                 else settings.lockout_minutes)
        self.allow_registration = settings.allow_registration if allow_registration is None else allow_registration
        self._now = now

    # -- Registration --------------------------------------------------------

    def register(self, username: str, email: str, password: str) -> dict:
        if not self.allow_registration:
            raise RegistrationError("Registration is disabled on this server.")
        username, email = (username or "").strip(), (email or "").strip().lower()
        problems = validate_registration(username, email, password)
        if problems:
            raise RegistrationError(" ".join(problems))
        try:
            user_id = self.db.create_user(username, email, hash_password(password))
        except sqlite3.IntegrityError:
            # One message for both cases - don't reveal which of the two exists.
            raise RegistrationError("That username or email is already registered.")
        logger.info("User registered: id=%s username=%s", user_id, username)
        return public_user(self.db.get_user(user_id))

    # -- Login / sessions ------------------------------------------------------

    def login(self, username_or_email: str, password: str) -> dict:
        """Returns {"token", "expires_at", "user"}. Raises InvalidCredentials / AccountLocked."""
        identifier = (username_or_email or "").strip()
        user = (self.db.get_user_by_email(identifier.lower()) if "@" in identifier
                else self.db.get_user_by_username(identifier))

        if user is None:
            verify_password(password or "", _dummy_hash())  # equalise timing
            raise InvalidCredentials("Invalid username or password.")

        now = self._now()
        if user["locked_until"] and datetime.fromisoformat(user["locked_until"]) > now:
            raise AccountLocked("Too many failed attempts. Try again later.")

        if not verify_password(password or "", user["password_hash"]):
            failures = (user["failed_logins"] or 0) + 1
            locked_until = None
            if failures >= self.max_failed:
                locked_until = (now + self.lockout).isoformat()
                failures = 0
                logger.warning("Account locked after repeated failures: user_id=%s", user["id"])
            self.db.update_login_state(user["id"], failures, locked_until)
            raise InvalidCredentials("Invalid username or password.")

        self.db.update_login_state(user["id"], 0, None)
        token = secrets.token_urlsafe(32)
        expires_at = (now + self.session_ttl).isoformat()
        self.db.create_session(hash_token(token), user["id"], expires_at)
        logger.info("User logged in: user_id=%s", user["id"])
        return {"token": token, "expires_at": expires_at, "user": public_user(user)}

    def authenticate(self, token: str) -> dict:
        """Token -> public user dict, or raises SessionInvalid."""
        if not token or not isinstance(token, str) or len(token) > 256:
            raise SessionInvalid("Not authenticated.")
        token_hash = hash_token(token)
        session = self.db.get_session(token_hash)
        if session is None:
            raise SessionInvalid("Not authenticated.")
        if datetime.fromisoformat(session["expires_at"]) <= self._now():
            self.db.delete_session(token_hash)
            raise SessionInvalid("Session expired. Please log in again.")
        user = self.db.get_user(session["user_id"])
        if user is None:
            self.db.delete_session(token_hash)
            raise SessionInvalid("Not authenticated.")
        return public_user(user)

    def logout(self, token: str) -> None:
        if token:
            self.db.delete_session(hash_token(token))

    def logout_everywhere(self, user_id: int) -> None:
        self.db.delete_user_sessions(user_id)

    def change_password(self, user_id: int, old_password: str, new_password: str) -> None:
        user = self.db.get_user(user_id)
        if user is None or not verify_password(old_password or "", user["password_hash"]):
            raise InvalidCredentials("Current password is incorrect.")
        problems = validate_registration(user["username"], user["email"], new_password)
        problems = [p for p in problems if "Username" not in p and "email" not in p]
        if problems:
            raise RegistrationError(" ".join(problems))
        self.db.update_password_hash(user_id, hash_password(new_password))
        self.db.delete_user_sessions(user_id)  # every existing session is revoked
