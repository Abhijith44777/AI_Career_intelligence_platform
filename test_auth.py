"""
test_auth.py
-------------
Milestone 4, Task 7: registration, login, authentication, session management.
"""

import sqlite3
from datetime import datetime, timedelta

from auth import (AccountLocked, AuthService, InvalidCredentials, RegistrationError, SessionInvalid,
                  hash_password, hash_token, validate_registration, verify_password)
from database import Database
from testutil import Env, raises


def make_auth(**kwargs):
    env = Env()
    return env, AuthService(env.db, **kwargs)


# -- password hashing ---------------------------------------------------------

def test_password_is_hashed_salted_and_verifiable():
    h1, h2 = hash_password("s3cretpass"), hash_password("s3cretpass")
    assert "s3cretpass" not in h1
    assert h1 != h2, "same password must hash differently (random salt)"
    assert verify_password("s3cretpass", h1) and verify_password("s3cretpass", h2)
    assert not verify_password("wrong-pass1", h1)


def test_verify_password_rejects_malformed_hashes():
    for bad in ("", "plaintext", "scrypt$1$2", "md5$a$b$c$d$e", None):
        assert verify_password("x", bad or "") is False


# -- registration ---------------------------------------------------------------

def test_register_success_returns_no_secrets():
    env, auth = make_auth()
    user = auth.register("Ravi", "Ravi@Example.com", "hunter2-hunter2")
    assert user["username"] == "Ravi" and user["email"] == "ravi@example.com"
    assert "password_hash" not in user and "password" not in user
    stored = env.db.get_user(user["id"])
    assert stored["password_hash"].startswith("scrypt$") and "hunter2" not in stored["password_hash"]


def test_register_validation_rules():
    assert validate_registration("ab", "a@b.co", "goodpass1")                # username too short
    assert validate_registration("valid_user", "not-an-email", "goodpass1")  # bad email
    assert validate_registration("valid_user", "a@b.co", "short1")           # too short
    assert validate_registration("valid_user", "a@b.co", "onlyletters")      # no digit
    assert validate_registration("valid_user", "a@b.co", "12345678")         # no letter
    assert validate_registration("valid_user", "a@b.co", "password123")      # too common
    assert validate_registration("bad name!", "a@b.co", "goodpass1")         # illegal chars
    assert validate_registration("x" * 40, "a@b.co", "goodpass1")            # too long
    assert not validate_registration("valid_user", "a@b.co", "goodpass1")


def test_register_rejects_duplicates_case_insensitively():
    env, auth = make_auth()
    auth.register("Priya", "priya@example.com", "goodpass1")
    with raises(RegistrationError):
        auth.register("priya", "other@example.com", "goodpass1")     # same username, different case
    with raises(RegistrationError):
        auth.register("priya2", "PRIYA@example.com", "goodpass1")    # same email, different case


def test_registration_can_be_disabled():
    env, auth = make_auth(allow_registration=False)
    with raises(RegistrationError):
        auth.register("someone", "s@example.com", "goodpass1")


def test_sql_injection_in_username_is_rejected_by_validation_and_harmless_in_login():
    env, auth = make_auth()
    with raises(RegistrationError):
        auth.register("x' OR '1'='1", "a@b.co", "goodpass1")
    auth.register("real_user", "r@example.com", "goodpass1")
    with raises(InvalidCredentials):
        auth.login("' OR '1'='1' --", "anything1")


# -- login ---------------------------------------------------------------------------

def test_login_by_username_or_email():
    env, auth = make_auth()
    auth.register("Amit", "amit@example.com", "goodpass1")
    assert auth.login("amit", "goodpass1")["user"]["username"] == "Amit"
    assert auth.login("AMIT@example.com", "goodpass1")["user"]["username"] == "Amit"


def test_login_errors_do_not_reveal_whether_user_exists():
    env, auth = make_auth()
    auth.register("Amit", "amit@example.com", "goodpass1")
    with raises(InvalidCredentials) as wrong_pw:
        auth.login("Amit", "badpassword1")
    with raises(InvalidCredentials) as no_user:
        auth.login("Nobody", "badpassword1")
    assert str(wrong_pw.value) == str(no_user.value)


def test_account_locks_after_repeated_failures_and_unlocks_later():
    clock = {"now": datetime(2026, 1, 1, 12, 0, 0)}
    env = Env()
    auth = AuthService(env.db, max_failed_logins=3, lockout_minutes=10, now=lambda: clock["now"])
    auth.register("Sara", "sara@example.com", "goodpass1")
    for _ in range(3):
        with raises(InvalidCredentials):
            auth.login("Sara", "wrongpass1")
    with raises(AccountLocked):                       # even the CORRECT password is refused while locked
        auth.login("Sara", "goodpass1")
    clock["now"] += timedelta(minutes=11)
    assert auth.login("Sara", "goodpass1")["token"]   # lockout expired


def test_successful_login_resets_failure_counter():
    env, auth = make_auth()
    user = auth.register("Dev", "dev@example.com", "goodpass1")
    for _ in range(2):
        with raises(InvalidCredentials):
            auth.login("Dev", "wrongpass1")
    auth.login("Dev", "goodpass1")
    assert env.db.get_user(user["id"])["failed_logins"] == 0


# -- sessions ------------------------------------------------------------------------------

def test_token_authenticates_and_only_hash_is_stored():
    env, auth = make_auth()
    auth.register("Kim", "kim@example.com", "goodpass1")
    session = auth.login("Kim", "goodpass1")
    assert auth.authenticate(session["token"])["username"] == "Kim"
    assert env.db.get_session(session["token"]) is None, "raw token must not be a DB key"
    assert env.db.get_session(hash_token(session["token"])) is not None
    assert len(session["token"]) >= 40


def test_invalid_tokens_are_rejected():
    env, auth = make_auth()
    for bad in ("", None, "garbage", "a" * 500, "Bearer x"):
        with raises(SessionInvalid):
            auth.authenticate(bad)


def test_session_expires():
    clock = {"now": datetime(2026, 1, 1, 9, 0, 0)}
    env = Env()
    auth = AuthService(env.db, session_ttl_hours=2, now=lambda: clock["now"])
    auth.register("Lee", "lee@example.com", "goodpass1")
    token = auth.login("Lee", "goodpass1")["token"]
    clock["now"] += timedelta(hours=1)
    assert auth.authenticate(token)
    clock["now"] += timedelta(hours=2)
    with raises(SessionInvalid):
        auth.authenticate(token)
    assert env.db.get_session(hash_token(token)) is None, "expired sessions are purged on use"


def test_logout_revokes_only_that_session():
    env, auth = make_auth()
    auth.register("Noor", "noor@example.com", "goodpass1")
    t1, t2 = auth.login("Noor", "goodpass1")["token"], auth.login("Noor", "goodpass1")["token"]
    auth.logout(t1)
    with raises(SessionInvalid):
        auth.authenticate(t1)
    assert auth.authenticate(t2)["username"] == "Noor"


def test_change_password_revokes_all_sessions_and_requires_old_password():
    env, auth = make_auth()
    user = auth.register("Omar", "omar@example.com", "goodpass1")
    token = auth.login("Omar", "goodpass1")["token"]
    with raises(InvalidCredentials):
        auth.change_password(user["id"], "not-the-old-one1", "brandnew123")
    with raises(RegistrationError):
        auth.change_password(user["id"], "goodpass1", "short")
    auth.change_password(user["id"], "goodpass1", "brandnew123")
    with raises(SessionInvalid):
        auth.authenticate(token)
    with raises(InvalidCredentials):
        auth.login("Omar", "goodpass1")
    assert auth.login("Omar", "brandnew123")["token"]


def test_expired_session_cleanup():
    clock = {"now": datetime(2026, 1, 1)}
    env = Env()
    auth = AuthService(env.db, session_ttl_hours=1, now=lambda: clock["now"])
    auth.register("Pat", "pat@example.com", "goodpass1")
    auth.login("Pat", "goodpass1")
    # DB compares against the real clock; the session above expires in 2026-01-01 (past) -> purged
    assert env.db.delete_expired_sessions() >= 1
