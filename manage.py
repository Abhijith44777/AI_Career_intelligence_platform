"""
manage.py - administration commands (Milestone 4, Tasks 7 & 10)
-----------------------------------------------------------------
    python manage.py check-config
    python manage.py create-user USERNAME EMAIL            (prompts for the password)
    python manage.py assign-legacy USERNAME                give pre-Milestone-4 meetings to USERNAME
    python manage.py reindex [USERNAME]                    rebuild embeddings (one user, or everyone)
    python manage.py purge-sessions                        delete expired login sessions

UPGRADING from Milestone 3: your existing meetings have no owner, so nobody can
see them until you run `assign-legacy`. That command also rebuilds their search
vectors so they carry the owner id used for access control.
"""

import argparse
import getpass
import sys

from auth import AuthService, RegistrationError
from config import settings
from database import Database
from logging_config import setup_logging


def cmd_check_config(args, db) -> int:
    print("Effective configuration (secrets masked):")
    for key, value in sorted(settings.redacted().items()):
        print(f"  {key:28} {value}")
    problems = settings.validate_for_production()
    for p in problems:
        print(f"  [PROBLEM] {p}")
    if not problems:
        print("\nNo configuration problems found.")
    return 1 if problems else 0


def cmd_create_user(args, db) -> int:
    password = args.password or getpass.getpass("Password: ")
    try:
        user = AuthService(db).register(args.username, args.email, password)
    except RegistrationError as e:
        print(f"Could not create user: {e}")
        return 1
    print(f"Created user '{user['username']}' (id {user['id']}).")
    return 0


def cmd_assign_legacy(args, db) -> int:
    user = db.get_user_by_username(args.username)
    if user is None:
        print(f"No such user: {args.username}")
        return 1
    count = db.assign_legacy_meetings(user["id"])
    print(f"Assigned {count} legacy meeting(s) to '{user['username']}'.")
    if count:
        from knowledge_repository import reindex_all_meetings
        results = reindex_all_meetings(db, user_id=user["id"])
        print(f"Re-indexed {len(results)} meeting(s) ({sum(results.values())} vectors).")
    return 0


def cmd_reindex(args, db) -> int:
    from knowledge_repository import reindex_all_meetings
    user_id = None
    if args.username:
        user = db.get_user_by_username(args.username)
        if user is None:
            print(f"No such user: {args.username}")
            return 1
        user_id = user["id"]
    results = reindex_all_meetings(db, user_id=user_id)
    print(f"Re-indexed {len(results)} meeting(s) ({sum(results.values())} vectors).")
    return 0


def cmd_purge_sessions(args, db) -> int:
    print(f"Deleted {db.delete_expired_sessions()} expired session(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config").set_defaults(func=cmd_check_config)
    p = sub.add_parser("create-user")
    p.add_argument("username")
    p.add_argument("email")
    p.add_argument("--password", help="omit to be prompted (recommended - avoids shell history)")
    p.set_defaults(func=cmd_create_user)
    p = sub.add_parser("assign-legacy")
    p.add_argument("username")
    p.set_defaults(func=cmd_assign_legacy)
    p = sub.add_parser("reindex")
    p.add_argument("username", nargs="?")
    p.set_defaults(func=cmd_reindex)
    sub.add_parser("purge-sessions").set_defaults(func=cmd_purge_sessions)
    return parser


def main(argv=None, db: Database = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args, db or Database())


if __name__ == "__main__":
    sys.exit(main())
