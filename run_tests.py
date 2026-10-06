"""
run_tests.py
-------------
Dependency-free test runner: discovers every `test_*.py` module and runs
each top-level `test_*` function, so the whole offline suite can run on a
machine without pytest installed:

    python run_tests.py                 # everything
    python run_tests.py test_auth       # one module (substring match)
    python run_tests.py -v              # print each test name

The tests are also plain pytest-compatible (`python -m pytest`).
"""
import glob
import importlib
import inspect
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Tests must never touch a developer's real database, logs or transcripts.
import tempfile  # noqa: E402
_SANDBOX = tempfile.mkdtemp(prefix="cip_tests_")
os.environ.setdefault("DATABASE_PATH", os.path.join(_SANDBOX, "test.db"))
os.environ.setdefault("LOG_DIR", os.path.join(_SANDBOX, "logs"))
os.environ.setdefault("TRANSCRIPTS_DIR", os.path.join(_SANDBOX, "transcripts"))
os.environ.setdefault("LOG_TO_FILE", "false")


def main(argv):
    verbose = "-v" in argv
    filters = [a for a in argv if not a.startswith("-")]
    modules = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(ROOT, "test_*.py")))
    if filters:
        modules = [m for m in modules if any(f in m for f in filters)]

    passed = failed = skipped = 0
    failures = []
    started = time.perf_counter()
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            failed += 1
            failures.append((name, "<import>", traceback.format_exc()))
            print(f"[IMPORT ERROR] {name}")
            continue
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if not fname.startswith("test_") or fn.__module__ != name:
                continue
            params = inspect.signature(fn).parameters
            if any(p.default is inspect.Parameter.empty for p in params.values()):
                skipped += 1  # needs a pytest fixture - covered when run under pytest
                if verbose:
                    print(f"[SKIP] {name}::{fname} (needs pytest fixture)")
                continue
            try:
                fn()
                passed += 1
                if verbose:
                    print(f"[PASS] {name}::{fname}")
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append((name, fname, traceback.format_exc()))
                print(f"[FAIL] {name}::{fname}")
    elapsed = time.perf_counter() - started
    for name, fname, tb in failures:
        print("\n" + "=" * 70 + f"\n{name}::{fname}\n" + tb)
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
