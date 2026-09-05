#!/usr/bin/env python3
"""Fail if the documentation and the code disagree about the public surface.

This exists because the documentation drifted badly once already: a stale line
count, two wrong test counts, a wrong CI matrix, an endpoint that appeared
nowhere, and a `.env.example` the README told people to copy that had never
been committed. Prose cannot be checked automatically, but the two lists that
drifted hardest can be:

  * environment variables the code reads
  * HTTP routes the app serves

Run from the repository root:  python scripts/check_docs_match_code.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Internal, deliberately undocumented in the user-facing config table: it is set
# by the test suite so importing the app needs no configuration. SECURITY.md
# does name it, as something never to set in a deployment.
INTERNAL_ENV = {"PDF_OCR_TESTING"}

DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "reference" / "configuration.md",
]
ENDPOINT_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "reference" / "endpoints.md",
]


def env_vars_in_code() -> set[str]:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    entrypoint = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")
    found = set(re.findall(r"env_int\(\s*'([A-Z_]+)'", app))
    found |= set(re.findall(r"environ\.get\(\s*'([A-Z_]+)'", app))
    found |= set(re.findall(r"\$\{([A-Z_]+)", entrypoint))
    return found - INTERNAL_ENV


def routes_in_code() -> set[str]:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    return set(re.findall(r"@app\.route\(\s*'([^']+)'", app))


def check(kind: str, expected: set[str], docs: list[Path]) -> list[str]:
    problems = []
    for doc in docs:
        if not doc.exists():
            problems.append(f"{doc.relative_to(ROOT)}: missing")
            continue
        text = doc.read_text(encoding="utf-8")
        for item in sorted(expected):
            if f"`{item}`" not in text:
                problems.append(
                    f"{doc.relative_to(ROOT)}: {kind} `{item}` is in the code but not documented"
                )
    return problems


def main() -> int:
    problems: list[str] = []

    env = env_vars_in_code()
    problems += check("env var", env, DOCS)

    routes = routes_in_code()
    problems += check("route", routes, ENDPOINT_DOCS)

    # The other direction: a documented env var that no longer exists.
    documented = set()
    for doc in DOCS:
        if doc.exists():
            documented |= set(re.findall(r"^\|\s*`([A-Z_]+)`", doc.read_text(encoding="utf-8"), re.M))
    for stale in sorted(documented - env - INTERNAL_ENV):
        problems.append(f"env var `{stale}` is documented but the code no longer reads it")

    if problems:
        print("Documentation does not match the code:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s).", file=sys.stderr)
        return 1

    print(f"Docs match the code: {len(env)} environment variables, {len(routes)} routes.")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
