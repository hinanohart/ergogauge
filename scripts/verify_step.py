#!/usr/bin/env python3
"""Honesty / hygiene checks runnable locally and in CI.

These are real assertions (not always-True): each check returns a boolean and the
script exits non-zero if any fails. A paired *negative* fixture in
tests/test_honest_marketing.py proves the denylist grep is live (rc=1 on a planted hit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hype / overclaim denylist. Word-boundary, case-insensitive. Includes 'first'.
DENYLIST = [
    r"\bpermanent\b",
    r"\bcomplete(?:ly)?\b",
    r"\bfirst\b",
    r"\bbest\b",
    r"\bguaranteed?\b",
    r"\bautomatic(?:ally)?\b",
    r"\bSOTA\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bsolves?\b",
    r"\boutperform(?:s|ed|ing)?\b",
    # Japanese equivalents
    "世界初",
    "業界初",
    "完全保証",
    "最高性能",
]

# Files to scan for marketing claims.
SCAN_GLOBS = ["README.md", "docs/*.md"]

# The mandatory verbatim disclaimer (a stable substring of it).
DISCLAIMER_NEEDLE = (
    "CPU-only, reference-free, decode-free, LLM-free diagnostic instrument that treats "
    "a generated codec-LM token stream as a finite-state Markov chain"
)

# Heuristic-not-theorem label must be present.
HEURISTIC_NEEDLE = "calibrated heuristic"


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for g in SCAN_GLOBS:
        files.extend(sorted(ROOT.glob(g)))
    return files


def check_denylist() -> bool:
    ok = True
    pats = [re.compile(p, re.IGNORECASE) for p in DENYLIST]
    for f in _scan_files():
        text = f.read_text(encoding="utf-8")
        for ln, line in enumerate(text.splitlines(), 1):
            # Allow denylist words only when explicitly negated in a NON-CLAIM context:
            # we keep the rule strict and instead phrase docs to avoid the words entirely.
            for pat in pats:
                if pat.search(line):
                    print(f"DENYLIST HIT: {f.relative_to(ROOT)}:{ln}: {line.strip()[:100]}")
                    ok = False
    return ok


def check_disclaimer() -> bool:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if DISCLAIMER_NEEDLE not in readme:
        print("MISSING verbatim disclaimer in README.md")
        return False
    return True


def check_heuristic_label() -> bool:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    nonclaim = (ROOT / "docs" / "NON-CLAIM.md").read_text(encoding="utf-8")
    if HEURISTIC_NEEDLE not in (readme + nonclaim):
        print("MISSING 'calibrated heuristic' label")
        return False
    return True


def check_no_placeholder_numbers() -> bool:
    """After S6, README must not contain unfilled MEASURED placeholders in shipped numbers.

    This check is advisory before S6 (placeholders expected) and enforced at S7 by passing
    --strict.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    n = readme.count("<!--MEASURED@S6")
    if "--strict" in sys.argv and n > 0:
        print(f"PLACEHOLDER still present in README ({n}) under --strict")
        return False
    return True


def main() -> int:
    checks = {
        "denylist_zero": check_denylist(),
        "disclaimer_present": check_disclaimer(),
        "heuristic_label_present": check_heuristic_label(),
        "no_placeholder_strict": check_no_placeholder_numbers(),
    }
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
