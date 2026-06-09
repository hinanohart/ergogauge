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

# Hype / overclaim denylist. Word-boundary, case-insensitive. Includes 'first'
# ('first-order'/'first order' are excluded as legitimate technical terms via lookahead).
DENYLIST = [
    r"\bpermanent\b",
    r"\bcomplete(?:ly)?\b",
    r"\bfirst\b(?![\- ]order)",
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

# A denylist word is *allowed* only when an explicit negation / NON-CLAIM / denylist-registry
# marker sits (a) within +/- WINDOW chars (so it survives line wrapping) AND (b) in the SAME
# sentence -- i.e. no sentence terminator (. ! ?) separates the negation from the hit. The
# same-sentence rule is what stops a wide window from silently excusing hype that merely sits
# near an unrelated negation in a *different* sentence (e.g. "It is NOT a benchmark. This is
# the best tool."). The negative-fixture test guarantees the patterns still fire on planted
# hype, so this is not a dead grep.
WINDOW = 90
NEGATION = re.compile(
    r"no claim|\bnot\b|n't|\bdenylist\b|without|neither|do(?:es)? not|cannot|"
    r"\brepair\b|aware of|makes no|fixes/solves|\bequivalents\b",
    re.IGNORECASE,
)
SENT_END = re.compile(r"[.!?]")


def _same_sentence(text: str, a0: int, a1: int, b0: int, b1: int) -> bool:
    """True if no sentence terminator lies strictly between spans [a0,a1) and [b0,b1)."""
    if a1 <= b0:
        gap = text[a1:b0]
    elif b1 <= a0:
        gap = text[b1:a0]
    else:
        return True  # overlapping spans -> same sentence
    return SENT_END.search(gap) is None


# Files to scan for marketing claims.
SCAN_GLOBS = ["README.md", "CHANGELOG.md", "docs/*.md"]

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


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def scan_text(text: str) -> list[tuple[int, str]]:
    """Return [(line, snippet)] for every denylist word used as a positive claim.

    A hit is *allowed* (skipped) only when a negation marker sits within +/- WINDOW chars
    AND in the same sentence (no sentence terminator between them). The window survives line
    wrapping; the same-sentence rule stops the window from excusing hype that merely sits near
    an unrelated negation in a different sentence. Pure function so tests can exercise the
    logic directly on planted strings, not just the regex patterns.
    """
    hits: list[tuple[int, str]] = []
    pats = [re.compile(p, re.IGNORECASE) for p in DENYLIST]
    for pat in pats:
        for mobj in pat.finditer(text):
            lo = max(0, mobj.start() - WINDOW)
            hi = min(len(text), mobj.end() + WINDOW)
            allowed = any(
                _same_sentence(text, mobj.start(), mobj.end(), lo + neg.start(), lo + neg.end())
                for neg in NEGATION.finditer(text[lo:hi])
            )
            if allowed:
                continue  # negation in the same sentence -> non-claim / registry context
            ln = _line_of(text, mobj.start())
            snippet = text[mobj.start() : mobj.start() + 80].splitlines()[0]
            hits.append((ln, snippet))
    return hits


def check_denylist() -> bool:
    ok = True
    for f in _scan_files():
        for ln, snippet in scan_text(f.read_text(encoding="utf-8")):
            print(f"DENYLIST HIT: {f.relative_to(ROOT)}:{ln}: ...{snippet}")
            ok = False
    return ok


def _normalize(s: str) -> str:
    """Collapse markdown blockquote markers and all whitespace so a wrapped disclaimer
    still matches as a contiguous substring."""
    return " ".join(s.replace(">", " ").split())


def check_disclaimer() -> bool:
    readme = _normalize((ROOT / "README.md").read_text(encoding="utf-8"))
    if _normalize(DISCLAIMER_NEEDLE) not in readme:
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
