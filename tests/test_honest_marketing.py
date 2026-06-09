"""G8: denylist grep is zero on shipped docs, and the grep is provably live (negative fixture)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import scripts.verify_step as vs  # noqa: E402  (verify_step is the single source of denylist)


def test_denylist_zero_on_docs() -> None:
    assert vs.check_denylist() is True


def test_disclaimer_and_heuristic_label_present() -> None:
    assert vs.check_disclaimer() is True
    assert vs.check_heuristic_label() is True


def test_denylist_is_live_negative_fixture() -> None:
    """A planted hype phrase MUST match the denylist (otherwise the grep is dead)."""
    pats = [re.compile(p, re.IGNORECASE) for p in vs.DENYLIST]
    for bad in (
        "the first tool to do this",
        "this is the best",
        "it solves the problem",
        "世界初の計測器",
    ):
        assert any(p.search(bad) for p in pats), f"denylist missed: {bad!r}"


def test_verify_step_script_passes_nonstrict() -> None:
    """The hygiene script itself exits 0 in non-strict mode (placeholders allowed pre-S7)."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_step.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
