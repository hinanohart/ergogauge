"""G10: gate.GateThresholds must equal docs/CLAIM.md, and TRAIN/TEST seeds disjoint."""

from __future__ import annotations

import re
from pathlib import Path

from tests.conftest import TEST_SEEDS, TRAIN_SEEDS

from ergogauge.gate import GateThresholds

CLAIM = Path(__file__).resolve().parent.parent / "docs" / "CLAIM.md"

# CLAIM.md key -> GateThresholds attribute
KEYS = {
    "min_pairs_per_state": "min_pairs_per_state",
    "min_total_pairs": "min_total_pairs",
    "max_sparsity_frac": "max_sparsity_frac",
    "other_collapse_min_count": "other_collapse_min_count",
    "max_states": "max_states",
    "gap_repetitive": "gap_repetitive",
    "phi_locked": "phi_locked",
    "collapse_max_support": "collapse_max_support",
    "collapse_occupancy": "collapse_occupancy",
    "support_eps": "support_eps",
    "over_random_ratio": "over_random_ratio",
    "healthy_struct_ratio": "healthy_struct_ratio",
    "bootstrap_reps": "bootstrap_reps",
    "block_mean_len": "block_mean_len",
    "ci_level": "ci_level",
    "alpha_nominal": "alpha_nominal",
    "corpus_min_utterances": "corpus_min_utterances",
}


def _parse_claim() -> dict[str, float]:
    text = CLAIM.read_text(encoding="utf-8")
    found: dict[str, float] = {}
    # rows like:  | `gap_repetitive` | 0.15 | ... |
    row = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*([0-9eE.+-]+)\s*\|", re.MULTILINE)
    for key, val in row.findall(text):
        if key in KEYS:
            found[key] = float(val)
    return found


def test_gate_thresholds_match_claim_md() -> None:
    claim = _parse_claim()
    thr = GateThresholds()
    missing = set(KEYS) - set(claim)
    assert not missing, f"keys missing from CLAIM.md: {missing}"
    for ckey, attr in KEYS.items():
        expected = claim[ckey]
        actual = float(getattr(thr, attr))
        assert actual == expected, f"{attr}: code={actual} != CLAIM.md={expected}"


def test_train_test_seeds_disjoint() -> None:
    assert set(TRAIN_SEEDS).isdisjoint(set(TEST_SEEDS))
    assert len(TRAIN_SEEDS) > 0 and len(TEST_SEEDS) > 0
