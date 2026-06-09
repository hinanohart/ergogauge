"""Primary correctness golden (S6): injected == detected on the held-out TEST seed grid.

Thresholds were calibrated on TRAIN seeds (docs/CLAIM.md); these assertions run only on
disjoint TEST seeds, so passing them is evidence of generalization, not of tuning.
"""

from __future__ import annotations

import pytest
from tests.conftest import EASY

from ergogauge import certify, certify_corpus
from ergogauge.synth import LABELS, generate_corpus

TEST_GRID_SEEDS = [1000, 1001, 1002]


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("seed", TEST_GRID_SEEDS)
def test_injected_equals_detected_easy_extreme(label: str, seed: int, fast_cfg) -> None:
    streams, gt = generate_corpus(label, seed=seed, **EASY)
    cert = certify_corpus(streams, config=fast_cfg)
    assert cert.aggregate["flags"][0] == gt, cert.to_json()


def test_easy_extreme_recall_at_least_090(fast_cfg) -> None:
    """Per-class recall >= 0.90 across the TEST grid (pre-registered acceptance)."""
    per_class: dict[str, list[bool]] = {lbl: [] for lbl in LABELS}
    for label in LABELS:
        for seed in TEST_GRID_SEEDS:
            streams, gt = generate_corpus(label, seed=seed, **EASY)
            cert = certify_corpus(streams, config=fast_cfg)
            per_class[gt].append(cert.aggregate["flags"][0] == gt)
    for lbl, hits in per_class.items():
        recall = sum(hits) / len(hits)
        assert recall >= 0.90, f"{lbl} recall {recall:.2f} < 0.90"


def test_single_short_utterance_abstains(fast_cfg) -> None:
    """G1: a single short utterance must fail closed to ABSTAIN (not a fabricated flag)."""
    cert = certify([1, 2, 3, 1, 2, 3, 1], config=fast_cfg)
    assert cert.aggregate["flags"][0] in ("ABSTAIN", "REPETITIVE", "COLLAPSED")
    # the identifiability gate itself should not be PASS on so few pairs
    assert cert.gate["status"] == "ABSTAIN"
