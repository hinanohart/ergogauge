"""Fail-closed contract: empty / degenerate / underdetermined input must ABSTAIN.

Regression guard for the v0.1.0a1 fail-OPEN bug where an empty corpus returned
HEALTHY/PASS and an empty single utterance returned COLLAPSED/PASS on zero data.
A pathological flag (COLLAPSED) is only allowed once enough transition pairs were
observed to trust the dominant-mass story (>= min_total_pairs).
"""

from __future__ import annotations

from ergogauge import certify, certify_corpus


def test_empty_corpus_abstains(fast_cfg) -> None:
    cert = certify_corpus([], config=fast_cfg)
    assert cert.aggregate["flags"] == ["ABSTAIN"]
    assert cert.aggregate["status"] == "ABSTAIN"
    assert cert.gate["status"] == "ABSTAIN"


def test_empty_single_utterance_abstains(fast_cfg) -> None:
    cert = certify([], config=fast_cfg)
    assert cert.aggregate["flags"] == ["ABSTAIN"]
    assert cert.aggregate["status"] == "ABSTAIN"


def test_tiny_degenerate_stream_abstains(fast_cfg) -> None:
    # 2 transition pairs is far below min_total_pairs -> cannot assert COLLAPSED.
    cert = certify([5, 5, 5], config=fast_cfg)
    assert cert.aggregate["flags"] == ["ABSTAIN"]
    assert cert.aggregate["status"] == "ABSTAIN"


def test_short_corpus_below_min_pairs_abstains(fast_cfg) -> None:
    # Several short utterances, still well under min_total_pairs=200 total pairs.
    cert = certify_corpus([[1, 2, 3], [3, 2, 1], [1, 1, 2]], config=fast_cfg)
    assert cert.aggregate["flags"] == ["ABSTAIN"]


def test_genuine_collapse_with_enough_data_is_flagged(fast_cfg) -> None:
    # 299 observed pairs (>= min_total_pairs) of a single dominant state -> COLLAPSED, not ABSTAIN.
    cert = certify([5] * 300, config=fast_cfg)
    assert cert.aggregate["flags"] == ["COLLAPSED"]
    assert cert.aggregate["status"] == "PASS"
