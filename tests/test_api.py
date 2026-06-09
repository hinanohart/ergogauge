"""Public API behavior: input forms, ABSTAIN on degenerate, config guards, determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ergogauge import Certificate, ErgogaugeConfig, certify, certify_corpus
from ergogauge.certificate import SCHEMA_VERSION


def test_flatten_rvq_mode_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ErgogaugeConfig(rvq_mode="flatten")


def test_unknown_rvq_mode_raises() -> None:
    with pytest.raises(ValueError):
        ErgogaugeConfig(rvq_mode="bogus")


def test_max_states_over_dense_ceiling_raises() -> None:
    with pytest.raises(ValueError, match="dense"):
        ErgogaugeConfig(max_states=20000)


def test_certify_list_of_ints_returns_certificate(fast_cfg) -> None:
    cert = certify([0, 1, 2, 3] * 50, config=fast_cfg)
    assert isinstance(cert, Certificate)
    assert cert.schema_version == SCHEMA_VERSION
    assert cert.to_dict()["levels"][0]["level"] == 0


def test_certify_multicodebook_two_levels(fast_cfg) -> None:
    rng = np.random.default_rng(0)
    lvl0 = rng.integers(0, 8, size=600).tolist()
    lvl1 = rng.integers(0, 8, size=600).tolist()
    cert = certify([lvl0, lvl1], config=fast_cfg)
    assert len(cert.to_dict()["levels"]) == 2


def test_degenerate_single_state_abstains_or_collapses(fast_cfg) -> None:
    cert = certify([5] * 300, config=fast_cfg)
    assert cert.aggregate["flags"][0] in ("COLLAPSED", "ABSTAIN")


def test_json_roundtrip_is_valid(fast_cfg) -> None:
    cert = certify([0, 1, 2, 3] * 50, config=fast_cfg)
    parsed = json.loads(cert.to_json())
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert "aggregate" in parsed and "gate" in parsed


def test_certify_corpus_path_load(tmp_path, fast_cfg) -> None:
    p = tmp_path / "toks.json"
    p.write_text(json.dumps({"tokens": [0, 1, 2, 3] * 100}), encoding="utf-8")
    cert = certify(str(p), config=fast_cfg)
    assert isinstance(cert, Certificate)


def test_html_report_self_contained(tmp_path, fast_cfg) -> None:
    from tests.conftest import EASY

    from ergogauge.synth import generate_corpus

    streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    cert = certify_corpus(streams, config=fast_cfg)
    out = tmp_path / "r.html"
    cert.to_html(out)
    text = out.read_text(encoding="utf-8")
    assert "<html" in text
    assert "http://" not in text and "https://" not in text  # no external resources
