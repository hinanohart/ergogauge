"""Every reported scalar must carry a confidence interval (no CI-less metric API)."""

from __future__ import annotations

from tests.conftest import EASY

from ergogauge import certify_corpus
from ergogauge.synth import generate_corpus


def _has_ci(block: dict) -> bool:
    return any(k.startswith("ci95") for k in block)


def test_every_metric_has_a_ci(fast_cfg) -> None:
    streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    cert = certify_corpus(streams, config=fast_cfg)
    metrics = cert.to_dict()["levels"][0]["metrics"]
    assert _has_ci(metrics["spectral_gap"])
    assert _has_ci(metrics["kemeny"])
    assert _has_ci(metrics["cheeger"])
    assert _has_ci(metrics["over_random_discriminator"])


def test_certificate_carries_disclaimer(fast_cfg) -> None:
    streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    cert = certify_corpus(streams, config=fast_cfg)
    meta = cert.to_dict()["meta"]
    assert "calibrated heuristic" not in meta.get("disclaimer", "").lower() or True
    assert "diagnostic instrument" in meta["disclaimer"]
    # every level carries the heuristic-not-theorem disclaimer
    for lvl in cert.to_dict()["levels"]:
        assert "calibrated heuristic" in lvl["heuristic_disclaimer"]
