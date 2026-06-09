"""Enforce docs/CLAIM.md section-5 acceptance criteria that previously had NO test:
severity monotonicity (Spearman >= 0.8) and the hard-extreme -> ABSTAIN regime.

Small seed/sweep counts keep CI fast; the extremes are separated widely enough that
the monotone trend is non-flaky. The full 100-seed protocol lives in scripts/gen_metrics.py.
"""

from __future__ import annotations

import numpy as np

from ergogauge import certify
from ergogauge.estimator import build_transition_model
from ergogauge.spectral import cheeger, spectral_gap
from ergogauge.synth import generate_corpus

EASY = {"V": 64, "L": 2000, "n_utt": 8}


def _spearman(x: list[float], y: list[float]) -> float:
    ax, ay = np.asarray(x, float), np.asarray(y, float)

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a), dtype=float)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    rx, ry = rank(ax) - rank(ax).mean(), rank(ay) - rank(ay).mean()
    d = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def test_repetitive_severity_monotonicity() -> None:
    """Calibrated instrument: stronger loop (rho up) -> slower mixing -> smaller spectral gap.
    We assert monotonicity of the invariant itself, not a saturating binary detection rate."""
    rhos = [0.5, 0.65, 0.8, 0.9, 0.95, 0.99]
    gaps = [
        float(
            np.mean(
                [
                    spectral_gap(
                        build_transition_model(
                            generate_corpus("REPETITIVE", seed=s, rho=rho, **EASY)[0]
                        ).P
                    )
                    for s in range(1000, 1003)
                ]
            )
        )
        for rho in rhos
    ]
    sp = _spearman(rhos, [-g for g in gaps])
    assert sp >= 0.8, f"rho vs -gap not monotone (sp={sp}): {list(zip(rhos, gaps, strict=True))}"


def test_locked_severity_monotonicity() -> None:
    """Calibrated instrument: tighter lock (delta down) -> smaller Cheeger phi, in the
    observable regime delta>=1e-3 at L=2000,n_utt=8. Tighter locks fall below sample
    resolution (the rare cross-transition is never sampled) and ABSTAIN -- a stated limit."""
    deltas = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    sev = [-np.log10(d) for d in deltas]
    phis = []
    for delta in deltas:
        vals = []
        for s in range(1000, 1003):
            m = build_transition_model(generate_corpus("LOCKED", seed=s, delta=delta, **EASY)[0])
            vals.append(cheeger(m.P, m.pi).conductance_phi)
        phis.append(float(np.mean(vals)))
    sp = _spearman(sev, [-p for p in phis])
    assert sp >= 0.8, (
        f"severity vs -phi not monotone (sp={sp}): {list(zip(sev, phis, strict=True))}"
    )


def test_hard_extreme_regime_abstains(fast_cfg) -> None:
    """Underdetermined noise (short stream, large alphabet) must fail closed, never a
    confident pathology flag."""
    n = ab = confident_wrong = 0
    for seed in range(1000, 1012):
        toks = np.random.default_rng(seed).integers(0, 256, size=80).tolist()
        cert = certify(toks, config=fast_cfg)
        n += 1
        ab += int(cert.gate["status"] == "ABSTAIN")
        flag, conf = cert.aggregate["flags"][0], cert.aggregate["confidence"]
        confident_wrong += int(flag in ("REPETITIVE", "LOCKED", "COLLAPSED") and conf == "high")
    assert ab / n >= 0.9, f"hard-extreme abstain rate too low: {ab}/{n}"
    assert confident_wrong == 0, "emitted a confident pathology flag on underdetermined noise"
