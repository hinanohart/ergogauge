"""Sensitivity gates G3, G4, G5, G7, G9."""

from __future__ import annotations

import numpy as np
from tests.conftest import EASY

from ergogauge import certify_corpus
from ergogauge.estimator import build_transition_model
from ergogauge.spectral import cheeger, kemeny, spectral_gap
from ergogauge.synth import generate_corpus


def test_g3_healthy_not_false_flagged(fast_cfg) -> None:
    """G3: HEALTHY corpora must not be flagged as a pathology (nominal false-flag rate)."""
    false_flags = 0
    n = 0
    for seed in (1000, 1001, 1002, 1003):
        streams, _ = generate_corpus("HEALTHY", seed=seed, **EASY)
        cert = certify_corpus(streams, config=fast_cfg)
        flag = cert.aggregate["flags"][0]
        n += 1
        if flag not in ("HEALTHY", "ABSTAIN"):
            false_flags += 1
    assert false_flags == 0, f"{false_flags}/{n} HEALTHY corpora mis-flagged"


def test_g4_determinism_rerun_byte_identical(fast_cfg) -> None:
    """G4: same input + same config -> byte-identical certificate JSON on rerun."""
    streams, _ = generate_corpus("LOCKED", seed=1000, **EASY)
    a = certify_corpus(streams, config=fast_cfg).to_json()
    b = certify_corpus(streams, config=fast_cfg).to_json()
    assert a == b


def test_g5_theorem_line_only_on_cheeger() -> None:
    """G5: is_theorem is True only for the Cheeger symmetrized-graph bound."""
    streams, _ = generate_corpus("LOCKED", seed=1000, **EASY)
    cert = certify_corpus(streams, seed=0)
    lvl = cert.to_dict()["levels"][0]["metrics"]
    assert lvl["cheeger"]["is_theorem"] is True
    assert lvl["spectral_gap"]["is_theorem"] is False
    assert lvl["kemeny"]["is_theorem"] is False
    assert lvl["over_random_discriminator"]["is_theorem"] is False


def test_g5_reversibility_residual_reported() -> None:
    streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    cert = certify_corpus(streams, seed=0)
    assert "reversibility_residual" in cert.to_dict()["levels"][0]


def test_g7_other_collapse_caps_states() -> None:
    """G7: a large alphabet collapses rare states; state count stays within cap+1."""
    streams, _ = generate_corpus("OVER_RANDOM", V=1024, L=4000, n_utt=8, seed=1000)
    m = build_transition_model(streams, max_states=256, other_collapse_min_count=5)
    assert m.n_states_after_collapse <= 256 + 1
    assert 0.0 <= m.other_mass <= 1.0


def test_g9_over_random_vs_healthy_separated(fast_cfg) -> None:
    """G9: entropy-ratio CIs of OVER_RANDOM and HEALTHY must not overlap (CI gap > 0)."""
    from ergogauge.calibrate import bootstrap_cis

    h_streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    o_streams, _ = generate_corpus("OVER_RANDOM", seed=1000, **EASY)
    h_ci = bootstrap_cis(h_streams, fast_cfg)["entropy_ratio"]
    o_ci = bootstrap_cis(o_streams, fast_cfg)["entropy_ratio"]
    # HEALTHY ratio is measurably below OVER_RANDOM: HEALTHY CI-high < OVER_RANDOM CI-low
    assert h_ci[1] < o_ci[0], f"no separation: HEALTHY {h_ci} vs OVER_RANDOM {o_ci}"


def test_cheeger_distinguishes_lock_from_loop() -> None:
    """The second axis is load-bearing: a near-reducible LOCKED chain and a slow REPETITIVE
    loop BOTH depress the spectral gap, so the gap alone cannot separate them. The Cheeger
    conductance phi does -- the loop is well-connected (large phi) while the lock has a
    genuine bottleneck cut (small phi). This is the real mechanism (not a 'gap-high /
    phi-low' wedge, which Cheeger's inequality forbids for the reversible symmetrized graph)."""
    lock, _ = generate_corpus("LOCKED", seed=1000, delta=1e-3, **EASY)
    loop, _ = generate_corpus("REPETITIVE", seed=1000, rho=0.97, **EASY)
    ml, mr = build_transition_model(lock), build_transition_model(loop)
    gap_lock = spectral_gap(ml.P, ml.eigvals)
    gap_loop = spectral_gap(mr.P, mr.eigvals)
    phi_lock = cheeger(ml.P, ml.pi).conductance_phi
    phi_loop = cheeger(mr.P, mr.pi).conductance_phi
    # both pathologies have a small spectral gap -> gap cannot tell them apart
    assert gap_lock < 0.15 and gap_loop < 0.15, f"expected both small-gap: {gap_lock}, {gap_loop}"
    # phi separates them: bottleneck (lock) vs well-connected cycle (loop)
    assert phi_lock < 0.05, f"lock conductance not small: {phi_lock}"
    assert phi_loop > 0.20, f"loop conductance not large: {phi_loop}"
    assert phi_loop > 10 * phi_lock, (
        f"phi fails to separate loop from lock: {phi_loop} vs {phi_lock}"
    )


def test_kemeny_crosscheck_finite_for_well_mixing() -> None:
    streams, _ = generate_corpus("HEALTHY", seed=1000, **EASY)
    m = build_transition_model(streams)
    k = kemeny(m.P, m.pi, m.eigvals)
    assert np.isfinite(k.value)
    assert not k.near_reducible
