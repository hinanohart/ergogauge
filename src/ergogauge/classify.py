"""Map invariants + CIs + identifiability gate to failure-mode flags (fail-closed).

The mapping is a *calibrated heuristic* (see NON-CLAIM.md), not a theorem. Decisions use
CI bounds, not point estimates, and fall through to ABSTAIN at every boundary. The order
matches docs/CLAIM.md section 3.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .entropy import EntropyResult
from .estimator import TransitionModel
from .gate import GateResult, GateThresholds
from .spectral import CheegerResult, KemenyResult

HEURISTIC_DISCLAIMER = (
    "Flags are a calibrated heuristic on a non-reversible empirical operator, not a "
    "theorem. Only the Cheeger phi<->lambda2(L_sym) two-sided bound on the symmetrized "
    "graph is a theorem; the lock/repetition inference is calibrated on a synthetic harness."
)


def _effective_support(pi: np.ndarray, eps: float) -> int:
    return int(np.sum(pi > eps))


def classify_level(
    *,
    model: TransitionModel,
    gate: GateResult,
    metrics: dict[str, float],
    cis: dict[str, tuple[float, float]],
    kem: KemenyResult,
    ch: CheegerResult,
    en: EntropyResult,
    vendi: float,
    thr: GateThresholds,
    is_corpus: bool,
) -> dict[str, object]:
    flags: list[str] = []
    confidence = "low"

    gap_lo, gap_hi = cis["spectral_gap"]
    phi_lo, phi_hi = cis["cheeger_phi"]
    ratio_lo, ratio_hi = cis["entropy_ratio"]

    max_occ = float(np.max(model.pi)) if model.pi.size else 1.0
    support = _effective_support(model.pi, thr.support_eps)
    is_collapsed = support <= thr.collapse_max_support or max_occ > thr.collapse_occupancy

    if is_collapsed:
        # Collapse is detectable from the (well-estimated) dominant stationary mass even
        # when the identifiability gate would otherwise ABSTAIN on the underobserved tail.
        flags = ["COLLAPSED"]
        confidence = "high" if is_corpus else "low"
    elif not gate.passed:
        flags = ["ABSTAIN"]
        confidence = "low"
    else:
        if kem.near_reducible or (np.isfinite(phi_hi) and phi_hi < thr.phi_locked):
            flags = ["LOCKED"]
        elif np.isfinite(gap_hi) and gap_hi < thr.gap_repetitive:
            flags = ["REPETITIVE"]
        elif np.isfinite(ratio_lo) and ratio_lo > thr.over_random_ratio:
            flags = ["OVER_RANDOM"]
        elif (
            np.isfinite(gap_lo)
            and np.isfinite(phi_lo)
            and np.isfinite(ratio_hi)
            and gap_lo >= thr.gap_repetitive
            and phi_lo >= thr.phi_locked
            and ratio_hi < thr.healthy_struct_ratio
        ):
            flags = ["HEALTHY"]
        else:
            flags = ["ABSTAIN"]

        # confidence: corpus + an unambiguous decision -> high; corpus + marginal -> medium
        if flags == ["ABSTAIN"]:
            confidence = "low"
        elif is_corpus:
            confidence = "high" if _decisive(flags[0], cis, thr) else "medium"
        else:
            confidence = "low"

    metric_block = {
        "spectral_gap": {
            "value": metrics["spectral_gap"],
            "ci95": list(cis["spectral_gap"]),
            "is_theorem": False,
        },
        "kemeny": {
            "value": kem.value,
            "ci95": list(cis["kemeny"]),
            "is_theorem": False,
            "method": kem.method,
            "eig_sum_crosscheck": kem.eig_sum_crosscheck,
            "near_reducible": kem.near_reducible,
        },
        "cheeger": {
            "fiedler_value": ch.fiedler_value,
            "conductance_phi": ch.conductance_phi,
            "ci95_phi": list(cis["cheeger_phi"]),
            "ci95_fiedler": list(cis["fiedler"]),
            "lower_bound": ch.lower_bound,
            "upper_bound": ch.upper_bound,
            "is_theorem": True,
            "graph": ch.graph,
        },
        "over_random_discriminator": {
            "transition_entropy": en.transition_entropy,
            "marginal_entropy": en.marginal_entropy,
            "ratio": en.ratio,
            "vs_uniform_null_z": en.vs_uniform_null_z,
            "ci95_ratio": list(cis["entropy_ratio"]),
            "is_theorem": False,
        },
        "vendi_baseline": {
            "value": vendi,
            "note": "order-independent set-diversity (sanity check, not a baseline to beat)",
        },
    }
    return {
        "flags": flags,
        "flag_confidence": confidence,
        "metrics": metric_block,
        "heuristic_disclaimer": HEURISTIC_DISCLAIMER,
    }


def _decisive(flag: str, cis: dict[str, tuple[float, float]], thr: GateThresholds) -> bool:
    """A decision is 'decisive' when the relevant CI clears the threshold with margin."""
    gap_lo, gap_hi = cis["spectral_gap"]
    _phi_lo, phi_hi = cis["cheeger_phi"]
    ratio_lo, ratio_hi = cis["entropy_ratio"]
    m = 0.02
    if flag == "REPETITIVE":
        return np.isfinite(gap_hi) and gap_hi < thr.gap_repetitive - m
    if flag == "LOCKED":
        return np.isfinite(phi_hi) and phi_hi < thr.phi_locked
    if flag == "OVER_RANDOM":
        return np.isfinite(ratio_lo) and ratio_lo > thr.over_random_ratio
    if flag == "HEALTHY":
        return (
            np.isfinite(gap_lo)
            and gap_lo >= thr.gap_repetitive + m
            and np.isfinite(ratio_hi)
            and ratio_hi < thr.healthy_struct_ratio
        )
    return flag == "COLLAPSED"


def aggregate_flags(level_results: list[dict[str, object]]) -> dict[str, object]:
    """Worst-level fail-closed aggregation across codebook levels."""
    severity = {
        "COLLAPSED": 4,
        "LOCKED": 3,
        "REPETITIVE": 3,
        "OVER_RANDOM": 2,
        "ABSTAIN": 1,
        "HEALTHY": 0,
    }
    worst = "HEALTHY"
    worst_sev = 0
    any_abstain = False
    confidences: list[str] = []
    for lvl in level_results:
        confidences.append(str(lvl["flag_confidence"]))
        flags_here = cast("list[str]", lvl["flags"])
        for f in flags_here:
            if f == "ABSTAIN":
                any_abstain = True
            sev = severity.get(str(f), 1)
            if sev > worst_sev:
                worst_sev = sev
                worst = str(f)
    if worst == "HEALTHY" and any_abstain:
        worst = "ABSTAIN"
    status = "ABSTAIN" if worst == "ABSTAIN" else "PASS"
    conf_rank = {"low": 0, "medium": 1, "high": 2}
    agg_conf = min(confidences, key=lambda c: conf_rank.get(c, 0)) if confidences else "low"
    return {"flags": [worst], "status": status, "confidence": agg_conf}
