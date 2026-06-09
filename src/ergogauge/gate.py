"""Pre-registered thresholds and the identifiability gate (fail-closed to ABSTAIN).

The values in :class:`GateThresholds` MUST equal ``docs/CLAIM.md`` (enforced by
``tests/test_preregistration.py``). They are *calibrated heuristics*, not theorems.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateThresholds:
    """Frozen, pre-registered decision constants (mirror of docs/CLAIM.md)."""

    # --- identifiability (fail-closed) ---
    min_pairs_per_state: int = 10
    min_total_pairs: int = 200
    max_sparsity_frac: float = 0.5
    other_collapse_min_count: int = 5
    max_states: int = 4096

    # --- flag thresholds (CI-gated calibrated heuristics) ---
    gap_repetitive: float = 0.15
    phi_locked: float = 0.05
    collapse_max_support: int = 2
    collapse_occupancy: float = 0.90
    support_eps: float = 1e-4
    over_random_ratio: float = 0.93
    healthy_struct_ratio: float = 0.88

    # --- confidence / calibration ---
    bootstrap_reps: int = 1000
    block_mean_len: int = 16
    ci_level: float = 0.95
    alpha_nominal: float = 0.05
    corpus_min_utterances: int = 8


@dataclass
class GateResult:
    """Outcome of the identifiability gate."""

    status: str  # "PASS" | "ABSTAIN"
    reasons: list[str] = field(default_factory=list)
    effective_n_pairs: int = 0
    support_coverage: float = 0.0
    sparsity: float = 0.0
    min_required_pairs: int = 0

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "effective_n_pairs": int(self.effective_n_pairs),
            "support_coverage": float(self.support_coverage),
            "sparsity": float(self.sparsity),
            "min_required_pairs": int(self.min_required_pairs),
        }


def run_identifiability_gate(
    *,
    n_obs_pairs: int,
    n_states_after_collapse: int,
    sparsity: float,
    support_coverage: float,
    thr: GateThresholds,
) -> GateResult:
    """Decide PASS / ABSTAIN purely from observed-sample sufficiency.

    No metric value is consulted here: this gate only asks whether the empirical
    operator is estimated from enough data to be trustworthy.
    """
    reasons: list[str] = []
    min_required = thr.min_pairs_per_state * max(1, n_states_after_collapse)

    if n_obs_pairs < thr.min_total_pairs:
        reasons.append(
            f"too few transition pairs: {n_obs_pairs} < min_total_pairs={thr.min_total_pairs}"
        )
    if n_obs_pairs < min_required:
        reasons.append(
            f"insufficient pairs per state: {n_obs_pairs} < "
            f"{thr.min_pairs_per_state}*{n_states_after_collapse}={min_required}"
        )
    if sparsity > thr.max_sparsity_frac:
        reasons.append(
            f"too sparse: {sparsity:.3f} of active states underobserved "
            f"> max_sparsity_frac={thr.max_sparsity_frac}"
        )

    status = "PASS" if not reasons else "ABSTAIN"
    return GateResult(
        status=status,
        reasons=reasons,
        effective_n_pairs=n_obs_pairs,
        support_coverage=support_coverage,
        sparsity=sparsity,
        min_required_pairs=min_required,
    )
