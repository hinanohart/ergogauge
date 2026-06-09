"""User-facing configuration for ergogauge."""

from __future__ import annotations

from dataclasses import dataclass, field

from .gate import GateThresholds


@dataclass
class ErgogaugeConfig:
    """Configuration for :func:`ergogauge.certify` / :func:`ergogauge.certify_corpus`.

    Defaults are the pre-registered values (see docs/CLAIM.md). ``rvq_mode='flatten'``
    is intentionally not implemented in v0.1 (raises ``NotImplementedError``).
    """

    rvq_mode: str = "per_level"  # 'per_level' only; 'flatten' -> NotImplementedError
    smoothing: str = "laplace"
    laplace_alpha: float = 1.0
    other_collapse_min_count: int = 5
    max_states: int = 4096
    bootstrap_reps: int = 1000
    block_mean_len: int = 16
    null: str = "uniform"
    reference_band: tuple[float, float] | None = None
    gate: GateThresholds = field(default_factory=GateThresholds)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rvq_mode not in ("per_level",):
            if self.rvq_mode == "flatten":
                raise NotImplementedError(
                    "rvq_mode='flatten' (joint multi-codebook chain) is ROADMAP, not v0.1; "
                    "use rvq_mode='per_level'."
                )
            raise ValueError(f"unknown rvq_mode={self.rvq_mode!r}")
        if self.smoothing not in ("laplace",):
            raise ValueError(f"unknown smoothing={self.smoothing!r}")
