"""A4 discriminator: structural vs. unstructured high-mixing chains.

Separates OVER_RANDOM (transition entropy ~ marginal entropy ~ log V: no measurable
conditional structure, statistically indistinguishable from iid-uniform) from HEALTHY
(high mixing yet measurably structured: transition entropy < marginal entropy).

Without this axis, a max-entropy iid stream and a well-mixing structured chain are
indistinguishable from the spectral gap alone (both have a large gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EntropyResult:
    transition_entropy: float  # H(X_{t+1} | X_t), nats
    marginal_entropy: float  # H(pi), nats
    ratio: float  # transition_entropy / marginal_entropy in [0, ~1]
    vs_uniform_null_z: float  # normalized gap of ratio below 1 (structure strength)


def _row_entropy(p: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return np.asarray(-terms.sum(axis=1), dtype=np.float64)


def entropy_discriminator(P: NDArray[np.float64], pi: NDArray[np.float64]) -> EntropyResult:
    n = P.shape[0]
    if n <= 1:
        return EntropyResult(0.0, 0.0, 1.0, 0.0)

    h_rows = _row_entropy(P)  # (n,) per-state conditional entropy
    transition_entropy = float(np.dot(pi, h_rows))

    with np.errstate(divide="ignore", invalid="ignore"):
        pterms = np.where(pi > 0, pi * np.log(pi), 0.0)
    marginal_entropy = float(-pterms.sum())

    ratio = transition_entropy / marginal_entropy if marginal_entropy > 1e-12 else 1.0
    ratio = float(np.clip(ratio, 0.0, 1.0))

    # structure strength: how far the ratio sits below the iid-uniform null (ratio==1),
    # scaled by log(n) so it is comparable across alphabet sizes.
    scale = np.log(n) if n > 1 else 1.0
    vs_uniform_null_z = float((marginal_entropy - transition_entropy) / scale)
    return EntropyResult(
        transition_entropy=transition_entropy,
        marginal_entropy=marginal_entropy,
        ratio=ratio,
        vs_uniform_null_z=vs_uniform_null_z,
    )
