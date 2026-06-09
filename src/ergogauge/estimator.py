"""Build the empirical token-transition operator P from integer token streams.

Pipeline: adjacent-pair counts -> OTHER-collapse of rare states -> row-normalize with
scaled Laplace smoothing (total pseudo-count `alpha` per row) -> stationary distribution.

The result is a dense fp64 row-stochastic matrix over at most ``max_states+1`` states
(the +1 is the merged OTHER state), which keeps the deterministic dense eigensolver path
viable (see docs/CLAIM.md section 6). All hot paths are vectorized (numpy), so the
bootstrap can rebuild P thousands of times cheaply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

OTHER_STATE = -1


@dataclass
class TransitionModel:
    """Estimated transition operator for one codebook level."""

    P: NDArray[np.float64]  # (S, S) row-stochastic
    pi: NDArray[np.float64]  # (S,) stationary distribution
    states: list[int]  # token id per row; OTHER_STATE (-1) for the merged bucket
    other_mass: float  # stationary mass on the OTHER state (0.0 if none)
    n_obs_pairs: int  # total adjacent transition pairs observed
    n_states_after_collapse: int
    support_coverage: float  # fraction of token occurrences kept as explicit states
    sparsity: float  # fraction of active states with < min_pairs_per_state outgoing obs
    reversible: bool
    reversibility_residual: float
    raw_counts: NDArray[np.float64]  # (S, S) integer counts before normalization
    eigvals: NDArray[np.complex128]  # eigenvalues of P (shared so spectral.py avoids re-eig)


def _degenerate_model() -> TransitionModel:
    return TransitionModel(
        P=np.ones((1, 1), dtype=np.float64),
        pi=np.ones(1, dtype=np.float64),
        states=[OTHER_STATE],
        other_mass=1.0,
        n_obs_pairs=0,
        n_states_after_collapse=1,
        support_coverage=0.0,
        sparsity=1.0,
        reversible=True,
        reversibility_residual=0.0,
        raw_counts=np.zeros((1, 1), dtype=np.float64),
        eigvals=np.array([1.0 + 0.0j], dtype=np.complex128),
    )


def build_transition_model(
    streams: Sequence[Sequence[int]],
    *,
    laplace_alpha: float = 1.0,
    other_collapse_min_count: int = 5,
    max_states: int = 4096,
    min_pairs_per_state: int = 10,
) -> TransitionModel:
    """Estimate :class:`TransitionModel` from one or more token streams (same level)."""
    arrs = [np.asarray(s, dtype=np.int64) for s in streams if len(s) > 0]
    if not arrs:
        return _degenerate_model()
    all_tok = np.concatenate(arrs)
    total_tokens = int(all_tok.size)
    if total_tokens == 0:
        return _degenerate_model()

    toks, tok_counts = np.unique(all_tok, return_counts=True)

    # --- explicit states: frequent enough, capped at max_states, deterministic order ---
    mask = tok_counts >= other_collapse_min_count
    freq_tokens = toks[mask]
    freq_counts = tok_counts[mask]
    order = np.lexsort((freq_tokens, -freq_counts))  # -count primary, token id secondary
    freq_tokens = freq_tokens[order]
    freq_counts = freq_counts[order]
    if freq_tokens.size > max_states:
        freq_tokens = freq_tokens[:max_states]
        freq_counts = freq_counts[:max_states]

    use_other = bool(freq_tokens.size < toks.size)
    n_states = int(freq_tokens.size + (1 if use_other else 0))
    if n_states == 0:
        return _degenerate_model()
    other_idx = freq_tokens.size if use_other else 0

    # lookup table token -> state index
    max_tok = int(all_tok.max())
    lut = np.full(max_tok + 1, other_idx, dtype=np.int64)
    lut[freq_tokens] = np.arange(freq_tokens.size, dtype=np.int64)

    # --- vectorized adjacent-pair counts (no cross-utterance pairs) ---
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for s in arrs:
        if s.size < 2:
            continue
        a = lut[s[:-1]]
        b = lut[s[1:]]
        flat = a * n_states + b
        counts += np.bincount(flat, minlength=n_states * n_states).reshape(n_states, n_states)
    n_obs_pairs = int(counts.sum())

    # --- scaled Laplace smoothing: total pseudo-count per row = alpha (Dirichlet(alpha/K)).
    # A fixed per-cell alpha would add alpha*K mass per row and, for large K, inflate
    # unobserved cross-state transitions enough to wash out genuine bottlenecks (LOCKED). ---
    row_tot = counts.sum(axis=1)
    active = row_tot > 0
    a_cell = laplace_alpha / n_states
    P = (counts + a_cell) / (row_tot[:, None] + laplace_alpha)
    if not np.all(active):
        P[~active] = 1.0 / n_states  # unobserved source row -> uniform (flagged via sparsity)

    states = [int(t) for t in freq_tokens.tolist()] + ([OTHER_STATE] if use_other else [])
    n_active = int(active.sum())
    underobserved = int(np.sum(active & (row_tot < min_pairs_per_state)))
    sparsity = underobserved / n_active if n_active > 0 else 1.0
    support_coverage = float(freq_counts.sum()) / total_tokens if total_tokens > 0 else 0.0

    pi, eigvals = _stationary_and_eigvals(P)
    other_mass = float(pi[other_idx]) if use_other else 0.0
    rev, resid = _reversibility(P, pi)

    return TransitionModel(
        P=P,
        pi=pi,
        states=states,
        other_mass=other_mass,
        n_obs_pairs=n_obs_pairs,
        n_states_after_collapse=n_states,
        support_coverage=support_coverage,
        sparsity=sparsity,
        reversible=rev,
        reversibility_residual=resid,
        raw_counts=counts,
        eigvals=eigvals,
    )


def _stationary_and_eigvals(
    P: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Left Perron eigenvector (stationary distribution) and the spectrum of P, in one eig.

    Eigenvalues of P and P^T coincide, so the same decomposition serves the stationary
    vector (a left eigenvector of P) and the spectral gap / Kemeny eig-sum downstream.
    """
    n = P.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64), np.array([1.0 + 0.0j], dtype=np.complex128)
    w, v = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(np.real(v[:, k]))
    s = pi.sum()
    pi = pi / s if s > 0 else np.full(n, 1.0 / n)
    return pi, np.asarray(w, dtype=np.complex128)


def _reversibility(
    P: NDArray[np.float64], pi: NDArray[np.float64], eps: float = 1e-8
) -> tuple[bool, float]:
    """Detailed-balance residual ||F - F^T||_F / ||F||_F where F_ij = pi_i P_ij."""
    F = pi[:, None] * P
    num = float(np.linalg.norm(F - F.T))
    den = float(np.linalg.norm(F))
    resid = num / den if den > 0 else 0.0
    return (resid < eps, resid)
