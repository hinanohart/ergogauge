"""COMMIT-1 synthetic generators — the primary correctness ground truth.

Each generator returns ``(stream, gt_label)`` where ``gt_label`` is the injected
pathology. Sampling uses a numpy Generator(PCG64) seeded for byte-reproducibility.

Generators:
  - HEALTHY:     irreducible, aperiodic, well-mixing, but measurably structured
  - REPETITIVE:  a deterministic cycle of period ``p`` followed with probability ``rho``
  - LOCKED:      two near-reducible blocks with cross-probability ``delta`` (mixes within
                 a block, so the spectral gap can miss it — Cheeger catches it)
  - OVER_RANDOM: iid near-uniform (high gap, no conditional structure)
  - COLLAPSED:   mass concentrated on one or two states
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LABELS = ("HEALTHY", "REPETITIVE", "LOCKED", "OVER_RANDOM", "COLLAPSED")


def _sample_chain(
    P: NDArray[np.float64], L: int, rng: np.random.Generator, start: int = 0
) -> list[int]:
    n = P.shape[0]
    out = np.empty(L, dtype=np.int64)
    s = start % n
    cdf = np.cumsum(P, axis=1)
    u = rng.random(L)
    for t in range(L):
        out[t] = s
        s = int(np.searchsorted(cdf[s], u[t], side="right"))
        if s >= n:
            s = n - 1
    return [int(x) for x in out.tolist()]


def gen_healthy(V: int, L: int, rng: np.random.Generator) -> tuple[list[int], str]:
    # Sparse-support structured chain: each state transitions to a random subset of size
    # ~V/8 with gamma weights. Row entropy ~ log(V/8) << log V (measurably structured),
    # yet the out-degree keeps the directed graph well-connected -> large spectral gap.
    deg = max(2, V // 8)
    P = np.zeros((V, V), dtype=np.float64)
    for i in range(V):
        targets = rng.choice(V, size=deg, replace=False)
        P[i, targets] = rng.gamma(shape=0.5, size=deg) + 1e-3
    P += 1e-4  # tiny floor for irreducibility/aperiodicity
    P /= P.sum(axis=1, keepdims=True)
    return _sample_chain(P, L, rng), "HEALTHY"


def gen_repetitive(
    V: int, L: int, rng: np.random.Generator, rho: float = 0.9, period: int = 4
) -> tuple[list[int], str]:
    period = min(period, V)
    cycle = list(range(period))
    M = rng.gamma(shape=0.4, size=(V, V)) + 1e-3
    Pr = M / M.sum(axis=1, keepdims=True)
    out: list[int] = []
    s = 0
    for _ in range(L):
        out.append(cycle[s % period] if s < period else int(rng.integers(V)))
        if rng.random() < rho:
            s = (s + 1) % period
        else:
            s = int(np.searchsorted(np.cumsum(Pr[out[-1]]), rng.random(), side="right"))
            s = min(s, V - 1)
    return out, "REPETITIVE"


def gen_locked(
    V: int, L: int, rng: np.random.Generator, delta: float = 1e-3
) -> tuple[list[int], str]:
    half = max(1, V // 2)
    P = np.zeros((V, V), dtype=np.float64)
    A = np.arange(half)
    B = np.arange(half, V)
    for blk in (A, B):
        if len(blk) == 0:
            continue
        M = rng.gamma(shape=0.5, size=(len(blk), len(blk))) + 1e-3
        sub = M / M.sum(axis=1, keepdims=True)
        for ii, i in enumerate(blk):
            P[i, blk] = sub[ii] * (1.0 - delta)
        # leak a tiny mass to the other block
        other = B if blk is A else A
        if len(other) > 0:
            P[np.ix_(blk, other)] += delta / len(other)
    P = P / P.sum(axis=1, keepdims=True)
    return _sample_chain(P, L, rng), "LOCKED"


def gen_over_random(V: int, L: int, rng: np.random.Generator) -> tuple[list[int], str]:
    return rng.integers(0, V, size=L).tolist(), "OVER_RANDOM"


def gen_collapsed(
    V: int, L: int, rng: np.random.Generator, n_active: int = 1
) -> tuple[list[int], str]:
    n_active = min(max(1, n_active), V)
    active = list(range(n_active))
    # ~99% on the active state(s), tiny leak elsewhere
    out = []
    for _ in range(L):
        if rng.random() < 0.99 or n_active == V:
            out.append(active[int(rng.integers(n_active))])
        else:
            out.append(int(rng.integers(V)))
    return out, "COLLAPSED"


def generate(
    label: str, V: int, L: int, seed: int, *, rho: float = 0.9, delta: float = 1e-3
) -> tuple[list[int], str]:
    rng = np.random.default_rng(seed)
    if label == "HEALTHY":
        return gen_healthy(V, L, rng)
    if label == "REPETITIVE":
        return gen_repetitive(V, L, rng, rho=rho)
    if label == "LOCKED":
        return gen_locked(V, L, rng, delta=delta)
    if label == "OVER_RANDOM":
        return gen_over_random(V, L, rng)
    if label == "COLLAPSED":
        return gen_collapsed(V, L, rng)
    raise ValueError(f"unknown label {label!r}")


def generate_corpus(
    label: str, V: int, L: int, n_utt: int, seed: int, *, rho: float = 0.9, delta: float = 1e-3
) -> tuple[list[list[int]], str]:
    streams = []
    for k in range(n_utt):
        s, _ = generate(label, V, L, seed * 100003 + k, rho=rho, delta=delta)
        streams.append(s)
    return streams, label
