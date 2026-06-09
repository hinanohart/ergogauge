"""Stationary block-bootstrap confidence intervals for every reported scalar.

Politis-Romano stationary bootstrap (geometric block lengths) for single long streams;
utterance-level resampling with replacement for corpora. Every public scalar is returned
with a CI; there is no CI-less metric API (enforced by tests/test_no_ci_less_api.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import ErgogaugeConfig
from .entropy import entropy_discriminator
from .estimator import TransitionModel, build_transition_model
from .spectral import cheeger, kemeny, spectral_gap

METRIC_KEYS = ("spectral_gap", "kemeny", "cheeger_phi", "fiedler", "entropy_ratio")


def compute_point_metrics(
    streams: Sequence[Sequence[int]], cfg: ErgogaugeConfig
) -> tuple[TransitionModel, dict[str, float]]:
    model = build_transition_model(
        streams,
        laplace_alpha=cfg.laplace_alpha,
        other_collapse_min_count=cfg.other_collapse_min_count,
        max_states=cfg.max_states,
        min_pairs_per_state=cfg.gate.min_pairs_per_state,
    )
    g = spectral_gap(model.P, model.eigvals)
    k = kemeny(model.P, model.pi, model.eigvals)
    ch = cheeger(model.P, model.pi)
    en = entropy_discriminator(model.P, model.pi)
    metrics = {
        "spectral_gap": g,
        "kemeny": k.value,
        "cheeger_phi": ch.conductance_phi,
        "fiedler": ch.fiedler_value,
        "entropy_ratio": en.ratio,
    }
    return model, metrics


def _stationary_resample_one(
    seq: Sequence[int], rng: np.random.Generator, block_mean_len: int
) -> list[int]:
    n = len(seq)
    if n == 0:
        return []
    p = 1.0 / max(1, block_mean_len)
    out: list[int] = []
    while len(out) < n:
        start = int(rng.integers(n))
        out.append(seq[start])
        cur = start
        while len(out) < n and rng.random() > p:
            cur = (cur + 1) % n
            out.append(seq[cur])
    return out[:n]


def _resample(
    streams: Sequence[Sequence[int]], rng: np.random.Generator, cfg: ErgogaugeConfig
) -> list[list[int]]:
    if len(streams) >= 2:
        # corpus: resample utterances with replacement
        idx = rng.integers(0, len(streams), size=len(streams))
        return [list(streams[i]) for i in idx]
    # single utterance: stationary block bootstrap
    return [_stationary_resample_one(streams[0], rng, cfg.block_mean_len)]


def bootstrap_cis(
    streams: Sequence[Sequence[int]], cfg: ErgogaugeConfig
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(cfg.seed)
    reps = max(2, cfg.bootstrap_reps)
    samples: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(reps):
        rs = _resample(streams, rng, cfg)
        if sum(len(s) for s in rs) < 2:
            continue
        _, m = compute_point_metrics(rs, cfg)
        for key in METRIC_KEYS:
            v = m[key]
            if np.isfinite(v):
                samples[key].append(v)
    lo_q = (1.0 - cfg.gate.ci_level) / 2.0 * 100.0
    hi_q = (1.0 + cfg.gate.ci_level) / 2.0 * 100.0
    cis: dict[str, tuple[float, float]] = {}
    for key in METRIC_KEYS:
        arr = np.asarray(samples[key], dtype=np.float64)
        if arr.size == 0:
            cis[key] = (float("nan"), float("nan"))
        else:
            cis[key] = (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
    return cis
