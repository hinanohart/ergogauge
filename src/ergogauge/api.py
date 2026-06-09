"""Top-level analysis: certify (single utterance) and certify_corpus (pooled)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .baseline_vendi import vendi_from_tokens
from .calibrate import bootstrap_cis, compute_point_metrics
from .certificate import Certificate
from .classify import aggregate_flags, classify_level
from .config import ErgogaugeConfig
from .entropy import entropy_discriminator
from .estimator import OTHER_STATE, TransitionModel
from .gate import run_identifiability_gate
from .io import coerce_utterance, load_file
from .spectral import cheeger, kemeny

# Mandatory verbatim disclaimer (also in README and release notes; grep-enforced).
DISCLAIMER = (
    "ergogauge is a CPU-only, reference-free, decode-free, LLM-free diagnostic instrument "
    "that treats a generated codec-LM token stream as a finite-state Markov chain and reports "
    "an ergodicity certificate (spectral gap, Kemeny constant, Cheeger/Fiedler near-reducibility) "
    "from the empirical token-transition operator. It is NOT a replacement for FAD/MMD/UTMOS "
    "(it is a complementary, reference-free, decode-free axis), NOT an audio-quality benchmark, "
    "and makes NO claim about perceptual quality or that it fixes/solves generation quality. "
    "The invariant->failure-mode mapping on non-reversible chains is a CALIBRATED HEURISTIC, "
    "not a theorem; flags are reported with bootstrap confidence intervals and fail-closed ABSTAIN "
    "below the pre-registered identifiability threshold. Primary correctness is demonstrated on a "
    "synthetic injected-pathology harness; any real codec-LM token demo is secondary and labelled "
    "synthetic-vs-real."
)


def _resolve_config(config: ErgogaugeConfig | None, seed: int) -> ErgogaugeConfig:
    if config is None:
        return ErgogaugeConfig(seed=seed)
    if seed and config.seed == 0:
        config.seed = seed
    return config


def _analyze_level(
    streams: Sequence[Sequence[int]], cfg: ErgogaugeConfig, is_corpus: bool, level_idx: int
) -> dict[str, Any]:
    model, metrics = compute_point_metrics(streams, cfg)
    assert isinstance(model, TransitionModel)
    kem = kemeny(model.P, model.pi, model.eigvals)
    ch = cheeger(model.P, model.pi)
    en = entropy_discriminator(model.P, model.pi)
    cis = bootstrap_cis(streams, cfg)

    pooled: list[int] = [t for s in streams for t in s]
    vendi = vendi_from_tokens(pooled)
    n_states_raw = len(set(pooled))

    gate = run_identifiability_gate(
        n_obs_pairs=model.n_obs_pairs,
        n_states_after_collapse=model.n_states_after_collapse,
        sparsity=model.sparsity,
        support_coverage=model.support_coverage,
        thr=cfg.gate,
    )
    cls = classify_level(
        model=model,
        gate=gate,
        metrics=metrics,
        cis=cis,
        kem=kem,
        ch=ch,
        en=en,
        vendi=vendi,
        thr=cfg.gate,
        is_corpus=is_corpus,
    )

    order = sorted(range(len(model.pi)), key=lambda i: -model.pi[i])[:10]
    occ_top = [
        [(-1 if model.states[i] == OTHER_STATE else int(model.states[i])), float(model.pi[i])]
        for i in order
    ]

    level_block: dict[str, Any] = {
        "level": level_idx,
        "n_states_raw": int(n_states_raw),
        "n_states_after_collapse": int(model.n_states_after_collapse),
        "other_mass": float(model.other_mass),
        "reversible": bool(model.reversible),
        "reversibility_residual": float(model.reversibility_residual),
        "occupancy_top": occ_top,
    }
    level_block.update(cls)
    return level_block


def _analyze(
    levels_corpus: list[list[list[int]]], cfg: ErgogaugeConfig, is_corpus: bool
) -> Certificate:
    level_results: list[dict[str, Any]] = []
    gate_summary: dict[str, Any] = {"status": "PASS", "reasons": []}
    n_utt = max((len(lc) for lc in levels_corpus), default=0)

    for li, streams in enumerate(levels_corpus):
        block = _analyze_level(streams, cfg, is_corpus, li)
        level_results.append(block)

    # surface a representative gate (worst-case) at the top level
    model0, _ = compute_point_metrics(levels_corpus[0], cfg) if levels_corpus else (None, {})
    if model0 is not None:
        g0 = run_identifiability_gate(
            n_obs_pairs=model0.n_obs_pairs,
            n_states_after_collapse=model0.n_states_after_collapse,
            sparsity=model0.sparsity,
            support_coverage=model0.support_coverage,
            thr=cfg.gate,
        )
        gate_summary = g0.to_dict()

    aggregate = aggregate_flags(level_results)
    meta = {
        "n_utterances": int(n_utt),
        "n_levels": len(levels_corpus),
        "rvq_mode": cfg.rvq_mode,
        "seed": cfg.seed,
        "is_corpus": is_corpus,
        "disclaimer": DISCLAIMER,
    }
    return Certificate(gate=gate_summary, levels=level_results, aggregate=aggregate, meta=meta)


def certify(
    streams: dict[str, Any] | list[Any] | str | Path,
    *,
    config: ErgogaugeConfig | None = None,
    seed: int = 0,
) -> Certificate:
    """Certify a single utterance (advisory / low-confidence by default).

    ``streams`` may be a ``list[int]`` (single level), ``list[list[int]]`` (multi-codebook),
    a ``{'tokens': ...}`` / ``{'codebooks': ...}`` dict, or a path to a ``.json`` / ``.npz``.
    """
    cfg = _resolve_config(config, seed)
    obj: Any = load_file(streams) if isinstance(streams, (str, Path)) else streams
    per_level = coerce_utterance(obj)  # list over levels of one utterance's tokens
    levels_corpus = [[lvl] for lvl in per_level]
    return _analyze(levels_corpus, cfg, is_corpus=False)


def certify_corpus(
    streams: Iterable[Any],
    *,
    config: ErgogaugeConfig | None = None,
    seed: int = 0,
) -> Certificate:
    """Certify a corpus of utterances (the confident path). Token pairs are pooled per
    codebook level across utterances."""
    cfg = _resolve_config(config, seed)
    items = list(streams)
    if len(items) == 1 and isinstance(items[0], (str, Path)):
        items = [load_file(items[0])]
    per_utt = [coerce_utterance(u) for u in items]
    n_levels = max((len(p) for p in per_utt), default=0)
    levels_corpus: list[list[list[int]]] = [[] for _ in range(n_levels)]
    for utt in per_utt:
        for li, lvl_tokens in enumerate(utt):
            levels_corpus[li].append(lvl_tokens)
    return _analyze(levels_corpus, cfg, is_corpus=True)
