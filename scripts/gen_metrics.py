#!/usr/bin/env python3
"""Generate results/<version>_metrics.json: the measured numbers the README quotes.

All correctness is synthetic (injected == detected on the held-out TEST seed grid).
Deterministic given the seeds; env-stamped. Run once at S6; the README is generated from
this file (no hand-written numbers).

This executes the pre-registered evaluation protocol of ``docs/CLAIM.md`` section 5:
  * easy-extreme per-class recall on the full 100-seed held-out TEST set,
  * a small grid-robustness sweep across V (regime, not a single point),
  * severity monotonicity (Spearman) for REPETITIVE (rho) and LOCKED (delta),
  * hard-extreme regime -> ABSTAIN is the correct answer,
  * G9 separation (entropy-ratio CI gap excludes 0) between HEALTHY / OVER_RANDOM.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

import ergogauge
from ergogauge import ErgogaugeConfig, certify, certify_corpus
from ergogauge.baseline_vendi import vendi_from_tokens
from ergogauge.certificate import Certificate
from ergogauge.estimator import build_transition_model
from ergogauge.spectral import cheeger, spectral_gap
from ergogauge.synth import LABELS, generate_corpus

ROOT = Path(__file__).resolve().parent.parent
EASY = {"V": 64, "L": 2000, "n_utt": 16}
TEST_SEEDS = list(range(1000, 1100))  # full pre-registered held-out TEST set (G10)
# Bootstrap reps for the bulk sweeps: the easy-extreme flag decision clears its threshold
# by a wide margin, so the CI-gated label is robust to reps; fewer reps only widens CIs
# (the conservative / fail-closed direction). Determinism is preserved (fixed seed).
CFG = ErgogaugeConfig(seed=0, bootstrap_reps=30)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (numpy-only; average-rank ties)."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)

    def _avg_rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # average tied ranks
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]

    rx = _avg_rank(ax)
    ry = _avg_rank(ay)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _flag_and_ratio_ci(cert: Certificate) -> tuple[str, tuple[float, float]]:
    flag = str(cert.aggregate["flags"][0])
    lvl0 = cert.levels[0]
    ci = lvl0["metrics"]["over_random_discriminator"]["ci95_ratio"]
    return flag, (float(ci[0]), float(ci[1]))


def confusion_and_recall() -> tuple[dict, dict, float, dict]:
    conf: dict[str, dict[str, int]] = {gt: dict.fromkeys([*LABELS, "ABSTAIN"], 0) for gt in LABELS}
    ratio_ci: dict[str, list[tuple[float, float]]] = {gt: [] for gt in LABELS}
    for label in LABELS:
        for seed in TEST_SEEDS:
            streams, gt = generate_corpus(label, seed=seed, **EASY)
            flag, rci = _flag_and_ratio_ci(certify_corpus(streams, config=CFG))
            conf[gt][flag] += 1
            ratio_ci[gt].append(rci)
    recall = {gt: conf[gt][gt] / sum(conf[gt].values()) for gt in LABELS}
    h = conf["HEALTHY"]
    healthy_fpr = sum(h[p] for p in ("REPETITIVE", "LOCKED", "COLLAPSED", "OVER_RANDOM")) / sum(
        h.values()
    )
    return conf, recall, healthy_fpr, ratio_ci


def grid_robustness(seeds: int = 5) -> dict:
    """Per-class recall at additional easy-extreme grid points across V (regime, not a
    single point). L>=2000, N_utt>=8 keeps us in the pre-registered easy regime."""
    out: dict[str, dict[str, float]] = {}
    for V in (16, 256):
        pts = {"V": V, "L": 2000, "n_utt": 8}
        conf = dict.fromkeys(LABELS, 0)
        tot = dict.fromkeys(LABELS, 0)
        for label in LABELS:
            for seed in range(1000, 1000 + seeds):
                streams, gt = generate_corpus(label, seed=seed, **pts)
                flag = certify_corpus(streams, config=CFG).aggregate["flags"][0]
                tot[gt] += 1
                conf[gt] += int(flag == gt)
        out[f"V{V}_L2000_Nutt8"] = {gt: round(conf[gt] / tot[gt], 4) for gt in LABELS}
    return out


def severity_monotonicity(seeds: int = 8) -> dict:
    """The instrument is calibrated: its load-bearing invariant moves monotonically with
    injected severity (Spearman >= 0.8). We measure the invariant itself (point estimate,
    no bootstrap) rather than a saturating binary detection rate.

      REPETITIVE: stronger loop ratio rho -> slower mixing -> smaller spectral gap.
      LOCKED:     smaller cross-block leak delta -> tighter bottleneck -> smaller Cheeger phi,
                  in the OBSERVABLE regime delta >= ~1/n_pairs. Below that the rare cross
                  transition is never sampled, the bottleneck is unobservable, and the
                  instrument correctly ABSTAINs (a stated resolution limit, not a miss).
    """
    rhos = [0.50, 0.65, 0.80, 0.90, 0.95, 0.99]
    gaps = []
    for rho in rhos:
        vals = [
            spectral_gap(
                build_transition_model(generate_corpus("REPETITIVE", seed=s, rho=rho, **EASY)[0]).P
            )
            for s in range(1000, 1000 + seeds)
        ]
        gaps.append(float(np.mean(vals)))
    # observable LOCKED regime at L=2000, n_utt=8 (>=~16 expected cross transitions at delta=1e-3)
    deltas = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    sev = [-float(np.log10(d)) for d in deltas]
    phis = []
    for delta in deltas:
        vals = []
        for s in range(1000, 1000 + seeds):
            m = build_transition_model(generate_corpus("LOCKED", seed=s, delta=delta, **EASY)[0])
            vals.append(cheeger(m.P, m.pi).conductance_phi)
        phis.append(float(np.mean(vals)))
    return {
        "metric": "Spearman of the load-bearing invariant vs injected severity (point estimate)",
        "repetitive": {
            "rho": rhos,
            "mean_spectral_gap": [round(g, 4) for g in gaps],
            "spearman_rho_vs_neg_gap": round(_spearman(rhos, [-g for g in gaps]), 4),
        },
        "locked": {
            "neg_log10_delta": [round(s, 3) for s in sev],
            "mean_cheeger_phi": [round(p, 4) for p in phis],
            "spearman_severity_vs_neg_phi": round(_spearman(sev, [-p for p in phis]), 4),
            "observable_regime_note": "delta>=1e-3 at L=2000,n_utt=8; tighter locks are below "
            "sample resolution and ABSTAIN (see docs/NON-CLAIM.md).",
        },
    }


def hard_extreme_abstain(seeds: int = 20) -> dict:
    """Hard-extreme regime: too little data to identify the operator -> ABSTAIN is correct
    (G3). We require the system NOT to emit a confident pathological flag on noise."""
    n = ab = confident_wrong = 0
    for seed in range(1000, 1000 + seeds):
        rng = np.random.default_rng(seed)
        toks = rng.integers(0, 256, size=80).tolist()  # short, large alphabet -> underdetermined
        cert = certify(toks, config=CFG)
        n += 1
        ab += int(cert.gate["status"] == "ABSTAIN")
        flag = cert.aggregate["flags"][0]
        conf = cert.aggregate["confidence"]
        confident_wrong += int(flag in ("REPETITIVE", "LOCKED", "COLLAPSED") and conf == "high")
    return {
        "abstain_rate": round(ab / n, 4),
        "confident_pathology_rate": round(confident_wrong / n, 4),
    }


def g9_separation(ratio_ci: dict) -> dict:
    """G9: entropy-ratio CI gap between HEALTHY and OVER_RANDOM excludes 0."""

    def agg(gt: str) -> tuple[float, float]:
        los = [lo for lo, _ in ratio_ci[gt]]
        his = [hi for _, hi in ratio_ci[gt]]
        return float(np.mean(los)), float(np.mean(his))

    h_lo, h_hi = agg("HEALTHY")
    o_lo, o_hi = agg("OVER_RANDOM")
    return {
        "healthy_ratio_ci_mean": [round(h_lo, 4), round(h_hi, 4)],
        "over_random_ratio_ci_mean": [round(o_lo, 4), round(o_hi, 4)],
        "gap_excludes_zero": bool(o_lo > h_hi),
        "gap": round(o_lo - h_hi, 4),
    }


def order_sensitivity() -> dict:
    """G6: ergogauge separates a loop from its shuffle; Vendi (order-independent) cannot."""
    rng = np.random.default_rng(0)
    loops, shufs = [], []
    for _ in range(20):
        loop = [i % 4 for i in range(4000)]
        shuf = loop[:]
        rng.shuffle(shuf)
        loops.append(loop)
        shufs.append(shuf)
    g_loop = [spectral_gap(build_transition_model([s]).P) for s in loops]
    g_shuf = [spectral_gap(build_transition_model([s]).P) for s in shufs]
    v_loop = [vendi_from_tokens(s) for s in loops]
    v_shuf = [vendi_from_tokens(s) for s in shufs]

    def auroc(neg: list[float], pos: list[float]) -> float:
        wins = sum(p > n for p in pos for n in neg)
        ties = sum(p == n for p in pos for n in neg)
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    return {
        "ergogauge_gap_auroc": round(auroc(g_loop, g_shuf), 4),
        "vendi_auroc": round(auroc(v_loop, v_shuf), 4),
        "note": "structural sanity check (order-sensitivity), not a superiority claim",
    }


def main() -> int:
    conf, recall, healthy_fpr, ratio_ci = confusion_and_recall()
    metrics = {
        "version": ergogauge.__version__,
        "mode": "synthetic",
        "regime": "easy-extreme",
        "grid": {
            **EASY,
            "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "n_test_seeds": len(TEST_SEEDS),
            "bootstrap_reps": CFG.bootstrap_reps,
        },
        "confusion_matrix": conf,
        "per_class_recall": {k: round(v, 4) for k, v in recall.items()},
        "macro_recall": round(sum(recall.values()) / len(recall), 4),
        "healthy_false_flag_rate": round(healthy_fpr, 4),
        "grid_robustness_recall": grid_robustness(),
        "severity_monotonicity": severity_monotonicity(),
        "hard_extreme": hard_extreme_abstain(),
        "g9_separation": g9_separation(ratio_ci),
        "order_sensitivity": order_sensitivity(),
        "vendi_wedge": False,
        "vendi_role": "complementary order-independent axis; not claimed to be beaten",
        "env": {
            "os": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seed": CFG.seed,
            "ci_method": "stationary-block-bootstrap",
        },
        "disclaimer": "Primary correctness is synthetic (injected==detected, held-out TEST seeds). "
        "Mapping is a calibrated heuristic, not a theorem.",
    }
    out = ROOT / "results" / f"{ergogauge.__version__}_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(metrics["per_class_recall"], indent=2))
    print(
        "macro_recall", metrics["macro_recall"], "healthy_fpr", metrics["healthy_false_flag_rate"]
    )
    print(
        "monotonicity",
        metrics["severity_monotonicity"]["repetitive"]["spearman_rho_vs_neg_gap"],
        metrics["severity_monotonicity"]["locked"]["spearman_severity_vs_neg_phi"],
    )
    print("hard_extreme", metrics["hard_extreme"])
    print("grid_robustness", metrics["grid_robustness_recall"])
    print("g9", metrics["g9_separation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
