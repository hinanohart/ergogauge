#!/usr/bin/env python3
"""Generate results/<version>_metrics.json: the measured numbers the README quotes.

All correctness is synthetic (injected == detected on the held-out TEST seed grid).
Deterministic given the seeds; env-stamped. Run once at S6; the README is generated from
this file (no hand-written numbers).
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
from ergogauge.estimator import build_transition_model
from ergogauge.spectral import spectral_gap
from ergogauge.synth import LABELS, generate_corpus

ROOT = Path(__file__).resolve().parent.parent
EASY = {"V": 64, "L": 2000, "n_utt": 16}
TEST_SEEDS = list(range(1000, 1003))
CFG = ErgogaugeConfig(seed=0, bootstrap_reps=60)


def confusion_and_recall() -> tuple[dict, dict, float]:
    conf: dict[str, dict[str, int]] = {gt: dict.fromkeys([*LABELS, "ABSTAIN"], 0) for gt in LABELS}
    for label in LABELS:
        for seed in TEST_SEEDS:
            streams, gt = generate_corpus(label, seed=seed, **EASY)
            flag = certify_corpus(streams, config=CFG).aggregate["flags"][0]
            conf[gt][flag] += 1
    recall = {gt: conf[gt][gt] / sum(conf[gt].values()) for gt in LABELS}
    # HEALTHY false-flag rate: HEALTHY mislabeled as a hard pathology
    h = conf["HEALTHY"]
    healthy_fpr = sum(h[p] for p in ("REPETITIVE", "LOCKED", "COLLAPSED", "OVER_RANDOM")) / sum(
        h.values()
    )
    return conf, recall, healthy_fpr


def abstain_rate_degenerate() -> float:
    """Single very short utterances should fail closed to ABSTAIN."""
    n = 0
    ab = 0
    for seed in TEST_SEEDS:
        rng = np.random.default_rng(seed)
        toks = rng.integers(0, 256, size=80).tolist()  # short, large alphabet -> underdetermined
        cert = certify(toks, config=CFG)
        n += 1
        ab += cert.gate["status"] == "ABSTAIN"
    return ab / n


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

    # AUROC of separating shuffle (positive) from loop (negative)
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
    conf, recall, healthy_fpr = confusion_and_recall()
    metrics = {
        "version": ergogauge.__version__,
        "mode": "synthetic",
        "regime": "easy-extreme",
        "grid": {**EASY, "test_seeds": TEST_SEEDS, "bootstrap_reps": CFG.bootstrap_reps},
        "confusion_matrix": conf,
        "per_class_recall": {k: round(v, 4) for k, v in recall.items()},
        "macro_recall": round(sum(recall.values()) / len(recall), 4),
        "healthy_false_flag_rate": round(healthy_fpr, 4),
        "abstain_rate_on_degenerate": round(abstain_rate_degenerate(), 4),
        "order_sensitivity": order_sensitivity(),
        "vendi_wedge": False,
        "vendi_role": "complementary order-independent axis; not claimed to be beaten",
        "env": {
            "os": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
