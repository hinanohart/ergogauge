#!/usr/bin/env python3
"""OPTIONAL real-token demo (secondary, illustrative — NOT a benchmark).

Requires the ``[demo]`` extra (``pip install -e ".[demo]"``: torch + encodec). The core
library and CI never import torch; this script lives outside the tested surface.

It encodes a *self-generated, license-free* audio signal (a synthetic chirp/noise mix — no
copyrighted material) into real EnCodec RVQ integer tokens, dumps them to an NPZ, and runs
ergogauge on those real codec indices. This is an *illustration* of the pipeline on real
codec structure; it is labelled synthetic-vs-real and makes no quantitative claim. Primary
correctness lives in the synthetic harness (tests/test_synth_invariants.py).

Usage:
    python examples/real_demo.py            # encode -> certify, print certificate
    python examples/real_demo.py --dump examples/data/real_tokens.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_license_free_audio(seconds: float = 6.0, sr: int = 24000) -> np.ndarray:
    """A self-generated chirp + colored noise mix (no copyrighted source)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    chirp = np.sin(2 * np.pi * (200 + 600 * t / seconds) * t)
    noise = 0.2 * np.cumsum(rng.standard_normal(t.size)) / np.sqrt(t.size)
    sig = 0.8 * chirp + noise
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig.astype(np.float32)


def encode_to_tokens(wav: np.ndarray) -> np.ndarray:
    """Encode mono 24 kHz audio to EnCodec RVQ integer codes -> (n_codebooks, T)."""
    import torch
    from encodec import EncodecModel

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    x = torch.from_numpy(wav)[None, None, :]  # (B, C, T)
    with torch.no_grad():
        encoded = model.encode(x)
    codes = encoded[0][0].squeeze(0).cpu().numpy().astype(np.int64)  # (n_q, T)
    return codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=str, default=None, help="write codebooks to this .npz")
    args = ap.parse_args()

    wav = make_license_free_audio()
    codes = encode_to_tokens(wav)
    print(f"encoded real EnCodec RVQ tokens: shape={codes.shape} (n_codebooks, T)")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump, codebooks=codes)
        print(f"dumped -> {args.dump}")

    # certify the real codec indices (per-codebook). Core stays torch-free.
    from ergogauge import certify

    cert = certify({"codebooks": [row.tolist() for row in codes]})
    print(cert.to_json())
    print(
        "\nNOTE: illustrative real-codec demo (synthetic-vs-real). Not a benchmark; "
        "primary correctness is the synthetic injected-pathology harness."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
