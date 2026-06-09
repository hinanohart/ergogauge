"""In-repo Vendi Score (arXiv:2210.02410), re-implemented from the definition.

Only numpy.linalg.eigvalsh is used; no external Vendi package is vendored or imported.

In ergogauge this is **not** a baseline to beat. With a token-identity kernel the Vendi
score depends only on the token multiset, so it is provably invariant to shuffling the
stream. ergogauge's transition invariants are not. The G6 order-sensitivity test uses
that asymmetry as a bug-catcher (a loop and its shuffle must give equal Vendi but a
different ergogauge gap), never as a superiority claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def vendi_from_similarity(K: NDArray[np.float64]) -> float:
    """Vendi score = exp(Shannon entropy of the eigenvalues of K/n) for a PSD kernel K
    with unit diagonal."""
    n = K.shape[0]
    if n == 0:
        return 0.0
    evals = np.linalg.eigvalsh(K / n)
    evals = np.clip(evals, 0.0, None)
    s = evals.sum()
    if s <= 0:
        return 0.0
    p = evals / s
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
    return float(np.exp(ent))


def vendi_from_tokens(tokens: Sequence[int]) -> float:
    """Order-independent diversity of a token stream under the token-identity kernel.

    This reduces analytically to exp(H(unigram distribution)) — the effective number of
    distinct tokens — and is therefore exactly invariant to any permutation of ``tokens``.
    """
    if len(tokens) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(tokens), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
    return float(np.exp(ent))
