"""Loading and normalizing integer token streams. No audio is ever decoded.

Accepted in-memory forms:
  - ``list[int]``                       a single utterance, single codebook level
  - ``list[list[int]]`` (via certify)   a single utterance, multiple codebook levels
  - ``{"tokens": [...]}``               a single utterance, single level
  - ``{"codebooks": [[...], ...]}``     a single utterance, multiple levels

File forms (``.json`` / ``.npz``): the JSON object follows the dict forms above; an NPZ
may contain ``tokens`` (1-D) or ``codebooks`` (2-D, level-major).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _validate(level: Any) -> list[int]:
    arr = np.asarray(level)
    if arr.ndim != 1:
        raise ValueError("each codebook level must be a 1-D integer sequence")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError("token ids must be integers")
        arr = arr.astype(np.int64)
    if arr.size and arr.min() < 0:
        raise ValueError("token ids must be non-negative")
    return [int(x) for x in arr.tolist()]


def coerce_utterance(u: Any) -> list[list[int]]:
    """Return per-level token lists for one utterance (outer index = codebook level)."""
    if isinstance(u, dict):
        if "codebooks" in u:
            return [_validate(lv) for lv in u["codebooks"]]
        if "tokens" in u:
            return [_validate(u["tokens"])]
        raise ValueError("dict utterance must have 'tokens' or 'codebooks'")
    if isinstance(u, np.ndarray):
        if u.ndim == 2:
            return [_validate(row) for row in u]
        return [_validate(u)]
    if isinstance(u, (list, tuple)):
        if len(u) > 0 and isinstance(u[0], (list, tuple, np.ndarray)):
            return [_validate(lv) for lv in u]  # list[list[int]] -> multi-codebook
        return [_validate(list(u))]  # list[int] -> single level
    raise TypeError(f"unsupported utterance type: {type(u)!r}")


def load_file(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        if "codebooks" in data:
            return {"codebooks": [data["codebooks"][i] for i in range(data["codebooks"].shape[0])]}
        if "tokens" in data:
            return {"tokens": data["tokens"]}
        raise ValueError("npz must contain 'tokens' or 'codebooks'")
    return json.loads(p.read_text(encoding="utf-8"))
