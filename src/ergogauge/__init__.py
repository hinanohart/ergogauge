"""ergogauge — reference-free ergodicity certificate for codec-LM token streams.

The core library is torch-free (numpy + scipy only). The optional real-token demo
lives in ``examples/`` behind the ``[demo]`` extra.

Public API is wired up incrementally during the build; ``__version__`` is always
available so ``import ergogauge`` succeeds with only the core dependencies installed.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

__all__ = ["__version__"]
