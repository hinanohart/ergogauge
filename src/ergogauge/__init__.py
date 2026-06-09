"""ergogauge — reference-free ergodicity certificate for codec-LM token streams.

The core library is torch-free (numpy only). The optional real-token demo lives in
``examples/`` behind the ``[demo]`` extra.
"""

from __future__ import annotations

__version__ = "0.1.0a2"

from .api import DISCLAIMER, certify, certify_corpus
from .certificate import Certificate
from .config import ErgogaugeConfig
from .estimator import TransitionModel
from .gate import GateThresholds

__all__ = [
    "__version__",
    "certify",
    "certify_corpus",
    "Certificate",
    "ErgogaugeConfig",
    "TransitionModel",
    "GateThresholds",
    "DISCLAIMER",
]
