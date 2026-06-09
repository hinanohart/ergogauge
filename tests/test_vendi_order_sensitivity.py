"""G6: order-sensitivity sanity check (bug-catcher, not a superiority claim).

A loop and its shuffle share the same token multiset, so Vendi (order-independent) must
return the same value, while ergogauge's spectral gap must change. Failure here means the
transition operator is accidentally order-blind (a bug), not that ergogauge "beats" Vendi.
"""

from __future__ import annotations

import numpy as np

from ergogauge.baseline_vendi import vendi_from_tokens
from ergogauge.estimator import build_transition_model
from ergogauge.spectral import spectral_gap


def test_vendi_invariant_to_shuffle_but_gap_is_not() -> None:
    rng = np.random.default_rng(0)
    loop = [i % 4 for i in range(4000)]
    shuf = loop[:]
    rng.shuffle(shuf)

    v_loop = vendi_from_tokens(loop)
    v_shuf = vendi_from_tokens(shuf)
    assert abs(v_loop - v_shuf) < 1e-9  # identical multiset -> identical Vendi

    g_loop = spectral_gap(build_transition_model([loop]).P)
    g_shuf = spectral_gap(build_transition_model([shuf]).P)
    assert g_shuf - g_loop > 0.3  # ergogauge sees the temporal difference Vendi cannot


def test_healthy_shuffle_rating_stable_in_distribution() -> None:
    """Shuffling a HEALTHY stream changes the chain, but Vendi (multiset) is unchanged."""
    from ergogauge.synth import generate

    stream, _ = generate("HEALTHY", V=64, L=4000, seed=1000)
    shuf = stream[:]
    np.random.default_rng(1).shuffle(shuf)
    assert abs(vendi_from_tokens(stream) - vendi_from_tokens(shuf)) < 1e-9
