from __future__ import annotations

import pytest

from ergogauge import ErgogaugeConfig

# Pre-registered seed partition (mirror of docs/CLAIM.md section 5).
TRAIN_SEEDS = range(0, 100)
TEST_SEEDS = range(1000, 1100)

EASY = {"V": 64, "L": 2000, "n_utt": 16}  # easy-extreme regime


@pytest.fixture
def fast_cfg() -> ErgogaugeConfig:
    # fewer bootstrap reps for test speed; correctness/separation is unchanged.
    return ErgogaugeConfig(seed=0, bootstrap_reps=60)
