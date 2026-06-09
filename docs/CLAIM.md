# Pre-registered gate thresholds and evaluation protocol (ergogauge v0.1.0a1)

> **Binding anti-theater control (G10).** This file fixes every decision threshold
> and the train/test seed partition *before* the estimator / spectral / classifier
> code is written. The git commit that introduces this file is recorded in
> `.ergogauge-progress.json:preregistration.claim_md_sha`. A test
> (`tests/test_preregistration.py`) asserts that `ergogauge.gate.GateThresholds`
> equals the values below, and that the TRAIN and TEST synthetic seed sets are
> disjoint. Thresholds are calibrated on the TRAIN seed grid only; the confusion
> matrix is reported on the disjoint TEST seed grid.
>
> These numbers are *calibrated heuristics* on the empirical token-transition
> operator, **not theorems** (see `NON-CLAIM.md`). All scalars are reported with
> stationary block-bootstrap confidence intervals; decisions use CI bounds and
> fail closed to `ABSTAIN`.

## 1. Identifiability gate (fail-closed to ABSTAIN)

| key | value | meaning |
|---|---|---|
| `min_pairs_per_state` | 10 | required observed adjacent transition pairs per active state (row) |
| `min_total_pairs` | 200 | absolute floor on total adjacent transition pairs |
| `max_sparsity_frac` | 0.5 | if > this fraction of active states have < `min_pairs_per_state` outgoing observations → ABSTAIN |
| `other_collapse_min_count` | 5 | states whose total occurrence count < this are merged into a single OTHER state |
| `max_states` | 4096 | hard cap on post-collapse state count (enables deterministic dense eigensolver) |

If the gate fails the certificate flag is `ABSTAIN` with machine-readable reasons;
no numeric flag is fabricated.

## 2. Flag thresholds (calibrated heuristics, CI-gated)

Decisions use the 95% bootstrap CI bound shown, never the point estimate alone.

| key | value | rule |
|---|---|---|
| `gap_repetitive` | 0.15 | `spectral_gap` CI-high < this → **REPETITIVE** (slow mixing / loop) |
| `phi_locked` | 0.05 | Cheeger `conductance_phi` CI-high < this → **LOCKED** (near-reducible / sub-vocabulary lock) |
| `collapse_max_support` | 2 | effective support (states with stationary mass > `support_eps`) ≤ this → **COLLAPSED** |
| `collapse_occupancy` | 0.90 | max stationary mass > this → **COLLAPSED** |
| `support_eps` | 1e-4 | stationary-mass threshold for counting a state as "effective support" |
| `over_random_ratio` | 0.98 | entropy ratio (transition/marginal) CI-low > this → **OVER_RANDOM** (no measurable conditional structure) |
| `healthy_struct_ratio` | 0.95 | entropy ratio CI-high < this → structurally distinguishable from iid-uniform (HEALTHY-eligible) |

The `over_random_ratio` / `healthy_struct_ratio` pair is the A4 discriminator that
separates OVER_RANDOM (transition entropy ≈ marginal entropy ≈ log V, no conditional
structure) from HEALTHY (well-mixing yet measurably structured: transition entropy
< marginal entropy). If, on the TRAIN grid, these two classes cannot be separated
with a CI gap that excludes zero (gate **G9**), the taxonomy degrades honestly to the
4-class set {HEALTHY, REPETITIVE, LOCKED, COLLAPSED} (pivot `P-OVER-RANDOM-UNSEPARABLE`).

## 3. Decision order (classify.py, fail-closed)

1. identifiability gate fails → `ABSTAIN`
2. `COLLAPSED` (effective support ≤ `collapse_max_support` OR max stationary mass > `collapse_occupancy`)
3. `LOCKED` (`phi` CI-high < `phi_locked`)
4. `REPETITIVE` (`gap` CI-high < `gap_repetitive`)
5. `OVER_RANDOM` (entropy ratio CI-low > `over_random_ratio`)
6. `HEALTHY` (`gap` CI-low ≥ `gap_repetitive` AND `phi` CI-low ≥ `phi_locked` AND entropy ratio CI-high < `healthy_struct_ratio`)
7. otherwise (boundary / CI straddles a threshold) → `ABSTAIN`

LOCKED is checked independently of `gap` precisely because a near-reducible chain can
mix quickly *within* blocks (gap not low) yet be globally trapped (phi low): this is
the gap-misses-but-Cheeger-catches case that makes the second axis load-bearing.

## 4. Confidence / calibration

| key | value | meaning |
|---|---|---|
| `bootstrap_reps` | 1000 | stationary block-bootstrap replicates for every scalar CI |
| `block_mean_len` | 16 | mean geometric block length (Politis–Romano stationary bootstrap) |
| `ci_level` | 0.95 | two-sided confidence level |
| `alpha_nominal` | 0.05 | max tolerated HEALTHY false-flag rate (gate G3) |
| `corpus_min_utterances` | 8 | corpus mode is the confident path at/above this; single-utterance defaults to low confidence + aggressive ABSTAIN |

## 5. Synthetic evaluation protocol (primary correctness)

- Generators (ground truth = generator label): `HEALTHY`, `REPETITIVE`, `LOCKED`,
  `OVER_RANDOM`, `COLLAPSED` (see `synth.py`).
- Grid: alphabet `V ∈ {16, 64, 256, 1024}`, length `L ∈ {200, 500, 2000, 10000}`,
  utterances `N_utt ∈ {1, 8, 64}`. RNG = numpy PCG64.
- **Seed partition (G10, no-overlap):**
  - TRAIN seeds = `range(0, 100)` — thresholds above are calibrated here only.
  - TEST seeds  = `range(1000, 1100)` — confusion matrix reported here only.
  - `TRAIN ∩ TEST = ∅` is asserted by `tests/test_preregistration.py`.
- Pre-registered acceptance (TEST grid):
  - "easy-extreme" regime (corpus `N_utt ≥ 8`, `L ≥ 2000`, severity high:
    loop ratio ρ ≥ 0.8, lock cross-prob δ ≤ 1e-3): per-class recall ≥ 0.90.
  - "hard-extreme" regime: ABSTAIN is the correct answer (not counted as a miss; G3).
  - severity monotonicity: Spearman correlation of detection vs injected severity ≥ 0.8.
  - G9 separation: OVER_RANDOM vs HEALTHY and COLLAPSED vs OVER_RANDOM entropy-ratio
    CI gap excludes 0 on the easy-extreme TEST regime.

## 6. Determinism (G4)

`N ≤ max_states`: dense `numpy.linalg.eig` (gap/Kemeny) and `numpy.linalg.eigh`
(symmetric normalized Laplacian / Fiedler). Above the cap: `scipy.sparse.linalg.eigs`
with a fixed start vector `v0` and pinned `tol`/`maxiter`. Certificates are required to
be byte-identical on reruns and across Ubuntu and Windows.
