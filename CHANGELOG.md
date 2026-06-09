# Changelog

All notable changes to ergogauge are documented here. This project is pre-alpha;
versions are `0.1.0aN` prereleases.

**Threshold provenance (precise).** The identifiability thresholds (§1), the
gap/phi/collapse flag thresholds (§2), and the calibration constants (§4) were committed
before the estimator at the pre-registration scaffold (SHA `aa34cbf`) and are unchanged
since. The two **A4 discriminator** thresholds (`over_random_ratio` 0.98→0.93,
`healthy_struct_ratio` 0.95→0.88) were **calibrated on the TRAIN seed grid (0–99, never
TEST)** during S2 and finalized with the estimator (SHA `7b6adc7`) — so they were frozen
before any TEST-seed evaluation, but were *not* fixed at the aa34cbf scaffold. **a2 changes
no threshold value**; the changes below are bug fixes, added evaluation, and honesty
refinements.

## 0.1.0a2

### Fixed (correctness)
- **Fail-closed contract violation on empty / degenerate input.** Previously an empty
  corpus returned `HEALTHY` / `PASS` and an empty single utterance returned `COLLAPSED`
  / `PASS` — fabricating a verdict on zero data. `COLLAPSED` is now asserted only when
  `n_obs_pairs >= min_total_pairs`; otherwise control falls through the identifiability
  gate and fails closed to `ABSTAIN`. Empty / underdetermined input now always `ABSTAIN`s.
  (`classify.py`, `api.py`; regression tests in `tests/test_fail_closed.py`.)

### Added (evaluation — fulfils pre-registered `docs/CLAIM.md` §5)
- `scripts/gen_metrics.py` now runs the **full 100-seed** held-out TEST set (was 3),
  adds a **grid-robustness** recall sweep across `V ∈ {16, 256}`, **severity monotonicity**
  (Spearman of the load-bearing invariant vs injected severity), a **hard-extreme →
  ABSTAIN** check, and **G9 separation** numbers. Results in `results/0.1.0a2_metrics.json`.
- New enforced tests for the previously-untested pre-registered criteria:
  `tests/test_pre_registered_eval.py` (monotonicity + hard-extreme).

### Changed (honesty / clarity)
- Severity monotonicity is operationalized as the **calibrated-instrument** criterion
  (the invariant moves monotonically with severity) rather than a saturating binary
  detection rate, and a **finite-sample resolution limit** is documented: a lock with
  `δ ≲ 1/n_pairs` is unobservable and correctly `ABSTAIN`s (`docs/NON-CLAIM.md` §11).
- Corrected the "second axis" narrative. The Cheeger conductance `φ` is load-bearing not
  via a (impossible-for-reversible) "gap-high / φ-low" wedge, but because **both** LOCKED
  and REPETITIVE depress the spectral gap while `φ` separates them (bottleneck vs
  well-connected cycle). Test renamed `test_cheeger_distinguishes_lock_from_loop` with a
  strong assertion (no trivially-true branch).
- Hardened the honest-marketing denylist: a hit near a negation is excused only when the
  negation is in the **same sentence**, closing a wide-window bypass.
- Dropped an unverifiable Related-Work citation (arXiv:2501.01638); the five frozen
  citations remain.

## 0.1.0a1

- Initial pre-alpha: empirical token-transition operator, ergodicity certificate
  (spectral gap → REPETITIVE, Cheeger/Fiedler conductance → LOCKED, A4 discriminator →
  OVER_RANDOM vs HEALTHY, Kemeny convenience scalar), fail-closed `ABSTAIN`,
  pre-registered thresholds (G10), synthetic injected-pathology correctness, numpy-only
  CPU core, MIT.
