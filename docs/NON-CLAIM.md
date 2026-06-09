# NON-CLAIMS (ergogauge)

These are stated verbatim and grep-enforced in CI. ergogauge is deliberately scoped.

1. **ergogauge is an instrument, not a fix.** It diagnoses; it does not improve, repair,
   or solve generation quality.

2. **Correctness is demonstrated on a synthetic injected-pathology ground truth**
   (held-out test seed grid, where injected == detected). Any real codec-LM output shown
   is illustrative, labelled synthetic-vs-real, and is not a quantitative benchmark.

3. The invariant → failure-mode mapping (spectral gap → repetition, conductance →
   sub-vocabulary lock) is a **calibrated heuristic on non-reversible empirical operators,
   not a theorem**. On a non-reversible empirical operator the spectral gap only loosely
   bounds the mixing time.

4. The Cheeger two-sided inequality (1/2)λ₂ ≤ φ ≤ √(2λ₂) is a **theorem about the
   conductance of the symmetrized co-occurrence graph itself**. The inference
   "small symmetrized conductance ⇒ the original directed chain is trapped" is a
   heuristic calibrated only on the synthetic harness.

5. **Kemeny constant** is reported as a convenience exploration summary with a CI. It
   does not localize and it overlaps the gap axis, so it is assigned **no dedicated
   headline failure mode**.

6. **Novelty, stated precisely:** a reference-free, CPU-only ergodicity certificate
   computed from the empirical token-transition operator. We are not aware of a shipped
   tool that produces this on generated codec-LM token streams (see Related Work).
   We claim **no** invention of the Markov framework, the Kemeny constant, the Cheeger
   inequality, or the spectral gap (cf. arXiv:2410.02724, 2012.14660, 2402.13512,
   2501.01638 — all theory).

7. **Relation to Vendi Score:** ergogauge measures a temporal / transition axis; Vendi
   measures an order-independent set-diversity axis — they are complementary. The
   permutation-pair separation is a structural sanity check, **not** a superiority claim.
   Comparative wording ("better"/"complements-beats") is used only when its bootstrap CI
   excludes zero on realistic mixed-pathology streams; otherwise the headline auto-reverts
   to a neutral "complementary temporal axis".

8. Every scalar carries a stationary block-bootstrap CI. When a decision boundary lies
   within the CI, or the effective sample is below the pre-registered identifiability
   threshold, ergogauge returns **ABSTAIN** rather than a fabricated flag.

9. ergogauge **does not require, decode, or access** reference audio, waveforms, GPUs,
   torch, or an LLM. It reads integer token IDs only. It is **not** a replacement for
   FAD / MMD / UTMOS and makes no perceptual-quality claim.

10. **Downgrade rule (grep-enforced):** if a secondary comparative claim's bootstrap CI
    crosses zero, the headline auto-reverts to a neutral "instrument" framing. The denylist
    {permanent, complete, first, best, guaranteed, automatic, SOTA, solves, outperform,
    and Japanese equivalents} must grep to zero hits in README/docs/code, verified by a
    negative-fixture test.
