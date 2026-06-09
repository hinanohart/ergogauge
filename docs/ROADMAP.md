# ROADMAP

ergogauge v0.1.0a2 is a pre-alpha instrument with a deliberately narrow scope. The
following are explicitly **out of scope for v0.1** and tracked here so the README makes
no premature claims about them.

## Planned / under consideration

- **Full RVQ-joint modeling.** v0.1 treats each residual-quantizer codebook level as an
  independent order-1 (single-step) chain (`rvq_mode='per_level'`). A joint multi-codebook operator
  (`rvq_mode='flatten'` currently raises `NotImplementedError`) is future work; the joint
  state space is large and needs dedicated estimation/identifiability handling.

- **General autoregressive-token generalization** (e.g. video/world-model token streams
  such as Genie-style or AR-video tokenizers). The math is token-modality agnostic, but
  validation harnesses and calibration for those domains are not yet built.

- **Good–Turing / smoothing alternatives.** v0.1 uses Laplace add-α on the observed
  submatrix plus OTHER-collapse. Good–Turing and held-out smoothing are candidates.

- **Sparse eigensolver for very large state counts.** v0.1 is dense-only; OTHER-collapse
  caps the post-collapse state count and `max_states` is bounded to a dense-safe ceiling
  (8192). A deterministic sparse (fixed-`v0` Lanczos/Arnoldi) path would lift that ceiling
  for very large alphabets; it is not shipped in v0.1 (which is why scipy is not a runtime
  dependency).

- **Larger real-data study.** v0.1 ships synthetic ground-truth correctness as the primary
  evidence and only an optional, clearly-labelled illustrative real-token demo
  (`examples/real_demo.py`, `[demo]` extra). A quantitative real-corpus study is future work.

- **PyPI distribution.** Deferred to a later release; v0.1.0a2 is installed from source /
  GitHub.

## Related work / orthogonal sibling

- `hinanohart/koopgauge` performs a **Koopman/DMD spectral audit on continuous latent
  states** of sequence foundation models. ergogauge is disjoint: it operates on a
  **stochastic transition operator over discrete integer token IDs**, shares no modules,
  and targets a different object.
