# ROADMAP

ergogauge v0.1.0a1 is a pre-alpha instrument with a deliberately narrow scope. The
following are explicitly **out of scope for v0.1** and tracked here so the README makes
no premature claims about them.

## Planned / under consideration

- **Full RVQ-joint modeling.** v0.1 treats each residual-quantizer codebook level as an
  independent first-order chain (`rvq_mode='per_level'`). A joint multi-codebook operator
  (`rvq_mode='flatten'` currently raises `NotImplementedError`) is future work; the joint
  state space is large and needs dedicated estimation/identifiability handling.

- **General autoregressive-token generalization** (e.g. video/world-model token streams
  such as Genie-style or AR-video tokenizers). The math is token-modality agnostic, but
  validation harnesses and calibration for those domains are not yet built.

- **Good–Turing / smoothing alternatives.** v0.1 uses Laplace add-α on the observed
  submatrix plus OTHER-collapse. Good–Turing and held-out smoothing are candidates.

- **Larger real-data study.** v0.1 ships synthetic ground-truth correctness as the primary
  evidence and only an optional, clearly-labelled illustrative real-token demo
  (`examples/real_demo.py`, `[demo]` extra). A quantitative real-corpus study is future work.

- **PyPI distribution.** Deferred to a later release; v0.1.0a1 is installed from source /
  GitHub.

## Related work / orthogonal sibling

- `hinanohart/koopgauge` performs a **Koopman/DMD spectral audit on continuous latent
  states** of sequence foundation models. ergogauge is disjoint: it operates on a
  **stochastic transition operator over discrete integer token IDs**, shares no modules,
  and targets a different object.
