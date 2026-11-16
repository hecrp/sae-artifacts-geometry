# Phase 0: Cross-Model Verification -- Results
Started: March 2026
Status: COMPLETED

## Objective
Determine whether the sign flip in r(||E||, cos(E,D)) and the stable negative tail
(>=6sigma) generalize beyond GPT-2 Small + Bloom SAEs.

## Gate G0 Decision: G0-PARTIAL

PROCEED but reframe. The sign flip is architecture-specific (ReLU+L1), but
absolutely anti-aligned features (cos<0) are present in BOTH GPT-2 and Llama 3.1 8B.
This STRENGTHENS the case for studying anti-aligned features -- they are NOT merely
an L1 artifact.

---

## Models Analyzed
- GPT-2 Small (ReLU+L1):     Bloom gpt2-small-res-jb, 12 layers, d_sae=24576 (32x)
- Gemma 2B (JumpReLU):       gemma-scope-2b-pt-res-canonical, 26 layers, d_sae=16384 (7x)
- Llama 3.1 8B (JumpReLU):   llama_scope_lxr_8x, 32 layers, d_sae=32768 (8x)

Software: sae_lens 6.37.6, Python 3.14, scipy, pandas, matplotlib

---

## Key Findings

### Finding 1: Sign Flip is Architecture-Specific (ReLU+L1)
- GPT-2 (ReLU+L1):   r_early=+0.486, r_late=-0.713 -> SIGN FLIP YES
- Gemma 2B (JumpReLU): r_early=-0.876, r_late=-0.768 -> no flip (negative at all depths)
- Llama 3.1 8B (JumpReLU): r_early=-0.835, r_late=-0.756 -> no flip (negative at all depths)

JumpReLU SAEs have r(||E||, cos) NEGATIVE from the very first layer. This means
the sign flip is not a universal depth effect -- it is specific to how ReLU+L1
training modifies the geometry of early vs late layers.

INTERPRETATION: In ReLU+L1 SAEs, early layers are embedding-dominated (structured,
low-rank: large encoders align with decoders -> positive r). JumpReLU eliminates
magnitude shrinkage, so encoder norms do not systematically decode geometric
relationships as they do under L1 pressure. Result: r is always negative (large
encoder norm -> more superposition pressure, less alignment) regardless of depth.

### Finding 2: Negative Mean Cosine Differs Dramatically Across Architectures
- GPT-2 (ReLU+L1):   mean mu = 0.37 (all layers)
- Gemma 2B (JumpReLU): mean mu = 0.75 (much higher)
- Llama 3.1 8B (JumpReLU): mean mu = 0.48 (intermediate)

JumpReLU SAEs preserve encoder-decoder alignment better on average. Gemma achieves
mu~0.78 in later layers -- features are more "canonical" (encoder and decoder
serve the same direction). This is consistent with JumpReLU removing L1 shrinkage.

### Finding 3: Absolutely Anti-Aligned Features (cos<0) ARE Present Cross-Architecture
- GPT-2 (ReLU+L1):   86 features [see ERRATA E1] at >=6sigma with cos<0 (layers 4-11, late layers)
- Gemma 2B (JumpReLU): 1 feature at >=6sigma with cos<0 (layer 1 only)
- Llama 3.1 8B (JumpReLU): 86 features [see ERRATA E1] at >=6sigma with cos<0 (layers 0-7, early)

The presence of cos<0 features in Llama STRONGLY CHALLENGES H1 (L1-specific artifact).
Anti-aligned features appear in JumpReLU SAEs too, concentrated in EARLY layers.

### Finding 4: Depth Pattern REVERSAL Between GPT-2 and Llama
- GPT-2: anti-aligned features emerge in LATER layers (4-11)
- Llama: anti-aligned features concentrated in EARLIEST layers (0-2)

This reversal is a genuine new empirical observation. It suggests:
- The mechanism depends on representation structure, not just depth
- Llama early layers pack many concepts into superposition more aggressively
- OR: Llama SAE training dynamics create early-layer artifacts

### Finding 5: Heavy Tails in Llama Early Layers
- Llama L0: excess kurtosis = 7.42 (EXTREME - much more than Gaussian)
- Llama L1: excess kurtosis = 5.90
- GPT-2 maximum: 3.36 (layer 10)
- Gemma maximum: 1.87 (layer 25)

The extreme kurtosis in Llama early layers implies the distribution is more
"explosive" -- most features are well-aligned but a tiny fraction is extremely
anti-aligned. This population structure is distinct from GPT-2.

### Finding 6: Gemma is Mostly Resistant to Absolute Anti-Alignment
Only 1 Gemma feature (L1, F3378, cos=-0.465) has absolutely negative cosine at 6sigma.
Gemma's statistical anomalies (6sigma below Gemma's mean) occur at cos~0.15-0.30
(positive -- just very low relative to Gemma's mean of ~0.78).

This makes Gemma the "cleanest" SAE in terms of absolute encoder-decoder alignment,
consistent with JumpReLU's design goals.

---

## GPT-2 Small -- Complete Layer Statistics (NEW: all 12 layers)

Layer | mu    | sigma | skew  | kurt  | r_pearson | 3sig | 6sig | min_cos
  L0  | 0.556 | 0.239 | -0.95 | -0.48 | +0.621    |  0   |  0   | -0.061
  L1  | 0.360 | 0.139 | -0.92 | -0.04 | +0.432    |  5   |  0   | -0.073
  L2  | 0.332 | 0.125 | -0.64 | -0.40 | +0.493    |  3   |  0   | -0.075
  L3  | 0.333 | 0.104 | -0.76 | +0.52 | +0.397    | 50   |  0   | -0.048
  L4  | 0.348 | 0.075 | -0.43 | +2.58 | -0.260    |321   |  2   | -0.361
  L5  | 0.356 | 0.071 | +0.34 | +2.22 | -0.638    | 62   | 10   | -0.171
  L6  | 0.369 | 0.076 | +0.61 | +1.51 | -0.736    | 11   |  6   | -0.353
  L7  | 0.379 | 0.079 | +0.68 | +1.21 | -0.756    |  7   |  3   | -0.359
  L8  | 0.368 | 0.079 | +0.69 | +2.03 | -0.747    |  5   |  4   | -0.526
  L9  | 0.368 | 0.079 | +0.65 | +2.13 | -0.756    |  4   |  1   | -0.736
  L10 | 0.368 | 0.076 | +0.49 | +3.36 | -0.712    |  8   |  3   | -0.684
  L11 | 0.349 | 0.075 | +0.63 | +1.90 | -0.639    | 40   |  6   | -0.202

Notable: L9 has the single most extreme feature: F18298, cos=-0.736 at -13.94sigma.
The previous hub_cosine_2 analysis focused on L8 (F19139 at -11.38sigma). L9 and L10
have even more extreme features -- update Table 2 in paper.

Skewness transition: negative in early layers (left-skewed: outliers below mean)
to positive in late layers (right-skewed: bulk of features shift but tail extends left).
This skewness flip coincides with the sign flip in r.

---

## Gate G0 Formal Criteria Assessment

G0-YES:  Sign flip in >=2 model families -> NOT MET
G0-PARTIAL: Sign flip in GPT-2 only, negative tail in >=1 other model -> MET
G0-NO:  Neither phenomenon outside GPT-2 -> NOT MET

Decision: G0-PARTIAL

Effect on subsequent phases:
- Phase 1 (distributional analysis): PROCEED. Now include all 3 models.
  The kurtosis data from Phase 0 already gives preliminary distributional insight.
- Phase 2 (toy model): PROCEED but revise hypothesis.
  H1 (L1-specific mechanism) is CHALLENGED by Llama data.
  Toy model should show that anti-alignment can arise WITHOUT L1 (add competing mechanism).
- Phase 3 (seed stability): PROCEED but add Llama L1 as test case alongside GPT-2 L8.
- Phase 4 (co-activation): PROCEED. Check BOTH GPT-2 late features AND Llama early features.
- Phase 5 (architectural comparison): REFRAME.
  No longer "does hard sparsity eliminate anti-alignment" (already answered: partially NO).
  Instead: "what mechanism produces early-layer anti-alignment in JumpReLU SAEs?"

Paper implications:
- Drop "L1-specific artifact" as primary framing
- Emphasize "depth-dependent pattern of anti-alignment varies by architecture"
- Two phenomena to explain: (a) sign flip [L1-specific], (b) anti-aligned features [universal but depth-varies]
- The Llama anomaly (early layers, very high kurtosis, very large encoder norms) deserves its own investigation

---

## Files Generated
- results/phase0/layer_stats.csv         -- 70 rows (12+26+32 layers)
- results/phase0/layer_stats_gpt2.csv    -- checkpoint (12 rows)
- results/phase0/layer_stats_gpt2_gemma.csv -- checkpoint (38 rows)
- results/phase0/extreme_features.csv   -- top-10 anti-aligned per layer per model
- results/phase0/sign_flip.csv          -- r values summary
- results/phase0/G0_decision.md         -- gate decision document
- figures/phase0/sign_flip_comparison.{png,pdf}
- figures/phase0/mean_cosine_by_depth.{png,pdf}
- figures/phase0/6sigma_tail_count.{png,pdf}
- figures/phase0/heatmap_r_pearson.{png,pdf}
- figures/phase0/cosine_distributions.{png,pdf}
- figures/phase0/skewness_kurtosis.{png,pdf}
- reports/PHASE0_REPORT.pdf             -- formal LaTeX report
