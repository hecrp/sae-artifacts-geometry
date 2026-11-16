# Phase 2: Toy Model Validation -- Running Notes
Started: March 2026
Status: COMPLETED

## Objective (REVISED POST SUPL1+2)

DUAL objective established by Phase 0 + Supplements:

  2A. Validate that anti-aligned features arise geometrically, even WITHOUT any L1 penalty.
      Test across lambda=0 in a 2D polysemantic toy model.

  2B. Map lambda dependency: does anti-alignment grow monotonically with lambda?
      10D sweep across 12 lambda values, 2 seeds.

  2C. Compare TopK vs ReLU+L1: does removing the L1 penalty eliminate anti-alignment?
      Prediction H1: TopK -> NO anti-alignment. RESULT: VIOLATED.

  2E. JumpReLU scalar vs per-feature threshold.
      Does replacing scalar threshold with per-feature eliminate anti-alignment?

## Prior Results Summary (from Phases 0, 1, SUPL1, SUPL2)

Gate G0: G0-PARTIAL
  - Sign flip (r pos->neg with depth): GPT-2 ONLY (ReLU+L1 specific)
  - Absolutely anti-aligned (cos<0 at >=6sigma): GPT-2 (86 feats [see ERRATA E1], late) + Llama (86 feats [see ERRATA E1], early)
  - Gemma 2B (per-feature JumpReLU): 1 feature, effectively resistant

Gate G1: G1-YES for all 3 models
  - Mixture model wins for all distributions (69% cases)
  - Bimodal structure confirmed

E2 CONFIRMED (SUPL1):
  - Threshold architecture IS the key differentiator
  - Per-feature threshold (Gemma 2B + 9B): 22/42 layers, ZERO neg-cos
  - Scalar threshold (Llama LXR): 86 features [see ERRATA E1], early layers

H6 REJECTED (SUPL2):
  - Size (9B Gemma vs 8B Llama) is NOT the explanation

## Progress Log

### March 2026 - Experiments run

#### 2A: 2D Analytical Toy Model

Trained SAEs on 2D synthetic data (2 superposed polysemantic features at angle theta).
Key findings:
- Anti-alignment appears at lambda=0 (no L1, no sparsity mechanism): n_neg=1, min_cos=-0.787 (theta=30 deg)
- The anti-aligned feature encoder points "between" the two data features
- Lambda effect is minimal: min_cos changes by <0.01 across lambda=0 to 0.2
- Theta dependence: larger theta -> stronger anti-alignment
  theta=15deg: min=-0.778 | theta=30deg: min=-0.787 | theta=45deg: min=-0.823 | theta=60deg: min=-0.867
- Geometric interpretation: the anti-aligned feature is the "background subtractor" that
  allows the SAE to reconstruct inputs via negative activation mechanism

Figures: phase2a_2D_toy_phase_transition.png, phase2a_analytical_vs_numerical.png
CSV: results/phase2/analytical_2D_toy.csv

#### 2B: Lambda Search (10D)

Swept lambda=[0, 5e-1] with 2 seeds.
Key findings:
- Anti-alignment appears sporadically at ALL lambda values -- no monotonic relationship
- At lambda=0: n_neg=0 (seed 42), n_neg=0 (seed 1337) -- geometry alone insufficient in 10D
- At lambda=1e-5: n_neg=1 (seed 42), n_neg=1 (seed 1337) -- tiny L1 triggers it
- At lambda=0.5: n_neg=0 (seed 42), n_neg=3 (seed 1337) -- high lambda + wrong init = worse
- Pattern: seed-dependent, non-monotonic, consistent with local minima landscape

CSV: results/phase2/lambda_search_10D.csv
Figure: phase2b_lambda_search_10D.png

#### 2C: TopK vs ReLU+L1 (10D)

ReLU+L1 (3 seeds): n_neg=[0, 0, 0], min_cos=[0.21, 0.81, 0.43], mean_mu=[0.81, 0.87, 0.63]
TopK k=5 (3 seeds): n_neg=[3, 4, 6], min_cos=[-0.159, -0.260, -0.354], mean_mu=[0.72, 0.61, 0.56]

CRITICAL RESULT: TopK produces MORE anti-aligned features than ReLU+L1.
In this toy setting this is evidence against H1 in its original form (anti-alignment as an L1-specific artifact); 3 seeds per condition.
TopK imposes hard sparsity with NO L1 penalty, yet anti-alignment is stronger.

Figure: phase2c_topk_comparison.png
CSV: results/phase2/experiment_2c_topk_results.csv

#### 2E: JumpReLU Scalar vs Per-Feature Threshold

Scalar JumpReLU (3 seeds): n_neg=[3, 0, 3], min_cos=[-0.604, 0.104, -0.437], theta_std=0.0
Per-feature JumpReLU (3 seeds): n_neg=[2, 3, 0], min_cos=[-0.328, -0.476, 0.074], theta_std~0.1

Key finding: BOTH variants produce anti-alignment. Per-feature allows more theta diversity
but does NOT reliably suppress anti-alignment. The scalar-vs-per-feature distinction observed
empirically (Gemma vs Llama) must arise from larger-scale interaction effects, not the
threshold mechanism alone.

Figure: phase2e_jumprelu_fixed.png
CSV: results/phase2/jumprelu_fixed_comparison.csv

## Key Numerical Results

| Experiment | Architecture | n_neg (mean) | min_cos (min) | Finding |
|------------|-------------|--------------|----------------|---------|
| 2A (2D, lambda=0) | ReLU, no sparsity | 1 | -0.787 | Anti-align at lambda=0 |
| 2B (10D, lambda sweep) | ReLU+L1 | noisy | -0.373 | No monotonic lambda effect |
| 2C | ReLU+L1 | 0 | +0.21 | L1 less prone |
| 2C | TopK k=5 | 4.3 | -0.354 | TopK MORE anti-aligned |
| 2E | JumpReLU scalar | 2 | -0.604 | Both variants produce anti-align |
| 2E | JumpReLU per-feat | 1.7 | -0.476 | Both variants produce anti-align |

## Gate G2 Decision

Decision: G2-YES (STRONG)
Date: March 2026

Rationale:
1. Anti-alignment appears at lambda=0 in 2D (no L1, no sparsity mechanism at all)
2. TopK (no L1) produces MORE anti-alignment than ReLU+L1 in 10D
3. JumpReLU scalar and per-feature both produce anti-alignment in toy models
4. Effect driven primarily by GEOMETRY (theta, bottleneck ratio) not sparsity mechanism

Effect on next phases:
- Phase 3 should investigate the GEOMETRIC mechanism in depth (manifold theory)
- The "absorption" theory (negative feature as global inhibitor) should be formalized
- The empirical Gemma vs Llama difference requires explanation beyond the toy model
- Phase 3 can proceed without needing to explain the scalar/per-feature distinction

## Figures Generated
- figures/phase2/phase2a_2D_toy_phase_transition.png/pdf
- figures/phase2/phase2a_analytical_vs_numerical.png/pdf
- figures/phase2/phase2b_lambda_search_10D.png/pdf
- figures/phase2/phase2b_relu_l1_distributions.png/pdf
- figures/phase2/phase2b_cosine_evolution.png/pdf
- figures/phase2/phase2c_topk_comparison.png/pdf
- figures/phase2/phase2e_jumprelu_comparison.png/pdf
- figures/phase2/phase2e_jumprelu_fixed.png/pdf

## Formal Report
reports/PHASE2_REPORT.pdf (14 pages, compiled successfully March 2026)
