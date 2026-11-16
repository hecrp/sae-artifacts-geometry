# Supplement 3: Connecting Toy Model to Real SAEs
# Priority: P2.5 (between Phase 2 and Phase 3) | Duration: 1 day | Compute: ZERO
# Created: March 2026 | Status: PLANNED

## Motivation

Phase 2 established that anti-alignment is a geometric phenomenon independent of sparsity
mechanism (G2-YES STRONG). However, three critical gaps remain before Phase 3 (Seed Stability):

1. **GAP 1 (Limitation 4)**: The "inhibitory bisector" toy model mechanism -- anti-aligned
   feature encoder E detects parent A while decoder D writes -A -- has never been tested
   against the real GPT-2 L8 anti-aligned features identified in Phase 0.
   If this test fails, the toy model captures the wrong mechanism and Phase 3 design needs
   to be revised before the expensive GPU training begins.

2. **GAP 2 (Limitation 5, Missing Exp 2D)**: The expansion ratio sweep (d_SAE/d_model) was
   planned but never executed. Real GPT-2 Bloom SAEs use 32x expansion (24576 / 768).
   Our toy model used ~3x expansion. Furthermore, Phase 2C found that TopK (no L1) produces
   MORE anti-alignment than L1, suggesting that MORE features may help or hurt anti-alignment.
   The discrepancy between the geometric theory ("more overcomplete = fewer anti-aligned")
   and the Phase 2C empirical result ("TopK = more anti-aligned") is unresolved.

3. **GAP 3 (Limitation 2)**: Anti-alignment is highly seed-dependent in 10D experiments. We
   do not have a reliable estimate of the fraction of seeds that produce anti-alignment.
   Without this, Phase 3's power analysis (how many seeds to train?) is uncalibrated.

## Sub-Experiments

### S3A: Bisector Test on Real GPT-2 SAE
**What**: Load the existing Bloom GPT-2 L8 SAE. For each of the 4 anti-aligned reference
features (F19139, F18573, F11969, F20804), test if the geometric absorption structure holds.

**Pre-registered test**:
For anti-aligned feature f with encoder E_f and decoder D_f:
  j_enc = argmax_j cos(E_f,  D_j)    [which aligned feature decoder is most detected by E_f?]
  j_dec = argmin_j cos(D_f,  D_j)    [which feature decoder is most opposed by D_f?]

Bisector-hedge prediction (from toy model):
  - j_enc == j_dec (OR highly overlapping) -> same parent being detected AND opposed
  - cos(E_f, D_{j_enc}) > 0.3          -> encoder genuinely detects the parent
  - cos(D_f, D_{j_dec}) < -0.3         -> decoder genuinely opposes a positive feature
  - cos(E_f, D_{j_dec}) > 0.1          -> encoder detects what decoder suppresses

Secondary check: activation frequency
  - If enc_norm of f is very small (< 0.5), check max activation over 10k tokens
  - Dead features (never activate) may show bisector structure by coincidence (H2)

**Expected results**:
  S3A-YES: bisector test passes for >= 2/4 features -> toy model mechanism = real mechanism
  S3A-PARTIAL: passes for 1/4 features -> mechanism heterogeneous (some H1, some H2)
  S3A-NO: passes for 0/4 features -> toy model captures different mechanism than reality

**Impact on Phase 3**:
  S3A-YES: Phase 3 proceeds as planned, focus on geometric framework
  S3A-PARTIAL/NO: Phase 3 should prioritize activation frequency test first (H2 check)

### S3B: Expansion Ratio Sweep (Missing Experiment 2D)
**What**: Train toy SAEs with d_SAE/d_model from 1x to 64x, measuring n_neg and min_cos.
d_model=10, 15 concepts, 20 seeds per expansion ratio. Lambda fixed at lambda=5e-3.

**Expansion ratios to test**: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
(32x matches the real GPT-2 L8 Bloom SAE exactly)

**Pre-registered prediction (theory)**:
As d_SAE/d_model increases, the SAE has MORE degrees of freedom, making it easier to
represent each concept with a dedicated feature rather than merging them. Prediction:
  n_neg DECREASES monotonically as d_SAE/d_model increases.
  At 32x expansion, n_neg should approach 0.

**Counter-prediction (from Phase 2C TopK finding)**:
TopK produced MORE anti-aligned features than L1 despite being "less constrained."
Possibly: more features = more opportunities for superposition interference.
  n_neg may be NON-MONOTONIC or even INCREASE with expansion ratio.

**The result directly calibrates Phase 3 expectations**:
  If n_neg ≈ 0 at 32x -> Phase 3 may find no anti-aligned features in new seeds
  If n_neg persists at 32x -> Phase 3 should find statistically stable anti-aligned features

### S3C: Seed Distribution Statistics
**What**: Run 100 seeds for the 10D toy model. Measure:
  - At lambda=0: fraction of seeds producing n_neg >= 1
  - At lambda=5e-3: fraction of seeds producing n_neg >= 1
  - Distribution of n_neg per seed (mean, std, P(n_neg >= 1))
  - Distribution of min_cos per seed (what is the typical minimum?)

**Why**: Phase 3 trains 3 new SAEs and expects to find anti-aligned features.
  If P(n_neg >= 1 | toy model, 10D) = 0.70, then training 3 real SAEs should give ~2.1 with
  anti-aligned features (if real SAEs follow toy model statistics).
  If P = 0.20, then 3 seeds is likely insufficient.

**Impact**: Directly determines the sample size recommendation for Phase 3.

## Output Specifications

### Files to create:
  docs/phases/SUPPLEMENT3_REPORT.md         (running notes)
  scripts/supplement3_toy_real_bridge.py          (all 3 sub-experiments)
  results/supplement3/
    bisector_test_results.csv    (S3A: per feature, per metric)
    expansion_ratio_sweep.csv    (S3B: per ratio, per seed)
    seed_distribution.csv        (S3C: per seed, lambda=0 and 5e-3)
  figures/supplement3/
    supplement3a_bisector_test.png/pdf
    supplement3b_expansion_ratio.png/pdf
    supplement3c_seed_distribution.png/pdf
  reports/SUPPLEMENT3_REPORT.tex (formal LaTeX)
  reports/SUPPLEMENT3_REPORT.pdf

### Gate for Phase 3:
  The supplement does NOT change G2-YES (already decided).
  It adds CONDITIONS to Phase 3:
  Condition C1: If S3A is NO -> Phase 3 must add activation frequency test before seed matching
  Condition C2: If S3B shows n_neg≈0 at 32x -> need more seeds (budget increase)
  Condition C3: If S3C P(n_neg>=1) < 0.30 -> revise Phase 3 null hypothesis

## Pre-registration

These predictions are pre-registered BEFORE running the experiments:
1. S3A: bisector test will PASS for F19139 (large enc_norm=0.727) but FAIL for F18573 (small enc_norm=0.233)
2. S3B: n_neg will DECREASE with expansion ratio (MORE features = easier to separate concepts)
3. S3C: P(n_neg>=1 | lambda=5e-3) > P(n_neg>=1 | lambda=0) [some L1 amplifies the effect]
   and P(n_neg>=1) will be between 0.30 and 0.70 (not extreme in either direction)
