# Phase 3: Seed Stability Analysis
# Priority: P3 | Duration: ~2 weeks | Compute: MEDIUM | Gate: G3
# REVISED POST PHASE 0 (March 2026)

## Objective -- REVISED (Two test cases now)

ORIGINAL: Test the 4 GPT-2 reference features across random seeds.
NEW: Also test Llama 3.1 8B Layer 1 features, but using a BUDGET APPROACH
     instead of full retraining (which is prohibitively expensive for 8B models).

### Test Case A: GPT-2 Small Layer 8 (unchanged from original plan)
Train 3 additional SAEs, seeds 1337/42/7.
Reference features: F18298 (L9), F19679 (L10), F19139 (L8), F18573 (L8)
[Note: L9/F18298 is now the primary reference, not L8/F19139]

### Test Case B: Llama 3.1 8B Layer 1 -- BUDGET seed stability
PROBLEM: Training a Llama SAE from scratch requires ~days of A100 compute.
SOLUTION: Use the existing multi-k Llama Scope checkpoints as a "pseudo-seed" test.
Llama Scope has 6 variants for each layer at 32K features:
  - llama_scope_lxr_8x (k=~64-100): our primary analysis
  - Different k values (50, 100, 200) at 16K and 32K widths represent
    different sparsity regimes -- not exactly seeds, but test robustness to
    training variation.

Budget stability protocol for Llama L1:
  1. Load 16K k=50, 16K k=100, 16K k=200 for layer 1 (3 checkpoints)
  2. Compute cos(E,D) distribution for each
  3. Check if 6sigma tail count is consistent: does it stay at 10-20 features?
  4. Find best decoder-cosine match for the 14 reference L1 features across checkpoints
  
This does NOT replace a proper seed stability test but gives a signal at zero
additional compute cost. True Llama seed stability = one expensive future experiment.

## Reference Features (UPDATED post Phase 0)

### GPT-2 Reference (expanded)
Previously: F19139, F18573, F11969, F20804 (all L8)
Updated top-5 (all layers):
  L9/F18298: cos=-0.736, -13.94sigma, ||E||=0.165 [small norm -- H2 candidate?]
  L10/F19679: cos=-0.684, -13.92sigma, ||E||=0.931 [large norm -- active hedge?]
  L10/F3268: cos=-0.573, -12.45sigma, ||E||=0.863 [large norm]
  L8/F19139: cos=-0.526, -11.38sigma, ||E||=0.727 [large norm]
  L8/F18573: cos=-0.421, -10.05sigma, ||E||=0.233 [small norm]

Note: L9/F18298 has SMALL norm (0.165) despite being the most extreme cosine.
This is INCONSISTENT with the large-norm anti-aligned features. May be H2/H4.

### Llama L1 Reference features (top 5)
  L1/F27745: cos=-0.646, -8.66sigma, ||E||=98.98 [very large norm]
  L1/F24381: cos=-0.614, -8.40sigma, ||E||=78.90 [very large norm]
  L1/F13765: cos=-0.591, -8.21sigma, ||E||=112.82 [very large norm]
  L2/F22038: cos=-0.585, -7.45sigma, ||E||=65.63 [large norm]
  L1/F4138:  cos=-0.582, -8.13sigma, ||E||=111.27 [very large norm]
All Llama features have VERY large encoder norms (60-113), suggesting they are
ACTIVE features (not dead), which CHALLENGES H2 for the Llama population.

## Determine whether the STATISTICAL PROPERTY (2-6+ features at >=6sigma per layer)
is stable even if the specific features differ.

---

## Background

belrose2025 demonstrated that only ~30% of SAE features are shared across seeds.
This means the specific features F19139 etc. may be training artifacts.

However, there are two distinct stability questions:
  1. FEATURE-LEVEL STABILITY: Do the same specific features appear?
     (measured by decoder cosine similarity matching)
  2. STATISTICAL STABILITY: Is the COUNT of anti-aligned features consistent?
     (measured by comparing 6sigma-tail counts)

Even if specific features differ across seeds, if every seed produces 3-6 features
at >=6sigma, the phenomenon is statistically stable -- it's a property of the
SAE training process, not a specific random seed artifact.

---

## Training Protocol

### What to train
3 additional SAEs on GPT-2 Small Layer 8 residual stream
Using EXACT Bloom hyperparameters (critical for reproducibility):
  - Architecture: ReLU + L1
  - Lambda_L1 = 8e-5
  - Learning rate = 4e-4
  - d_sae = 24576 (32x expansion)
  - d_model = 768
  - Training tokens: 300M
  - Ghost gradients: YES (same as Bloom training)
  - Seeds: 1337, 42, 7 (Bloom's original seed unknown -- pick distinct values)

### Training setup with SAELens
```python
# pip install sae-lens==X.X.X  (pin the version)
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    # Data
    model_name="gpt2",
    hook_name="blocks.8.hook_resid_post",
    hook_layer=8,
    d_in=768,
    dataset_path="NeelNanda/pile-10k",  # CHECK: match the exact dataset Bloom used
    # NOTE: Bloom used The Pile. "NeelNanda/pile-10k" may not be the full 300M tokens.
    # Use context_size=128 and enough steps for 300M tokens
    
    # SAE Architecture
    expansion_factor=32,
    b_dec_to_identity=True,
    normalize_activations="expected_average_only_in",  # verify Bloom's exact norm
    
    # Training
    lr=4e-4,
    l1_coefficient=8e-5,
    use_ghost_grads=True,
    train_batch_size_tokens=4096,
    
    # For seed: set torch.manual_seed(SEED) before runner
)
```

IMPORTANT NOTES:
  1. Verify the exact Bloom training config from: jbloom/GPT2-Small-SAEs model card on HF
  2. The training data must be THE SAME as Bloom's. Check if the pile subset is available.
  3. 300M tokens at 4096 batch size = ~73K steps. At ~1-2 ms/step on A100: ~73-146 GPU minutes.
     So ~2-3 hours per SAE on A100, not "2-3 days" as originally estimated.
     More likely: 6-9 hours total for 3 SAEs on a single A100.

### Compute options
  - Lambda Labs A100: ~$1-2/hour, ~$6-18 total for 3 SAEs
  - Modal: serverless GPU, similar cost
  - Google Colab Pro+: A100 available, may time out on long runs

---

## Analysis Protocol

### Step 1: Compute PW-MCC between all 4 SAEs

PW-MCC (song2025) = Pairwise Maximum Cosine Correlation.
For two SAEs with decoder matrices D_1 and D_2 (each d_sae x d_model, unit-norm rows):

```python
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_pw_mcc(D1, D2, batch_size=1000):
    """
    D1, D2: tensors of shape (d_sae, d_model) with unit-norm rows
    Returns: PW-MCC score in [0, 1]
    """
    d_sae = D1.shape[0]
    # Compute full cosine similarity matrix (d_sae x d_sae)
    # For 24576 features: matrix is 24576 x 24576 = ~4.8GB float32 -- TOO BIG
    # Use approximate: for each feature in D1, find best match in D2
    
    max_cosines = []
    for i in range(0, d_sae, batch_size):
        batch = D1[i:i+batch_size]  # batch_size x d_model
        sims = batch @ D2.T  # batch_size x d_sae
        max_cosines.append(sims.max(dim=1).values)
    
    max_cosines = torch.cat(max_cosines)
    pw_mcc = max_cosines.mean().item()
    return pw_mcc, max_cosines

# Compute for all pairs:
# (original, seed1337), (original, seed42), (original, seed7)
# (seed1337, seed42), (seed1337, seed7), (seed42, seed7)
```

Expected PW-MCC from belrose2025 for similar SAE size: ~0.30 (only 30% shared).
Our anti-aligned features are at the EXTREME of the distribution -- they might
be MORE or LESS stable than average features.

### Step 2: Match reference features across seeds

For each reference feature F_ref in {F19139, F18573, F11969, F20804}:
  1. Get its decoder direction: D_ref = W_dec[F_ref] (from original Bloom SAE)
  2. For each new seed SAE, find the feature with highest decoder cosine similarity:
     F_match = argmax_i cos(D_ref, D_new[i])
  3. Record the match quality (cosine similarity)
  4. Check if F_match has cos(E,D) <= mu_new - 3*sigma_new in the new SAE

```python
def find_matching_features(D_ref, D_new, W_enc_new, W_dec_new):
    """Find best-match features and check if they're also anti-aligned"""
    # Find best decoder match
    cosines_with_ref = (D_new @ D_ref.T)  # d_sae
    best_match_idx = cosines_with_ref.argmax().item()
    match_quality = cosines_with_ref[best_match_idx].item()
    
    # Check if best match is also anti-aligned in new SAE
    E_new = W_enc_new.T  # d_sae x d_model
    cos_ED_new = torch.nn.functional.cosine_similarity(E_new, W_dec_new, dim=1)
    mu_new = cos_ED_new.mean().item()
    sigma_new = cos_ED_new.std().item()
    match_cos_ED = cos_ED_new[best_match_idx].item()
    match_sigma_dev = (match_cos_ED - mu_new) / sigma_new
    
    return {
        'best_match_idx': best_match_idx,
        'decoder_cosine_with_ref': match_quality,
        'cos_ED_in_new_sae': match_cos_ED,
        'sigma_deviation_in_new_sae': match_sigma_dev,
        'is_3sigma_antialigned': match_sigma_dev < -3
    }
```

GATE CRITERIA for feature-level stability:
  Strong match: decoder_cosine_with_ref > 0.7 AND sigma_deviation < -3
  Weak match: decoder_cosine_with_ref > 0.5 AND sigma_deviation < -3
  No match: decoder_cosine_with_ref < 0.5 OR sigma_deviation >= -3

### Step 3: Count 6sigma-tail features per seed

For each new SAE, compute the full cos(E,D) distribution and count features at >=6sigma:
  - Compute all 24576 cosine similarities
  - Compute mu and sigma
  - Count features at <= mu - 6*sigma

Build a table:
  SAE seed | 6sigma-count | 3sigma-count | mu | sigma

If original Bloom SAE has 4 features at 6sigma, and all new seeds also have 3-5
features at 6sigma, statistical stability is confirmed.

Tightened gate (from reviewer feedback in paper): count should be within 1*sigma
of the original count, where sigma is computed from a Poisson or Binomial null.

---

## Expected Outputs (save to results/phase3/)

- results/phase3/pw_mcc_matrix.csv  (4x4 PW-MCC between all SAE pairs)
- results/phase3/feature_matching.csv  (per reference feature, per seed: match quality)
- results/phase3/tail_counts_per_seed.csv  (6sigma count per seed)
- results/phase3/G3_decision.md
- figures/phase3/feature_stability_heatmap.png  (reference feature x seeds)
- figures/phase3/tail_count_comparison.png  (bar chart per seed)

---

## Gate G3 Decision Criteria

H_STABLE:  >= 3/4 reference features (F19139, F18573, F11969, F20804) have a strong
           match (decoder cosine > 0.7) in >= 2/3 new seeds AND matched feature is
           also anti-aligned (sigma_dev < -3) in the new SAE.
           -> The SPECIFIC FEATURES are stable. Proceed to co-activation analysis (Phase 4)
              focused on these specific features.

H_STAT_STABLE: < 3/4 reference features match across seeds, BUT 6sigma-tail count
               is consistently 2-8 features per seed (within 1 sigma of original count).
               -> The STATISTICAL PROPERTY is stable, not the specific features.
                  Reframe: "a stable number of anti-aligned features exists per training
                  run, but specific feature identity varies." Proceed to Phase 4 with
                  the most extreme features from EACH seed.

H_UNSTABLE: < 2 features at 6sigma in >= 2/3 new seeds.
            -> Phenomenon is not reproducible.
               Publish as negative/instability result.
               Contribution: "The Bloom SAE's anti-aligned features are a specific
               training artifact compounding the general seed instability of SAEs
               (belrose2025)."

---

## Pre-Phase Setup

Before starting training, verify:
  [ ] SAELens installation and version pinned
  [ ] Access to Bloom training data (The Pile, same subset)
  [ ] Original Bloom SAE downloaded and cached
  [ ] GPU resources reserved (budget: ~$20 for Lambda Labs)
  [ ] SAELens config exactly matches Bloom hyperparameters (verify from HF model card)
  [ ] Baseline measurements: run Phase 1 analysis on original Bloom SAE first

---

## Known Risks

RISK 1: Training data mismatch
  The exact subset of The Pile used by Bloom may not be publicly available.
  If different data is used, differences in feature distributions could be
  confounded with seed effects.
  Mitigation: Use the SAELens-provided data loading utilities for gpt2 training.

RISK 2: SAELens version changes
  SAELens has changed substantially since Bloom trained. Ghost gradient implementation
  and normalization schemes may differ.
  Mitigation: Pin SAELens to a specific version. Document in requirements.txt.

RISK 3: Training instability
  3 training runs might produce degenerate SAEs (too many dead features, etc.)
  Mitigation: Monitor training with W&B logging. Stop early if >50% dead features.

RISK 4: Metric confusion
  PW-MCC as defined in song2025 may differ from the "pairwise MCC" referred to
  in belrose2025. Read both papers carefully before computing.
  song2025 (arXiv:2505.20254) is the primary reference for our implementation.
