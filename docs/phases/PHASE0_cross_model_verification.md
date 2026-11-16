# Phase 0: Cross-Model Verification
# Priority: P0 | Duration: ~3 days | Compute: ZERO | Gate: G0

## Objective
Determine whether the sign flip in r(||E||, cos(E,D)) and the stable negative tail
(>=6sigma) generalize beyond GPT-2 Small + Bloom SAEs.

This is the most important gate in the research program. If negative, fast-fail on day 3.

---

## Motivation
All data so far comes from one model (GPT-2 Small, 124M params) and one SAE training
run (Bloom 2024, HuggingFace: jbloom/GPT2-Small-SAEs). A phenomenon specific to this
combination has limited scientific value.

The two target model families use DIFFERENT SAE architectures:
- Gemma Scope: JumpReLU (no L1 penalty, no magnitude shrinkage)
- Llama Scope: TopK (hard sparsity, exact k nonzero per input)

This creates an important confound: absence of the phenomenon in these SAEs could mean
EITHER (a) model-specific OR (b) architecture-specific (L1 vs hard sparsity).
We must track this carefully -- even the confounded result is informative.

---

## Resources

### Primary: Gemma Scope (lieberum2024)
- HuggingFace: google/gemma-scope-2b-pt-res
- Architecture: JumpReLU (hard sparsity, learned threshold, no L1 shrinkage)
- Model: Gemma 2 2B (~2.5B parameters)
- Coverage: all layers, multiple widths (16K, 32K, 65K, 131K, 262K features)
- Start with 16K width to match expansion ratio to GPT-2 analysis (32x of 768=24576)
  For Gemma 2B: d_model=2304, so 32x would be ~73K. Use 65K as closest.
  Alternatively start with 16K for speed and expand.

### Secondary: Llama Scope (he2024)
- HuggingFace: see he2024 paper for exact repo names
- Architecture: TopK
- Model: Llama 3.1 8B
- Coverage: all layers, 32K and 128K widths

---

## Protocol

### Step 1: Download SAE weights
```python
# Gemma Scope (residual stream SAEs)
# pip install transformer_lens huggingface_hub sae_lens

from huggingface_hub import snapshot_download
# Download specific SAE checkpoints for a subset of layers
# gemma-scope-2b-pt-res/layer_X/width_16k/average_l0_Y/
```

TARGET LAYERS for Gemma 2B (12 transformer blocks):
  - Sample every 2nd layer: 0, 2, 4, 6, 8, 10 (same pattern as GPT-2 analysis)
  - Check if Gemma Scope provides residual stream SAEs for these layers
  - If not, use whatever layers are available

### Step 2: Compute core metrics for each SAE
```python
import torch

def compute_cosine_stats(sae_weights):
    """
    sae_weights: dict with 'W_enc' (d_model x d_sae) and 'W_dec' (d_sae x d_model)
    Returns: per-feature cosine similarities and encoder norms
    """
    W_enc = sae_weights['W_enc']  # or encoder.weight
    W_dec = sae_weights['W_dec']  # or decoder.weight (normalized rows)

    # Encoder directions: rows of W_enc.T (shape: d_sae x d_model)
    E = W_enc.T  # d_sae x d_model
    D = W_dec    # d_sae x d_model (should be unit-norm rows)

    # Cosine similarities (feature-wise)
    cos_sim = torch.nn.functional.cosine_similarity(E, D, dim=1)

    # Encoder norms
    enc_norms = E.norm(dim=1)

    return cos_sim, enc_norms

def compute_layer_stats(cos_sim, enc_norms):
    """Compute all summary statistics for one layer"""
    mu = cos_sim.mean().item()
    sigma = cos_sim.std().item()
    skewness = ((cos_sim - mu) / sigma).pow(3).mean().item()
    kurtosis = ((cos_sim - mu) / sigma).pow(4).mean().item() - 3  # excess kurtosis

    r_pearson = torch.corrcoef(torch.stack([enc_norms, cos_sim]))[0, 1].item()

    threshold_3sig = mu - 3 * sigma
    threshold_6sig = mu - 6 * sigma
    count_3sig = (cos_sim < threshold_3sig).sum().item()
    count_6sig = (cos_sim < threshold_6sig).sum().item()

    return {
        'mu': mu, 'sigma': sigma,
        'skewness': skewness, 'kurtosis': kurtosis,
        'r_pearson': r_pearson,
        'count_3sig': count_3sig, 'count_6sig': count_6sig,
        'min_cos': cos_sim.min().item(),
        'n_features': len(cos_sim)
    }
```

### Step 3: Compare sign flip pattern across models

Build a table:
  Layer | Model | Architecture | r(||E||, cos) | mu | sigma | 6sigma-count
  
Compare sign (positive/negative) of r per layer, for each model+architecture combo.

KEY COMPARISON:
  - GPT-2 Small (ReLU+L1): sign flip from +0.62 (L0) to -0.75 (L8) -- established
  - Gemma 2B (JumpReLU): sign flip present/absent?
  - Llama 3.1 8B (TopK): sign flip present/absent?

### Step 4: Characterize the negative tail (if present)

For each model/architecture:
  - Report mu, sigma, 6sigma-tail count per layer (same Table 1 format as paper)
  - Identify most extreme anti-aligned features (equiv. of Table 2 in paper)
  - If tail absent: note explicitly (supports H1 if L1-specific)

---

## Expected Outputs (save to results/phase0/)

- results/phase0/gemma_scope_stats.csv  (layer x model statistics)
- results/phase0/llama_scope_stats.csv
- results/phase0/sign_flip_comparison.csv  (r values across layers x models)
- results/phase0/G0_decision.md  (document the gate decision and rationale)
- figures/phase0/sign_flip_comparison.png  (multi-model plot)

---

## Gate G0 Decision Criteria

G0-YES:  Sign flip (transition from positive to negative r) present in >=2 model families
         -> PROCEED with all phases. Strong paper.
         
G0-PARTIAL: Sign flip present in GPT-2 only, but negative tail (>=6sigma features)
            present in >=1 other model family
         -> PROCEED but reframe. Focus on tail mechanism, not sign flip universality.
         Note carefully: tail in JumpReLU SAEs would mean H1 is WRONG
         (L1 not necessary for anti-alignment), supporting H3 or H5.
         
G0-NO:  Neither sign flip nor negative tail found outside GPT-2 Small + Bloom SAEs
        -> TERMINATE. Write brief blog post or Alignment Forum note.
        Useful message: "SAE geometric artifacts may be model+training-run-specific,
        compounding seed instability (belrose2025) at the cross-model level."

Document the decision in results/phase0/G0_decision.md with:
  1. The exact measurements
  2. Why the criterion is met/not met
  3. Effect on subsequent phases

---

## Important Notes

NOTE 1: JumpReLU ARCHITECTURAL CONFOUND
If the sign flip is absent in Gemma Scope but present in a hypothetical ReLU+L1
SAE on Gemma 2B, that would be strong evidence for H1 (L1-specific artifact).
We cannot train new Gemma SAEs in Phase 0 (zero compute). But we should note this
confound explicitly and plan for it:
  If G0-PARTIAL (GPT-2 only), Phase 2 toy model becomes critical for establishing
  theoretical basis that the phenomenon IS L1-specific.

NOTE 2: DIFFERENT EXPANSION RATIOS
GPT-2 analysis: d_sae=24576, d_model=768, ratio=32x
Gemma Scope 2B: d_model=2304, pick 65K (28x) or 16K (7x)
The sigma-threshold method (6sigma) is ratio-agnostic, but total feature count
affects the expected tail count for any given distribution.

NOTE 3: LAYER CORRESPONDENCE
GPT-2 has 12 layers, Gemma 2B has 26 layers, Llama 3.1 8B has 32 layers.
Use RELATIVE depth (layer_i / total_layers) to compare sign flip position.
The sign flip in GPT-2 occurs around layer 4/12 = 33% depth.
Expected transition zone in Gemma 2B: around layer 9 (9/26 = 35% depth).

---

## Script Template

See: scripts/phase0_cross_model.py (to be created when implementation begins)

Key imports:
  transformer_lens, sae_lens, torch, scipy.stats, pandas, matplotlib
  
Approximate runtime: 10-30 minutes per SAE (weights download: ~500MB-2GB per SAE)
