# Phase 5: Architectural Comparison -- REFRAMED
# Priority: P5 | Duration: 5-7 days | Compute: LOW-MEDIUM | Gate: None
# REVISED POST PHASE 0 (March 2026)

## Objective -- COMPLETELY REFRAMED

ORIGINAL QUESTION: "Does hard sparsity eliminate anti-alignment?"
This question is ALREADY ANSWERED by Phase 0: NO. JumpReLU Llama SAEs have 86 [see ERRATA E1]
absolutely anti-aligned features. Hard sparsity does not eliminate the phenomenon.

NEW PRIMARY QUESTION:
"What produces the early-layer anti-alignment concentration in Llama but NOT in Gemma,
even though both use JumpReLU?"

This is the central architectural mystery:
  - Both Gemma 2B and Llama 3.1 8B use JumpReLU SAEs via SAELens
  - Gemma 2B mean cosine ~0.75, almost no absolute anti-alignment (1 feature)
  - Llama 3.1 8B mean cosine ~0.48, 86 absolutely anti-aligned features [see ERRATA E1] in early layers
  - What accounts for this difference?

CANDIDATE EXPLANATIONS (hypotheses to test):
  E1 MODEL SIZE: Llama 8B > Gemma 2B. Larger models may have earlier superposition onset.
     Test: Gemma 9B early layers (d_model=3584, 42 layers) -- if anti-alignment appears
     in early Gemma 9B layers, size explains the Gemma 2B-Llama gap.
     Prediction from H6: Gemma 9B should show early-layer anti-alignment.

  E2 THRESHOLD ARCHITECTURE: Gemma uses per-feature threshold (learned per feature),
     Llama uses scalar threshold (same for all features). Per-feature threshold may
     allow finer-grained alignment preservation.
     Test: compare threshold statistics between Gemma and Llama SAEs.

  E3 ACTIVATION NORMALIZATION: Llama SAEs use "expected_average_only_in" normalization.
     This scales input activations, potentially amplifying early-layer feature competition.
     Test: compute cos(E,D) for Llama with and without the normalization applied.

  E4 EXPANSION RATIO: Gemma canonical 16K = 7x expansion (d_model=2304).
     Llama 32K = 8x expansion (d_model=4096). Very similar. Unlikely to explain gap.

  E5 TRAINING DATA/PROCEDURE: Different data, different training dynamics.
     Hard to test without re-training.

NEW SECONDARY QUESTION:
"Within the ReLU+L1 (GPT-2) setting, does reducing lambda eliminate late-layer anti-alignment?
Does increasing lambda (more L1 shrinkage) increase the count and push it earlier?"
This is the residual Phase 5 question for H1.

---

## Background

Three main SAE architectures:
  1. ReLU + L1 (Bricken 2023, Bloom 2024): original, magnitude shrinkage artifact
     z_i = ReLU(x @ W_enc + b_enc)_i, loss = recon + lambda * ||z||_1
     
  2. TopK / BatchTopK (Gao 2024): exactly k features per token, no magnitude penalty
     z_i = topk_relu(x @ W_enc + b_enc, k), loss = recon only
     
  3. JumpReLU (Rajamanoharan 2024): learned threshold, L0 directly optimized
     z_i = step(pre_act_i - theta_i) * pre_act_i, loss = recon + lambda * ||z||_0

KEY PREDICTION (H1 implies):
  ReLU+L1 -> anti-aligned features PRESENT (magnitude shrinkage drives hedging)
  TopK     -> anti-aligned features ABSENT (no shrinkage, magnitude unconstrained)
  JumpReLU -> anti-aligned features ABSENT (no shrinkage, hard threshold)

KEY PREDICTION (H3 implies):
  All architectures -> anti-aligned features PRESENT (superposition-driven)

---

## Option A: Train BatchTopK SAE on GPT-2 Small Layer 8

### When to use
Use if Phase 0 Gemma Scope analysis is ambiguous (JumpReLU vs model confound)
OR if compute is available.

### Protocol
Train a BatchTopK SAE with SAELens, matching all Bloom hyperparameters EXCEPT:
  - Architecture: BatchTopK instead of ReLU+L1
  - Set k such that expected mean L0 matches Bloom's (check from Bloom model card)
    Bloom's L1 coefficient 8e-5 produces some average L0 -- match this L0 with k

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

# From gao2024: BatchTopK uses k=average expected activations
# If Bloom's SAEs have mean L0 = 32 (to be checked), set k=32 for BatchTopK

cfg_topk = LanguageModelSAERunnerConfig(
    model_name="gpt2",
    hook_name="blocks.8.hook_resid_post",
    hook_layer=8,
    d_in=768,
    expansion_factor=32,
    architecture="topk",  # SAELens BatchTopK
    activation_fn_kwargs={"k": 32},  # adjust based on Bloom L0
    l1_coefficient=0,  # NO L1 penalty for TopK
    train_batch_size_tokens=4096,
    # ... other params matching Bloom
)
```

### Analysis
After training, compute SAME metrics as Phase 1:
  - Full cos(E,D) distribution
  - r(||E||, cos(E,D)) per layer
  - Count at >=6sigma

COMPARE directly with ReLU+L1 results (same model, same layer, only architecture differs).

---

## Option B: Zero-Compute Comparison (preferred first attempt)

### Step 1: Use Gemma Scope results from Phase 0 (JumpReLU)
If Phase 0 found that Gemma Scope (JumpReLU) has NO anti-aligned features:
  This confirms L1 specificity IF we can rule out the model confound.
  Write: "Anti-aligned features absent in JumpReLU Gemma Scope SAEs, consistent with
  H1 (L1 artifact), though the model confound cannot be fully ruled out without
  training matched ReLU+L1 SAEs on Gemma 2B."

If Phase 0 found that Gemma Scope (JumpReLU) DOES have anti-aligned features:
  This contradicts H1 (L1-specific). Supports H3 or H5.
  This would be a significant finding and would reframe the paper.

### Step 2: Check SAELens model registry for TopK SAEs
  From SAELens: https://github.com/jbloom/SAELens
  Check if any TopK SAEs for GPT-2 Small are publicly available.
  If yes: download and analyze (zero additional training cost).

### Step 3: Eleuther GPT-2 SAEs (if any with different architecture)
  Check if Eleuther AI or other groups have published TopK SAEs for GPT-2.

---

## Expected Comparison Table

Architecture | Model     | d_sae  | mean_cos | 6sigma_count | r(L8) | r(L0)
ReLU+L1      | GPT-2 Sm  | 24576  |  0.368   |      4       | -0.747| +0.621
BatchTopK    | GPT-2 Sm  | 24576  |    ?     |      ?       |   ?   |   ?
JumpReLU     | Gemma 2B  | 65536  |    ?     |      ?       |   ?   |   ?
TopK         | Llama 3.1 | 131072 |    ?     |      ?       |   ?   |   ?

NOTE: For cross-model comparisons, normalize 6sigma_count by d_sae if needed.
      The sign flip pattern (positive L0, negative L8) is architecture-agnostic
      as a PATTERN even if exact r values differ.

---

## Interpretation Framework

CASE 1: ReLU+L1 has 4 features, TopK/JumpReLU have 0 features
  -> H1 STRONGLY SUPPORTED. L1 magnitude shrinkage is the mechanism.
  -> Paper title: "Anti-aligned features are L1-specific geometric artifacts"
  -> Recommendation for SAE design: avoid ReLU+L1 if interpretability matters

CASE 2: ReLU+L1 has 4 features, TopK/JumpReLU have 1-2 features
  -> H1 PARTIALLY SUPPORTED. L1 amplifies the phenomenon but doesn't create it alone.
  -> May indicate both H1 and H3 contribute.
  -> Paper: "L1 amplification of a baseline superposition-driven misalignment"

CASE 3: All architectures have similar anti-aligned feature counts
  -> H3 or H5 SUPPORTED. Architecture-independent mechanism.
  -> Major finding: anti-alignment is a fundamental property of SAEs, not L1 artifact.
  -> More alarming for the interpretability field.

CASE 4: Hard sparsity has MORE anti-aligned features than ReLU+L1
  -> Very unexpected. Would require new theoretical framework.
  -> Possible explanation: without L1, the encoder has more freedom to drift?

---

## Additional Analysis: Gated SAE (if available)

Gated SAEs (Rajamanoharan et al., same paper as JumpReLU) use a gate mechanism
that decouples feature detection from magnitude estimation:
  gate_pre_act = x @ W_gate + b_gate
  mag_pre_act = x @ W_mag + b_mag (with weight tying to W_dec)
  
  z_i = ReLU(gate_pre_act_i) * ReLU(mag_pre_act_i)  (element-wise)

The gate and magnitude heads can in principle have different alignment properties.
If Gated SAEs are available in Phase 0 (Gemma Scope includes them), compare.

---

## Expected Outputs (save to results/phase5/)

- results/phase5/architecture_comparison.csv
- results/phase5/sign_flip_by_architecture.csv
- results/phase5/G5_conclusion.md  (architectural comparison conclusion)
- figures/phase5/cosine_dist_comparison.png  (ReLU+L1 vs TopK vs JumpReLU histograms)
- figures/phase5/sign_flip_comparison.png  (r values: all architectures x all layers)

---

## Connection to Paper

Phase 5 results directly inform:
  - Section 4.1 (H1 predictions) -- specifically prediction #5: "Absent with hard sparsity"
  - Section 4.3 (H3 predictions) -- specifically prediction #4: "Present across architectures"
  - Section 6.5 (Phase 5 description in paper)
  - Section 7 (scenario analysis -- probability estimates update based on results)
  - Section 9 (Limitations -- SAE architecture section)
  - Conclusion: whether we recommend architectural changes for interpretability work

If Phase 5 confirms H1 (L1-specific), the practical recommendation is clear:
  USE JumpReLU OR TopK INSTEAD OF ReLU+L1 for any interpretability work where
  encoder-decoder alignment matters (feature attribution, steering vectors, etc.)
