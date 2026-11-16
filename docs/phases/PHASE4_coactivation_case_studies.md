# Phase 4: Co-activation Case Studies
# Priority: P4 | Duration: 1-2 weeks | Compute: LOW | Gate: None (conditional on G0-G3)

## Objective
Determine the behavioral and co-activation properties of anti-aligned features to
discriminate between hypotheses H1-H5, especially H1 (subtractive hedges) vs H2 (dead decay).

The single most discriminating measurement: ACTIVATION FREQUENCY.
  - Near-zero frequency -> H2 (dead feature decay)
  - Moderate frequency  -> H1 (subtractive hedge) or H3/H5

---

## Prerequisites
  - GPT-2 Small model cached
  - Bloom SAEs cached (all layers, especially L8)
  - 10M tokens of text data (can use The Pile or any large text corpus)

---

## Sub-phase 4a: Activation Frequency Measurement

### What to compute
For all features with cos(E,D) <= mu - 3*sigma (~10-14 per layer for L4-L10):
  - ACTIVATION FREQUENCY: fraction of tokens where z_i > 0
    (= firing rate, "density" in neuronpedia terminology)
  - MEAN ACTIVATION: E[z_i | z_i > 0]
    (average activation magnitude when feature fires)
  - MAX ACTIVATION: max_t(z_i(t))
    (largest activation observed in the 10M token corpus)

### Implementation

```python
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE  # or load directly from HF

# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2")
# Load Bloom GPT-2 Small SAE for layer 8
# sae = SAE.from_pretrained("jbloom/GPT2-Small-SAEs", "blocks.8.hook_resid_post")

def compute_activation_stats(model, sae, dataloader, target_features, device='cuda'):
    """
    Compute activation stats for a list of target feature indices.
    
    target_features: list of feature indices to track (e.g., [19139, 18573, 11969, 20804])
    dataloader: yields batches of token sequences
    """
    fire_counts = {f: 0 for f in target_features}
    total_activations = {f: [] for f in target_features}
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch.to(device)  # shape: (batch, seq_len)
            batch_tokens = tokens.shape[0] * tokens.shape[1]
            total_tokens += batch_tokens
            
            # Get residual stream activations at layer 8
            _, cache = model.run_with_cache(tokens, names_filter="blocks.8.hook_resid_post")
            resid = cache["blocks.8.hook_resid_post"]  # shape: (batch, seq_len, d_model)
            
            # Flatten to (batch*seq_len, d_model)
            resid_flat = resid.reshape(-1, 768)
            
            # Run through SAE encoder
            z = torch.relu(resid_flat @ sae.W_enc + sae.b_enc)  # (N, d_sae)
            
            # Track target features
            for feat_idx in target_features:
                feat_acts = z[:, feat_idx]  # (N,)
                fire_counts[feat_idx] += (feat_acts > 0).sum().item()
                # Store activations where feature fires
                active_acts = feat_acts[feat_acts > 0].cpu().numpy()
                total_activations[feat_idx].extend(active_acts.tolist())
    
    results = {}
    for feat_idx in target_features:
        acts = np.array(total_activations[feat_idx])
        results[feat_idx] = {
            'activation_frequency': fire_counts[feat_idx] / total_tokens,
            'mean_activation': acts.mean() if len(acts) > 0 else 0.0,
            'max_activation': acts.max() if len(acts) > 0 else 0.0,
            'fire_count': fire_counts[feat_idx],
            'total_tokens': total_tokens
        }
    
    return results
```

### Expected results and implications

Feature F19139 (cos=-0.526, ||E||=0.727):
  LARGE encoder norm suggests the feature may fire frequently (large encoder -> strong detection)
  If activation_frequency > 0.001 (1 in 1000 tokens): H1 supported (active hedge)
  If activation_frequency < 0.0001: H2 supported (near-dead feature)

Feature F18573 (cos=-0.421, ||E||=0.233):
  SMALL encoder norm suggests near-dead (see paper open question)
  If activation_frequency < 0.0001: strongly suggests H2
  This feature may represent the dead-feature-decay mechanism

NOTE: The two large-norm features (F19139, F20804) and two small-norm features
(F18573, F11969) may represent two different mechanisms operating simultaneously.
This is an open question highlighted in the paper.

---

## Sub-phase 4b: Co-activation Analysis

For each anti-aligned feature f_neg, compute co-activation statistics with all other features.

### Co-occurrence matrix (approximate, memory-efficient)

```python
def compute_coactivation_for_feature(model, sae, dataloader, f_neg, top_k=100, device='cuda'):
    """
    Compute P(f_j > 0 | f_neg > 0) and PMI for the top-k most co-active features.
    
    Instead of storing full co-activation matrix (24576 x 24576 = too large),
    we use streaming: accumulate only for f_neg > 0 tokens.
    """
    n_features = sae.W_dec.shape[0]
    
    # Counters
    count_fneg = 0        # tokens where f_neg fires
    count_joint = torch.zeros(n_features)  # tokens where f_neg AND f_j fire
    count_fj = torch.zeros(n_features)    # total tokens where f_j fires
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch.cuda()
            _, cache = model.run_with_cache(tokens, names_filter="blocks.8.hook_resid_post")
            resid = cache["blocks.8.hook_resid_post"].reshape(-1, 768)
            
            z = torch.relu(resid @ sae.W_enc + sae.b_enc)  # (N, d_sae)
            binary = (z > 0).float()  # (N, d_sae)
            
            # Global frequency for P(f_j)
            count_fj += binary.sum(0).cpu()
            total_tokens += binary.shape[0]
            
            # Co-activation with f_neg
            fneg_active = binary[:, f_neg]  # (N,)
            count_fneg += fneg_active.sum().item()
            
            # For tokens where f_neg fires, count co-activations
            fneg_mask = fneg_active > 0  # (N,)
            if fneg_mask.any():
                count_joint += binary[fneg_mask].sum(0).cpu()
    
    # Compute conditional probabilities
    p_fneg = count_fneg / total_tokens
    p_fj = count_fj / total_tokens
    
    if count_fneg > 0:
        p_fj_given_fneg = count_joint / count_fneg
    else:
        p_fj_given_fneg = torch.zeros(n_features)
    
    # PMI = log(P(f_neg, f_j) / (P(f_neg) * P(f_j)))
    p_joint = count_joint / total_tokens
    pmi = torch.log(p_joint / (p_fneg * p_fj + 1e-10) + 1e-10)
    pmi[p_joint == 0] = float('-inf')
    
    return {
        'p_fneg': p_fneg,
        'p_fj_given_fneg': p_fj_given_fneg,
        'p_fneg_given_fj': count_joint / (count_fj + 1e-10),
        'pmi': pmi,
        'count_fneg': count_fneg
    }
```

### Identify putative parent

```python
def find_parent_feature(coact_stats, f_neg):
    """Identify putative parent: argmax_j PMI(f_neg, f_j), excluding f_neg itself"""
    pmi = coact_stats['pmi'].clone()
    pmi[f_neg] = float('-inf')  # exclude self
    
    parent_idx = pmi.argmax().item()
    
    # Test H1 asymmetry criterion
    p_parent_given_neg = coact_stats['p_fj_given_fneg'][parent_idx].item()
    p_neg_given_parent = coact_stats['p_fneg_given_fj'][parent_idx].item()
    
    asymmetry_ratio = p_parent_given_neg / (p_neg_given_parent + 1e-10)
    
    # Pre-registered threshold: asymmetry ratio > 10x to confirm H1
    h1_confirmed = (p_parent_given_neg > 0.8) and (asymmetry_ratio > 10)
    
    return {
        'parent_idx': parent_idx,
        'p_parent_given_neg': p_parent_given_neg,  # should be ~1 for H1
        'p_neg_given_parent': p_neg_given_parent,   # should be <<1 for H1
        'asymmetry_ratio': asymmetry_ratio,
        'pmi_with_parent': coact_stats['pmi'][parent_idx].item(),
        'h1_confirmed': h1_confirmed  # asymmetry_ratio > 10
    }
```

### Pre-registered thresholds (from paper Section 6.4b)

H1 CONFIRMATION requires:
  - P(f_parent > 0 | f_neg > 0) > 0.8 (parent almost always fires when hedge fires)
  - P(f_neg > 0 | f_parent > 0) < 0.1 (hedge only fires in a minority of parent activations)
  - asymmetry_ratio = P(parent|neg) / P(neg|parent) > 10

---

## Sub-phase 4c: Decoder Projection Analysis

For each (f_neg, f_parent) pair identified in 4b:

```python
def decoder_projection_analysis(sae, f_neg, f_parent):
    """
    Compute the projection of f_neg's decoder onto f_parent's decoder.
    Pre-registered threshold: pi_dec < -0.3 to confirm decoder opposition (H1).
    """
    D_neg = sae.W_dec[f_neg]  # d_model vector (unit norm)
    D_parent = sae.W_dec[f_parent]  # d_model vector (unit norm)
    
    # Both should be unit norm already (Bloom training enforces this)
    pi_dec = torch.dot(D_neg / D_neg.norm(), D_parent / D_parent.norm()).item()
    
    # pi_dec < -0.3: decoder of f_neg opposes decoder of parent -> H1 supported
    # pi_dec ≈ 0: no geometric relationship -> H3/H5 more likely
    # pi_dec > 0: decoders aligned (would contradict the hedging mechanism)
    
    return {
        'f_neg': f_neg,
        'f_parent': f_parent,
        'pi_dec': pi_dec,
        'h1_decoder_confirmed': pi_dec < -0.3
    }
```

---

## Sub-phase 4d: Reconstruction Contribution Analysis

Does the anti-aligned feature HELP or HURT reconstruction?

```python
def ablation_analysis(model, sae, dataloader, f_neg, device='cuda'):
    """
    For tokens where f_neg fires:
    - Compute reconstruction error WITH f_neg: ||x - x_hat||^2
    - Compute reconstruction error WITHOUT f_neg: ||x - (x_hat - z_neg * D_neg)||^2
    
    If error_with < error_without: f_neg is a useful correction term (H1)
    If error_with > error_without: f_neg is harmful (H2/H4)
    """
    errors_with = []
    errors_without = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, cache = model.run_with_cache(batch.cuda(), names_filter="blocks.8.hook_resid_post")
            resid = cache["blocks.8.hook_resid_post"].reshape(-1, 768)
            
            # SAE forward pass
            z = torch.relu(resid @ sae.W_enc + sae.b_enc)
            x_hat = z @ sae.W_dec + sae.b_dec
            
            # Find tokens where f_neg fires
            fneg_acts = z[:, f_neg]  # (N,)
            fneg_mask = fneg_acts > 0
            
            if not fneg_mask.any():
                continue
            
            # Get relevant slices
            x_active = resid[fneg_mask]        # tokens where f_neg fires
            xhat_active = x_hat[fneg_mask]
            z_neg = fneg_acts[fneg_mask]       # f_neg activation values
            D_neg = sae.W_dec[f_neg]           # decoder direction of f_neg
            
            # error WITH f_neg (current reconstruction)
            err_with = (x_active - xhat_active).pow(2).sum(1)
            
            # error WITHOUT f_neg (ablate f_neg contribution)
            xhat_ablated = xhat_active - z_neg.unsqueeze(1) * D_neg.unsqueeze(0)
            err_without = (x_active - xhat_ablated).pow(2).sum(1)
            
            errors_with.extend(err_with.cpu().tolist())
            errors_without.extend(err_without.cpu().tolist())
    
    errors_with = np.array(errors_with)
    errors_without = np.array(errors_without)
    
    return {
        'mean_error_with': errors_with.mean(),
        'mean_error_without': errors_without.mean(),
        'fraction_helpful': (errors_with < errors_without).mean(),
        'mean_error_reduction': (errors_without - errors_with).mean(),
        'h1_helpful': errors_with.mean() < errors_without.mean()
    }
```

---

## Expected Outputs (save to results/phase4/)

- results/phase4/activation_frequencies.csv  (per feature: freq, mean_act, max_act)
- results/phase4/coactivation_stats_F{19139,18573,11969,20804}.csv
- results/phase4/parent_identification.csv  (putative parent for each anti-aligned feat)
- results/phase4/decoder_projections.csv  (pi_dec for each (neg, parent) pair)
- results/phase4/ablation_results.csv  (reconstruction error with/without)
- results/phase4/hypothesis_scorecard.md  (which hypothesis each test supports)
- figures/phase4/activation_frequency_comparison.png  (bar chart: 3sigma vs 6sigma features)
- figures/phase4/coactivation_heatmap_F19139.png  (top co-activating features)
- figures/phase4/reconstruction_ablation.png  (error with vs without, per feature)

---

## Hypothesis Scorecard (pre-registered)

Build this table as results come in:

Test                        | H1(Hedge) | H2(Dead) | H3(Interf) | H4(Ghost) | H5(Comp)
Activation frequency        |    ?      |    ?     |     ?      |     ?     |    ?
Single dominant parent      |    ?      |    ?     |     ?      |     ?     |    ?
pi_dec < -0.3               |    ?      |    ?     |     ?      |     ?     |    ?
Error reduced when active   |    ?      |    ?     |     ?      |     ?     |    ?
Asymmetry ratio > 10        |    ?      |    ?     |     ?      |     ?     |    ?

After filling in, write a narrative in hypothesis_scorecard.md explaining which
hypothesis best accounts for ALL observations (not just cherry-picked tests).

---

## Data Requirements

Corpus: 10M tokens for activation measurement
  Options:
    - The Pile (same dataset as Bloom training): most principled
    - OpenWebText: accessible alternative, likely similar results
    - Note: 10M tokens at 128 seq length = ~78K sequences. 30-60 min of compute.
  
  Use the SAME corpus for 4a, 4b, 4c, 4d to ensure consistency.
  Save the token dataset seed for reproducibility.
