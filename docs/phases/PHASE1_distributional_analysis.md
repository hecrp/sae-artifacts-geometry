# Phase 1: Distributional Analysis
# Priority: P1 | Duration: 3-5 days | Compute: ZERO | Gate: G1
# REVISED (post Phase 0, March 2026)

## Objective
Establish whether the observed anti-aligned features in each model are statistically
anomalous relative to a reasonable distributional null model, or are simply expected
extreme values under a heavy-tailed distribution.

Phase 0 PROVIDED the key head-start: skewness and excess kurtosis are already measured
for all 70 layers. This transforms Phase 1 from a "compute from scratch" task into a
"formal statistical modeling on already-rich data" task.

The question differs by model:
  GPT-2 late layers: is the 6sigma tail anomalous despite positive kurtosis (2-3)?
  Llama early layers: is the 6sigma tail anomalous despite EXTREME kurtosis (5-7)?
  Gemma: are the 6sigma features (POSITIVE cos) anomalous within Gemma's distribution?

---

## What Phase 0 Already Gave Us

### GPT-2 small -- full 12-layer statistics (ALL metrics computed):
  L0:  mu=0.556, sigma=0.239, skew=-0.95, kurt=-0.48, r=+0.621, 3sig=0,   6sig=0
  L1:  mu=0.360, sigma=0.139, skew=-0.92, kurt=-0.04, r=+0.432, 3sig=5,   6sig=0
  L2:  mu=0.332, sigma=0.125, skew=-0.64, kurt=-0.40, r=+0.493, 3sig=3,   6sig=0
  L3:  mu=0.333, sigma=0.104, skew=-0.76, kurt=+0.52, r=+0.397, 3sig=50,  6sig=0
  L4:  mu=0.348, sigma=0.075, skew=-0.43, kurt=+2.58, r=-0.260, 3sig=321, 6sig=2
  L5:  mu=0.356, sigma=0.071, skew=+0.34, kurt=+2.22, r=-0.638, 3sig=62,  6sig=10
  L6:  mu=0.369, sigma=0.076, skew=+0.61, kurt=+1.51, r=-0.736, 3sig=11,  6sig=6
  L7:  mu=0.379, sigma=0.079, skew=+0.68, kurt=+1.21, r=-0.756, 3sig=7,   6sig=3
  L8:  mu=0.368, sigma=0.079, skew=+0.69, kurt=+2.03, r=-0.747, 3sig=5,   6sig=4
  L9:  mu=0.368, sigma=0.079, skew=+0.65, kurt=+2.13, r=-0.756, 3sig=4,   6sig=1
  L10: mu=0.368, sigma=0.076, skew=+0.49, kurt=+3.36, r=-0.712, 3sig=8,   6sig=3
  L11: mu=0.349, sigma=0.075, skew=+0.63, kurt=+1.90, r=-0.639, 3sig=40,  6sig=6

Key patterns:
  - Early layers (L0-L3): kurtosis near 0 or negative, light-tailed
  - Late layers (L4-L11): kurtosis 1.2-3.4, HEAVY-TAILED -> this matters for G1
  - The 6sigma features (35 total) are concentrated in L4-L11

### Llama 3.1 8B -- key early layers:
  L0: mu=0.365, sigma=0.100, skew=+1.54, kurt=+7.42, r=-0.683, 3sig=156, 6sig=12
  L1: mu=0.383, sigma=0.119, skew=+1.28, kurt=+5.90, r=-0.786, 3sig=119, 6sig=14
  L2: mu=0.414, sigma=0.134, skew=+1.68, kurt=+2.98, r=-0.854, 3sig=13,  6sig=2
  L3: mu=0.449, sigma=0.156, skew=+1.25, kurt=+0.66, r=-0.895, 3sig=2,   6sig=0

CRITICAL INSIGHT: Llama L0 kurtosis=7.42 is extreme. Under a Student-t distribution
with 3 degrees of freedom (which has kurtosis=6), the EXPECTED count of 6sigma
features would be dramatically higher than under Gaussian. Phase 1 must determine
whether the 12 features at L0 are anomalous or merely expected given this kurtosis.

### Gemma 2B -- selected layers from Phase 0:
  L0:  mu=0.534, sigma=0.158, skew=+1.70, kurt=+1.39, 6sig=0 [no neg-cos]
  L11: mu=0.778, sigma=0.080, skew=-0.77, kurt=+0.92, 6sig=3 [cos~0.18 is 7.5sigma below]
  L25: mu=0.823, sigma=0.077, skew=-1.10, kurt=+1.85, 6sig=4 [cos~0.28 is 7.1sigma below]
  -> Gemma 6sigma features have POSITIVE cosine (0.14-0.30), not negative.
     This means for Gemma, Phase 1 should assess whether cos~0.15-0.30 is anomalous
     for a distribution with mu=0.78, sigma=0.08. This is a fundamentally different
     question from "is cos<0 anomalous" in GPT-2.

---

## What Phase 1 Needs to Add

### Missing data: the RAW cos(E,D) arrays
Phase 0 saved statistics (mu, sigma, etc.) but NOT the raw distributions.
Phase 1 needs to reload each SAE and save the raw cos(E,D) arrays for fitting.

GOOD NEWS: All SAEs are cached in HuggingFace hub (~/.cache/huggingface/).
Reloading = fast (no download, just disk I/O), < 1 minute per SAE.

Plan: phase1_main.py loads each SAE, saves cos_sim arrays to:
  results/phase1/cos_sim_gpt2_L{layer}.npy
  results/phase1/cos_sim_gemma_L{layer}.npy
  results/phase1/cos_sim_llama_L{layer}.npy

---

## Analysis Protocol (5 steps, now with Phase 0 context)

### Step 1: Full histograms (log y-axis)

For EACH layer of EACH model, plot:
  - Histogram with log y-axis (reveals tail structure)
  - Mark mu, mu-3sigma, mu-6sigma thresholds
  - Annotate the extreme features (cos value + sigma deviation)
  
Key visualizations to create:
  a) GPT-2 late layers (L4-L11): log-scale histograms with 6sigma features marked
  b) Llama L0-L2: log-scale histograms -- does the distribution have a VISIBLE
     separate population at the negative tail, or is it a smooth tail?
  c) Gemma selected layers: emphasize that 6sigma features are POSITIVE-cosine

Interpretation test: if the Llama L0 distribution shows a continuous,
smooth tail extending to -0.4, the 6sigma features may not be a distinct population.
If there is a visible gap or bimodal structure, they are distinct.

### Step 2: Distributional model fitting

For EACH layer of EACH model, fit 3 null models:

MODEL A: Gaussian N(mu, sigma^2)
  - MLE: mu_hat = mean, sigma_hat = std
  - Log-likelihood at each data point
  
MODEL B: Beta (shifted to [-1, 1])
  - Transform y = (cos + 1) / 2, fit Beta(alpha, beta) to y in [0,1]
  - Jacobian adjustment for the [-1,1] transform
  
MODEL C: Two-component Gaussian Mixture
  pi * N(mu1, sigma1^2) + (1-pi) * N(mu2, sigma2^2)
  - Fit via EM algorithm or scipy.optimize
  - 5 parameters vs 2 (Gaussian), so BIC penalizes more
  
MODEL D [NEW -- motivated by Llama kurtosis]: Student-t
  - t(nu) distribution: captures heavy tails with a single df parameter
  - For Llama L0 with kurtosis~7.4, the fitted nu ~ 3 (since kurtosis = 6/nu-4 -> nu~3.4)
  - This is a cleaner heavy-tail model than Gaussian mixture for smooth distributions

Model comparison: BIC, AIC. Lower = better fit.
The WINNING MODEL determines the null against which we assess anomaly.

```python
from scipy import stats
from scipy.optimize import minimize
import numpy as np

def fit_all_models(cos_np):
    n = len(cos_np)
    mu = cos_np.mean()
    sigma = cos_np.std(ddof=1)
    
    # Model A: Gaussian
    ll_gauss = stats.norm.logpdf(cos_np, mu, sigma).sum()
    aic_gauss = 2*2 - 2*ll_gauss
    bic_gauss = 2*np.log(n) - 2*ll_gauss
    
    # Model B: Beta [-1, 1]
    y = (cos_np + 1) / 2
    y = np.clip(y, 1e-6, 1-1e-6)
    a, b, loc, scale = stats.beta.fit(y, floc=0, fscale=1)
    ll_beta = (stats.beta.logpdf(y, a, b) - np.log(2)).sum()
    aic_beta = 2*2 - 2*ll_beta
    bic_beta = 2*np.log(n) - 2*ll_beta
    
    # Model D: Student-t
    df_hat, loc_hat, scale_hat = stats.t.fit(cos_np)
    ll_t = stats.t.logpdf(cos_np, df_hat, loc_hat, scale_hat).sum()
    aic_t = 2*3 - 2*ll_t
    bic_t = 3*np.log(n) - 2*ll_t
    
    # Model C: Gaussian mixture (EM)
    # [see phase1_main.py for full EM implementation]
    
    return {
        'gaussian': {'ll': ll_gauss, 'aic': aic_gauss, 'bic': bic_gauss,
                     'params': {'mu': mu, 'sigma': sigma}},
        'beta':     {'ll': ll_beta, 'aic': aic_beta, 'bic': bic_beta,
                     'params': {'alpha': a, 'beta': b}},
        'student_t': {'ll': ll_t, 'aic': aic_t, 'bic': bic_t,
                      'params': {'df': df_hat, 'loc': loc_hat, 'scale': scale_hat}},
    }
```

### Step 3: Tail p-values under best-fitting model

For each observed extreme feature (all features at >=6sigma), compute:
  expected_count = n_features * P(X <= cos_observed | best_model)

CRITICAL CALCULATION FOR LLAMA L0:
  n = 32768, 12 features at cos <= mu - 6*sigma = 0.365 - 0.6 = -0.235
  Under Student-t(df~3.4): P(X <= -0.235 | t-distribution) = ?
  If expected_count >= 12 --> G1-NO for Llama early layers
  If expected_count << 12 --> G1-YES for Llama early layers

This is the KEY calculation that determines whether Llama early-layer anti-alignment
is a genuine mechanistic phenomenon or simply an expected heavy-tail effect.

FORMULA for expected count under Student-t:
```python
df, loc, scale = student_t_params
threshold = mu - 6 * sigma  # the 6sigma threshold value (in cos units)
p_below = stats.t.cdf(threshold, df, loc, scale)
expected_count = n_features * p_below
# Compare to observed_count
anomaly_ratio = observed_count / expected_count
# > 10x: G1-YES; 2-10x: G1-MARGINAL; ~1x: G1-NO
```

### Step 4: Q-Q plots

For each model (winning distributional model vs empirical data):
  - Focus on LEFT TAIL (negative cos end)
  - Departures in the left tail identify anomalous features

Plot for each combination: GPT-2 L5, L8, L9 | Llama L0, L1 | Gemma L11, L25

### Step 5: Cross-model tail comparison

Build a unified table:
  Model | Layer | Best-fit model | df (if t) | Expected 6sig | Observed 6sig | Anomaly ratio
  
This table is the central result of Phase 1 and the basis for the G1 decision.

---

## New Analysis: Gemma Threshold Investigation

Since Gemma has per-feature threshold and Llama has scalar threshold (key for H6/E2),
Phase 1 can extract threshold statistics from the SAE weights at zero extra cost:

```python
from sae_lens import SAE

def analyze_threshold(release, sae_id):
    sae = SAE.from_pretrained(release, sae_id, device='cpu')
    if hasattr(sae, 'threshold'):
        thresh = sae.threshold.detach().cpu().numpy()
        return {
            'has_threshold': True,
            'n_threshold_values': len(thresh) if thresh.ndim > 0 else 1,
            'threshold_mean': float(thresh.mean()),
            'threshold_std': float(thresh.std()),
            'threshold_min': float(thresh.min()),
            'threshold_max': float(thresh.max()),
        }
    return {'has_threshold': False}
```

For each model, record:
  Gemma 2B: thresh shape (16384,) -- per-feature, check std vs mean
  Llama (SAELens): thresh shape (32768,) -- check if constant (scalar broadcast)

If Gemma thresholds have high std (each feature has very different threshold) but
Llama thresholds are nearly constant -- this supports E2 (per-feature threshold
provides alignment protection that scalar threshold cannot).

---

## Gate G1 Decision Criteria (updated)

The gate decision applies SEPARATELY to each model's phenomenon:

### G1 for GPT-2 late layers (main paper):
  G1-YES:  Tail features anomalous under Student-t or best model with ratio > 10x
           -> Proceed with Phase 2 toy model + Phase 4 co-activation for GPT-2
  G1-MARGINAL: Ratio 2-10x
           -> Cautious proceed. Sign flip is still a valid contribution.
  G1-NO:   Tail fully explained by heavy-tailed distribution
           -> Reframe: study the distributional structure itself, not the tail.

### G1 for Llama early layers (new question):
  G1-YES:  12-14 features anomalous even under t(df~3) null
           -> Strong finding: early-layer anti-alignment is mechanistically special.
              Phase 4 co-activation on Llama L1 becomes high priority.
  G1-MARGINAL:
           -> Llama early-layer features may be heavy-tail noise.
              Focus analysis on GPT-2. Llama becomes context/motivation only.
  G1-NO:   Student-t explains L0-L1 tail entirely
           -> Important negative: Llama early-layer anomaly is distributional noise.
              The depth-inverted kurtosis (early in Llama, late in GPT-2) is still
              interesting but the tail features are not mechanistically special.
              Could still publish: "distributional structure of SAE cosine distributions."

### G1 for Gemma (separate question):
  Gemma's 6sigma features have cos>0 -- they are "relatively misaligned" not
  "absolutely anti-aligned." Phase 1 should report their p-value but these are
  NOT the primary subject of the paper. Likely G1-NO or G1-MARGINAL for Gemma
  (the phenomenon is different: it's tail of a high-mean distribution, not extreme negatives).

---

## Expected Outputs

results/phase1/
  cos_sim_gpt2_L{0-11}.npy          -- raw arrays (12 files, ~100KB each)
  cos_sim_gemma_L{0-25}.npy         -- raw arrays (26 files, ~65KB each)
  cos_sim_llama_L{0-31}.npy         -- raw arrays (32 files, ~128KB each)
  distributional_fits.csv           -- model comparison per layer per model
  tail_pvalues.csv                  -- expected vs observed counts, anomaly ratios
  threshold_stats.csv               -- JumpReLU threshold analysis (Gemma vs Llama)
  G1_decision_gpt2.md               -- gate decision for GPT-2
  G1_decision_llama.md              -- gate decision for Llama
  G1_decision_synthesis.md          -- overall Phase 1 synthesis

figures/phase1/
  hist_log_gpt2_L{5,8,9}.{png,pdf}  -- log-scale histograms, late GPT-2 layers
  hist_log_llama_L{0,1}.{png,pdf}   -- log-scale histograms, early Llama layers
  hist_log_gemma_L{11,25}.{png,pdf} -- Gemma late layers
  qq_gpt2_L{5,8,9}.{png,pdf}        -- Q-Q plots for GPT-2
  qq_llama_L{0,1}.{png,pdf}         -- Q-Q plots for Llama
  model_comparison_bic.{png,pdf}    -- BIC comparison across all layers x models
  anomaly_ratio_table.{png,pdf}     -- heatmap: observed/expected per layer per model
  threshold_distribution.{png,pdf}  -- Gemma vs Llama threshold statistics

reports/PHASE1_REPORT.tex           -- formal LaTeX report
reports/PHASE1_REPORT.pdf           -- compiled PDF

---

## Implementation Notes

### Data extraction approach
Phase 1 must re-extract raw cos_sim arrays. For efficiency, modify the Phase 0 approach
to SAVE the raw arrays. The phase1_main.py script:
1. First pass: load each SAE, compute cos_sim, save .npy, compute distributions, delete SAE
2. Second pass: load .npy files, fit distributional models, compute tail p-values
3. Third pass: generate all figures from .npy + CSV data

This is FAST: all SAEs are cached, the computation per SAE is < 5 seconds.

### Gaussian Mixture EM implementation
The mixture model needs careful initialization to avoid local minima:
  - Initialize mu1 near the overall mean, mu2 = mu1 - 4*sigma (near the 4sigma point)
  - Run 10 different random initializations, take best log-likelihood
  - Use scipy.optimize for 5-parameter optimization, NOT EM (simpler for small n_components=2)

### Focus layers for detailed analysis
Full analysis for ALL layers, BUT detailed manual review for:
  GPT-2: L4 (spike layer, 321 at 3sigma), L5 (peak 6sigma count=10), L9 (most extreme feature)
  Llama: L0 (kurtosis=7.42, 12 features), L1 (kurtosis=5.90, 14 features)
  Gemma: L25 (last layer, 4 features, 178 at 3sigma)

---

## Connection to Overall Paper

Phase 1 results directly update:
  - Table 1 in the project proposal: add skewness, kurtosis (already computed in Phase 0)
  - Section 5.3 (Model comparison): update with actual BIC/AIC numbers
  - The G1 gate (Section 5 boxed question): document the decision for all three models
  - Phase 0 report Table discussion: add tail p-value context to the raw counts

The Phase 1 result most likely to reshape the paper:
  If G1-YES for GPT-2 and G1-MARGINAL/NO for Llama:
    "The late-layer anti-alignment in GPT-2 is statistically anomalous and likely
    mechanistic. The early-layer Llama phenomenon, while visually similar, may be
    explained by the heavier-tailed distribution of JumpReLU early layers."
    
  If G1-YES for BOTH GPT-2 and Llama:
    "Anti-aligned features are statistically anomalous across architectures and
    depth positions, providing strong evidence for a universal but depth-variable mechanism."
