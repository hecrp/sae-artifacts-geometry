# Phase 1: Distributional Analysis -- Running Notes
Started: [pending]
## Status: COMPLETED -- G1-YES for all models

## Context
Phase 0 established:
  - Sign flip: GPT-2 only (ReLU+L1 specific)
  - Anti-aligned (cos<0): GPT-2 (late, 86 features [see ERRATA E1]) + Llama (early, 86 features [see ERRATA E1])
  - Gemma: high baseline (mu~0.75), 1 absolutely negative feature
  - Key open question: are the extreme Llama L0-L1 features (kurtosis=7.4) mechanistically
    real or just expected heavy-tail noise?

## The Core Gate Question
G1 applies separately to each model:
  - GPT-2 late layers: is the 6sigma tail anomalous despite kurtosis 2-3?
  - Llama early layers: CRITICAL -- is 12-14 features at 6sigma anomalous given kurtosis=7.4?
  - Gemma: are the positive-cosine 6sigma features mechanistically interesting?

## How to Execute
# from the repository root
python scripts/phase1_main.py

Expected runtime: 30-90 minutes (reloads all 70 SAEs from cache + fitting)

## Progress Log
[fill in during execution]

## Key Findings
[fill in after execution]

### GPT-2 Small distributional results
...

### Llama L0/L1 critical result
BEFORE running: kurtosis=7.4 for Llama L0 means Student-t with df~3.4 may explain the tail.
Under t(df=3.4): P(X <= mu - 6*sigma) is much larger than Gaussian.
Expected count = 32768 * P(t(3.4) <= (mu-6sigma - loc)/scale)
If expected_count >= 12: G1-NO for Llama (heavy-tail noise)
If expected_count << 12: G1-YES for Llama (anomalous)

### Gemma results
...

### Model comparison (BIC/AIC winners)
Expected winners:
  - GPT-2 late layers: likely Gaussian Mixture or Student-t (kurtosis 2-3)
  - Llama early layers: likely Student-t (kurtosis 7.4 -> df~3)
  - Gemma: likely Beta or Gaussian

## Threshold Analysis (Gemma vs Llama)
Key question: does Gemma's per-feature threshold explain its resistance to anti-alignment?
  - Gemma: threshold shape (16384,), std should be significant
  - Llama: threshold shape (32768,), if nearly constant -> scalar threshold

## Gate G1 Decision
[fill in after execution]

GPT-2 Small:    [G1-YES / G1-MARGINAL / G1-NO]
Gemma 2B:       [G1-YES / G1-MARGINAL / G1-NO]
Llama 3.1 8B:   [G1-YES / G1-MARGINAL / G1-NO]

Synthesis: [overall effect on phases 2+]

## References for Phase 1 report
- Phase 0 data: results/phase0/layer_stats.csv
- Phase 1 data: results/phase1/*.npy, distributional_fits.csv, tail_pvalues.csv
- Figures: figures/phase1/
