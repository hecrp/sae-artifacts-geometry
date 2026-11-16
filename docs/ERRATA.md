# Errata

This file records errors found after the reports and notes were first written. In November 2026 the affected reports in `reports/` were corrected, and Supplement 4 was withdrawn and rewritten (E6). Each corrected report has a correction or revision note below its abstract. All tables in Phase 0-2 and Supplements 1-3 were checked against `results/` and match the stored outputs; the errors below are the ones found. The running notes in `docs/phases/` are kept as recorded, with `[see ERRATA E<n>]` markers. Numbers below were recomputed from the full per-layer cosine arrays in `results/phase1/cos_sim_*_L*.npy` and from the Bloom GPT-2 layer-8 SAE weights.

## E1. The "86 anti-aligned features" count was an artefact of truncation

**Original claim.** GPT-2 Small and Llama 3.1 8B each have 86 "absolutely anti-aligned" features (cos(E, D) < 0 at or beyond 6 sigma), 173 in total across the three models.

**Cause.** `results/phase0/extreme_features.csv` stores only the 10 most extreme features per layer: 120 rows for GPT-2 (12 layers) and 320 for Llama (32 layers). 86 is the number of rows with cos(E, D) < 0 *in that truncated table* for each model. The two values being equal is a coincidence.

**Corrected counts.**

| Model | SAE | Features with cos(E, D) < 0 | cos(E, D) < 0 and <= mu - 6 sigma | Layers with 6-sigma negative features |
|-------|-----|-----------------------------|-----------------------------------|----------------------------------------|
| GPT-2 Small | Bloom `gpt2-small-res-jb`, 24,576 per layer | 196 | 35 | L4 (2), L5 (10), L6 (6), L7 (3), L8 (4), L9 (1), L10 (3), L11 (6) |
| Llama 3.1 8B | Llama Scope LXR-8x, 32,768 per layer | 232 | 29 | L0 (12), L1 (14), L2 (2), L7 (1) |
| Llama 3.1 8B | Llama Scope LXR-32x, 131,072 per layer (L0, L1, L10 only) | 788 | 207 | L0 (35), L1 (172) |
| Gemma 2 2B | Gemma Scope 16k canonical | 1 | 1 | L1 (1) |
| Gemma 2 9B | Gemma Scope 16k (22 of 42 layers) | 0 | 0 | - |

Remarks:

- In GPT-2 and Llama every 6-sigma feature also has a negative cosine. In Gemma 2B, 16 of the 17 6-sigma features have a positive cosine.
- For GPT-2, most of the 196 negative-cosine features are in L1 (72) and L2 (36). Those layers have wide distributions, so these features are not 6-sigma outliers. "Concentrated in late layers" holds for the 6-sigma population only.
- The qualitative depth contrast (GPT-2 6-sigma tail in L4-L11, Llama 6-sigma tail in L0-L2) is unaffected.

**Corrected in:** `reports/PHASE0_REPORT`, `PHASE2_REPORT`, `SUPPLEMENT1_REPORT`, `SUPPLEMENT2_REPORT` (Table "Complete comparison" now reports all cos < 0 counts). Supplement 4 was rewritten (E6). Also corrected in `scripts/phase2_toy_model.py` (console and figure text) and `scripts/supplement2_stream.py`, which now computes the counts from the Phase 1 arrays instead of using hard-coded constants. `figures/supplement2/neg_cos_comparison` was regenerated. `scripts/neuronpedia/feature_registry.py` still lists the 86 + 1 + 86 features from the truncated tables, and its docstring now says so.

`figures/phase2/phase2e_jumprelu_comparison` was regenerated with the corrected title. Re-running Phase 2e reproduced `results/phase2/jumprelu_comparison.csv` exactly.

## E2. The Supplement 3 "quantitative match" with the real SAE did not hold

**Original claim.** The 10-D toy model at 32x expansion (n_neg = 82.95 +/- 4.96) closely matches the "~86 anti-aligned features" of the Bloom GPT-2 layer-8 SAE. The report presented this as strong evidence for a shared mechanism and used it to rule out H_UNSTABLE for Phase 3.

**Correction.** The two quantities are not comparable:

1. The real GPT-2 layer-8 SAE has 4 features with cos(E, D) < 0, not ~86 (see E1).
2. The toy count is out of 320 dictionary features (25.9%). The real count is out of 24,576 (0.016% at L8, or 0.066% across all 12 layers).

The match claim has been removed from `reports/SUPPLEMENT3_REPORT`; the rewritten Supplement 4 does not repeat it. The Phase 3 power argument is now stated for the toy model only: H_UNSTABLE is "unlikely", not "ruled out". Supplement 3 still supports these points: in the toy model the probability of at least one anti-aligned feature rises sharply with expansion ratio, and lambda has no measurable effect at 3x.

## E3. Decoder-decoder cosines in Supplement 4 were wrong

**Original claim.** Supplement 4 reported decoder-decoder cosines for the parent cluster and the hedge family (Table "Decoder-decoder cosine similarity matrix", inversion tables, per-feature profiles). It described a "clean two-cluster structure" with hedge cluster {F19139, F11969, F18573} and cross-cluster cosines between -0.705 and -0.091.

**Cause.** Several values were taken from the Supplement 3 bisector test, which measures cosines against a different partner feature, and were attached to the wrong pairs. Others did not match the SAE weights at all. The issue was first noticed in the Supplement 5 local-PCA audit.

**Corrected values** (computed from `W_dec` of `gpt2-small-res-jb`, `blocks.8.hook_resid_pre`):

|        | F17487 | F21174 | F16152 | F19139 | F11969 | F18573 |
|--------|--------|--------|--------|--------|--------|--------|
| F17487 | -      | 0.695  | 0.671  | -0.705 | -0.480 | -0.511 |
| F21174 | 0.695  | -      | 0.939  | -0.990 | -0.863 | 0.158  |
| F16152 | 0.671  | 0.939  | -      | -0.908 | -0.926 | 0.224  |
| F19139 | -0.705 | -0.990 | -0.908 | -      | 0.842  | -0.148 |
| F11969 | -0.480 | -0.863 | -0.926 | 0.842  | -      | -0.499 |
| F18573 | -0.511 | 0.158  | 0.224  | -0.148 | -0.499 | -      |

Consequences:

- Cross-cluster opposition between the parent cluster and {F19139, F11969} is much stronger than reported (-0.48 to -0.99).
- F18573 is not part of the hedge cluster in decoder space, although it passes the encoder-side bisector test in Supplement 3.
- The logit-inversion counts remain strongly, but not perfectly, rank-correlated with the corrected decoder cosines (Spearman rho = 0.94 over six pairs, recomputed from the weights).

**Resolution:** superseded by the rewrite of Supplement 4 (E6), which computes the matrix directly from `W_dec` (`figures/supplement4/s4_decoder_cosine_matrix`).

## E4. Threshold type was presented as causal

Supplements 1 and 2 concluded that per-feature JumpReLU thresholds (Gemma Scope) *protect against* anti-alignment and called this "definitively confirmed" (hypothesis E2). Phase 2 then found that both scalar and per-feature JumpReLU produce anti-aligned features in toy models (3 seeds each). The Gemma-Llama difference is an empirical observation about specific public SAE families, which differ in more than the threshold parameterisation. `reports/SUPPLEMENT1_REPORT` and `SUPPLEMENT2_REPORT` now describe it as correlational. The rejection of H6 (model size) stands.

## E5. Overstated toy-model conclusion in Phase 2

`reports/PHASE2_REPORT` stated that Experiment 2C "definitively rules out L1 as the causal mechanism". The experiment shows that an L1 penalty is not *required* for anti-alignment in a 10-D toy setting with 3 seeds per condition. The sentence has been revised accordingly.

## E6. Supplement 4 (March 2026 version) withdrawn and rewritten

An audit of Supplement 4 against the SAE weights and the stored Neuronpedia snapshot (`results/supplement4/full_analysis_20260304_203128.json`) found errors that affected its main conclusions:

1. **Mislabelled comparison features.** Four randomly selected layer-8 features (F5977 "pent", F22561 "Soros", F6165 "Walsh", F3696 "Dem") were presented as "active anti-aligned features". The query script lists them as a random sample. Their cos(E, D) is 0.33-0.44, at the layer mean (-0.5 to +0.9 sigma).
2. **Feature-profile table without a data source.** The table "Dead vs. active breakdown" gave cos(E, D) and sigma values that match neither the weights nor Phase 0. Examples: F22561 was listed as -0.418 at -10.02 sigma (actual +0.347); the parents as +0.25 to +0.32 (actual +0.48 to +0.82); Gemma F3378 as -0.312 with frequency 2.6e-4 and explanation "repetition, lists" (actual -0.465, frequency 1.7e-2, "references to software libraries and coding frameworks"). Several listed features (F1579, F7893, F9283, F4421, F14988, F15702, F23001, F100, F500, F1000) were not queried in any stored Neuronpedia run, and none of them is anti-aligned in layer 8 (cos(E, D) between 0.23 and 0.46).
3. **Layer-mixed explanation analysis.** `download_and_analyze_explanations.py` looked up layer-8 explanations for feature indices that belong to layers 4, 6, 7, 9, 10 and 11, so 11 of its 15 "anti-aligned" entries described unrelated features. The output (`s3_explanations_analysis.json`) was withdrawn and the script was fixed.
4. **Wrong decoder cosines** (E3).
5. **Overstated logit-inversion evidence.** The inversion is a linear consequence of nearly antiparallel decoders. It was presented as independent "functional" confirmation of the hedge mechanism.

**Consequences for the conclusions.**
- "Anti-alignment is not synonymous with feature death" is not supported for GPT-2. All seven strongly anti-aligned GPT-2 features queried are dead or nearly dead (frequency <= 3.2e-7), and so are the three parent features. The only active anti-aligned feature found is Gemma 2B F3378.
- "Anti-aligned features share no semantic theme" is not supported. It was based on the random features. Among the anti-aligned features, only F19139 and Gemma F3378 have explanations.
- H2 (dead-feature decay) is strengthened for the most extreme GPT-2 anti-aligned features.

**Resolution.** The original report, its seven figures and two Markdown analyses are no longer in the repository. `reports/SUPPLEMENT4_NEURONPEDIA_REPORT` is a new 6-page report whose numbers all come from `scripts/supplement4_verified_analysis.py`. The comments in `scripts/neuronpedia/run_full_analysis.py` that described non-anti-aligned features as "extreme" were corrected.

## E7. Supplement 5 inherited the mislabelled group

The Supplement 5 local-PCA scripts and Neuronpedia neighbour-graph outputs used the same four random layer-8 features as an "active anti-aligned" group. The group was renamed `random_l8`. Both preserved Supplement 5 scripts were re-run, and their numerical outputs are identical to the originals apart from the label. Outputs of the unpreserved Supplement 5 scripts were relabelled by hand and carry a note. Prediction P4 of the H7 evidence table ("active anti-aligned features sit in dense neighbourhoods") is now marked untested. The conclusion that local dimensionality is not specific to anti-aligned features stands, and is if anything strengthened, because the comparison group was random. `reports/SUPPLEMENT5_GOODFIRE_NEURAL_GEOMETRY_REPORT` has a correction note, and its statements about Supplement 4 were revised.
