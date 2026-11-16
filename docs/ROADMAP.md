# Research plan

Version 3.1 (November 2026). Supersedes the March-May 2026 working versions.

## Question

Why do some SAE features have encoders and decoders that point in opposite directions (cos(E, D) < 0)? Is this a property of the model's representations, an artefact of the SAE training objective or architecture, or a property of specific training runs?

Two empirical signatures are studied separately:

1. **Sign flip.** The correlation r(‖E‖, cos(E, D)) changes sign with depth in GPT-2 Small ReLU+L1 SAEs. It is negative at every depth in the JumpReLU SAEs examined.
2. **Anti-aligned tail.** A small population of features has cos(E, D) < 0 at 6 or more standard deviations below the layer mean. It appears in late GPT-2 layers and early Llama 3.1 8B layers, and is almost absent in Gemma Scope.

Current results and hypotheses are summarised in the top-level `README.md`. Corrections are in `docs/ERRATA.md`.

## Phases and decision gates

Each phase had a pre-registered gate, written before the phase was run (see `docs/phases/PHASE*_*.md`).

| Phase | Content | Compute | Status | Gate outcome |
|-------|---------|---------|--------|--------------|
| 0 | Cross-model verification of the sign flip and the tail | none (public SAE weights) | done | G0-PARTIAL: sign flip GPT-2 only; tail present in GPT-2 and Llama |
| 1 | Distributional analysis (histograms, moments, Gaussian/Beta/Student-t/mixture fits, tail p-values) | none | done | G1-YES for all three models |
| S1, S2 | Gemma 2 9B and Llama 32x SAEs; LXR threshold architecture | none | done | H6 (model size) rejected |
| 2 | Toy models: 2-D analytical, 10-D lambda sweep, TopK, JumpReLU | CPU | done | G2-YES: anti-alignment arises without L1 |
| S3 | Toy-to-real bridge: bisector test, expansion-ratio sweep, seed distribution | CPU | done | see Errata E2 |
| S4 | Hedge-family geometry, logit effects and activity (rewritten Nov 2026) | API snapshot + CPU | done | most extreme GPT-2 anti-aligned features and their parents are dead (E6) |
| S5 | Neural-geometry framing (H7), local PCA diagnostics | CPU | done (theory and first diagnostics) | Supplement 6 deferred |
| 3 | Seed stability | GPU | not started | G3 pending |
| 4 | Activation frequency and co-activation case studies | GPU (forward passes) | not started | - |
| 5 | Architectural comparison (TopK, scalar vs per-feature threshold) | low to medium | not started | - |

## Remaining phases

### Phase 3: seed stability

Train three additional SAEs on GPT-2 Small layer 8 with the Bloom hyperparameters (seeds 1337, 42, 7). Compute pairwise mean-max cosine (PW-MCC; Song et al. 2025) between decoder matrices. For each reference feature (F19139, F18573, F11969, F20804), find its best decoder match in each new SAE and check whether the match is also in the 3-sigma tail.

Gate G3:

- H_STABLE: at least 3 of the 4 reference features match in at least 2 of the 3 seeds. Specific features are stable.
- H_STAT_STABLE: each seed has 2-8 strongly negative features, but fewer than 3 of 4 reference features match. The tail is a statistical property, not a set of specific features.
- H_UNSTABLE: fewer than 2 strongly negative features in at least 2 of the 3 seeds.

Supplement 3 found that in toy models the probability of at least one anti-aligned feature approaches 1 at high expansion ratios, so H_UNSTABLE is unlikely. H_STAT_STABLE is the expected outcome.

Requirements: SAELens version pinned to reproduce Bloom training; the same 300M-token subset of The Pile; about 2-3 A100-days per SAE. Record whether matched anti-aligned features are dead in each run, and train at least one run without ghost gradients to test H4.

### Phase 4: co-activation case studies

- 4a. Activation frequency, mean and maximum over 10M tokens for every feature in the 3-sigma tail. This distinguishes H1 (active hedges) from H2 (dead features). Supplement 4 found all seven strongly anti-aligned GPT-2 features queried on Neuronpedia to be dead or nearly dead, so 4a is now the first priority: it decides whether the anti-aligned population is mainly dead features, and whether any active ones exist to test H1.
- 4b. Co-activation: P(f_j > 0 | f_neg > 0) and PMI(f_neg, f_j) for all j. The putative parent is argmax_j PMI. Pre-registered threshold: an asymmetry ratio above 10x indicates an H1-type relationship.
- 4c. Decoder projection onto the putative parent. Pre-registered threshold: pi_dec < -0.3.
- 4d. Reconstruction contribution: reconstruction error with and without z_neg * D_neg on tokens where the feature fires.
- H7 diagnostics: compare geodesic and chord distances on activation neighbourhoods; local PCA on co-activation geometry.

Compute: about 2-4 GPU-hours of forward passes on GPT-2 Small.

### Phase 5: architectural comparison

Main question: is the Gemma-Llama difference caused by threshold parameterisation (per-feature vs scalar), or by other differences between the SAE families? The toy results in Phase 2 do not settle this (Errata E4).

- Option A: train matched BatchTopK and JumpReLU SAEs (scalar and per-feature thresholds) on GPT-2 Small layer 8, keeping all other hyperparameters fixed.
- Option B (no training): use public TopK SAEs (e.g. Llama Scope TopK variants, once accessible) and public JumpReLU variants.

Belrose et al. (2025) report higher seed variability for TopK SAEs, so Phase 5 comparisons should include more than one seed per architecture.

## Data and models

- GPT-2 Small: `jbloom/GPT2-Small-SAEs` (`gpt2-small-res-jb`). Residual stream, 24,576 features per layer (32x), ReLU + L1 (lambda = 8e-5), ghost gradients.
- Gemma 2 2B / 9B: Gemma Scope residual SAEs (`width_16k/canonical`). JumpReLU with per-feature thresholds.
- Llama 3.1 8B: Llama Scope residual SAEs (LXR-8x and 32x). JumpReLU with an effectively scalar threshold.

## Compute summary

| Phase | Compute |
|-------|---------|
| 0-2, S1-S5 | negligible (weight downloads and CPU analysis) |
| 3 | 3 SAE training runs, about 2-3 A100-days each |
| 4 | about 2-4 GPU-hours |
| 5 | 0 (option B) to 1-3 SAE training runs (option A) |
