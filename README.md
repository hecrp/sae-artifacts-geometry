# The Geometry of SAE Artifacts: Anti-Aligned Encoder-Decoder Features

In a sparse autoencoder (SAE), feature *i* has an encoder vector E_i, which detects a direction in the residual stream, and a decoder vector D_i, which writes a direction back. The two are usually positively aligned. This project studies the small population of features whose encoder and decoder point in *opposite* directions (cos(E_i, D_i) < 0), which we call **anti-aligned features**. We also study how the encoder-decoder relationship changes with depth.

The work uses publicly released SAEs only (GPT-2 Small, Gemma 2 2B/9B, Llama 3.1 8B) and small synthetic toy models. No new SAEs have been trained on real model activations yet.

**Status:** work in progress. Phases 0-2 and Supplements 1-5 are complete. Phases 3-5 (seed stability, co-activation case studies, architectural comparison) have not been run. In November 2026 every report table was checked against `results/`. Several numbers were corrected and Supplement 4 was rewritten; [`docs/ERRATA.md`](docs/ERRATA.md) lists what changed and why.

## Main observations

All numbers are from `results/`. SAEs: Bloom `gpt2-small-res-jb` (12 layers, 24,576 features each), Gemma Scope 2B `width_16k/canonical` (26 layers), Llama Scope LXR-8x (32 layers, 32,768 features each).

1. **Depth-dependent sign flip in GPT-2 Small.** The Pearson correlation between encoder norm and cos(E, D) goes from +0.62 at layer 0 to between -0.64 and -0.76 from layer 5 onward. The sign changes between layers 3 and 4. In the Gemma Scope and Llama Scope SAEs the correlation is negative at every layer (early/late means of -0.88/-0.77 for Gemma and -0.84/-0.76 for Llama). The flip is specific to the ReLU+L1 SAEs in this sample. ([Phase 0](reports/PHASE0_REPORT.pdf))

2. **Anti-aligned tails exist in two of the three SAE families.** These are features that have cos(E, D) < 0 and are also at least 6 standard deviations below the layer mean:

   | Model | Mean cos(E, D) | cos < 0 | cos < 0 and <= mu - 6 sigma | Where |
   |-------|----------------|---------|------------------------------|-------|
   | GPT-2 Small (ReLU+L1) | 0.37 | 196 | 35 | layers 4-11 |
   | Llama 3.1 8B (JumpReLU, scalar threshold) | 0.48 | 232 | 29 | layers 0-2 (plus one at L7) |
   | Gemma 2 2B (JumpReLU, per-feature threshold) | 0.75 | 1 | 1 | layer 1 |

   The depth pattern is reversed between GPT-2 (late) and Llama (early). In Gemma 2 9B (22 layers checked) no feature has cos(E, D) < 0; the minimum is 0.16. In the Llama 32x SAEs the early-layer tail is larger (172 6-sigma features at L1). ([Phase 0](reports/PHASE0_REPORT.pdf), [Supplement 1](reports/SUPPLEMENT1_REPORT.pdf), [Supplement 2](reports/SUPPLEMENT2_REPORT.pdf))

3. **The tails are not explained by the fitted null distributions.** For each layer, Gaussian, Beta, Student-t and two-component mixture models were fitted to the cos(E, D) distribution. The observed 6-sigma counts exceed the best-fitting null's expectation by more than 10x in 7 of 8 GPT-2 layers that have such features, and in 3 of 4 such Llama layers. ([Phase 1](reports/PHASE1_REPORT.pdf))

4. **Toy models produce anti-alignment without an L1 penalty.** In a 2-D model with two superposed features, a ReLU SAE trained with lambda = 0 learns one anti-aligned feature (min cos = -0.79 at theta = 30 degrees). In a 10-D model, TopK (k = 5) SAEs produced 3-6 anti-aligned features per seed, versus 0 for ReLU+L1. Both scalar and per-feature JumpReLU also produced them. Each comparison uses 3 seeds, so these are existence results in toy settings, not estimates of effect sizes. ([Phase 2](reports/PHASE2_REPORT.pdf))

5. **Toy-to-real bridge.** On the real GPT-2 L8 SAE, 3 of the 4 reference anti-aligned features (F19139, F18573, F11969) pass a bisector test for the "hedge" geometry predicted by the toy model. All four point to the same putative parent cluster {F17487, F21174, F16152}. In decoder space, F19139 and F11969 are strongly opposed to that cluster (cosines -0.48 to -0.99), while F18573 is not. In the toy model, the probability of at least one anti-aligned feature rises sharply with expansion ratio: 0 at 4x or less, 0.45 at 6x, 0.80 at 8x and 1.0 at 12x or more (20 seeds per ratio). The toy counts are not quantitatively comparable to the real SAE (25.9% of a 320-feature toy dictionary versus 0.016% at GPT-2 L8). ([Supplement 3](reports/SUPPLEMENT3_REPORT.pdf), [Supplement 4](reports/SUPPLEMENT4_NEURONPEDIA_REPORT.pdf))

6. **The most extreme GPT-2 anti-aligned features are dead.** All seven strongly anti-aligned GPT-2 features queried on Neuronpedia (the four in layer 8 and the three most extreme overall, in layers 9-10) are dead or nearly dead (activation frequency at most 3.2e-7). The three parent features are also dead. F19139 and parent F21174 have nearly antiparallel decoders (cosine -0.99), so their logit effects are inverted (Pearson r = -0.985 over the vocabulary). This follows from the weights and is not evidence of inference-time correction. The only active anti-aligned feature found is Gemma 2B F3378 (layer 1, 1.7% of tokens). ([Supplement 4](reports/SUPPLEMENT4_NEURONPEDIA_REPORT.pdf))

7. **Interpretive update.** [Supplement 5](reports/SUPPLEMENT5_GOODFIRE_NEURAL_GEOMETRY_REPORT.pdf) adds hypothesis H7: some anti-aligned features may be local corrective patches of curved concept manifolds that a global sparse dictionary fragments. First diagnostics (local PCA on decoder neighbourhoods against matched random baselines) show that the hedge and parent clusters are geometrically atypical. The pattern looks more like two tight, opposed clusters than a smooth higher-dimensional manifold, so H7 is untested beyond this point. See `results/supplement5/`.

## Hypotheses

| ID | Hypothesis | Current assessment |
|----|------------|--------------------|
| H1 | Subtractive hedges: anti-aligned features are correction terms for an absorbed parent concept. | The weight geometry fits for F19139 and F11969 (bisector test, decoder cosines), but hedges and parents are dead, so no inference-time role has been observed. Not L1-specific in toy models. |
| H2 | Dead-feature decay: rarely active features whose encoder has drifted. | Supported for the most extreme GPT-2 features (7 of 7 dead or nearly dead on Neuronpedia). Needs confirmation on a fixed corpus (Phase 4a) and for less extreme features. |
| H3 | Superposition interference. | Consistent with cross-architecture presence; not directly tested. |
| H4 | Ghost-gradient artefacts. | Cannot explain the Llama tail (no ghost gradients there). Could contribute to the dead GPT-2 hedge-parent system; untested. |
| H5 | Compositional artefacts. | Not directly tested. |
| H6 | Model size explains the early-layer Llama tail. | Rejected: Gemma 2 9B has no negative-cosine features. |
| H7 | Manifold fragmentation by the SAE dictionary. | Plausible framing; first diagnostics inconclusive. |

The full research plan and decision gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Repository layout

```
Makefile          make reports | make check | make clean
reports/          Per-phase and supplement reports (LaTeX sources and compiled PDFs)
scripts/          Analysis scripts, one per phase/supplement
scripts/neuronpedia/  Neuronpedia API clients and queries (Supplement 4)
results/          Numerical outputs (CSV, JSON, per-layer cosine arrays as .npy)
figures/          Figures used in the reports (PNG and PDF)
docs/ROADMAP.md   Research plan, hypotheses and decision gates
docs/ERRATA.md    Corrections made to the reports and notes
docs/phases/      Pre-registered phase plans and running lab notes
docs/literature/  Literature summaries and BibTeX
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root and write to `results/<phase>/` and `figures/<phase>/`.

| Phase | Script | Needs |
|-------|--------|-------|
| 0: cross-model cosine statistics | `scripts/phase0_main.py` | Downloads GPT-2, Gemma Scope and Llama Scope SAE weights via SAELens / Hugging Face (tens of GB) |
| 1: distributional analysis | `scripts/phase1_main.py` | Same SAEs as Phase 0 (cached) |
| Supplement 1: Gemma 2 9B, Llama 32x | `scripts/supplement1_extended_models.py` | Additional SAE downloads |
| Supplement 2: Gemma 2 9B full depth | `scripts/supplement2_gemma9b_full.py`, `scripts/supplement2_stream.py` (low-disk variant) | Additional SAE downloads |
| 2: toy models | `scripts/phase2_toy_model.py`, `scripts/phase2_analytical_and_highK.py` | CPU only |
| Supplement 3: toy-to-real bridge | `scripts/supplement3_toy_real_bridge.py` | GPT-2 L8 SAE + CPU |
| Supplement 4: geometry, logits, activity | `scripts/supplement4_verified_analysis.py` | GPT-2 + GPT-2 L8 SAE (CPU) and the stored Neuronpedia snapshot |
| Supplement 4: Neuronpedia queries | `scripts/neuronpedia/` (see its README) | Neuronpedia API key |
| Supplement 5: local PCA diagnostics | `scripts/supplement5_local_pca_no_training.py`, `scripts/supplement5_random_baseline_local_pca.py` | GPT-2 L8 SAE + the Neuronpedia neighbour-graph files already in `results/supplement5/` |

Gemma Scope SAEs require accepting the Gemma licence on Hugging Face; set `HF_TOKEN` in the environment.

`make reports` builds every report with `pdflatex` and `bibtex`; `make check` byte-compiles all scripts.

Reproducibility check (November 2026): re-running Phase 2 (2A analytical, 2B, 2C, 2E), Supplement 3 (all three parts, about 20 minutes on CPU), the Supplement 4 analysis and both Supplement 5 scripts reproduced the stored CSV/JSON outputs exactly. Phase 0, Phase 1 and Supplements 1-2 were not re-run, because they need tens of GB of SAE downloads; their report tables were checked against the stored outputs.

## Known limitations

- All real-model results come from a single public SAE per model and layer. Seed stability (Phase 3) has not been tested, so it is unknown whether specific anti-aligned features or only their count are stable.
- Activation frequencies and co-activation statistics on a fixed corpus (Phase 4) have not been measured. Claims about dead features rely on Neuronpedia's activation sample (March 2026 snapshot).
- Comparisons across SAE families differ in more than one variable (architecture, threshold parameterisation, model, training data, expansion ratio).
- Toy-model experiments use few seeds and low dimensions.
- `scripts/neuronpedia/feature_registry.py` lists features from a truncated top-10-per-layer table; see Errata E1.
- Some generating scripts were not preserved: those for the Neuronpedia neighbour graph, the F1803 audit and the H7 evidence tables in `results/supplement5/`, and for `results/supplement4/deep_analysis_results.json`. Those outputs are kept as recorded, with notes where labels were corrected.

## Relation to prior work in this line

This project grew out of earlier exploratory work on encoder-decoder cosine distributions and the layer-0 MLP of GPT-2 Small. That work is not included here. Where results from it are cited (for example the F1803 temporal-feature audit in `results/supplement5/`), the source files are marked as not included.

## License

Code is released under the MIT License (see `LICENSE`). Reports, documentation and figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party papers referenced in `docs/literature/` are not redistributed.

## Citation

See `CITATION.cff`.
