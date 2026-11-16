# Next Steps: Neural Geometry / H7 Without New SAE Training

Generated: 2026-05-09 22:23:43 
Paths below are relative to the repository root.

> Scope: continue H7 rigorously using existing local results, Neuronpedia, public SAE dashboards/metadata, public SAE weights if needed, and literature up to May 2026. Do **not** train new SAEs.

## Goal

Test H7 conservatively:

> Some anti-aligned SAE features may be local/corrective shards of curved or multidimensional conceptual manifolds fragmented by sparse linear dictionaries.

Do not try to prove H7 in one step. The goal is to produce falsifiable, lightweight diagnostics that can reject or constrain H7.

## Evidence discipline

Use these labels in all reports:

- **Observation:** directly measured from data/API/local files.
- **Inference:** reasonable interpretation of observations.
- **Hypothesis:** proposed explanation requiring additional tests.
- **Prediction:** testable expected result under H7.
- **Speculation:** plausible but currently weak idea.

Required conservative wording:

- Say “compatible with H7”, “motivates H7”, or “supports testing H7”.
- Do **not** say “H7 demonstrated” unless at least two independent diagnostics pass and alternatives are addressed.

## Available no-training resources

### Local results

- `results/supplement3/bisector_test_results.csv`
- `results/supplement3/expansion_ratio_sweep.csv`
- `results/phase0/extreme_features.csv`
- `results/phase1/*.npy`, `threshold_stats.csv`, `distributional_fits.csv`
- `results/supplement4/full_analysis_20260304_203128.json`
- `results/supplement4/reference_features_20260304_201510.json`
- `results/supplement4/s3_explanations_analysis.json`
- `results/supplement5/h7_preliminary_neuronpedia_audit.json`

### External/public resources

- Neuronpedia feature pages/API for GPT-2 and Gemma.
- Public Bloom GPT-2 SAEs via SAELens / HuggingFace.
- Public Gemma Scope / Llama Scope metadata where accessible.
- Goodfire Neural Geometry and arXiv literature listed in Supplement 5.

## Workstream A: Expand Neuronpedia neighborhood graph

### Objective

Build a larger local graph around:

- Hedge family: F19139, F18573, F11969, F20804
- Parent cluster: F17487, F21174, F16152
- Random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7]: F22561, F6165, F5977, F3696
- Control features: F1803, F1442, F10666

### Method

1. Query/parse Neuronpedia `topkCosSimIndices` and `topkCosSimValues` for each seed feature.
2. Recursively query top neighbors to depth 1 or 2, with a hard cap (e.g. <=300 features) to avoid uncontrolled API use.
3. Build directed and symmetrized graphs.
4. Compute:
   - connected components;
   - clustering coefficient;
   - nearest-neighbor overlap;
   - parent/hedge separation;
   - whether random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7] sit in dense neighborhoods or isolated leaves.

### Falsification criteria

H7 is weakened if anti-aligned features are graph-isolated, have no stable local neighborhoods, and do not cluster with semantically/geometrically coherent neighbors.

### Output

- `results/supplement5/neuronpedia_neighbor_graph.json`
- `results/supplement5/neuronpedia_neighbor_graph_summary.md`
- optional figure: `figures/supplement5/neuronpedia_neighbor_graph.png`

## Workstream B: Geodesic-vs-chord on decoder-neighbor graph

### Objective

Test whether local graph distances differ meaningfully from direct cosine/chord distances.

### Method

1. Use the graph from Workstream A.
2. Edge length: `1 - cosine` for positive-neighbor edges.
3. For selected pairs, compute:
   - direct cosine if available;
   - shortest path distance;
   - geodesic/chord proxy ratio.
4. Compare hedge/parent families against controls.

### Falsification criteria

H7 is weakened if graph geodesics add no information over direct nearest-neighbor cosine and neighborhoods are star-like rather than path/cycle/manifold-like.

### Caveat

Neuronpedia top-k neighbors are not full vectors. This is a graph-proxy test, not a true manifold proof.

## Workstream C: Local PCA if vectors are locally available

### Objective

Measure intrinsic dimensionality around target neighborhoods.

### Method options

1. If public SAE decoder vectors are locally loadable, use actual `W_dec` and `W_enc`.
2. If only Neuronpedia auto-interp embeddings are available, treat them as explanation-space probes, not decoder geometry.
3. Compute local PCA / participation ratio for:
   - hedge+parent cluster;
   - random layer-8 neighborhoods [labelled 'active anti-aligned' in the original run; see ERRATA E7];
   - controls F1803/F1442/F10666.

### Prediction under H7

Some anti-aligned neighborhoods should show dimension >1, especially if they are shards of a curved concept manifold.

### Falsification criteria

H7 is weakened if local neighborhoods are essentially rank-1 and no richer structure appears beyond ordinary nearest-neighbor similarity.

## Workstream D: Re-read F1803 as temporal manifold coordinate

### Objective

Treat F1803 not only as a temporal specificity feature, but as a coordinate/shard of a temporal representation manifold.

### Existing observations

- F1803 acts as temporal specificity gate.
- Temporal PCA from COS3 showed PC1=30.5%, PC2=26.6%, PC3=23.7%, PC4=19.1%; PC2/PC1=0.87.
- 2023 anomaly is explained by BPE tokenization: `[' 20','23']` vs dedicated `[' 2024']`.
- L0_MLP routes year-token information strongly to F1803.

### Lightweight next tests

1. Locate existing COS3 JSON/CSV from `hub_cosine_2` or `proyecto_hub_cosine`.
2. Recompute/verify PCA and year trajectories without new model passes if cached activations exist.
3. Test whether years form a path, branch, bifurcation, or product-like geometry.
4. Add tokenization as a geometric confound axis.

### Conservative interpretation

F1803 is currently **compatible with** a manifold-coordinate view. It is not yet evidence that anti-alignment in the hedge family is caused by manifold fragmentation.

## Workstream E: Literature synthesis table

### Objective

Make a rigorous mapping from each H7 prediction to prior literature and available data.

Columns:

- Prediction
- Literature basis
- Local data available
- Test possible without training?
- Expected outcome under H7
- Alternative explanations
- Status

Output:

- `results/supplement5/H7_PREDICTION_EVIDENCE_TABLE.md`

## Supplement 6 decision rule

Do not create Supplement 6 yet. Create it only if at least one of these is true:

1. Expanded Neuronpedia graph produces a nontrivial falsifiable result.
2. Local PCA with actual SAE vectors produces a meaningful dimensionality result.
3. F1803 cached-data reanalysis produces a clear temporal-geometric result.
4. A no-training public-SAE analysis yields new evidence that changes the H7 scorecard.

If created, Supplement 6 should be titled conservatively, e.g.:

`SUPPLEMENT6_NO_TRAINING_H7_DIAGNOSTICS_REPORT.tex`

and should include a scorecard with labels: supports / weakens / inconclusive.

## Immediate next 2-hour plan

1. Build expanded Neuronpedia neighbor graph for GPT-2 L8 target features.
2. Write `results/supplement5/neuronpedia_neighbor_graph_summary.md`.
3. Search for cached F1803 COS3 data in sibling projects.
4. Draft `results/supplement5/H7_PREDICTION_EVIDENCE_TABLE.md`.
5. Decide whether evidence is substantial enough for Supplement 6. Default answer should be **no** unless results are clearly new.

## Current status

A preliminary H7 audit has started:

- `results/supplement5/H7_PRELIMINARY_NEURONPEDIA_AUDIT.md`
- `results/supplement5/h7_preliminary_neuronpedia_audit.json`

Initial conclusion: existing Supplement 3/4 data motivate H7 but do not demonstrate manifold fragmentation. Expanded graph/local geometry tests are the next falsifiable step.
