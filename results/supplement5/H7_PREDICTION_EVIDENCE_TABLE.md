> Note (November 2026): the group `random_l8` (F22561, F6165, F5977, F3696) was labelled "active anti-aligned" when this file was produced. These are randomly selected layer-8 features with cos(E, D) near the layer mean, not anti-aligned features. See docs/ERRATA.md, E7.

# H7 Prediction / Evidence Table (No-Training, May 2026)

Status: conservative scorecard. This table distinguishes observations, inferences, hypotheses, alternatives, and next tests.

| Prediction | Literature basis | Observation(s) | Inference | Alternatives / confounds | Status | Next test |
|---|---|---|---|---|---|---|
| P1: anti-aligned features have structured local neighborhoods, not random isolation | Goodfire neural geometry; Hindupur 2025; Supplement 3 hedge-family mechanism | Neuronpedia depth-1 graph has 113 nodes / 238 thresholded undirected edges. Hedge-family mean degree=10; F19139-F11969 decoder cosine≈0.8423. Parent cluster internal cosines≈0.67-0.94. | Structured neighborhoods exist for queried hedge/parent features. | Ordinary decoder feature-family clustering; dead-feature artifacts; Neuronpedia top-k bias. | weak support / motivates H7, not diagnostic | Expand neighbor crawl and compare against random anti-aligned and aligned controls. |
| P2: local PCA around hedge-parent clusters has intrinsic dimension > 1 | Engels 2024; Modell 2025; Shafran 2026 local regions | No actual W_dec/W_enc vectors loaded in this continuation. Neuronpedia graph alone cannot estimate PCA dimension. | Untested. | Rank-1 local feature families; simple parent-child hedges without manifold. | untested | Load public Bloom GPT-2 L8 SAE vectors and run local PCA / participation ratio on top-k neighborhoods. |
| P3: geodesic graph distance differs from direct chord/cosine distance in curved neighborhoods | Goodfire manifold steering; representation manifolds; graph manifold learning | Depth-1 graph built from topkCosSim is available, but direct pairwise cosines are sparse and graph is query-biased. Several seed groups fall into separate components; largest component includes F19139/F11969/F20804, F10666, F22561. | Graph-proxy test is feasible but currently underpowered. | Top-k graph construction artifact; component structure from query seed choice rather than manifold. | inconclusive | Crawl neighbor graph to depth 2 with cap; compute geodesic/chord ratios only where direct cosine is available or vectors are loaded. |
| P4: active anti-aligned features sit in dense neighborhoods or transition regions | Goodfire SAE shattering; Shafran local geometry | Random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7] F22561, F6165, F5977, F3696 each have degree≈10 in depth-1 graph. They belong to components of size 50 or 11, but semantic explanations are diverse. | Random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7] are not graph-isolated in this sample. | Any queried Neuronpedia feature has 10 top neighbors by construction; degree alone is weak. | untested: the four features examined (F22561, F6165, F5977, F3696) are random layer-8 features, not anti-aligned | Compare component sizes/clustering to matched active aligned features and random L8 features. |
| P5: F1803 behaves like a coordinate/shard of temporal manifold rather than isolated scalar feature | Engels 2024 multidimensional features; Goodfire neural geometry; Wurgaft 2026 steering | COS3 cached JSON exists. PCA variance: PC1=0.3054, PC2=0.2663, PC3=0.2374, PC4=0.1908; PC2/PC1=0.872. ANOM_2023 JSON exists and supports tokenization confound. | F1803 temporal data are compatible with a multidimensional temporal manifold reading. | PCA over chosen prompts may reflect prompt-template/tokenization variation; F1803 may still be a specificity gate with surrounding confounds, not a manifold shard. | moderate support for temporal-manifold framing; not direct support for anti-aligned H7 | Parse COS3 per-prompt coordinates, test year trajectory geometry, separate tokenization/template axes, and compare with F1803 decoder-neighbor graph. |
| P6: manifold-aware steering is cleaner than linear single-feature steering | Wurgaft 2026 / Goodfire manifold steering | Prior report says PC1 steering increased logit_diff by ≈+0.43, but no side-by-side linear-vs-manifold steering benchmark exists here. | Causal relevance of temporal PC1 is suggestive only. | PC1 may be ordinary linear direction; steering metric may not test off-manifold naturalness. | untested for H7 | Define behavior-space metric and compare F1803 decoder steering, PC1 steering, and geodesic/tangent steering. |
| P7: Supplement 6 should only be created if H7 evidence becomes substantial | Project non-retroactivity rule; systematic phase documentation | Current outputs are preliminary graph/cached-data audits; no vector PCA, no expanded depth-2 graph, no causal steering comparison. | Enough for next-step planning and maybe a short results note; insufficient for formal Supplement 6. | Could write Supplement 6 as a negative/inconclusive report, but that would be premature before completing the planned no-training diagnostics. | do not create Supplement 6 yet | Complete Workstreams A-D first. |

## Conservative decision
H7 is still a live and useful hypothesis, but it is not demonstrated. Current evidence supports structured neighborhoods and a strong temporal-manifold case study for F1803, while the core manifold-fragmentation claim requires vector-based local PCA, broader graph tests, or causal steering comparisons.

## Local PCA update after public SAE vector loading

Source files:

- `scripts/supplement5_local_pca_no_training.py`
- `results/supplement5/local_pca_decoder_neighborhoods.json`
- `results/supplement5/LOCAL_PCA_DECODER_NEIGHBORHOODS.md`
- `figures/supplement5/local_pca_seed_groups_decoder.png/pdf`

### Observation

The public Bloom GPT-2 Small L8 residual SAE was loaded with SAELens without training. Shapes:

- W_enc: `[768, 24576]`
- W_dec: `[24576, 768]`
- global cos(E,D) mean/std: 0.368286 / 0.078535

Vector-based local PCA results:

| Group | n | Decoder PR | Decoder PC2/PC1 | Decoder rank95 | Encoder PR | Encoder PC2/PC1 |
|---|---:|---:|---:|---:|---:|---:|
| hedge_family | 4 | 1.937 | 0.472 | 2 | 1.888 | 0.559 |
| parent_cluster | 3 | 1.296 | 0.151 | 2 | 1.599 | 0.333 |
| random_l8 | 4 | 2.960 | 0.872 | 3 | 2.695 | 0.609 |
| controls | 3 | 1.995 | 0.904 | 2 | 1.996 | 0.912 |
| hedge_plus_parent | 7 | 2.226 | 0.363 | 3 | 3.495 | 0.526 |

### Inference

- The hedge+parent set is not rank-1 by decoder PCA (PR≈2.23, PC2/PC1≈0.36). This is compatible with local multidimensional geometry.
- However, controls and random layer-8 neighborhoods [labelled 'active anti-aligned' in the original run; see ERRATA E7] also show high local PR values. Therefore high dimensionality is **not specific** to anti-alignment in this first pass.
- Parent cluster alone is comparatively close to a line/plane dominated by PC1 (decoder PR≈1.30, PC2/PC1≈0.15), consistent with a tight parent direction cluster.

### Important correction / audit finding

Actual W_dec pair cosines for some Supplement 4 hedge-parent pairs are much more negative than the values manually listed in the earlier comparison table. Example:

- F19139-F21174 actual W_dec cosine: -0.9901.
- F11969-F16152 actual W_dec cosine: -0.9255.
- F11969-F21174 actual W_dec cosine: -0.8632.

This suggests the earlier Supplement 4 table mixed different S3 quantities for at least some rows (e.g. `cos_Df_with_jenc` vs `cos_Df_Djdec`). This does **not** weaken the hedge mechanism; it makes decoder opposition stronger. But it should be documented before any Supplement 6 formalization.

### Updated status

H7 receives **weak-to-moderate motivation**, not proof. Local PCA now gives a real-vector diagnostic, but it needs matched random/control baselines and broader neighborhoods before any claim of manifold fragmentation.

## Random/matched baseline update for local PCA

Source files:

- `scripts/supplement5_random_baseline_local_pca.py`
- `results/supplement5/random_baseline_local_pca.json`
- `results/supplement5/RANDOM_BASELINE_LOCAL_PCA.md`
- `results/supplement5/RANDOM_BASELINE_LOCAL_PCA_INTERPRETATION.md`

### Observation

Against simple cos(E,D)/encoder-norm matched baselines (1000 samples per group):

| Group | Decoder PR pct | Decoder PC2/PC1 pct | Decoder mean-cos pct | Encoder PR pct | Encoder mean-cos pct |
|---|---:|---:|---:|---:|---:|
| hedge_family | 0.0 | 0.1 | 91.7 | 0.2 | 100.0 |
| parent_cluster | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| random_l8 | 13.9 | 15.2 | 64.6 | 26.5 | 98.1 |
| controls | 44.4 | 44.4 | 28.6 | 60.0 | 69.3 |
| hedge_plus_parent | 0.0 | 0.0 | 0.0 | 45.8 | 100.0 |

### Inference

The hedge/parent system is highly atypical relative to matched random features, but mainly as **structured low-dimensional/opposed clusters**, not as generic high-dimensional manifold evidence. This strengthens the existing hedge-family geometry and motivates H7, but does not demonstrate shattering.

### Updated H7 status

H7 remains live. The random baseline makes the result more rigorous by showing the hedge/parent geometry is non-random, but the specific pattern currently favors a conservative phrasing: “structured corrective subclusters compatible with a manifold-fragmentation framing,” not “manifold fragmentation demonstrated.”
