# Supplement 6 decision note

Decision: DO NOT CREATE SUPPLEMENT 6 YET.

Rationale:

- The expanded Neuronpedia graph is useful but still a proxy built from top-k decoder-neighbor metadata.
- The graph shows structured neighborhoods, especially hedge and parent clusters, but this is also predicted by the existing hedge-family mechanism and ordinary feature clustering.
- No vector-based local PCA has been run yet.
- No true geodesic-vs-chord test with complete pairwise distances has been run yet.
- No manifold-vs-linear steering comparison has been run yet.
- F1803 cached data strongly motivates a temporal-manifold reanalysis, but it is not direct evidence that anti-aligned features are manifold shards.

Create Supplement 6 only after at least one of these is completed:

1. Depth-2 Neuronpedia graph with control/random comparison and a clear result.
2. Local PCA / participation-ratio analysis using actual public SAE vectors.
3. Reproducible COS3/F1803 cached-data reanalysis with tokenization/template controls.
4. A steering comparison showing manifold-aware intervention improves naturalness over a linear feature/vector baseline.

Current outputs belong in `results/supplement5/` and future planning docs, not a formal supplement.

## Update after local PCA run

A real-vector local PCA diagnostic has now been run with the public Bloom GPT-2 Small L8 SAE:

- `results/supplement5/LOCAL_PCA_DECODER_NEIGHBORHOODS.md`
- `results/supplement5/local_pca_decoder_neighborhoods.json`

Decision remains: **DO NOT CREATE SUPPLEMENT 6 YET**.

Reason:

- The hedge+parent set has decoder PR≈2.226 and PC2/PC1≈0.363, which is compatible with local multidimensional geometry.
- But controls and random layer-8 neighborhoods [labelled 'active anti-aligned' in the original run; see ERRATA E7] also show high PR, so dimensionality is not specific to H7 in this first pass.
- A useful audit finding emerged: some Supplement 4 reported hedge-parent cosine values appear to mix S3 metrics; actual W_dec pair cosines are more negative. This should be corrected/annotated in any formal future supplement.

Supplement 6 should wait for matched random baselines and/or a broader local-neighborhood PCA.

## Update after random/matched local PCA baseline

A matched random baseline has now been run:

- `scripts/supplement5_random_baseline_local_pca.py`
- `results/supplement5/random_baseline_local_pca.json`
- `results/supplement5/RANDOM_BASELINE_LOCAL_PCA.md`
- `results/supplement5/RANDOM_BASELINE_LOCAL_PCA_INTERPRETATION.md`

Decision remains: **DO NOT CREATE SUPPLEMENT 6 YET**.

Reason:

- The hedge/parent system is clearly non-random under the baseline.
- But the atypicality mostly indicates tight/opposed low-dimensional structure, already compatible with the hedge-family mechanism.
- H7 as manifold fragmentation remains a plausible broader interpretation, but still needs either activation/coactivation geometry, broader graph geodesics, or F1803 manifold reanalysis before a formal Supplement 6.
