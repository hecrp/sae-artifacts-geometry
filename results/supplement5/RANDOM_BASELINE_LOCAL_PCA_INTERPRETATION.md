> Note (November 2026): the group `random_l8` (F22561, F6165, F5977, F3696) was labelled "active anti-aligned" when this file was produced. These are randomly selected layer-8 features with cos(E, D) near the layer mean, not anti-aligned features. See docs/ERRATA.md, E7.

# Random baseline interpretation for H7 local PCA

Status: conservative interpretation of `random_baseline_local_pca.json`.

## Summary
Against simple cos/norm matched baselines, hedge_plus_parent decoder PR=2.226 is at percentile 0.0, and PC2/PC1=0.363 is at percentile 0.0. This calibrates the local-dimensionality result. Interpret as evidence about geometric atypicality only, not as proof of H7.

## Key observations
- Parent cluster is extremely decoder-coherent relative to matched baseline: mean decoder cosine percentile 100.0, PR and PC2/PC1 percentiles 0.0. This indicates a tight near-1D/low-dimensional parent direction cluster, consistent with existing hedge-family interpretation.
- Hedge family is also unusually low-dimensional by decoder PR (0.0 percentile) but has high mean decoder cosine percentile 91.7; it is structured, not random, but less tight than the parent cluster.
- Hedge+parent combined has very low decoder PR/PC2 percentiles and very low mean cosine percentile, reflecting two opposed structured subclusters rather than a single smooth high-dimensional manifold.
- Controls are not extreme on matched decoder PR/PC2. Random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7] are moderately low percentile but semantically heterogeneous; this does not isolate H7.
- Encoder space shows very high mean cosine percentiles for hedge, parent, and hedge+parent, suggesting encoder-side commonality/detection structure even where decoder structure is oppositional.

## H7 update
The baseline strengthens the claim that hedge/parent features are geometrically atypical, but the direction of atypicality is more consistent with structured opposed clusters / corrective hedge geometry than with a smooth high-dimensional manifold by itself. H7 remains plausible as a broader framing, but the random baseline does not prove SAE shattering.

## Supplement 6 decision
Do not create Supplement 6 yet. The result is substantial enough for a Supplement 5 results note, but formal Supplement 6 should wait for either (a) broader depth-2 graph + random controls, (b) activation/coactivation geometry, or (c) F1803 reproducible manifold reanalysis.