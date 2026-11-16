# Supplement 5: Goodfire Neural Geometry Integration

Status: PLANNED / THEORY-INTEGRATION
Started: 2026-05-09
Primary source: https://www.goodfire.ai/research/the-world-inside-neural-networks

## Non-retroactivity policy

This supplement does not reopen closed gates G0, G1, or G2, and does not modify completed reports PHASE0, PHASE1, PHASE2, SUPPLEMENT3, or SUPPLEMENT4.

It adds a new conceptual frame and a pre-registered experimental agenda for later work.

## Motivation

Goodfire's Neural Geometry Series argues that neural networks contain structured inner worlds: representation spaces recapitulate the structure of the external world and behavior spaces often share this geometry. This directly matters for SAE interpretability because SAEs may fragment smooth manifolds into local sparse features.

## Research questions

1. Can anti-aligned SAE features be explained as local correction shards of curved representation manifolds?
2. Does the F1803 temporal manifold behave more like a geometric object than a single linear feature?
3. Are hedge-parent families stable as geometric neighborhoods even if exact feature IDs vary across seeds?
4. Does manifold-aware steering outperform linear steering for temporal/year concepts?
5. Which SAE architectures shard manifolds most strongly, and which preserve hierarchy/local charts?

## H7: Manifold Fragmentation / SAE Shattering

Anti-aligned SAE features are sometimes local linear shards of a curved or multidimensional concept manifold. The SAE objective decomposes the manifold into sparse dictionary elements. Some shards become corrective: their encoder detects a local region or parent cluster, while their decoder writes an opposing vector that keeps reconstruction consistent with the global manifold.

## Pre-registered predictions

- P1: Anti-aligned features will have structured local neighborhoods, not random isolation.
- P2: Local PCA around hedge-parent clusters will show intrinsic dimension > 1.
- P3: Geodesic distances on decoder/coactivation graphs will differ from direct cosine/chord distances.
- P4: For temporal concepts, active SAE features will form a smooth trajectory or low-dimensional graph as year/context varies.
- P5: Linear steering through a single SAE feature will produce more off-target behavior than manifold/tangent steering.
- P6: Architectures with explicit local geometry or hierarchy will reduce apparent shattering.

## Data inputs

- Bloom GPT-2 Small residual SAEs: `gpt2-small-res-jb`.
- Existing Phase 0-2 results.
- Supplement 3 bisector/expansion-ratio results.
- Supplement 4 Neuronpedia feature data and logit inversion results.
- F1803 temporal manifold results from `hub_cosine_2` / `feature_steering`.
- Goodfire source pages (read locally; not redistributed). See `docs/literature/GOODFIRE_NEURAL_GEOMETRY_SUMMARY.md` for links.

## Metrics

- Local PCA intrinsic dimension.
- Participation ratio / explained-variance ratios.
- Decoder-neighborhood continuity.
- Encoder-decoder tangent/normal decomposition.
- Geodesic-vs-chord ratio on kNN graphs.
- Coactivation graph modularity and path/cycle structure.
- Behavior-space Hellinger distance for steering trajectories.
- Off-target probability and distance-to-natural-activation manifold.

## Proposed experiments

### S5A: Geodesic-vs-chord diagnostic

Build kNN graphs over decoder directions and coactivation patterns. Compare graph geodesic distance with direct Euclidean/cosine distance.

### S5B: Local tangent/normal decomposition

For anti-aligned features and parents, fit local PCA to neighborhoods. Decompose encoder and decoder directions into tangent and normal components.

### S5C: SAE shard reconstruction

Use temporal prompts or calendar concepts to see whether active SAE features form a path/cycle/surface rather than unrelated points.

### S5D: Manifold steering vs linear steering

Compare single-vector steering with tangent/geodesic steering on F1803-like temporal representations.

### S5E: Alternative decompositions

Compare SAE features with MFA/local-region decompositions and hierarchical SAE variants.

## Expected outputs

- `reports/SUPPLEMENT5_GOODFIRE_NEURAL_GEOMETRY_REPORT.tex/pdf`
- `docs/literature/GOODFIRE_NEURAL_GEOMETRY_SUMMARY.md`
- Future scripts under `scripts/supplement5_*` if experiments are run.
- Future results under `results/supplement5/` and figures under `figures/supplement5/`.

## Failure modes

- Apparent manifolds may be artifacts of dimensionality reduction.
- Coactivation graphs may reflect tokenization rather than conceptual geometry.
- Some anti-aligned features are genuinely dead and may not have meaningful local geometry.
- Manifold steering may be infeasible without robust behavior-space definitions.

## Criteria for stronger claim

H7 should be promoted only if at least two independent diagnostics agree, e.g. local intrinsic dimension > 1 plus geodesic/chord mismatch plus causal steering improvement.
