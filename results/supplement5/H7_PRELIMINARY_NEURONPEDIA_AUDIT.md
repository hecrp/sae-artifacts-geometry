> Note (November 2026): the group `random_l8` (F22561, F6165, F5977, F3696) was labelled "active anti-aligned" when this file was produced. These are randomly selected layer-8 features with cos(E, D) near the layer mean, not anti-aligned features. See docs/ERRATA.md, E7.

# H7 preliminary Neuronpedia/local-data audit

Status: exploratory; no training; not evidence that H7 is true.

## Conservative conclusion
Existing Supplement 3/4 data support structured hedge/parent neighborhoods and motivate H7. They do not demonstrate manifold fragmentation. Next falsifiable step is expanding local neighborhoods and computing graph/local-geometry diagnostics.

## Family summaries
### hedge_family
Nodes: [19139, 18573, 11969, 20804]
Known pairwise cosines: [{'i': 19139, 'j': 11969, 'cos': 0.8423}]
Mean/max/min known cosine: 0.8423 / 0.8423 / 0.8423

### parent_cluster
Nodes: [17487, 21174, 16152]
Known pairwise cosines: [{'i': 17487, 'j': 21174, 'cos': 0.6954}, {'i': 17487, 'j': 16152, 'cos': 0.6712}, {'i': 21174, 'j': 16152, 'cos': 0.9385}]
Mean/max/min known cosine: 0.7683666666666666 / 0.9385 / 0.6712

### controls
Nodes: [1803, 1442, 10666]
Known pairwise cosines: []
Mean/max/min known cosine: None / None / None

### random_l8
Nodes: [5977, 22561, 6165, 3696]
Known pairwise cosines: []
Mean/max/min known cosine: None / None / None

### mixed_extreme
Nodes: [18298, 19679, 3268, 4855, 18458, 7685, 21750]
Known pairwise cosines: []
Mean/max/min known cosine: None / None / None

## Observation / inference / hypothesis separation
- **observation — parent_cluster_tightness**: Parent cluster has high positive decoder-neighbor cosines where observed; supports structured neighborhood but not H7 alone.
- **observation — hedge_family_tightness**: F19139-F11969 is a strong positive hedge-hedge neighbor; F18573 also points to parent features. This supports neighborhood structure but could also be ordinary hedge-family geometry.
- **observation — manual_negative_cross_edges_from_supplement4**: Hedge-parent relations include negative decoder opposition; this is direct hedge evidence and a candidate local-normal correction under H7, but not proof of manifold shattering.
- **observation — random_l8_features_have_local_neighbors**: Several random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7] have positive nearest-neighbor clusters, but semantics are diverse; this supports testing H7 across families rather than claiming one global manifold.
- **inference — geodesic_chord_feasibility_limited**: With only top-k neighbors for a small queried set, graph tests are underpowered. Need expanded Neuronpedia neighbor crawl or local SAE vectors.

## Next required data
- Expanded local Neuronpedia neighbor crawl for hedge/parent/random layer-8 features [labelled 'active anti-aligned' in the original run; see ERRATA E7].
- If vectors are available, local PCA on decoder/encoder neighborhoods.
- If activation data is available, coactivation graph and temporal trajectory tests for F1803.
- Manifold steering tests are future work; current evidence is insufficient.