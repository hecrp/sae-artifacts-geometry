# Goodfire neural-geometry series: summary and relevance to this project

Last updated: 2026-05-09

Primary source: https://www.goodfire.ai/research/the-world-inside-neural-networks

The source pages are not redistributed in this repository; see the links above and in Section 3.

## 1. Central thesis

Goodfire argues that neural networks contain geometrically structured internal "worlds". These are not only collections of linear features but geometric objects that reflect regularities of the world and of the training data: circles, curves, helices, surfaces, regions, hierarchies, graphs and manifolds.

For this project the conceptual change is:

- Before: one concept corresponds to one direction or one SAE feature.
- After: a concept may be a geometric object; a single SAE feature may be only a local patch, a coordinate, a correction term or a shard of that object.

## 2. Concepts used in this project

**Neural geometry.** The study of the internal geometry of activations: how models organise concepts and relations in representation space.

**Representation manifold.** The subset of activations associated with variations of a concept. Examples: days of the week as a circle, years as a curve, numbers as helices, colours as surfaces.

**Behaviour manifold.** Geometric structure in the space of behaviours or output distributions. For language models it can be measured on the probability simplex using the Hellinger distance.

**Manifold steering.** Intervening along a natural trajectory in representation space instead of adding a single global linear vector. A linear intervention is a local approximation; if the concept is curved, it can leave the natural manifold and produce incoherent outputs.

**SAE manifold shattering.** SAEs can fragment a global manifold into many local features. Automatic labels for each feature may be locally correct while losing the global semantics and topology.

## 3. Literature considered

- Goodfire, "The World Inside Neural Networks" (2026): the neural-geometry research programme; models recapitulate structure in the world.
- Wurgaft et al., arXiv:2605.05115: manifold steering; representation and behaviour geometries are causally aligned.
- Engels et al., arXiv:2405.14860: not all features are one-dimensional and linear; days and months form multidimensional circles.
- Modell et al., arXiv:2505.18235: formalises representation manifolds as metric/topological spaces.
- Karkada et al., arXiv:2602.15029: statistical symmetries of language give rise to representation geometry.
- Prieto et al., arXiv:2603.09972: correlations shape superposition; interference can be constructive.
- Shafran et al., arXiv:2602.02464: local regions and geometry via mixtures of factor analysers.
- Hindupur et al., arXiv:2503.01822: each SAE architecture imposes geometric assumptions.
- Menon et al., arXiv:2410.11767: interpretable SAE features do not guarantee causal relevance.
- Muchane et al., arXiv:2506.01197: hierarchical SAEs for hierarchical semantics.
- Kantamneni and Tegmark, arXiv:2502.00873: addition implemented with helices and trigonometry.
- Levy and Geva, arXiv:2410.11781: numbers represented as circular structures per base-10 digit.
- Fu et al., arXiv:2604.20817: separating spectral signal, geometric separability and functional use.

## 4. Implications for existing results

**Anti-alignment.** Phase 2 indicated that anti-alignment can arise from geometry alone in toy models. H7 adds the possibility that some anti-aligned features are local corrections of a manifold fragmented by the SAE.

**F1803 temporal structure (prior project).** F1803 should be treated as a local shard or coordinate of a multidimensional temporal manifold rather than as a single "temporal feature":

- A PC2/PC1 ratio of 0.87 suggests the structure is not one-dimensional.
- The 2023 anomaly is explained by BPE tokenisation, which links representation geometry to the tokenizer.
- The layer-0 MLP, previously characterised as a multi-channel router, may build the initial coordinates of this temporal geometry.

**Neuronpedia logit inversion.** The F19139-F21174 logit inversion can be read as a local correction: an encoder detects a region or parent cluster and its decoder writes an opposing vector that keeps the reconstruction on the natural manifold. The inversion itself follows from the nearly antiparallel decoders, and both features are dead on Neuronpedia's activation sample (revised Supplement 4), so this reading describes the weights rather than observed behaviour.

**Phases 3-5.**

- Phase 3: measure stability of regions and neighbourhoods, not only of individual feature IDs.
- Phase 4: analyse co-activation as a graph or manifold, not only as parent-child pairs.
- Phase 5: compare SAE architectures as geometric priors.

## 5. Hypothesis H7: manifold fragmentation / SAE shattering

Anti-aligned SAE features are sometimes local linear shards of a curved or multidimensional concept manifold. By imposing a global sparse dictionary, the SAE fragments the geometry and produces corrective features whose encoder and decoder can end up anti-aligned.

Predictions:

1. Anti-aligned features are not isolated outliers but nodes in dense neighbourhoods.
2. They appear near regions of high curvature or transitions of the manifold.
3. Graph geodesic distance differs from direct Euclidean or cosine distance.
4. Local PCA around these features shows dimension greater than one.
5. Manifold steering is cleaner than linear steering.
6. Local or hierarchical methods reduce apparent shattering.

Status: plausible, not established. See `results/supplement5/` for the first diagnostics and `docs/phases/SUPPLEMENT6_DECISION_NOTE.md` for the current assessment.

## 6. Related files

- `reports/SUPPLEMENT5_GOODFIRE_NEURAL_GEOMETRY_REPORT.tex` / `.pdf`
- `reports/goodfire_neural_geometry.bib`
- `docs/phases/SUPPLEMENT5_goodfire_neural_geometry.md`

## 7. Methodological rule

Goodfire's work is not used as retroactive evidence for phases whose gates are already closed. It is used as:

1. an updated theoretical framework;
2. a source of hypotheses;
3. motivation for new experiments;
4. a bridge between anti-alignment, F1803, SAEs and manifold steering.
