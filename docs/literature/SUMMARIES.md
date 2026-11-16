# Paper Summaries: The Geometry of SAE Artifacts Project

This document contains summaries of all papers referenced in the project proposal and reports.
Papers are grouped by type: arXiv (PDF available), and web/blog (no PDF).

---

## arXiv Papers (PDFs in this directory)

---

### [belrose2025] Sparse Autoencoders Trained on the Same Data Learn Different Features
Authors: Goncalo Paulo, Nora Belrose
Year: 2025 | arXiv: 2501.16615
PDF: belrose2025_sae_different_features.pdf

This paper investigates a fundamental question in SAE interpretability: do SAEs recover
a unique, objectively "true" set of features used by a language model, or do they produce
arbitrary decompositions that vary with training conditions? The problem it addresses is
the implicit assumption that SAEs converge to a canonical, universal feature basis when
trained on the same model and data. The authors train multiple SAEs with identical
hyperparameters but different random initialization seeds on the same LLM activations
(feedforward networks in Llama 3 8B, plus two other LLMs) and measure how many features
are shared across runs using cosine similarity and activation pattern matching.

MAIN FINDING: For a 131K-latent SAE on Llama 3 8B, only ~30% of features were
consistently shared across seeds. This seed-dependence was observed across multiple
layers, three LLMs, two training datasets, and several SAE architectures. Architectural
finding: ReLU SAEs with L1 exhibited greater cross-seed stability, while TopK SAEs were
substantially more seed-dependent, even controlling for sparsity level.

The paper argues SAEs should be understood as pragmatically useful decompositions rather
than universal catalogues of "true" features. This reframes their epistemic status.

RELEVANCE: Directly challenges the uniqueness hypothesis underlying SAE interpretability.
Feature instability implies artifact geometry is partially a function of the training
process. Provides the PW-MCC-precursor metric and motivates our Phase 3 (seed stability)
work. The TopK vs ReLU stability difference is key for our Phase 5 architectural comparison.

---

### [chanin2024] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders
Authors: David Chanin, James Wilken-Smith, Tomas Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom
Year: 2024 | arXiv: 2409.14507
PDF: chanin2024_absorption.pdf

This paper identifies and systematically characterizes "feature absorption" -- a failure
mode where, when a general feature (e.g., "starts with L") and a specific feature
(e.g., "the token 'lion'") frequently co-occur, the L1 penalty causes the SAE to absorb
the general concept into the specific feature. The general feature then fails to fire on
tokens where the specific feature is active, even though the general concept is present.

KEY METHOD: Novel detection metric quantifying how often a feature fails to fire when
it should. Validated across hundreds of SAEs on Gemma 2 2B. Shows absorption arises
systematically from hierarchical structure + sparsity, not random failure.

CRITICAL NEGATIVE RESULT: Simply adjusting SAE size or sparsity does not resolve
absorption; it persists and sometimes worsens with scaling.

RELEVANCE: Identifies the behavioral counterpart to our geometric phenomenon. We
hypothesize (H1) that anti-aligned features are the geometric trace of absorption.
Specifically, absorbed features may leave a "hedge" feature with cos(E,D) < 0 that
subtracts the absorbed parent concept. Critical reference for Section 1.5 and H1.

---

### [gao2024] Scaling and Evaluating Sparse Autoencoders
Authors: Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu (OpenAI)
Year: 2024 | arXiv: 2406.04093
PDF: gao2024_scaling_evaluating_sae.pdf

Addresses two challenges: scaling SAEs to capture all features in large LLMs, and
rigorously evaluating feature quality. Prior L1-based training had dead latents, tuning
difficulties, and opaque reconstruction-sparsity tradeoffs.

KEY INNOVATION: TopK activation function -- exactly k features activate per input,
eliminating need to balance reconstruction vs L1 penalty. Also nearly eliminates dead
latents at scale. Establishes power-law scaling relationships between SAE size and
quality. Trains 16M-latent SAE on GPT-4 activations (40B tokens, largest at time).

EVALUATION FRAMEWORK: Multiple new metrics including hypothesized feature recovery,
LLM-based interpretability scoring, and sparsity of downstream causal effects.

RELEVANCE: Introduces ghost gradients, which form our H4 hypothesis -- ghost gradients
periodically reinitialize dead features, potentially creating encoder-decoder misalignment
during resurrection. TopK/BatchTopK becomes the architectural alternative in our
Phase 5 comparison. Scaling laws framework motivates our Phase 0 (larger models).

---

### [he2024] Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders
Authors: Zhengfu He et al.
Year: 2024 | arXiv: 2410.20526
PDF: he2024_llama_scope.pdf

Provides a comprehensive, publicly released suite of 256 SAEs covering every layer and
sublayer (attention, MLP, residual stream) of Llama-3.1-8B-Base, at 32K and 128K
feature sizes. Built on TopK SAEs with various architectural modifications evaluated
across multiple quality dimensions.

KEY FINDING: Feature splitting across SAE sizes is geometrically confirmable -- 32K to
128K produces genuinely new, finer-grained feature directions rather than redundant
duplicates. Base-model SAEs transfer meaningfully to instruction-tuned variants.

RELEVANCE: Primary infrastructure resource for Phase 0 (cross-model verification).
Llama-3.1-8B SAEs at multiple layers provide the second model family (alongside Gemma
Scope) to test generalization of the sign-flip and anti-aligned tail phenomena.
Multi-scale SAEs enable studying how geometric artifacts change with expansion factor.

---

### [rajamanoharan2024] Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders
Authors: Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, Janos Kramar, Neel Nanda (DeepMind)
Year: 2024 | arXiv: 2407.14435
PDF: rajamanoharan2024_jumprelu.pdf

Addresses shrinkage artifacts in L1-trained SAEs: the L1 penalty biases feature
activations toward zero, causing learned features to underestimate true magnitude.

KEY INNOVATION: JumpReLU -- a discontinuous ReLU variant with a learnable threshold
jump. Activations below threshold become zero; above pass through without modification
(unlike standard ReLU/TopK). Allows sharp, learnable cutoff without L1's magnitude bias.
Uses straight-through estimators to train through discontinuity. Directly optimizes L0
(count of nonzero activations) instead of L1 proxy -- eliminates root cause of shrinkage.

BENCHMARKED on Gemma 2 9B against Gated SAE and TopK: state-of-the-art reconstruction
at equivalent sparsity. Validated with both manual and LLM-scored interpretability.

RELEVANCE: JumpReLU is the architecture used in Gemma Scope -- our primary Phase 0
cross-model target. Critical for architectural comparison (Phase 5): if anti-aligned
features are absent in JumpReLU SAEs, it supports H1 (L1-specific shrinkage artifact).
The shrinkage mechanism explained here is the geometric foundation of our toy model.

---

### [leask2025] Sparse Autoencoders Do Not Find Canonical Units of Analysis
Authors: Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, Neel Nanda
Year: 2025 | arXiv: 2502.04878
PDF: leask2025_canonical_units.pdf

Challenges the assumption that a correct dictionary size exists yielding stable,
indivisible atomic features. Introduces two techniques:
- SAE Stitching: inserts/swaps latents from larger SAE into smaller one, classifying
  as "novel" (improves performance, proving smaller SAE incomplete) or "reconstruction"
  (replaceable equivalents).
- Meta-SAEs: trains new SAE on decoder weight matrix of existing SAE. When larger SAE
  latents decompose into combinations of smaller-SAE latents, proves lack of atomicity.

FINDING: Features decompose semantically coherently -- "Einstein" -> "scientist",
"Germany", "famous person". No privileged scale: continuous hierarchy, no atomic level.

RELEVANCE: Compositional structure (H5 hypothesis) -- anti-aligned features may be
correction terms for non-linear interactions across the compositional hierarchy.
Larger SAEs having MORE anti-aligned features would support this. The geometric
relationships between parent and child feature decoder directions is central to our work.

---

### [lieberum2024] Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2
Authors: Tom Lieberum, Senthooran Rajamanoharan et al. (DeepMind)
Year: 2024 | arXiv: 2408.05147
PDF: lieberum2024_gemma_scope.pdf

Open suite of JumpReLU SAEs trained on all layers and sub-layers of Gemma 2 2B and 9B,
with select layers of 27B. Multiple dictionary widths at each location. SAEs also trained
on instruction-tuned Gemma 2 9B.

KEY CONTRIBUTIONS: Comprehensive layer coverage enabling longitudinal studies. Multiple
widths per location enabling direct study of how feature geometry changes with expansion
factor. Standard evaluation on L0, variance explained, and downstream performance.

RELEVANCE: PRIMARY RESOURCE for Phase 0. Gemma Scope's standardized, multi-scale SAE
collection allows systematic characterization of geometric artifacts across architecturally
matched models. JumpReLU vs ReLU comparison is central to Phase 5. Multi-width settings
permit studying how anti-aligned feature count changes with expansion factor.

---

### [peter2025] Learning Multi-Level Features with Matryoshka Sparse Autoencoders
Authors: Peter et al.
Year: 2025 | arXiv: 2503.17547
PDF: peter2025_matryoshka_sae.pdf

Addresses the tension in SAE design where increasing dictionary size causes general
high-level features to be split into overly specific sub-features or absorbed.

KEY INNOVATION: Matryoshka SAEs simultaneously train multiple nested dictionaries of
increasing size (e.g., 512, 2048, 8192 latents) sharing the same input, with inductive
bias that smaller dictionaries independently reconstruct without the larger ones. Small
dicts learn broad concepts; larger dicts refine without cannibalizing high-level features.

RESULTS on Gemma 2 2B and TinyStories: superior sparse probing and concept erasure,
more disentangled representations, significantly reduced feature absorption, minor
reconstruction quality trade-off.

RELEVANCE: Makes hierarchical geometric structure explicit and trainable. The nested
dictionary structure encodes multi-scale geometry: same concept represented at multiple
resolutions, angular relationships between high-level and child feature vectors in
decoder space become measurable. Direct relevance to H5 (compositional artifacts)
and to understanding how anti-aligned features relate to scale.

---

### [song2025] Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs
Authors: Song et al.
Year: 2025 | arXiv: 2505.20254
PDF: song2025_feature_consistency.pdf

Confronts reproducibility crisis: independent SAE runs on identical conditions learn
different feature sets. Proposes feature consistency as primary optimization target.

KEY METRIC: Pairwise Dictionary Mean Correlation Coefficient (PW-MCC) -- average
best-match correlation between latent directions across two independently trained SAEs.
Validated via synthetic model organism. On real LLMs, TopK SAEs achieve PW-MCC=0.80.
High PW-MCC strongly correlates with semantic similarity of automated feature explanations.

RELEVANCE: PW-MCC is the primary metric for Phase 3 (seed stability analysis). The
paper provides both the metric definition and validated implementation details needed
to characterize whether our specific anti-aligned features (F19139, F18573, F11969,
F20804) are stable across seeds or seed-specific artifacts. PW-MCC is fundamentally
a geometric measure (directional alignment of decoder vectors).

---

## Web-Only Papers (no PDF available)

---

### [elhage2022] Toy Models of Superposition
Authors: Nelson Elhage, Tristan Hume, Catherine Olsson et al. (Anthropic)
Year: 2022 | URL: https://transformer-circuits.pub/2022/toy_model/index.html

FOUNDATIONAL PAPER. Establishes the superposition hypothesis: networks represent
more features than dimensions by using nearly-orthogonal directions in high-dimensional
space. When features are sparse, interference costs of packing multiple features into
fewer dimensions are outweighed by representational benefits.

METHODS: Small ReLU toy networks on synthetic sparse inputs with known ground-truth
features, enabling direct measurement of geometric feature arrangements.

KEY FINDINGS:
- Superposition is real and empirically observable
- Phase transition between non-superposition and superposition based on feature
  importance and sparsity (2D phase diagram)
- Features organize into configurations matching uniform polytopes (triangles,
  pentagons, tetrahedra, etc.)
- Some computation possible within superposition
- Monosemantic neurons arise when feature importance exceeds superposition threshold

RELEVANCE: Establishes the theoretical foundation for ALL our work. The uniform
polytope geometry of superposition is precisely what SAEs try to disentangle. SAE
artifacts (dead latents, feature splitting) are consequences of this geometry.
The polytope structures motivate studying geometric arrangement of SAE decoder directions.

---

### [bricken2023] Towards Monosemanticity: Decomposing Language Models with Dictionary Learning
Authors: Trenton Bricken, Adly Templeton, Joshua Batson et al. (Anthropic)
Year: 2023 | URL: https://transformer-circuits.pub/2023/monosemantic-features/index.html

First large-scale empirical demonstration that SAEs can decompose a real language model
into monosemantic features. Targets polysemanticity in a one-layer transformer's MLP.

METHODS: SAEs on MLP activations of 1-layer transformer with 512-neuron MLP. 8B data
points. Expansion factors 1x to 256x. Encoder: learned weights + ReLU. Decoder maps
back. Loss: reconstruction + L1 sparsity penalty. Decoder columns constrained to unit norm.

EVIDENCE FOR MONOSEMANTICITY: Manual feature investigation (Arabic, DNA, base64,
Hebrew features), human interpretability ratings, automated scores, logit effects.

KEY PHENOMENA:
- Feature splitting: lower expansion factor features split into related sub-features
- Universality: similar features appear across independent SAE runs
- Feature neighborhoods: UMAP visualization shows global geometric structure of features

ESTABLISHES: Decoder unit-norm constraint, L1 penalty motivation, canonical ReLU SAE
architecture. This is the reference implementation for the Bloom GPT-2 SAEs we analyze.

RELEVANCE: Defines the standard SAE setup we study. Feature splitting as geometric
phenomenon (a direction splintering into nearby directions) is directly related to our
encoder-decoder misalignment question. Bricken's decoder normalization convention
is what makes cos(E,D) a meaningful quantity.

---

### [templeton2024] Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
Authors: Adly Templeton, Tom Conerly, Jonathan Marcus et al. (Anthropic)
Year: 2024 | URL: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

Demonstrates SAEs scale to Claude 3 Sonnet (production-grade model). Establishes
scaling laws for SAE quality analogous to LLM scaling laws.

KEY FINDINGS:
- SAEs produce highly interpretable features at scale, including multilingual and
  multimodal features (same concept fires across languages and modalities)
- Systematic relationship between concept frequency and dictionary size needed to resolve it
- Features causally steer model behavior
- Safety-relevant features identified (deception, sycophancy, bias, dangerous content)

GEOMETRY CONTRIBUTION: Feature neighborhood analysis -- features that are geometrically
proximal in decoder space form coherent semantic clusters. Case study: "Inner monologue"
feature cluster with semantically coherent neighbors. This is an early empirical result
that decoder geometry is meaningful and structured.

RELEVANCE: Strongest empirical evidence that decoder direction neighborhoods are not
random. Our sign-flip diagnostic could complement their neighborhood analysis as a
lightweight geometric quality metric. The scaling laws framework positions our work:
if sign-flip is architecture-specific (L1 vs JumpReLU), it has design implications.

---

### [karvonen2024] An Intuitive Explanation of Sparse Autoencoders for LLM Interpretability
Author: Adam Karvonen
Year: 2024 | URL: https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html

Blog post bridging popular and technical understanding of SAEs for LLM interpretability.

TECHNICAL COVERAGE: Polysemanticity examples (neurons responding to academic citations,
English dialogue, HTTP requests, Korean text simultaneously). Superposition via
compressed sensing analogy. Concrete dimension examples (GPT-3's 12,288-dim activations
expanded to 49,512 with 4x expansion). L1 penalty and reconstruction trade-off.
Reference PyTorch implementation. Update recommends BatchTopK/JumpReLU over vanilla ReLU.

KEY NOTE: Uses median encoder-decoder cosine similarity of ~0.5 as baseline reference.
Our paper finds ~0.35 for the Bloom GPT-2 SAEs, suggesting Karvonen's figure comes
from a different training regime. This discrepancy needs to be acknowledged in Section 2.1.

RELEVANCE: Cited in our paper for the cos(E,D) ~ 0.5 empirical baseline (karvonen2024).
Provides accessible geometric intuition: decoder columns as unit vectors in activation
space, encoder as detection directions, sparsity controlling which directions are active.

---

### [smith2025] Negative Results for Sparse Autoencoders on Downstream Tasks
Authors: Lewis Smith, Senthooran Rajamanoharan, Arthur Conmy et al. (Google DeepMind)
Year: 2025 | URL: Google DeepMind blog post
NOTE: No arXiv ID found. Blog post only.

Documents negative results when applying Gemma Scope SAEs to downstream safety and
interpretability tasks (e.g., concept detection, circuit analysis, behavior steering).
SAE-based approaches failed to outperform simpler baselines on several benchmark tasks.

RELEVANCE: Provides broader context for our work. If SAE features fail on downstream
tasks, characterizing artifacts (like anti-aligned features) becomes more urgent. The
architectural evolution the paper implies (away from L1 toward harder sparsity) is
the motivation for our Phase 5 architectural comparison. Cited in Section 1.5 as
motivation for understanding L1-specific geometric artifacts.

---

## Key Relationships Between Papers (for implementation)

SUPERPOSITION FOUNDATION:
  elhage2022 -> bricken2023 -> bloom2024 (GPT-2 SAEs we analyze)

ARCHITECTURE EVOLUTION (L1 -> Hard Sparsity):
  bricken2023 (ReLU+L1) -> gao2024 (TopK) -> rajamanoharan2024 (JumpReLU)

CROSS-MODEL RESOURCES:
  lieberum2024 (Gemma Scope, JumpReLU) -> Phase 0 primary target
  he2024 (Llama Scope, TopK) -> Phase 0 secondary target

ARTIFACTS AND QUALITY:
  chanin2024 (absorption) -> H1 hypothesis (geometric trace of absorption)
  belrose2025 (seed instability) -> Phase 3 (seed stability)
  leask2025 (non-canonical) -> H5 hypothesis (compositional artifacts)
  song2025 (PW-MCC) -> Phase 3 metric

SCALING BEHAVIOR:
  templeton2024 (Claude 3 Sonnet) -> sign-flip as diagnostic at scale
  peter2025 (Matryoshka) -> hierarchical geometry, H5

BASELINE / REFERENCE VALUES:
  karvonen2024: cos(E,D) ~ 0.5 median (Bloom GPT-2 has ~0.35 -- discrepancy)
---

## Goodfire Neural Geometry Series (added May 2026)

Full synthesis: `docs/literature/GOODFIRE_NEURAL_GEOMETRY_SUMMARY.md`

### [geiger2026] The World Inside Neural Networks
Authors: Atticus Geiger, Ekdeep Singh Lubana, Thomas Fel, Jack Merullo, Michael Jae Byun, Owen Lewis, Tom McGrath
Year: 2026 | URL: https://www.goodfire.ai/research/the-world-inside-neural-networks

Goodfire's central claim is that neural networks develop structured inner worlds: representation geometry recapitulates structure in the external world and in training data. The post surveys evidence across LLMs, image models, genomic models, and biomedical models: days/months as cycles, years as curves, colors as surfaces, phylogeny as a manifold, and biomarkers as clean geometric structures.

RELEVANCE: Provides the conceptual frame for Supplement 5. SAE features should be treated as probes rather than final ontology. A concept may be a manifold; a single SAE feature may be a patch, coordinate, or correction term. This directly motivates H7: Manifold Fragmentation / SAE Shattering.

### [wurgaft2026] Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior
Authors: Daniel Wurgaft, Can Rager, Matthew Kowal, Vasudev Shyam, Sheridan Feucht, Usha Bhalla, Tal Haklay, Eric Bigelow, et al.
Year: 2026 | arXiv:2605.05115 | URL: https://arxiv.org/abs/2605.05115

Shows that steering along representation manifolds produces more natural behavioral trajectories than linear steering. In calendar examples, the representation manifold and behavior manifold share circular structure; moving along the representation manifold shifts output distributions along the natural behavior manifold.

RELEVANCE: Provides the experimental template for testing F1803 temporal geometry: compare linear steering with tangent/geodesic steering, evaluate behavior-space trajectories with Hellinger distance, and measure off-manifold artifacts.

### [engels2024] Not All Language Model Features Are One-Dimensionally Linear
Year: 2024 | arXiv:2405.14860

Shows that some language-model features are irreducibly multidimensional. Calendar concepts can be represented as circles rather than single directions.

RELEVANCE: Directly supports interpreting F1803 / temporal features as a multidimensional manifold rather than a single scalar feature. Connects to PC2/PC1=0.87 from the temporal manifold analysis.

### [modell2025] The Origins of Representation Manifolds in Large Language Models
Year: 2025 | arXiv:2505.18235

Formalizes features as metric/topological objects and explains representation manifolds as learned geometric correspondences.

RELEVANCE: Supplies the mathematical vocabulary for H7: concept space, representation manifold, geodesic distance, topology, and intrinsic dimension.

### [karkada2026] Symmetry in Language Statistics Shapes the Geometry of Model Representations
Year: 2026 | arXiv:2602.15029

Argues that symmetries in language statistics generate internal representation geometry. Calendar and geographic geometries arise from regularities in co-occurrence statistics.

RELEVANCE: Suggests a source for temporal manifolds in GPT-2 and for F1803's year-specific geometry.

### [prieto2026] From Data Statistics to Feature Geometry: How Correlations Shape Superposition
Year: 2026 | arXiv:2603.09972

Shows that correlated features can make interference constructive and shape superposition geometry.

RELEVANCE: Reframes some SAE artifacts as consequences of data correlations interacting with compression geometry, not merely failures of the SAE objective.

### [shafran2026] From Directions to Regions
Year: 2026 | arXiv:2602.02464

Uses local regions / Mixture of Factor Analyzers as an alternative to global directions.

RELEVANCE: Provides a baseline for local-geometry analysis of anti-aligned neighborhoods and manifold shards.

### [hindupur2025] Projecting Assumptions
Year: 2025 | arXiv:2503.01822

Argues that different SAE architectures impose different geometric assumptions.

RELEVANCE: Supports treating architecture comparison as a comparison of geometric priors: ReLU+L1, TopK, JumpReLU, hierarchical and local-region approaches all expose or hide different concept geometries.

UPDATED RELATIONSHIP MAP:

NEURAL GEOMETRY:
  Goodfire 2026 -> manifolds as internal data structures -> H7
  Engels 2024 + Modell 2025 -> features as multidimensional metric/topological objects
  Karkada 2026 + Prieto 2026 -> data statistics/correlations as source of geometry

CONTROL:
  Wurgaft 2026 -> manifold steering > linear steering when concept geometry is curved

SAE LIMITS:
  Hindupur 2025 + Menon 2024 + Goodfire 2026 -> SAEs are useful but biased probes; they can shatter manifolds and produce correlational rather than causal features

PROJECT IMPACT:
  Anti-alignment may be a local correction shard of a curved manifold.
  F1803 is now best framed as a temporal representation manifold problem, not only as a single SAE feature.
  Phase 3 should track geometric neighborhoods across seeds, not only exact feature IDs.
