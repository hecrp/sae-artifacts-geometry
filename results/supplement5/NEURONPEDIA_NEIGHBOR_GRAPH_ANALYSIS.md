> Note (November 2026): the group `random_l8` (F22561, F6165, F5977, F3696) was labelled "active anti-aligned" when this file was produced. These are randomly selected layer-8 features with cos(E, D) near the layer mean, not anti-aligned features. See docs/ERRATA.md, E7.

# Neuronpedia neighbor graph analysis for H7

Status: no-training, graph-proxy analysis. Conservative: motivates tests, does not prove H7.

## Metadata
- source_cache: results/supplement5/neuronpedia_neighbor_profiles_gpt2_l8_depth1.json
- n_nodes: 113
- n_directed_edges: 391
- n_undirected_edges_thresholded: 238
- threshold: 0.13
- note: No SAE training. Graph built from Neuronpedia topkCosSim fields for queried GPT-2 L8 features and depth-1 neighbors.

## Group aggregates
| Group | n | mean degree | mean clustering | mean component size | degrees | component sizes |
|---|---:|---:|---:|---:|---|---|
| hedge_family | 4 | 10 | 0.13727106227106228 | 39.5 | [14, 6, 13, 7] | [50, 8, 50, 50] |
| parent_cluster | 3 | 2.6666666666666665 | 0.7777777777777778 | 8 | [2, 3, 3] | [8, 8, 8] |
| random_l8 | 4 | 10 | 0.22777777777777777 | 20.75 | [10, 10, 10, 10] | [50, 11, 11, 11] |
| controls | 3 | 10 | 0.4666666666666667 | 24 | [10, 10, 10] | [11, 11, 50] |

## Target features
| Feature | Group | frac_nonzero | degree | clustering | component size | explanation | top neighbors |
|---:|---|---:|---:|---:|---:|---|---|
| F1442 | controls | 0.0002895991007486979 | 10 | 0.4222222222222222 | 11 | locations or cities specifically denoted as "capital" in the text | F12080(0.582,neighbor), F21625(0.525,neighbor), F19490(0.450,neighbor), F9101(0.412,neighbor), F2281(0.401,neighbor) |
| F1803 | controls | 0.0003646214803059896 | 10 | 0.5111111111111111 | 11 | years related to future predictions or targets | F14355(0.510,neighbor), F23746(0.500,neighbor), F23638(0.482,neighbor), F7520(0.463,neighbor), F10687(0.457,neighbor) |
| F3696 | random_l8 | 0.0002768834431966146 | 10 | 0.3111111111111111 | 11 | mentions of the name "Dem" followed by other characters or words | F6853(0.631,neighbor), F5517(0.433,neighbor), F15971(0.412,neighbor), F23955(0.408,neighbor), F21026(0.403,neighbor) |
| F5977 | random_l8 | 0.0003449122111002604 | 10 | 0.15555555555555556 | 11 | instances of the word "pent" along with related words or phrases | F6674(0.397,neighbor), F24049(0.391,neighbor), F17285(0.390,neighbor), F20546(0.374,neighbor), F11176(0.371,neighbor) |
| F6165 | random_l8 | 0.00091552734375 | 10 | 0.15555555555555556 | 11 | proper nouns, specifically the surname "Walsh" | F13600(0.278,neighbor), F5245(0.263,neighbor), F3095(0.262,neighbor), F15359(0.261,neighbor), F23054(0.255,neighbor) |
| F10666 | controls | 0.0006621678670247396 | 10 | 0.4666666666666667 | 50 | expressions of admiration or appreciation | F11978(0.643,neighbor), F22337(0.598,neighbor), F3959(0.562,neighbor), F16269(0.496,neighbor), F9837(0.466,neighbor) |
| F11969 | hedge_family | 0 | 13 | 0.19230769230769232 | 50 |  | F19139(0.842,hedge_family), F15019(0.164,neighbor), F7145(0.157,neighbor), F16114(0.153,neighbor), F3787(0.153,neighbor) |
| F16152 | parent_cluster | 0 | 3 | 0.6666666666666666 | 8 |  | F21174(0.939,parent_cluster), F17487(0.671,parent_cluster), F18573(0.224,hedge_family) |
| F17487 | parent_cluster | 0 | 2 | 1.0 | 8 |  | F21174(0.695,parent_cluster), F16152(0.671,parent_cluster) |
| F18573 | hedge_family | 0 | 6 | 0.13333333333333333 | 8 |  | F16152(0.224,parent_cluster), F21174(0.158,parent_cluster), F8488(0.142,neighbor), F5970(0.138,neighbor), F8683(0.137,neighbor) |
| F19139 | hedge_family | 3.178914388020833e-07 | 14 | 0.17582417582417584 | 50 | words related to legal or criminal actions | F11969(0.842,hedge_family), F21829(0.169,neighbor), F21714(0.163,neighbor), F4872(0.163,neighbor), F3441(0.161,neighbor) |
| F20804 | hedge_family | 0 | 7 | 0.047619047619047616 | 50 |  | F8341(0.158,neighbor), F4061(0.147,neighbor), F4012(0.137,neighbor), F7406(0.136,neighbor), F3781(0.132,neighbor) |
| F21174 | parent_cluster | 0 | 3 | 0.6666666666666666 | 8 |  | F16152(0.939,parent_cluster), F17487(0.695,parent_cluster), F18573(0.158,hedge_family) |
| F22561 | random_l8 | 0.001110712687174479 | 10 | 0.28888888888888886 | 50 | references to a specific individual named "Soros" | F15944(0.441,neighbor), F9152(0.418,neighbor), F12047(0.391,neighbor), F8233(0.384,neighbor), F16579(0.370,neighbor) |

## Geodesic-vs-chord proxy pairs
| Pair | Groups | direct cos known | graph geodesic | ratio | path |
|---|---|---:|---:|---:|---|
| F1442-F1803 | controls / controls | None | None | None | [] |
| F1442-F10666 | controls / controls | None | None | None | [] |
| F1803-F10666 | controls / controls | None | None | None | [] |
| F3696-F5977 | random_l8 / random_l8 | None | None | None | [] |
| F3696-F6165 | random_l8 / random_l8 | None | None | None | [] |
| F3696-F22561 | random_l8 / random_l8 | None | None | None | [] |
| F5977-F6165 | random_l8 / random_l8 | None | None | None | [] |
| F5977-F22561 | random_l8 / random_l8 | None | None | None | [] |
| F6165-F22561 | random_l8 / random_l8 | None | None | None | [] |
| F11969-F16152 | hedge_family / parent_cluster | None | None | None | [] |
| F11969-F17487 | hedge_family / parent_cluster | None | None | None | [] |
| F11969-F18573 | hedge_family / hedge_family | None | None | None | [] |
| F11969-F19139 | hedge_family / hedge_family | 0.8423 | 0.15769999999999995 | 1.0 | [11969, 19139] |
| F11969-F20804 | hedge_family / hedge_family | None | 2.4111000000000002 | None | [11969, 21714, 4012, 20804] |
| F11969-F21174 | hedge_family / parent_cluster | None | None | None | [] |
| F16152-F17487 | parent_cluster / parent_cluster | 0.6712 | 0.3288 | 1.0 | [16152, 17487] |
| F16152-F18573 | parent_cluster / hedge_family | 0.2236 | 0.7764 | 1.0 | [16152, 18573] |
| F16152-F19139 | parent_cluster / hedge_family | None | None | None | [] |
| F16152-F20804 | parent_cluster / hedge_family | None | None | None | [] |
| F16152-F21174 | parent_cluster / parent_cluster | 0.9385 | 0.0615 | 1.0 | [16152, 21174] |
| F17487-F18573 | parent_cluster / hedge_family | None | 1.1052 | None | [17487, 16152, 18573] |
| F17487-F19139 | parent_cluster / hedge_family | None | None | None | [] |
| F17487-F20804 | parent_cluster / hedge_family | None | None | None | [] |
| F17487-F21174 | parent_cluster / parent_cluster | 0.6954 | 0.3046 | 1.0 | [17487, 21174] |
| F18573-F19139 | hedge_family / hedge_family | None | None | None | [] |
| F18573-F20804 | hedge_family / hedge_family | None | None | None | [] |
| F18573-F21174 | hedge_family / parent_cluster | 0.1577 | 0.8379 | 0.9947762080018995 | [18573, 16152, 21174] |
| F19139-F20804 | hedge_family / hedge_family | None | 2.3840000000000003 | None | [19139, 21714, 4012, 20804] |
| F19139-F21174 | hedge_family / parent_cluster | None | None | None | [] |
| F20804-F21174 | hedge_family / parent_cluster | None | None | None | [] |

## Observation / inference / hypothesis separation
- **observation**: Parent cluster nodes occupy component ids [6]; hedge-family nodes occupy component ids [0, 6] under threshold 0.13. **Implication:** If parent and hedge nodes share/approach components this motivates local graph tests; if separated it weakens simple single-manifold story.
- **observation**: F19139 and F11969 remain a very strong decoder-neighbor pair in Neuronpedia top-k data (cos≈0.8423). **Implication:** Supports a structured hedge sub-neighborhood; not sufficient for H7 because ordinary feature-family clustering also predicts this.
- **observation**: Parent cluster has high internal decoder-neighbor cosines where known (F21174-F16152≈0.9385; F17487 with both≈0.67-0.70). **Implication:** Supports a parent cluster, already consistent with Supplement 3/4 hedge interpretation.
- **observation**: F1803 has degree 10, component size 11, and top neighbors [14355, 23746, 23638, 7520, 10687]. **Implication:** F1803 has a local decoder neighborhood suitable for no-training follow-up, but this graph alone does not establish a temporal manifold.

## Conservative conclusion
Expanded Neuronpedia graph provides stronger descriptive evidence that hedge/parent/control features live in nontrivial local neighborhoods. It does not prove H7. The graph is a top-k decoder-neighbor proxy and must be followed by vector-based local PCA or broader API crawl before making manifold-fragmentation claims.