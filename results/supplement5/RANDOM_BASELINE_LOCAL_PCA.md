# Random / matched baseline for local PCA H7 diagnostics

Status: no-training baseline using public GPT-2 Small L8 SAE. Conservative: calibrates local PCA, does not prove H7.

## Metadata
- release: gpt2-small-res-jb
- sae_id: blocks.8.hook_resid_pre
- n_random: 1000
- rng_seed: 20260509
- W_enc_shape: [768, 24576]
- W_dec_shape: [24576, 768]
- global_cos_mean: 0.36828580498695374
- global_cos_std: 0.07853515446186066
- note: No SAE training. Random and simple cos/norm matched baselines for local PCA metrics.

## Main decoder metrics vs matched baseline
| Group | k | obs PR | matched PR pct | obs PC2/PC1 | matched PC2/PC1 pct | obs mean cos | matched mean cos pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| hedge_family | 4 | 1.937 | 0.0 | 0.472 | 0.1 | 0.057 | 91.7 |
| parent_cluster | 3 | 1.296 | 0.0 | 0.151 | 0.0 | 0.768 | 100.0 |
| random_l8 | 4 | 2.960 | 13.9 | 0.872 | 15.2 | 0.014 | 64.6 |
| controls | 3 | 1.995 | 44.4 | 0.904 | 44.4 | -0.011 | 28.6 |
| hedge_plus_parent | 7 | 2.226 | 0.0 | 0.363 | 0.0 | -0.131 | 0.0 |

## Main encoder metrics vs matched baseline
| Group | k | obs PR | matched PR pct | obs PC2/PC1 | matched PC2/PC1 pct | obs mean cos | matched mean cos pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| hedge_family | 4 | 1.888 | 0.2 | 0.559 | 7.5 | 0.839 | 100.0 |
| parent_cluster | 3 | 1.599 | 0.0 | 0.333 | 0.0 | 0.791 | 100.0 |
| random_l8 | 4 | 2.695 | 26.5 | 0.609 | 25.8 | 0.670 | 98.1 |
| controls | 3 | 1.996 | 60.0 | 0.912 | 60.0 | 0.643 | 69.3 |
| hedge_plus_parent | 7 | 3.495 | 45.8 | 0.526 | 61.9 | 0.821 | 100.0 |

## Conservative conclusion
Against simple cos/norm matched baselines, hedge_plus_parent decoder PR=2.226 is at percentile 0.0, and PC2/PC1=0.363 is at percentile 0.0. This calibrates the local-dimensionality result. Interpret as evidence about geometric atypicality only, not as proof of H7.

## Interpretation discipline
- Observation: percentiles compare target groups against random/matched groups of the same size.
- Inference: extreme percentiles indicate atypical local geometry under the chosen baseline.
- Hypothesis: H7 becomes stronger only if anti-aligned/hedge neighborhoods are atypical in ways predicted by manifold fragmentation and not mirrored by controls.
- Not claimed: atypical PCA metrics alone do not prove manifold shattering.