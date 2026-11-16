# Neuronpedia queries (Supplement 4)

Scripts used to retrieve feature metadata from the [Neuronpedia](https://www.neuronpedia.org) API and its public S3 exports: activation frequency, auto-interpretability explanations, logit effects and decoder neighbours. The snapshot they produced on 2026-03-04 is stored in `results/supplement4/`.

The Supplement 4 analysis itself does not depend on these scripts. `scripts/supplement4_verified_analysis.py` recomputes geometry and logit effects from the SAE weights and reads only activation frequencies and explanations from the stored snapshot.

## Setup

An API key is required for the live API (not for the S3 exports).

```bash
cp scripts/neuronpedia/.env.example scripts/neuronpedia/.env
# set NEURONPEDIA_API_KEY in .env (https://www.neuronpedia.org/account)
```

The clients read `NEURONPEDIA_API_KEY` from the environment or from `scripts/neuronpedia/.env`. The `.env` file is git-ignored.

## Scripts

| File | Purpose | Output |
|------|---------|--------|
| `run_full_analysis.py` | Profiles for the layer-8 hedge and parent features, extreme features, controls and random layer-8 features. Built on `neuronpedia_client_v3.py`. | `results/supplement4/full_analysis_*.json` |
| `query_anomalous_enhanced.py` | Earlier query tool built on `neuronpedia_client_v2.py`. | `results/supplement4/reference_features_*.json` |
| `download_and_analyze_explanations.py` | Downloads the layer-8 explanation export from S3 and summarises it for the layer-8 groups. Fixed in November 2026 to use layer-8 indices only. The earlier output mixed layers and was withdrawn; the fixed script has not been re-run. | `results/supplement4/s3_explanations_analysis.json` (not present) |
| `feature_registry.py` | Static list of negative-cosine features from the Phase 0 top-10-per-layer tables. | - |
| `query_anomalous.py`, `batch_analyze.py`, `neuronpedia_client.py` | First-version tools, kept for provenance. | - |

`results/supplement4/deep_analysis_results.json` contains explanation-search and top-k-by-token queries from an interactive session; the script that produced it was not preserved.

## Feature groups in the snapshot

Labels in `run_full_analysis.py` do not all match the measured cos(E, D). `results/supplement4/queried_features.csv` gives the measured value for every queried feature.

| Group in the snapshot | Features | Anti-aligned (cos < 0, <= mu - 6 sigma)? |
|-----------------------|----------|-------------------------------------------|
| `gpt2_hedge_family` | L8: 19139, 18573, 11969, 20804 | yes |
| `gpt2_parents` | L8: 17487, 21174, 16152 | no (positive, up to +5.8 sigma) |
| `gpt2_extreme` | L9: 18298; L10: 19679, 3268 | yes |
| `gpt2_extreme` | L11: 4855; L4: 18458; L6: 7685; L7: 21750 | no |
| `gpt2_controls` | L8: 1803, 1442, 10666 | no |
| `gpt2_extra_l8` | L8: 5977, 22561, 6165, 3696 (random sample) | no |
| `llama_extreme` | L1: 27745, 24381 | yes (no Neuronpedia data returned) |
| `llama_extreme` | L0: 15649; L2: 30420 | no (no Neuronpedia data returned) |
| `gemma_single` | L1: 3378 | yes |

## Known API limitations (March 2026)

- The connected-neurons (sparsity) endpoint only served GPT-2 layers 0-7.
- Dead features have no activating examples and therefore no explanations.
- Llama Scope SAEs were listed on Neuronpedia but returned server errors; their profiles in the snapshot are empty.
