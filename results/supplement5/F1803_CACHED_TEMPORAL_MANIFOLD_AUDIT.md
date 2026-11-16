# F1803 cached temporal-manifold audit for H7

Status: cached-data audit only; no new model passes; conservative.

## Available files
- `<prior project, not included>/proyecto_hub_cosine/resultados/COS3_multidim_temporal.json` — exists=True, bytes=206562
- `<prior project, not included>/proyecto_hub_cosine/resultados/ANOM_2023_results.json` — exists=True, bytes=21479
- `<prior project, not included>/proyecto_hub_cosine/figuras/COS3_temporal_manifold_pca.png` — exists=True, bytes=510479
- `<prior project, not included>/proyecto_hub_cosine/figuras/COS3_year_trajectory.png` — exists=True, bytes=504888

## Observation / inference / hypothesis separation
- **observation**: Cached COS3 result exists with script, JSON, and two figures in proyecto_hub_cosine. Basis: <prior project, not included>/proyecto_hub_cosine/resultados/COS3_multidim_temporal.json
- **observation**: Prior report states F1803 temporal PCA: PC1=30.5%, PC2=26.6%, PC3=23.7%, PC4=19.1%; PC2/PC1≈0.87, above Engels multidimensional threshold 0.30. Basis: proyecto_hub_cosine/reportes/REPORTE_MAESTRO.md and project memory; verify fields in COS3 JSON before formal reuse.
- **observation**: Prior report states steering along PC1 increased logit_diff by about +0.43 (2.13 to 2.55). Basis: proyecto_hub_cosine/reportes/REPORTE_MAESTRO.md; this is causal relevance evidence for temporal geometry but not H7 proof.
- **observation**: ANOM_2023 cached JSON exists and prior result resolves 2023 anomaly as BPE tokenization plus weak L0_MLP routing. Basis: <prior project, not included>/proyecto_hub_cosine/resultados/ANOM_2023_results.json
- **inference**: F1803 is a strong candidate for Workstream D: a temporal-coordinate/manifold case study that can be reanalyzed without new SAE training if cached vectors/PCA scores are recoverable. Basis: COS3 JSON + report + figures.
- **hypothesis**: F1803 may be a coordinate/shard of a temporal representation manifold rather than a single isolated scalar feature. Basis: Compatible with multidimensional PCA and Goodfire/Engels framing; not demonstrated by current audit alone.

## COS3 relevant JSON fields sampled
- `/metadata/years`: `[1990, 1995, 2000, 2005, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2030]`
- `/metadata/n_prompts_per_year`: `20`
- `/metadata/cutoff_year`: `2019`
- `/pca_primary`: `dict[6]`
- `/pca_primary/explained_variance`: `[0.30542798377775077, 0.26633120849904407, 0.23739646467100872, 0.19084434305219652]`
- `/pca_primary/explained_variance_cumsum`: `[0.30542798377775077, 0.5717591922767948, 0.8091556569478036, 1.0]`
- `/pca_primary/metrics/ratio_PC2_PC1`: `0.8719934735673859`
- `/pca_primary/metrics/spearman_PC1_year`: `0.5759613441744416`
- `/pca_primary/metrics/corr_PC1_PC2`: `1.734723475976807e-17`
- `/pca_primary/metrics/mean_PC1_ID`: `-0.45784655966490506`
- `/pca_primary/metrics/mean_PC1_OOD`: `0.6540665138070085`
- `/pca_cluster`: `dict[6]`
- `/pca_cluster/explained_variance`: `[0.13696214952472205, 0.11426663715953438, 0.08504941367289028, 0.07140846981016949]`
- `/pca_cluster/explained_variance_cumsum`: `[0.13696214952472205, 0.2512287866842564, 0.3362782003571467, 0.4076866701673162]`
- `/pca_cluster/metrics/ratio_PC2_PC1`: `0.8342935442825313`
- `/pca_cluster/metrics/spearman_PC1_year`: `0.6602306739284082`
- `/pca_cluster/metrics/corr_PC1_PC2`: `-3.885780586188048e-16`
- `/pca_cluster/metrics/mean_PC1_ID`: `-0.9109930251108761`
- `/pca_cluster/metrics/mean_PC1_OOD`: `1.3014186073012524`
- `/activation_mean_by_year`: `dict[17]`
- `/activation_mean_by_year/1990/F1803`: `0.0`
- `/activation_mean_by_year/1995/F1803`: `0.0`
- `/activation_mean_by_year/2000/F1803`: `0.03302620947360992`
- `/activation_mean_by_year/2005/F1803`: `0.0`
- `/activation_mean_by_year/2010/F1803`: `0.0`
- `/activation_mean_by_year/2015/F1803`: `0.0051861807703971865`
- `/activation_mean_by_year/2016/F1803`: `0.018425148725509644`
- `/activation_mean_by_year/2017/F1803`: `0.047941967844963074`
- `/activation_mean_by_year/2018/F1803`: `0.052466651797294615`
- `/activation_mean_by_year/2019/F1803`: `18.976384973526002`
- `/activation_mean_by_year/2020/F1803`: `34.84526405334473`
- `/activation_mean_by_year/2021/F1803`: `33.621073150634764`
- `/activation_mean_by_year/2022/F1803`: `29.973648452758788`
- `/activation_mean_by_year/2023`: `dict[7]`
- `/activation_mean_by_year/2023/F1803`: `1.78777776658535`
- `/activation_mean_by_year/2024/F1803`: `31.69462823867798`
- `/activation_mean_by_year/2025/F1803`: `38.789368629455566`
- `/activation_mean_by_year/2030/F1803`: `40.07375564575195`
- `/trajectory_primary_pca`: `dict[17]`
- `/trajectory_primary_pca/2023`: `[0.013547878956203158, -0.9856840498109284, -0.4662283349936289, -0.034803847690985594]`

## ANOM_2023 relevant JSON fields sampled
- `/b_enc_f1803`: `-0.2641870379447937`
- `/T1_tokenizacion`: `dict[12]`
- `/T1_tokenizacion/2019/n_tokens`: `1`
- `/T1_tokenizacion/2019/token_strings`: `[' 2019']`
- `/T1_tokenizacion/2019/token_ids`: `[13130]`
- `/T1_tokenizacion/2020/n_tokens`: `1`
- `/T1_tokenizacion/2020/token_strings`: `[' 2020']`
- `/T1_tokenizacion/2020/token_ids`: `[12131]`
- `/T1_tokenizacion/2021/n_tokens`: `1`
- `/T1_tokenizacion/2021/token_strings`: `[' 2021']`
- `/T1_tokenizacion/2021/token_ids`: `[33448]`
- `/T1_tokenizacion/2022/n_tokens`: `1`
- `/T1_tokenizacion/2022/token_strings`: `[' 2022']`
- `/T1_tokenizacion/2022/token_ids`: `[33160]`
- `/T1_tokenizacion/2023`: `dict[4]`
- `/T1_tokenizacion/2023/n_tokens`: `2`
- `/T1_tokenizacion/2023/token_strings`: `[' 20', '23']`
- `/T1_tokenizacion/2023/token_ids`: `[1160, 1954]`
- `/T1_tokenizacion/2024`: `dict[4]`
- `/T1_tokenizacion/2024/n_tokens`: `1`
- `/T1_tokenizacion/2024/token_strings`: `[' 2024']`
- `/T1_tokenizacion/2024/token_ids`: `[48609]`
- `/T1_tokenizacion/2025/n_tokens`: `1`
- `/T1_tokenizacion/2025/token_strings`: `[' 2025']`
- `/T1_tokenizacion/2025/token_ids`: `[32190]`
- `/T1_tokenizacion/2026/n_tokens`: `2`
- `/T1_tokenizacion/2026/token_strings`: `[' 20', '26']`
- `/T1_tokenizacion/2026/token_ids`: `[1160, 2075]`
- `/T1_tokenizacion/2027/n_tokens`: `2`
- `/T1_tokenizacion/2027/token_strings`: `[' 20', '27']`
- `/T1_tokenizacion/2027/token_ids`: `[1160, 1983]`
- `/T1_tokenizacion/2028/n_tokens`: `2`
- `/T1_tokenizacion/2028/token_strings`: `[' 20', '28']`
- `/T1_tokenizacion/2028/token_ids`: `[1160, 2078]`
- `/T1_tokenizacion/2029/n_tokens`: `2`
- `/T1_tokenizacion/2029/token_strings`: `[' 20', '29']`
- `/T1_tokenizacion/2029/token_ids`: `[1160, 1959]`
- `/T1_tokenizacion/2030/n_tokens`: `1`
- `/T1_tokenizacion/2030/token_strings`: `[' 2030']`
- `/T1_tokenizacion/2030/token_ids`: `[25054]`

## Conservative interpretation
Cached F1803/COS3 assets are sufficient for a no-training reanalysis plan and possibly a Supplement 6 section, but not yet enough to claim H7. The next step is to parse COS3 JSON into a reproducible table of PCA variance, year coordinates, tokenization confounds, and steering metrics.