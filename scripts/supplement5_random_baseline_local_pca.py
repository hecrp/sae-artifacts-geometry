#!/usr/bin/env python3
"""
Supplement 5 / H7 random baseline for local PCA diagnostics.

No SAE training. Loads the public Bloom GPT-2 Small L8 residual SAE and compares
H7 target groups against random and simple matched baselines.

Outputs:
  results/supplement5/random_baseline_local_pca.json
  results/supplement5/RANDOM_BASELINE_LOCAL_PCA.md
  figures/supplement5/random_baseline_local_pca_{metric}.png/pdf
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "supplement5"
FIGURES = PROJECT / "figures" / "supplement5"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
RNG_SEED = 20260509
N_RANDOM = 1000

SEED_GROUPS = {
    "hedge_family": [19139, 18573, 11969, 20804],
    "parent_cluster": [17487, 21174, 16152],
    "random_l8": [22561, 6165, 5977, 3696],
    "controls": [1803, 1442, 10666],
}
COMPOSITE_GROUPS = {
    "hedge_plus_parent": SEED_GROUPS["hedge_family"] + SEED_GROUPS["parent_cluster"],
}
ALL_GROUPS = {**SEED_GROUPS, **COMPOSITE_GROUPS}


def load_sae_weights():
    from sae_lens import SAE
    print(f"Loading SAE: {RELEASE} / {SAE_ID}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sae = SAE.from_pretrained(RELEASE, SAE_ID, device="cpu")
    W_enc = sae.W_enc.detach().float().cpu()  # d_model x d_sae
    W_dec = sae.W_dec.detach().float().cpu()  # d_sae x d_model
    return W_enc, W_dec


def pca_metrics(X: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        return {"participation_ratio": None, "pc2_pc1_ratio": None, "rank95": None, "rank99": None, "evr": []}
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    eig = (s ** 2) / max(n - 1, 1)
    total = eig.sum()
    evr = eig / total if total > 0 else np.zeros_like(eig)
    pr = float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12)) if total > 0 else 0.0
    cumsum = np.cumsum(evr)
    return {
        "participation_ratio": pr,
        "pc2_pc1_ratio": float(evr[1] / evr[0]) if len(evr) > 1 and evr[0] > 0 else None,
        "rank95": int(np.searchsorted(cumsum, 0.95) + 1) if len(cumsum) else None,
        "rank99": int(np.searchsorted(cumsum, 0.99) + 1) if len(cumsum) else None,
        "evr": [float(x) for x in evr[:10]],
    }


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-9)


def group_metrics(V: torch.Tensor, nodes: List[int]) -> Dict[str, float]:
    X = V[nodes].numpy()
    pca = pca_metrics(X)
    U = normalize_rows(V[nodes])
    C = (U @ U.T).numpy()
    vals = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            vals.append(float(C[i, j]))
    pca.update({
        "mean_pairwise_cos": float(np.mean(vals)) if vals else None,
        "std_pairwise_cos": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min_pairwise_cos": float(np.min(vals)) if vals else None,
        "max_pairwise_cos": float(np.max(vals)) if vals else None,
    })
    return pca


def percentile(value: float, samples: np.ndarray) -> float:
    return float(100.0 * np.mean(samples <= value))


def two_sided_extremeness(value: float, samples: np.ndarray) -> float:
    p = percentile(value, samples) / 100.0
    return float(2 * min(p, 1 - p))


def sample_random_groups(rng, d_sae: int, k: int, excluded: set[int], n: int) -> List[List[int]]:
    universe = np.array([i for i in range(d_sae) if i not in excluded], dtype=np.int64)
    groups = []
    for _ in range(n):
        groups.append([int(x) for x in rng.choice(universe, size=k, replace=False)])
    return groups


def sample_matched_groups(rng, cos_ed: np.ndarray, enc_norms: np.ndarray, target_nodes: List[int], excluded: set[int], n: int) -> List[List[int]]:
    """Simple matching: for each target feature, sample from same cos_ED decile and enc_norm decile."""
    d_sae = len(cos_ed)
    cos_bins = np.quantile(cos_ed, np.linspace(0, 1, 11))
    norm_bins = np.quantile(enc_norms, np.linspace(0, 1, 11))
    candidates_by_target = []
    for t in target_nodes:
        c = cos_ed[t]
        en = enc_norms[t]
        cb = min(9, max(0, int(np.searchsorted(cos_bins, c, side="right") - 1)))
        nb = min(9, max(0, int(np.searchsorted(norm_bins, en, side="right") - 1)))
        lo_c, hi_c = cos_bins[cb], cos_bins[cb + 1]
        lo_n, hi_n = norm_bins[nb], norm_bins[nb + 1]
        mask = (cos_ed >= lo_c) & (cos_ed <= hi_c) & (enc_norms >= lo_n) & (enc_norms <= hi_n)
        cand = [int(i) for i in np.where(mask)[0] if int(i) not in excluded and int(i) not in target_nodes]
        # If bin too small, relax to cos decile only.
        if len(cand) < 20:
            mask = (cos_ed >= lo_c) & (cos_ed <= hi_c)
            cand = [int(i) for i in np.where(mask)[0] if int(i) not in excluded and int(i) not in target_nodes]
        # If still too small, use full universe.
        if len(cand) < 20:
            cand = [int(i) for i in range(d_sae) if int(i) not in excluded and int(i) not in target_nodes]
        candidates_by_target.append(cand)
    groups = []
    for _ in range(n):
        chosen = []
        used = set()
        for cand in candidates_by_target:
            # avoid duplicates inside group
            available = [x for x in cand if x not in used]
            if not available:
                available = [int(i) for i in range(d_sae) if int(i) not in used and int(i) not in excluded]
            x = int(rng.choice(available))
            chosen.append(x)
            used.add(x)
        groups.append(chosen)
    return groups


def run_baseline_for_group(name: str, nodes: List[int], V_dec: torch.Tensor, V_enc: torch.Tensor, cos_ed: np.ndarray, enc_norms: np.ndarray, rng) -> Dict:
    d_sae = V_dec.shape[0]
    excluded = set(n for group in ALL_GROUPS.values() for n in group)
    k = len(nodes)
    observed_dec = group_metrics(V_dec, nodes)
    observed_enc = group_metrics(V_enc, nodes)
    random_groups = sample_random_groups(rng, d_sae, k, excluded, N_RANDOM)
    matched_groups = sample_matched_groups(rng, cos_ed, enc_norms, nodes, excluded, N_RANDOM)

    def collect(groups, V):
        metrics = [group_metrics(V, g) for g in groups]
        return {key: np.array([m[key] for m in metrics if m[key] is not None], dtype=np.float64)
                for key in ["participation_ratio", "pc2_pc1_ratio", "rank95", "mean_pairwise_cos", "max_pairwise_cos", "min_pairwise_cos"]}

    rand_dec = collect(random_groups, V_dec)
    match_dec = collect(matched_groups, V_dec)
    rand_enc = collect(random_groups, V_enc)
    match_enc = collect(matched_groups, V_enc)

    def summarize_dist(dist, observed):
        out = {}
        for key, arr in dist.items():
            val = observed.get(key)
            out[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "p05": float(np.quantile(arr, 0.05)),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "observed": val,
                "percentile": percentile(val, arr) if val is not None else None,
                "two_sided_extremeness": two_sided_extremeness(val, arr) if val is not None else None,
            }
        return out

    return {
        "name": name,
        "nodes": nodes,
        "k": k,
        "observed_decoder": observed_dec,
        "observed_encoder": observed_enc,
        "random_decoder": summarize_dist(rand_dec, observed_dec),
        "matched_decoder": summarize_dist(match_dec, observed_dec),
        "random_encoder": summarize_dist(rand_enc, observed_enc),
        "matched_encoder": summarize_dist(match_enc, observed_enc),
    }


def plot_metric(results: Dict[str, Dict], metric: str, baseline: str, space: str):
    labels = list(results.keys())
    obs = [results[g][f"observed_{space}"][metric] for g in labels]
    means = [results[g][f"{baseline}_{space}"][metric]["mean"] for g in labels]
    p05 = [results[g][f"{baseline}_{space}"][metric]["p05"] for g in labels]
    p95 = [results[g][f"{baseline}_{space}"][metric]["p95"] for g in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(p05), np.array(p95) - np.array(means)], fmt="o", label=f"{baseline} baseline 5-95%")
    ax.scatter(x, obs, color="red", zorder=3, label="observed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{space} {metric}: observed vs {baseline} baseline")
    ax.legend()
    fig.tight_layout()
    stem = FIGURES / f"random_baseline_local_pca_{space}_{baseline}_{metric}"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    W_enc, W_dec = load_sae_weights()
    V_enc = W_enc.T.contiguous()
    V_dec = W_dec.contiguous()
    cos_ed = F.cosine_similarity(V_enc, V_dec, dim=1).numpy()
    enc_norms = V_enc.norm(dim=1).numpy()
    rng = np.random.default_rng(RNG_SEED)

    results = {}
    for name, nodes in ALL_GROUPS.items():
        print("Analyzing", name, nodes)
        results[name] = run_baseline_for_group(name, nodes, V_dec, V_enc, cos_ed, enc_norms, rng)

    package = {
        "metadata": {
            "release": RELEASE,
            "sae_id": SAE_ID,
            "n_random": N_RANDOM,
            "rng_seed": RNG_SEED,
            "W_enc_shape": list(W_enc.shape),
            "W_dec_shape": list(W_dec.shape),
            "global_cos_mean": float(np.mean(cos_ed)),
            "global_cos_std": float(np.std(cos_ed, ddof=1)),
            "note": "No SAE training. Random and simple cos/norm matched baselines for local PCA metrics.",
        },
        "results": results,
        "conservative_conclusion": "",
    }

    # Generate conservative conclusion from hedge_plus_parent.
    hp = results["hedge_plus_parent"]
    hp_pr = hp["observed_decoder"]["participation_ratio"]
    hp_pr_pct = hp["matched_decoder"]["participation_ratio"]["percentile"]
    hp_pc2 = hp["observed_decoder"]["pc2_pc1_ratio"]
    hp_pc2_pct = hp["matched_decoder"]["pc2_pc1_ratio"]["percentile"]
    package["conservative_conclusion"] = (
        f"Against simple cos/norm matched baselines, hedge_plus_parent decoder PR={hp_pr:.3f} "
        f"is at percentile {hp_pr_pct:.1f}, and PC2/PC1={hp_pc2:.3f} is at percentile {hp_pc2_pct:.1f}. "
        "This calibrates the local-dimensionality result. Interpret as evidence about geometric atypicality only, not as proof of H7."
    )

    json_path = RESULTS / "random_baseline_local_pca.json"
    json_path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figures.
    for metric in ["participation_ratio", "pc2_pc1_ratio", "mean_pairwise_cos"]:
        for baseline in ["random", "matched"]:
            for space in ["decoder", "encoder"]:
                plot_metric(results, metric, baseline, space)

    # Markdown.
    md = []
    md.append("# Random / matched baseline for local PCA H7 diagnostics")
    md.append("")
    md.append("Status: no-training baseline using public GPT-2 Small L8 SAE. Conservative: calibrates local PCA, does not prove H7.")
    md.append("")
    md.append("## Metadata")
    for k, v in package["metadata"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Main decoder metrics vs matched baseline")
    md.append("| Group | k | obs PR | matched PR pct | obs PC2/PC1 | matched PC2/PC1 pct | obs mean cos | matched mean cos pct |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        md.append(
            f"| {name} | {r['k']} | {r['observed_decoder']['participation_ratio']:.3f} | {r['matched_decoder']['participation_ratio']['percentile']:.1f} | "
            f"{r['observed_decoder']['pc2_pc1_ratio']:.3f} | {r['matched_decoder']['pc2_pc1_ratio']['percentile']:.1f} | "
            f"{r['observed_decoder']['mean_pairwise_cos']:.3f} | {r['matched_decoder']['mean_pairwise_cos']['percentile']:.1f} |"
        )
    md.append("")
    md.append("## Main encoder metrics vs matched baseline")
    md.append("| Group | k | obs PR | matched PR pct | obs PC2/PC1 | matched PC2/PC1 pct | obs mean cos | matched mean cos pct |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        md.append(
            f"| {name} | {r['k']} | {r['observed_encoder']['participation_ratio']:.3f} | {r['matched_encoder']['participation_ratio']['percentile']:.1f} | "
            f"{r['observed_encoder']['pc2_pc1_ratio']:.3f} | {r['matched_encoder']['pc2_pc1_ratio']['percentile']:.1f} | "
            f"{r['observed_encoder']['mean_pairwise_cos']:.3f} | {r['matched_encoder']['mean_pairwise_cos']['percentile']:.1f} |"
        )
    md.append("")
    md.append("## Conservative conclusion")
    md.append(package["conservative_conclusion"])
    md.append("")
    md.append("## Interpretation discipline")
    md.append("- Observation: percentiles compare target groups against random/matched groups of the same size.")
    md.append("- Inference: extreme percentiles indicate atypical local geometry under the chosen baseline.")
    md.append("- Hypothesis: H7 becomes stronger only if anti-aligned/hedge neighborhoods are atypical in ways predicted by manifold fragmentation and not mirrored by controls.")
    md.append("- Not claimed: atypical PCA metrics alone do not prove manifold shattering.")
    md_path = RESULTS / "RANDOM_BASELINE_LOCAL_PCA.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(package["conservative_conclusion"])


if __name__ == "__main__":
    main()
