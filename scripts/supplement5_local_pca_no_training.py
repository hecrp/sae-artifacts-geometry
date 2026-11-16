#!/usr/bin/env python3
"""
Supplement 5 / H7 no-training local PCA diagnostics.

Loads the public Bloom GPT-2 Small L8 residual SAE with SAELens and analyzes
local decoder/encoder geometry for H7 candidate neighborhoods. No SAE training.

Outputs:
  results/supplement5/local_pca_decoder_neighborhoods.json
  results/supplement5/LOCAL_PCA_DECODER_NEIGHBORHOODS.md
  figures/supplement5/local_pca_*.png/pdf
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
NEURONPEDIA_GRAPH = RESULTS / "neuronpedia_neighbor_graph_analysis.json"
NEURONPEDIA_PROFILES = RESULTS / "neuronpedia_neighbor_profiles_gpt2_l8_depth1.json"

SEED_GROUPS = {
    "hedge_family": [19139, 18573, 11969, 20804],
    "parent_cluster": [17487, 21174, 16152],
    "random_l8": [22561, 6165, 5977, 3696],
    "controls": [1803, 1442, 10666],
}

MANUAL_HEDGE_PARENT_NEGATIVE = [
    (19139, 21174, -0.705),
    (11969, 16152, -0.480),
    (11969, 21174, -0.312),
    (19139, 16152, -0.198),
    (19139, 17487, -0.105),
    (11969, 17487, -0.091),
]


def load_sae_weights():
    from sae_lens import SAE

    print(f"Loading SAE: release={RELEASE}, sae_id={SAE_ID}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sae = SAE.from_pretrained(RELEASE, SAE_ID, device="cpu")
    W_enc = sae.W_enc.detach().float().cpu()  # (d_model, d_sae)
    W_dec = sae.W_dec.detach().float().cpu()  # (d_sae, d_model)
    print("W_enc", tuple(W_enc.shape), "W_dec", tuple(W_dec.shape))
    return W_enc, W_dec


def normalize_rows(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def pca_stats(X: np.ndarray, center: bool = True) -> Dict:
    """PCA via SVD. X shape: n_points x d."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if center:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X.copy()
    # If n=1, PCA undefined
    if n < 2:
        return {
            "n": n,
            "d": d,
            "explained_variance": [],
            "participation_ratio": None,
            "rank_95": None,
            "rank_99": None,
            "pc2_pc1_ratio": None,
        }
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = (s ** 2) / max(n - 1, 1)
    total = float(eig.sum())
    evr = eig / total if total > 0 else np.zeros_like(eig)
    pr = float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12)) if total > 0 else 0.0
    cumsum = np.cumsum(evr)
    rank95 = int(np.searchsorted(cumsum, 0.95) + 1) if len(cumsum) else None
    rank99 = int(np.searchsorted(cumsum, 0.99) + 1) if len(cumsum) else None
    ratio = float(evr[1] / evr[0]) if len(evr) > 1 and evr[0] > 0 else None
    return {
        "n": int(n),
        "d": int(d),
        "explained_variance": [float(x) for x in evr[:10]],
        "participation_ratio": pr,
        "rank_95": rank95,
        "rank_99": rank99,
        "pc2_pc1_ratio": ratio,
    }


def pairwise_cos_stats(V: torch.Tensor, nodes: List[int]) -> Dict:
    if len(nodes) < 2:
        return {"n_pairs": 0}
    X = normalize_rows(V[nodes])
    C = (X @ X.T).numpy()
    vals = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            vals.append(float(C[i, j]))
    return {
        "n_pairs": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "values": vals,
    }


def group_vectors(W_enc: torch.Tensor, W_dec: torch.Tensor, nodes: List[int]) -> Dict:
    E = W_enc.T  # d_sae x d_model
    D = W_dec
    return {
        "decoder_pca": pca_stats(D[nodes].numpy(), center=True),
        "encoder_pca": pca_stats(E[nodes].numpy(), center=True),
        "decoder_pairwise_cos": pairwise_cos_stats(D, nodes),
        "encoder_pairwise_cos": pairwise_cos_stats(E, nodes),
    }


def load_profiles() -> Dict[int, Dict]:
    if not NEURONPEDIA_PROFILES.exists():
        return {}
    data = json.loads(NEURONPEDIA_PROFILES.read_text())
    return {int(k): v for k, v in data.get("profiles", {}).items()}


def neighbor_nodes_from_profiles(profiles: Dict[int, Dict], seed: int, k: int = 10, threshold: float = 0.13) -> List[int]:
    p = profiles.get(seed, {})
    nodes = [seed]
    for idx, val in zip(p.get("top_similar_indices") or [], p.get("top_similar_values") or []):
        try:
            idx = int(idx)
            val = float(val)
        except Exception:
            continue
        if idx == seed:
            continue
        if val >= threshold:
            nodes.append(idx)
        if len(nodes) >= k + 1:
            break
    # stable unique
    out = []
    seen = set()
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def tangent_normal_for_pair(W_enc: torch.Tensor, W_dec: torch.Tensor, feature: int, neighborhood: List[int], n_tangent: int = 2) -> Dict:
    """Decompose E_f and D_f relative to local decoder PCA tangent subspace."""
    E = W_enc.T
    D = W_dec
    if len(neighborhood) < 3:
        return {"available": False, "reason": "neighborhood too small"}
    X = D[neighborhood].numpy().astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_tangent, vt.shape[0])
    basis = torch.tensor(vt[:k], dtype=torch.float32)  # k x d_model

    def decomp(v: torch.Tensor) -> Tuple[float, float, float]:
        v = v.float()
        coeff = basis @ v
        par = coeff @ basis
        perp = v - par
        return float(v.norm()), float(par.norm()), float(perp.norm())

    E_norm, E_par, E_perp = decomp(E[feature])
    D_norm, D_par, D_perp = decomp(D[feature])
    return {
        "available": True,
        "feature": feature,
        "neighborhood": neighborhood,
        "n_tangent": k,
        "singular_values": [float(x) for x in s[:10]],
        "E_norm": E_norm,
        "E_parallel_norm": E_par,
        "E_perp_norm": E_perp,
        "E_parallel_frac": E_par / E_norm if E_norm > 0 else None,
        "D_norm": D_norm,
        "D_parallel_norm": D_par,
        "D_perp_norm": D_perp,
        "D_parallel_frac": D_par / D_norm if D_norm > 0 else None,
    }


def plot_group_pca(W_dec: torch.Tensor, groups: Dict[str, List[int]], path_stem: Path):
    all_nodes = []
    labels = []
    for g, nodes in groups.items():
        for n in nodes:
            all_nodes.append(n)
            labels.append(g)
    X = W_dec[all_nodes].numpy().astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt[:2].T
    colors = {
        "hedge_family": "#d62728",
        "parent_cluster": "#1f77b4",
        "random_l8": "#9467bd",
        "controls": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in sorted(set(labels)):
        idxs = [i for i, lab in enumerate(labels) if lab == g]
        ax.scatter(coords[idxs, 0], coords[idxs, 1], label=g, s=70, color=colors.get(g, None), alpha=0.85)
        for i in idxs:
            ax.text(coords[i, 0], coords[i, 1], f"F{all_nodes[i]}", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("Decoder PCA: H7 seed feature groups (GPT-2 L8 SAE)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    W_enc, W_dec = load_sae_weights()
    profiles = load_profiles()
    result = {
        "metadata": {
            "release": RELEASE,
            "sae_id": SAE_ID,
            "W_enc_shape": list(W_enc.shape),
            "W_dec_shape": list(W_dec.shape),
            "status": "no SAE training; public pretrained SAE loaded for local geometry diagnostics",
        },
        "seed_groups": SEED_GROUPS,
        "group_pca": {},
        "neighborhood_pca": {},
        "tangent_normal": {},
        "manual_pair_cosines": [],
        "conservative_conclusion": "",
    }

    E = W_enc.T
    D = W_dec
    cos_ed = F.cosine_similarity(E, D, dim=1).numpy()
    result["metadata"]["global_cos_mean"] = float(np.mean(cos_ed))
    result["metadata"]["global_cos_std"] = float(np.std(cos_ed, ddof=1))

    # Group-level PCA.
    for group, nodes in SEED_GROUPS.items():
        result["group_pca"][group] = group_vectors(W_enc, W_dec, nodes)

    # Combined hedge + parent.
    result["group_pca"]["hedge_plus_parent"] = group_vectors(W_enc, W_dec, SEED_GROUPS["hedge_family"] + SEED_GROUPS["parent_cluster"])
    result["group_pca"]["all_seed_groups"] = group_vectors(W_enc, W_dec, sorted({n for nodes in SEED_GROUPS.values() for n in nodes}))

    # Neighborhood-level PCA for each target seed.
    for group, seeds in SEED_GROUPS.items():
        for seed in seeds:
            neigh = neighbor_nodes_from_profiles(profiles, seed, k=10, threshold=0.13)
            # keep only valid feature ids in SAE range
            neigh = [n for n in neigh if 0 <= n < W_dec.shape[0]]
            result["neighborhood_pca"][str(seed)] = {
                "group": group,
                "nodes": neigh,
                **group_vectors(W_enc, W_dec, neigh),
            }
            result["tangent_normal"][str(seed)] = tangent_normal_for_pair(W_enc, W_dec, seed, neigh, n_tangent=2)

    # Manual hedge-parent pair cosine from actual vectors.
    for a, b, supplement4_cos in MANUAL_HEDGE_PARENT_NEGATIVE:
        actual = float(F.cosine_similarity(D[a].unsqueeze(0), D[b].unsqueeze(0), dim=1).item())
        result["manual_pair_cosines"].append({"i": a, "j": b, "supplement4_reported_cos": supplement4_cos, "actual_Wdec_cos": actual, "abs_diff": abs(actual - supplement4_cos)})

    # Conservative interpretation.
    hedge_parent_pr = result["group_pca"]["hedge_plus_parent"]["decoder_pca"]["participation_ratio"]
    hedge_parent_pc2 = result["group_pca"]["hedge_plus_parent"]["decoder_pca"]["pc2_pc1_ratio"]
    parent_pr = result["group_pca"]["parent_cluster"]["decoder_pca"]["participation_ratio"]
    result["conservative_conclusion"] = (
        "Vector-based local PCA is now available for the public GPT-2 L8 SAE. "
        f"The hedge+parent decoder set has participation_ratio={hedge_parent_pr:.3f} and PC2/PC1={hedge_parent_pc2:.3f}; "
        f"the parent cluster has participation_ratio={parent_pr:.3f}. "
        "These values should be interpreted as local geometric diagnostics, not proof of H7. "
        "They can strengthen or weaken H7 only after comparison to matched controls and after checking whether the same structure persists under broader neighborhoods."
    )

    # Save JSON.
    json_path = RESULTS / "local_pca_decoder_neighborhoods.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plot.
    plot_group_pca(W_dec, SEED_GROUPS, FIGURES / "local_pca_seed_groups_decoder")

    # Markdown report.
    md = []
    md.append("# Local PCA decoder/encoder neighborhoods for H7")
    md.append("")
    md.append("Status: no-training vector-based diagnostic using public Bloom GPT-2 Small L8 SAE. Conservative: not proof of H7.")
    md.append("")
    md.append("## Metadata")
    for k, v in result["metadata"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Group-level PCA")
    md.append("| Group | n | Decoder PR | Decoder PC2/PC1 | Decoder rank95 | Encoder PR | Encoder PC2/PC1 | Mean decoder cosine |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for group, stats in result["group_pca"].items():
        dp = stats["decoder_pca"]
        ep = stats["encoder_pca"]
        dc = stats["decoder_pairwise_cos"]
        md.append(
            f"| {group} | {dp['n']} | {dp['participation_ratio']} | {dp['pc2_pc1_ratio']} | {dp['rank_95']} | "
            f"{ep['participation_ratio']} | {ep['pc2_pc1_ratio']} | {dc.get('mean')} |"
        )
    md.append("")
    md.append("## Seed-neighborhood PCA")
    md.append("| Feature | Group | n nodes | Decoder PR | Decoder PC2/PC1 | Encoder PR | Top nodes |")
    md.append("|---:|---|---:|---:|---:|---:|---|")
    for seed, stats in result["neighborhood_pca"].items():
        dp = stats["decoder_pca"]
        ep = stats["encoder_pca"]
        md.append(f"| F{seed} | {stats['group']} | {len(stats['nodes'])} | {dp['participation_ratio']} | {dp['pc2_pc1_ratio']} | {ep['participation_ratio']} | {stats['nodes'][:8]} |")
    md.append("")
    md.append("## Tangent/normal decomposition relative to local decoder PCA")
    md.append("| Feature | available | E_parallel_frac | D_parallel_frac | n_tangent |")
    md.append("|---:|---|---:|---:|---:|")
    for seed, stats in result["tangent_normal"].items():
        md.append(f"| F{seed} | {stats.get('available')} | {stats.get('E_parallel_frac')} | {stats.get('D_parallel_frac')} | {stats.get('n_tangent')} |")
    md.append("")
    md.append("## Actual W_dec pair cosines for Supplement 4 hedge-parent pairs")
    md.append("| Pair | Supplement 4 reported (withdrawn March 2026 version) | Actual W_dec cosine | abs diff |")
    md.append("|---|---:|---:|---:|")
    for row in result["manual_pair_cosines"]:
        md.append(f"| F{row['i']}-F{row['j']} | {row['supplement4_reported_cos']} | {row['actual_Wdec_cos']} | {row['abs_diff']} |")
    md.append("")
    md.append("## Conservative conclusion")
    md.append(result["conservative_conclusion"])
    md.append("")
    md.append("## Interpretation discipline")
    md.append("- Observation: local PCA/PR values are direct calculations on public SAE vectors.")
    md.append("- Inference: values >1 indicate local dimensionality beyond a single line, but small-n PCA is noisy.")
    md.append("- Hypothesis: if anti-aligned neighborhoods are consistently higher-dimensional than controls, this supports H7.")
    md.append("- Not claimed: this run alone does not demonstrate manifold fragmentation or SAE shattering.")
    md_path = RESULTS / "LOCAL_PCA_DECODER_NEIGHBORHOODS.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {FIGURES / 'local_pca_seed_groups_decoder.png'}")
    print(result["conservative_conclusion"])


if __name__ == "__main__":
    main()
