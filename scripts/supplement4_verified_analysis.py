#!/usr/bin/env python3
"""
Supplement 4 (revised): hedge-family geometry, logit effects and activity.

Recomputes every quantitative result of Supplement 4 from primary sources:

  - SAE weights: Bloom GPT-2 Small residual SAE, layer 8 (gpt2-small-res-jb,
    blocks.8.hook_resid_pre), loaded with SAELens.
  - Unembedding: GPT-2 Small W_U as loaded by TransformerLens (folded layer
    norm, centred unembedding). This reproduces Neuronpedia's logit lists exactly.
  - Per-layer cos(E, D) arrays from Phase 1 (results/phase1/cos_sim_*_L*.npy).
  - Neuronpedia snapshot of 2026-03-04 (results/supplement4/full_analysis_*.json),
    used only for activation frequency (frac_nonzero) and auto-interp explanations.

Outputs:
  results/supplement4/verified_analysis.json
  results/supplement4/queried_features.csv
  figures/supplement4/s4_decoder_cosine_matrix.{png,pdf}
  figures/supplement4/s4_logit_scatter_F19139_F21174.{png,pdf}
  figures/supplement4/s4_activity_vs_sigma.{png,pdf}

Usage (from the repository root):
  python scripts/supplement4_verified_analysis.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, "results", "supplement4")
FIGURES = os.path.join(PROJECT, "figures", "supplement4")
PHASE1 = os.path.join(PROJECT, "results", "phase1")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

HEDGES = [19139, 11969, 18573, 20804]
PARENTS = [17487, 21174, 16152]
PAIRS = [(h, p) for h in (19139, 11969) for p in PARENTS]
TOP_K = 10  # Neuronpedia stores the top-10 positive and top-10 negative logits

plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  saved figures/supplement4/{stem}.{{png,pdf}}")


def layer_cos(model, layer):
    return np.load(os.path.join(PHASE1, f"cos_sim_{model}_L{layer}.npy"))


def resolve_layer(group, sae_id):
    """Return (phase1 array prefix, layer) for a Neuronpedia sae_id."""
    if sae_id.endswith("-res-jb"):
        return "gpt2", int(sae_id.split("-")[0])
    if "gemmascope" in sae_id:
        return "gemma", int(sae_id.split("-")[0])
    m = re.match(r"l(\d+)r_8x", sae_id)
    if m:
        return "llama", int(m.group(1))
    raise ValueError(sae_id)


def explanation_text(e):
    return e if isinstance(e, str) else (e.get("description") if isinstance(e, dict) else str(e))


def main():
    np_file = sorted(glob.glob(os.path.join(RESULTS, "full_analysis_*.json")))[-1]
    snapshot = json.load(open(np_file))

    # ------------------------------------------------------------------ #
    # 1. Queried features: true cos(E, D) and sigma, activity, explanation
    # ------------------------------------------------------------------ #
    rows = []
    for group, profiles in snapshot["profiles"].items():
        for p in profiles:
            prefix, layer = resolve_layer(group, p["sae_id"])
            c = layer_cos(prefix, layer)
            cos = float(c[p["index"]])
            sigma = (cos - c.mean()) / c.std()
            expl = [explanation_text(e) for e in (p.get("explanations") or [])]
            # Neuronpedia returned errors for Llama Scope SAEs: such profiles have no logits and
            # no neighbours, and their zero frequency means "no data", not "dead".
            has_data = bool(p.get("pos_logit_tokens") or p.get("top_similar_indices"))
            rows.append({
                "neuronpedia_group": group, "model": prefix, "layer": layer, "feature": p["index"],
                "cos_ED": round(cos, 4), "sigma": round(float(sigma), 2),
                "anti_aligned_6sigma": bool(cos < 0 and sigma <= -6),
                "frac_nonzero": p.get("frac_nonzero") if has_data else None,
                "explanation": expl[0] if expl else "",
            })
    feats = pd.DataFrame(rows)
    feats.to_csv(os.path.join(RESULTS, "queried_features.csv"), index=False)
    print(feats.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2. Weights: decoder geometry and logit effects (GPT-2 L8)
    # ------------------------------------------------------------------ #
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    loaded = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre", device="cpu")
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    W_dec = sae.W_dec.detach().float()
    W_enc = sae.W_enc.detach().float()
    W_U = model.W_U.detach().float()
    tok = model.tokenizer

    ids = PARENTS + HEDGES
    Dn = torch.nn.functional.normalize(W_dec[ids], dim=1)
    dec_cos = (Dn @ Dn.T).numpy()
    enc_dec = torch.nn.functional.cosine_similarity(W_enc.T[ids], W_dec[ids], dim=1).numpy()

    logits = (W_dec[ids] @ W_U).numpy()  # (n_features, d_vocab)
    L = {f: logits[i] for i, f in enumerate(ids)}

    def top_tokens(v, k=TOP_K):
        pos = np.argsort(-v)[:k]
        neg = np.argsort(v)[:k]
        return pos, neg

    pairs = []
    for h, p in PAIRS:
        hp, hn = top_tokens(L[h])
        pp, pn = top_tokens(L[p])
        inv_pos = sorted(set(hp) & set(pn))  # promoted by hedge, suppressed by parent
        inv_neg = sorted(set(hn) & set(pp))  # suppressed by hedge, promoted by parent
        inv = inv_pos + inv_neg
        mean_abs_delta = float(np.mean([abs(L[h][t] + L[p][t]) for t in inv])) if inv else None
        pairs.append({
            "hedge": h, "parent": p,
            "decoder_cos": round(float(dec_cos[ids.index(h), ids.index(p)]), 4),
            "logit_vector_pearson": round(float(pearsonr(L[h], L[p])[0]), 4),
            "inversions_of_20": len(inv),
            "mean_abs_delta_inverted": None if mean_abs_delta is None else round(mean_abs_delta, 4),
            "inverted_tokens": [tok.decode([int(t)]) for t in inv],
        })
    pairs_df = pd.DataFrame(pairs)
    print(pairs_df.drop(columns="inverted_tokens").to_string(index=False))
    rho, rho_p = spearmanr(pairs_df.inversions_of_20, pairs_df.decoder_cos.abs())

    # Cross-check against the Neuronpedia snapshot logit lists
    snap = {p["index"]: p for g in ("gpt2_hedge_family", "gpt2_parents") for p in snapshot["profiles"][g]}
    np_match = {}
    for f in ids:
        pos, _ = top_tokens(L[f])
        ours = [tok.decode([int(t)]) for t in pos]
        theirs = snap[f]["pos_logit_tokens"][:TOP_K]
        np_match[f] = float(np.mean([a == b or b.startswith("ã") for a, b in zip(ours, theirs)]))

    # ------------------------------------------------------------------ #
    # 3. Figures
    # ------------------------------------------------------------------ #
    labels = [f"F{f}" for f in ids]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(dec_cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ids)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), labels)
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j:
                ax.text(j, i, f"{dec_cos[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(dec_cos[i, j]) > 0.6 else "black")
    ax.axhline(2.5, color="k", lw=0.8)
    ax.axvline(2.5, color="k", lw=0.8)
    ax.set_title("Decoder cosine, GPT-2 L8 (parents | anti-aligned)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "s4_decoder_cosine_matrix")

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(L[21174], L[19139], s=2, alpha=0.25, color="#555555", rasterized=True)
    r = pearsonr(L[21174], L[19139])[0]
    ax.set_xlabel("Logit effect of F21174 decoder (parent)")
    ax.set_ylabel("Logit effect of F19139 decoder (anti-aligned)")
    ax.set_title(f"All {L[19139].size:,} tokens, Pearson r = {r:.3f}")
    lim = max(np.abs(L[21174]).max(), np.abs(L[19139]).max()) * 1.05
    ax.plot([-lim, lim], [lim, -lim], color="#c0392b", lw=0.8, ls="--", label="y = -x")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "s4_logit_scatter_F19139_F21174")

    g2 = feats[feats.model == "gpt2"].copy()
    eps = 1e-8
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for flag, color, lab in [(True, "#c0392b", "cos(E,D) < 0 and <= mu - 6 sigma"),
                             (False, "#2c7fb8", "other queried features")]:
        s = g2[g2.anti_aligned_6sigma == flag]
        ax.scatter(s.sigma, np.log10(s.frac_nonzero.astype(float) + eps), color=color, label=lab, s=28)
    ax.axhline(np.log10(eps), color="grey", lw=0.6, ls=":")
    ax.text(ax.get_xlim()[0], np.log10(eps) + 0.15, "frac_nonzero = 0", fontsize=8, color="grey")
    ax.set_xlabel("Deviation of cos(E, D) from layer mean (sigma)")
    ax.set_ylabel("log10(fraction of tokens active)")
    ax.set_title("GPT-2 features queried on Neuronpedia")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    save(fig, "s4_activity_vs_sigma")

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    aa = feats[feats.anti_aligned_6sigma]
    summary = {
        "neuronpedia_snapshot": os.path.basename(np_file),
        "features": ids,
        "cos_ED": dict(zip(map(str, ids), np.round(enc_dec, 4).tolist())),
        "decoder_cosine_matrix": np.round(dec_cos, 4).tolist(),
        "pairs": pairs,
        "spearman_inversions_vs_abs_decoder_cos": {"rho": round(float(rho), 4), "p": round(float(rho_p), 4)},
        "neuronpedia_top10_positive_logit_agreement": np_match,
        "anti_aligned_queried": {
            "n": int(len(aa)),
            "by_model": aa.groupby("model").size().to_dict(),
            "without_neuronpedia_data": aa[aa.frac_nonzero.isna()][["model", "layer", "feature"]].to_dict("records"),
            "active_frac_gt_1e-6": aa[aa.frac_nonzero.astype(float) > 1e-6][["model", "layer", "feature", "frac_nonzero", "explanation"]].to_dict("records"),
            "max_frac_gpt2": float(aa[aa.model == "gpt2"].frac_nonzero.astype(float).max()),
        },
        "queried_but_not_anti_aligned": feats[~feats.anti_aligned_6sigma][["neuronpedia_group", "model", "layer", "feature", "cos_ED", "sigma"]].to_dict("records"),
    }
    with open(os.path.join(RESULTS, "verified_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  saved results/supplement4/verified_analysis.json")
    print(json.dumps({k: summary[k] for k in ("spearman_inversions_vs_abs_decoder_cos", "neuronpedia_top10_positive_logit_agreement")}, indent=1))


if __name__ == "__main__":
    main()
