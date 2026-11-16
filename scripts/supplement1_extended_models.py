"""
Phase 0/1 Supplement: Architectural Validation (v2)
Geometry of SAE Artifacts Research Program

Addresses the two most critical limitations from Phase 1:

1. LLAMA ARCHITECTURE ROBUSTNESS: The lxr_8x release uses JumpReLU (fnlp/Llama3_1-8B-Base-LXR-8x).
   We validate using lxr_32x (DIFFERENT training run, same JumpReLU architecture, 131K features).
   If BOTH 8x and 32x show bimodal anti-alignment, the phenomenon is robust across training configs.
   Note: the original TopK SAEs (fnlp/llama-scope-l{L}-res-{W}k) appear to require authentication.

2. SIZE vs THRESHOLD (H6 vs E2): Test Gemma 9B (d_model=3584, 42 layers, per-feature threshold,
   ~9B params similar size to Llama 8B) for early-layer anti-alignment.
   If Gemma 9B ALSO lacks absolute anti-alignment (cos<0) -> per-feature threshold protects (E2).
   If Gemma 9B shows early anti-alignment -> model size matters too (H6).
"""

import os, gc, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.stats as stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT, 'results', 'supplement1')
FIGURES_DIR = os.path.join(PROJECT, 'figures', 'supplement1')
RESULTS0 = os.path.join(PROJECT, 'results', 'phase0')
RESULTS1 = os.path.join(PROJECT, 'results', 'phase1')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
COLORS = {
    'gpt2':        '#1f77b4',
    'gemma2b':     '#d62728',
    'gemma9b':     '#ff7f0e',
    'llama8x':     '#2ca02c',
    'llama32x':    '#9467bd',
}

def save_fig(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES_DIR, f'{stem}.{ext}'))
    plt.close(fig)


# ── Core stats from raw cos_sim numpy array ───────────────────────────────────
def stats_from_cos(cos_np):
    """Compute all key statistics directly from 1D cos_sim array."""
    mu    = float(cos_np.mean())
    sigma = float(cos_np.std(ddof=1))
    skew  = float(stats.skew(cos_np))
    kurt  = float(stats.kurtosis(cos_np))
    n6sig = int(np.sum(cos_np < mu - 6*sigma))
    n3sig = int(np.sum(cos_np < mu - 3*sigma))
    return {
        'mu': mu, 'sigma': sigma, 'skewness': skew, 'excess_kurtosis': kurt,
        'count_6sig': n6sig, 'count_3sig': n3sig,
        'min_cos': float(cos_np.min()), 'n_features': len(cos_np)
    }


def fit_mixture_and_anomaly(cos_np, n_init=5, maxiter=800):
    """Fit 2-component Gaussian Mixture, return (delta_bic_vs_gauss, anomaly_ratio)."""
    n = len(cos_np)
    mu0, sig0 = cos_np.mean(), cos_np.std(ddof=1)
    thresh_6sig = mu0 - 6*sig0

    # Gaussian baseline
    gauss_ll = stats.norm.logpdf(cos_np, mu0, sig0).sum()
    gauss_bic = 2*np.log(n) - 2*gauss_ll

    best_ll, best_p = -np.inf, None
    for seed in range(n_init):
        np.random.seed(seed*17 + 3)
        pi0 = 0.93 + 0.04*np.random.rand()
        m1  = mu0 + 0.05*sig0*np.random.randn()
        m2  = mu0 - (3+2*np.random.rand())*sig0
        s1  = sig0*(0.8+0.3*np.random.rand())
        s2  = sig0*(0.3+0.3*np.random.rand())
        def neg_ll(p):
            pi,m1,s1,m2,s2 = p
            if not (0.01<pi<0.99 and s1>0.001 and s2>0.001): return 1e10
            return -np.log(pi*stats.norm.pdf(cos_np,m1,s1)+
                           (1-pi)*stats.norm.pdf(cos_np,m2,s2)+1e-300).sum()
        try:
            r = minimize(neg_ll, [pi0,m1,s1,m2,s2], method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol':1e-7,'fatol':1e-7})
            if -r.fun > best_ll: best_ll=-r.fun; best_p=r.x
        except: continue

    if best_p is None:
        return gauss_bic - gauss_bic, 0.0, {}
    pi,m1,s1,m2,s2 = best_p
    k = 5
    mix_bic = k*np.log(n) - 2*best_ll
    delta_bic = gauss_bic - mix_bic  # positive = mixture better

    p_below = pi*stats.norm.cdf(thresh_6sig,m1,s1)+(1-pi)*stats.norm.cdf(thresh_6sig,m2,s2)
    expected = n * p_below
    obs = int(np.sum(cos_np <= thresh_6sig))
    ratio = obs / max(expected, 1e-10) if obs > 0 else 0.0

    return delta_bic, ratio, {'pi':pi,'mu1':m1,'sigma1':s1,'mu2':m2,'sigma2':s2,
                              'expected_6sig':expected, 'p_below':p_below}


def analyze_one_sae(releae, sae_id, layer_idx, model_key, label, n_layers_total,
                    all_rows, save_npy=True):
    """Load SAE, compute stats, optionally save npy."""
    from sae_lens import SAE
    print(f'  Loading L{layer_idx}...', end=' ', flush=True)
    t0 = time.time()
    try:
        sae = SAE.from_pretrained(releae, sae_id, device='cpu')
        E = sae.W_enc.detach().float().T   # (d_sae, d_model)
        D = sae.W_dec.detach().float()     # (d_sae, d_model)
        cos_np = F.cosine_similarity(E, D, dim=1).cpu().numpy()
        enc_norms = E.norm(dim=1).cpu().numpy()

        # Threshold info
        thresh_info = None
        if hasattr(sae, 'threshold'):
            t = sae.threshold.detach().numpy()
            thresh_info = {
                'model': model_key, 'layer_idx': layer_idx,
                'n_thresholds': len(t), 'mean': float(t.mean()),
                'std': float(t.std()), 'min': float(t.min()),
                'max': float(t.max()), 'is_scalar': bool(t.std() < 1e-5)
            }

        del sae, E, D; gc.collect()

        if save_npy:
            npy_path = os.path.join(RESULTS_DIR, f'cos_sim_{model_key}_L{layer_idx}.npy')
            np.save(npy_path, cos_np)

        # Stats
        st = stats_from_cos(cos_np)
        r_pearson = float(np.corrcoef(enc_norms, cos_np)[0,1]) if len(np.unique(enc_norms))>1 else 0.0
        st['r_pearson'] = r_pearson

        # Mixture fit (only if there are 6sig features, for speed)
        if st['count_6sig'] > 0 or st['excess_kurtosis'] > 1.5:
            delta_bic, ratio, mix_p = fit_mixture_and_anomaly(cos_np)
        else:
            delta_bic, ratio, mix_p = 0.0, 0.0, {}

        gate = ('G1-YES' if ratio>10 else 'G1-MARGINAL' if ratio>2 else
                'G1-NO' if st['count_6sig']>0 else 'no 6sig')

        elapsed = time.time() - t0
        print(f'ok ({elapsed:.1f}s) | mu={st["mu"]:.3f} kurt={st["excess_kurtosis"]:.2f} '
              f'6sig={st["count_6sig"]} ratio={ratio:.1f} ΔBIC={delta_bic:+.0f} | {gate}')

        row = {
            'model_key': model_key, 'label': label,
            'layer_idx': layer_idx,
            'rel_depth': layer_idx / max(n_layers_total-1, 1),
            **st, 'r_pearson': r_pearson,
            'mix_delta_bic': delta_bic,
            'anomaly_ratio': ratio, 'gate': gate,
        }
        all_rows.append(row)
        return thresh_info

    except Exception as e:
        print(f'FAILED -- {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: GEMMA 9B
# ─────────────────────────────────────────────────────────────────────────────
def analyze_gemma9b(all_rows):
    print('\n' + '='*60)
    print('GEMMA 9B (JumpReLU per-feature thresh, d_model=3584, 42 layers)')
    print('Layers: 0,1,2 (early) | 10 (early-mid) | 20 (mid) | 41 (last)')
    print('KEY QUESTION: Does per-feature threshold protect even at 9B scale?')
    print('='*60)

    layers = [0, 1, 2, 10, 20, 41]
    thresh_rows = []

    for layer in layers:
        sae_id = f'layer_{layer}/width_16k/canonical'
        info = analyze_one_sae(
            'gemma-scope-9b-pt-res-canonical', sae_id, layer,
            'gemma9b', 'Gemma 9B (JumpReLU per-feat)', 42,
            all_rows, save_npy=True
        )
        if info:
            thresh_rows.append(info)

    if thresh_rows:
        df_t = pd.DataFrame(thresh_rows)
        print('\n  Threshold statistics (all per-feature, std>>0):')
        print(df_t[['layer_idx','mean','std','min','max','is_scalar']].to_string(index=False))

    return thresh_rows


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: LLAMA 32x ROBUSTNESS CHECK
# ─────────────────────────────────────────────────────────────────────────────
def analyze_llama32x(all_rows):
    print('\n' + '='*60)
    print('LLAMA 32x (JumpReLU, 131K features, DIFFERENT training run from 8x)')
    print('Layers: 0 (G1-MARGINAL in 8x) | 1 (G1-YES in 8x) | 10 | 16')
    print('KEY QUESTION: Is the bimodal anti-alignment robust across Llama training runs?')
    print('='*60)

    layers = [0, 1, 10, 16]
    thresh_rows = []

    for layer in layers:
        sae_id = f'l{layer}r_32x'
        info = analyze_one_sae(
            'llama_scope_lxr_32x', sae_id, layer,
            'llama32x', 'Llama 8B 32x (JumpReLU scal)', 32,
            all_rows, save_npy=True
        )
        if info:
            thresh_rows.append(info)

    if thresh_rows:
        df_t = pd.DataFrame(thresh_rows)
        print('\n  Threshold statistics:')
        print(df_t[['layer_idx','mean','std','is_scalar']].to_string(index=False))

    return thresh_rows


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def figure_gemma_size_comparison():
    """Gemma 2B vs Gemma 9B early layers: test size hypothesis."""
    layers = [0, 1, 2]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Gemma 2B vs Gemma 9B: Early Layer Distributions\n'
                 'Both use per-feature JumpReLU threshold — testing H6 (SIZE) vs E2 (THRESHOLD)',
                 fontsize=11)

    for mi, (model_key, color, label) in enumerate([
        ('gemma', COLORS['gemma2b'], 'Gemma 2B (JumpReLU, 2.5B)'),
        ('gemma9b', COLORS['gemma9b'], 'Gemma 9B (JumpReLU, 9.2B)'),
    ]):
        for li, layer in enumerate(layers):
            ax = axes[mi, li]
            # For gemma2b use phase1 data, for gemma9b use supplement
            base = RESULTS1 if model_key == 'gemma' else RESULTS_DIR
            npy = os.path.join(base, f'cos_sim_{model_key}_L{layer}.npy')
            if not os.path.exists(npy):
                ax.text(0.5, 0.5, f'No data\nL{layer}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='grey')
                continue
            cos_np = np.load(npy)
            mu, sigma = cos_np.mean(), cos_np.std(ddof=1)
            n6 = int(np.sum(cos_np < mu - 6*sigma))
            n_neg_cos = int(np.sum(cos_np < 0))
            ax.hist(cos_np, bins=80, color=color, alpha=0.75, density=True, edgecolor='none')
            ax.axvline(mu-6*sigma, color='red', lw=1.2, linestyle='--',
                       label=f'μ−6σ (n={n6})')
            ax.axvline(0, color='black', lw=1, linestyle=':', alpha=0.7,
                       label=f'cos=0 (n_neg={n_neg_cos})')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\cos(E_i,D_i)$', fontsize=9)
            ax.set_title(f'L{layer} | μ={mu:.3f} | ≤6σ: {n6} | neg: {n_neg_cos}', fontsize=9)
            if li == 0:
                ax.set_ylabel(f'{label}\nDensity (log)', fontsize=9)
            ax.legend(fontsize=7.5, loc='upper left')
            ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_fig(fig, 'gemma_2b_vs_9b_comparison')
    print('  Saved: gemma_2b_vs_9b_comparison.{png,pdf}')


def figure_llama_robustness():
    """Llama 8x (Phase 1) vs Llama 32x (new) at L0 and L1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Llama 8x vs 32x SAEs: Robustness of Early-Layer Anti-Alignment\n'
                 '(Different training runs, same JumpReLU architecture with scalar threshold)',
                 fontsize=11)

    combos = [(0, '8x', RESULTS1, 'llama', COLORS['llama8x']),
              (0, '32x', RESULTS_DIR, 'llama32x', COLORS['llama32x']),
              (1, '8x', RESULTS1, 'llama', COLORS['llama8x']),
              (1, '32x', RESULTS_DIR, 'llama32x', COLORS['llama32x'])]

    for (layer, variant, base, model_key, color), ax in zip(combos, axes.flat):
        npy = os.path.join(base, f'cos_sim_{model_key}_L{layer}.npy')
        if not os.path.exists(npy):
            ax.text(0.5, 0.5, f'No data\nL{layer} {variant}', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
            continue
        cos_np = np.load(npy)
        mu, sigma = cos_np.mean(), cos_np.std(ddof=1)
        n6 = int(np.sum(cos_np < mu - 6*sigma))
        n_neg = int(np.sum(cos_np < 0))
        ax.hist(cos_np, bins=100, color=color, alpha=0.75, density=True, edgecolor='none')
        ax.axvline(mu-6*sigma, color='red', lw=1.2, linestyle='--', label=f'μ−6σ (n={n6})')
        ax.axvline(0, color='black', lw=0.8, linestyle=':', alpha=0.6, label=f'cos=0 (n={n_neg})')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\cos(E_i,D_i)$', fontsize=9)
        ax.set_title(f'Llama L{layer} {variant} | μ={mu:.3f} | ≤6σ: {n6}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_fig(fig, 'llama_8x_vs_32x_robustness')
    print('  Saved: llama_8x_vs_32x_robustness.{png,pdf}')


def figure_full_comparison(df_supp):
    """Summary r(||E||,cos) for all 5 model settings."""
    if df_supp.empty:
        print('  No data for full comparison figure')
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Phase 0 data for the original 3 models
    df0 = pd.read_csv(os.path.join(RESULTS0, 'layer_stats.csv'))
    for model_str, color, label, n_total in [
        ('GPT-2', COLORS['gpt2'], 'GPT-2 Small (ReLU+L1 12L)', 12),
        ('Gemma 2B', COLORS['gemma2b'], 'Gemma 2B (JumpReLU per-feat 26L)', 26),
        ('Llama 3.1 8B', COLORS['llama8x'], 'Llama 8B 8x (JumpReLU scal 32L)', 32),
    ]:
        sub = df0[df0['model'].str.contains(model_str.split()[0])].sort_values('relative_depth')
        if sub.empty: continue
        ax.plot(sub['relative_depth'], sub['r_pearson'],
                color=color, marker='o', markersize=4, linewidth=1.5, label=label)

    # Supplement data
    for model_key, color, label, n_total in [
        ('gemma9b',  COLORS['gemma9b'],  'Gemma 9B (JumpReLU per-feat 42L)', 42),
        ('llama32x', COLORS['llama32x'], 'Llama 8B 32x (JumpReLU scal 32L)', 32),
    ]:
        sub = df_supp[df_supp['model_key'] == model_key].sort_values('layer_idx')
        if sub.empty: continue
        ax.plot(sub['rel_depth'], sub['r_pearson'],
                color=color, marker='s', markersize=7, linewidth=1.5,
                linestyle='--', label=label)

    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Relative Layer Depth')
    ax.set_ylabel(r'Pearson $r(\|E\|, \cos(E,D))$')
    ax.set_title('Sign Flip Pattern: 5 Model/Architecture Combinations\n'
                 '(dashed = new supplement data)')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, 'extended_sign_flip_all_models')
    print('  Saved: extended_sign_flip_all_models.{png,pdf}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('='*60)
    print('PHASE 0/1 SUPPLEMENT: ARCHITECTURAL VALIDATION (v2)')
    print('='*60)
    print('1. Gemma 9B: H6 (size) vs E2 (per-feature threshold)')
    print('2. Llama 32x: Robustness of bimodal anti-alignment across training runs')

    all_rows = []

    thresh_9b = analyze_gemma9b(all_rows)
    thresh_32x = analyze_llama32x(all_rows)

    # ── Save results
    print('\n' + '='*60)
    print('SAVING RESULTS')
    print('='*60)

    df_supp = pd.DataFrame(all_rows)
    df_supp.to_csv(os.path.join(RESULTS_DIR, 'supplement_stats.csv'), index=False)
    print(f'Saved: supplement_stats.csv ({len(df_supp)} rows)')

    all_thresh = thresh_9b + thresh_32x
    if all_thresh:
        pd.DataFrame(all_thresh).to_csv(os.path.join(RESULTS_DIR, 'threshold_comparison.csv'), index=False)
        print('Saved: threshold_comparison.csv')

    # ── Figures
    print('\n' + '='*60)
    print('GENERATING FIGURES')
    print('='*60)

    figure_gemma_size_comparison()
    figure_llama_robustness()
    figure_full_comparison(df_supp)

    # ── Synthesis
    print('\n' + '='*60)
    print('SUPPLEMENT CONCLUSIONS')
    print('='*60)

    # Gemma 9B analysis
    g9b = df_supp[df_supp['model_key'] == 'gemma9b']
    print('\n[1] GEMMA 9B (H6 vs E2):')
    if not g9b.empty:
        for _, r in g9b.iterrows():
            lidx = int(r['layer_idx'])
            npy_p = os.path.join(RESULTS_DIR, f'cos_sim_gemma9b_L{lidx}.npy')
            neg_count = int(np.sum(np.load(npy_p) < 0)) if os.path.exists(npy_p) else -1
            print(f"  L{lidx}: mu={r['mu']:.3f} min_cos={r['min_cos']:.3f} "
                  f"6sig={int(r['count_6sig'])} neg_cos={neg_count}")
        early_9b_neg = g9b[g9b['layer_idx'] <= 2]['min_cos'].min()
        if early_9b_neg < 0:
            print('  → H6+E2: Gemma 9B DOES have cos<0 in early layers')
            print('    Size AND threshold type may both contribute')
        else:
            print(f'  → E2 CONFIRMED: Gemma 9B min_cos(L0-L2)={early_9b_neg:.3f} >= 0')
            print('    Per-feature threshold PROTECTS regardless of model size (2B -> 9B)')
            print('    The Llama-Gemma gap is due to THRESHOLD ARCHITECTURE, not size')

    # Llama 32x
    l32x = df_supp[df_supp['model_key'] == 'llama32x']
    print('\n[2] LLAMA 32x ROBUSTNESS:')
    if not l32x.empty:
        for _, r in l32x.iterrows():
            print(f"  L{int(r['layer_idx'])}: {r['gate']} (ratio={r['anomaly_ratio']:.1f} "
                  f"6sig={int(r['count_6sig'])} mu={r['mu']:.3f} ΔBIC={r['mix_delta_bic']:+.0f})")
        l32x_early = l32x[l32x['layer_idx'] <= 1]
        if (l32x_early['count_6sig'] > 0).any():
            print('  → ROBUST: Llama 32x ALSO shows early-layer anti-alignment')
            print('    Bimodal structure is not specific to the 8x training run')
        else:
            print('  → CAUTION: Llama 32x does not show early anti-alignment')
            print('    Phase 0/1 results may be specific to the lxr_8x training run')

    # Write synthesis
    synth = {
        'gemma9b_early_min_cos': float(g9b[g9b['layer_idx']<=2]['min_cos'].min()) if len(g9b)>0 else float('nan'),
        'gemma9b_has_abs_antialign': bool((g9b['min_cos'] < 0).any()) if len(g9b)>0 else None,
        'llama32x_l0_6sig': int(l32x[l32x['layer_idx']==0]['count_6sig'].values[0]) if 0 in l32x['layer_idx'].values else None,
        'llama32x_l1_6sig': int(l32x[l32x['layer_idx']==1]['count_6sig'].values[0]) if 1 in l32x['layer_idx'].values else None,
        'llama32x_l1_gate': l32x[l32x['layer_idx']==1]['gate'].values[0] if 1 in l32x['layer_idx'].values else None,
    }
    import json
    with open(os.path.join(RESULTS_DIR, 'supplement_synthesis.json'), 'w') as f:
        json.dump(synth, f, indent=2, default=str)

    print('\n>>> SUPPLEMENT COMPLETE <<<')
    return df_supp


if __name__ == '__main__':
    df = main()
