"""
Phase 1: Distributional Analysis
Geometry of SAE Artifacts Research Program

Tests whether anti-aligned features are statistically anomalous relative to
well-fitted null distributional models (Gaussian, Beta, Student-t, mixture).
Applies to all 3 models: GPT-2 Small, Gemma 2B, Llama 3.1 8B.

Key question by model:
  GPT-2 late layers: anomalous under Student-t/mixture given kurtosis 2-3?
  Llama early layers: anomalous given extreme kurtosis 5.9-7.4?
  Gemma: are statistically-low (but positive!) features anomalous?

Gate G1 (separate per model population):
  G1-YES: observed tail features > 10x expected under best-fit null
  G1-MARGINAL: 2-10x
  G1-NO: ~1x (distributional noise)

Usage:
  # from the repository root
  python scripts/phase1_main.py

Outputs:
  results/phase1/cos_sim_*.npy          -- raw cos(E,D) arrays
  results/phase1/distributional_fits.csv
  results/phase1/tail_pvalues.csv
  results/phase1/threshold_stats.csv
  figures/phase1/*.{png,pdf}
  results/phase1/G1_decision_*.md
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS0 = os.path.join(PROJECT, 'results', 'phase0')
RESULTS1 = os.path.join(PROJECT, 'results', 'phase1')
FIGURES1 = os.path.join(PROJECT, 'figures', 'phase1')
os.makedirs(RESULTS1, exist_ok=True)
os.makedirs(FIGURES1, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
COLORS = {'gpt2': '#1f77b4', 'gemma': '#d62728', 'llama': '#2ca02c'}

def save_fig(fig, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES1, f'{stem}.{ext}'))
    plt.close(fig)


# ── Core: load SAE and save raw cos_sim array ─────────────────────────────────
def extract_and_save_cos_sim(release, sae_id, out_path, model_key):
    """Load SAE, compute cos(E_i,D_i) for all features, save as .npy."""
    if os.path.exists(out_path):
        print(f"  Cached: {os.path.basename(out_path)}")
        return np.load(out_path)
    from sae_lens import SAE
    print(f"  Loading {sae_id}...", end=' ', flush=True)
    sae = SAE.from_pretrained(release, sae_id, device='cpu')
    E = sae.W_enc.detach().float().T    # (d_sae, d_model)
    D = sae.W_dec.detach().float()      # (d_sae, d_model)
    cos_sim = F.cosine_similarity(E, D, dim=1).numpy()
    np.save(out_path, cos_sim)

    # Also extract threshold info (JumpReLU)
    thresh_info = None
    if hasattr(sae, 'threshold'):
        t = sae.threshold.detach().cpu().numpy()
        thresh_info = {
            'model': model_key, 'sae_id': sae_id,
            'n_thresholds': len(t) if t.ndim > 0 else 1,
            'mean': float(t.mean()), 'std': float(t.std()),
            'min': float(t.min()), 'max': float(t.max()),
            'is_scalar': bool(t.std() < 1e-6),
        }
    del sae, E, D
    gc.collect()
    print(f"ok ({len(cos_sim)} features, range [{cos_sim.min():.3f}, {cos_sim.max():.3f}])")
    return cos_sim, thresh_info


# ── Distributional fitting ────────────────────────────────────────────────────
def fit_gaussian(x):
    mu, sigma = x.mean(), x.std(ddof=1)
    ll = stats.norm.logpdf(x, mu, sigma).sum()
    k = 2
    return {'name': 'Gaussian', 'params': {'mu': mu, 'sigma': sigma},
            'll': ll, 'aic': 2*k - 2*ll, 'bic': k*np.log(len(x)) - 2*ll}


def fit_beta(x):
    y = np.clip((x + 1) / 2, 1e-6, 1 - 1e-6)
    try:
        a, b, _, _ = stats.beta.fit(y, floc=0, fscale=1)
        ll = (stats.beta.logpdf(y, a, b) - np.log(2)).sum()
        k = 2
        return {'name': 'Beta', 'params': {'alpha': a, 'beta': b},
                'll': ll, 'aic': 2*k - 2*ll, 'bic': k*np.log(len(x)) - 2*ll}
    except Exception:
        return {'name': 'Beta', 'params': {}, 'll': -np.inf, 'aic': np.inf, 'bic': np.inf}


def fit_student_t(x):
    try:
        df, loc, scale = stats.t.fit(x)
        ll = stats.t.logpdf(x, df, loc, scale).sum()
        k = 3
        return {'name': 'Student-t', 'params': {'df': df, 'loc': loc, 'scale': scale},
                'll': ll, 'aic': 2*k - 2*ll, 'bic': k*np.log(len(x)) - 2*ll}
    except Exception:
        return {'name': 'Student-t', 'params': {}, 'll': -np.inf, 'aic': np.inf, 'bic': np.inf}


def fit_gaussian_mixture(x, n_init=8):
    """Fit 2-component Gaussian mixture via optimization."""
    n = len(x)
    mu0, sig0 = x.mean(), x.std(ddof=1)
    best_ll, best_params = -np.inf, None

    for seed in range(n_init):
        np.random.seed(seed)
        init_pi = 0.95
        init_mu1 = mu0 + 0.1 * sig0 * np.random.randn()
        init_mu2 = mu0 - (4 + 2*np.random.rand()) * sig0
        init_s1  = sig0 * (0.8 + 0.4 * np.random.rand())
        init_s2  = sig0 * (0.5 + 0.5 * np.random.rand())
        p0 = [init_pi, init_mu1, init_s1, init_mu2, init_s2]

        def neg_ll(p):
            pi, m1, s1, m2, s2 = p
            if not (0.01 < pi < 0.99 and s1 > 0.001 and s2 > 0.001):
                return 1e10
            l1 = pi * stats.norm.pdf(x, m1, s1)
            l2 = (1-pi) * stats.norm.pdf(x, m2, s2)
            lsum = np.log(l1 + l2 + 1e-300).sum()
            return -lsum

        try:
            res = minimize(neg_ll, p0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_ll:
                best_ll = -res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return {'name': 'Mixture', 'params': {}, 'll': -np.inf, 'aic': np.inf, 'bic': np.inf}
    pi, m1, s1, m2, s2 = best_params
    k = 5
    return {'name': 'Mixture', 'params': {'pi': pi, 'mu1': m1, 'sigma1': s1,
                                           'mu2': m2, 'sigma2': s2},
            'll': best_ll, 'aic': 2*k - 2*best_ll, 'bic': k*np.log(n) - 2*best_ll}


def fit_all_models(cos_np):
    models = [
        fit_gaussian(cos_np),
        fit_beta(cos_np),
        fit_student_t(cos_np),
        fit_gaussian_mixture(cos_np),
    ]
    winner = min(models, key=lambda m: m['bic'])
    return models, winner


def tail_pvalue(cos_np, winner, mu, sigma):
    """For each extreme feature (<=mu-6sigma), compute expected count under winner."""
    n = len(cos_np)
    threshold_6sig = mu - 6 * sigma
    observed_6sig  = int(np.sum(cos_np <= threshold_6sig))

    def cdf_below(threshold, model):
        p = model['params']
        name = model['name']
        if name == 'Gaussian':
            return stats.norm.cdf(threshold, p['mu'], p['sigma'])
        elif name == 'Beta':
            y_thresh = (threshold + 1) / 2
            return stats.beta.cdf(y_thresh, p['alpha'], p['beta'])
        elif name == 'Student-t':
            return stats.t.cdf(threshold, p['df'], p['loc'], p['scale'])
        elif name == 'Mixture':
            if not p:
                return 0.0
            c1 = p['pi'] * stats.norm.cdf(threshold, p['mu1'], p['sigma1'])
            c2 = (1-p['pi']) * stats.norm.cdf(threshold, p['mu2'], p['sigma2'])
            return c1 + c2
        return 0.0

    p_below = cdf_below(threshold_6sig, winner)
    expected_6sig = n * p_below
    anomaly_ratio = observed_6sig / max(expected_6sig, 1e-10)

    return {
        'threshold_6sig': threshold_6sig,
        'observed_6sig': observed_6sig,
        'expected_6sig': expected_6sig,
        'anomaly_ratio': anomaly_ratio,
        'p_below': p_below,
    }


# ── Figure functions ──────────────────────────────────────────────────────────
def plot_histogram_log(cos_np, mu, sigma, title, stem, color, n_extreme):
    """Log-scale histogram with threshold annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, yscale in zip(axes, ['linear', 'log']):
        ax.hist(cos_np, bins=120, color=color, alpha=0.75,
                edgecolor='none', density=True)
        ax.axvline(mu, color='black', lw=1.5, linestyle='-', label=f'μ={mu:.3f}')
        ax.axvline(mu - 3*sigma, color='orange', lw=1.2, linestyle='--',
                   label=f'μ−3σ={mu-3*sigma:.3f}')
        ax.axvline(mu - 6*sigma, color='red', lw=1.5, linestyle='--',
                   label=f'μ−6σ={mu-6*sigma:.3f} (n={n_extreme})')
        ax.set_xlabel(r'$\cos(E_i, D_i)$')
        ax.set_ylabel('Density' if yscale == 'linear' else 'Density (log)')
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_title(title + '\n(log scale)')
        else:
            ax.set_title(title + '\n(linear scale)')
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    save_fig(fig, stem)


def plot_qq(cos_np, winner, title, stem, color):
    """Q-Q plot against winning distributional model."""
    n = len(cos_np)
    # Use at most 1000 quantile points for speed (still accurate for tail inspection)
    n_q = min(1000, n)
    theoretical_q = np.linspace(0.001, 0.999, n_q)
    # Empirical: sample n_q evenly-spaced order statistics
    empirical_q = np.sort(cos_np)[np.linspace(0, n-1, n_q, dtype=int)]

    p = winner['params']
    name = winner['name']
    if name == 'Gaussian':
        theoretical = stats.norm.ppf(theoretical_q, p['mu'], p['sigma'])
    elif name == 'Beta':
        y_q = stats.beta.ppf(theoretical_q, p['alpha'], p['beta'])
        theoretical = y_q * 2 - 1
    elif name == 'Student-t':
        theoretical = stats.t.ppf(theoretical_q, p['df'], p['loc'], p['scale'])
    elif name == 'Mixture' and p:
        # Vectorized CDF inversion via brentq over n_q points (already small: <=1000)
        from scipy.optimize import brentq
        combined_ppf = []
        pi_, mu1, s1, mu2, s2 = p['pi'], p['mu1'], p['sigma1'], p['mu2'], p['sigma2']
        for q in theoretical_q:
            try:
                x_val = brentq(
                    lambda x: pi_*stats.norm.cdf(x,mu1,s1)+(1-pi_)*stats.norm.cdf(x,mu2,s2)-q,
                    -2.0, 2.0, xtol=1e-4, maxiter=50)
            except Exception:
                x_val = np.nan
            combined_ppf.append(x_val)
        theoretical = np.array(combined_ppf)
    else:
        theoretical = stats.norm.ppf(theoretical_q)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, empirical_q, s=1.5, color=color, alpha=0.5)
    lims = [min(theoretical.min(), empirical_q.min()),
            max(theoretical.max(), empirical_q.max())]
    ax.plot(lims, lims, 'k--', lw=1.2, label='y=x (perfect fit)')

    # Highlight the left tail
    tail_mask = empirical_q < np.percentile(empirical_q, 0.5)
    ax.scatter(theoretical[tail_mask], empirical_q[tail_mask],
               s=20, color='red', alpha=0.9, label='Left tail (<0.5%ile)')

    ax.set_xlabel(f'Theoretical ({name}) Quantiles')
    ax.set_ylabel(r'Empirical $\cos(E,D)$ Quantiles')
    ax.set_title(title + f'\nQ-Q: {name}')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_bic_comparison(df_fits):
    """Heatmap of BIC advantage of winning model vs Gaussian."""
    # For each (model, layer), delta_BIC = BIC_Gaussian - BIC_winner
    # Positive = winner is better
    models_list = df_fits['model_display'].unique()
    fig, axes = plt.subplots(1, len(models_list), figsize=(13, 4), sharey=False)
    if len(models_list) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models_list):
        sub = df_fits[df_fits['model_display'] == model_name].sort_values('layer_idx')
        ax.barh(sub['layer_idx'], sub['delta_bic_vs_gaussian'],
                color=COLORS.get(model_name.split()[0].lower().replace('-', '').replace('.',''),
                                 'steelblue'),
                alpha=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('ΔBIC (Gaussian − Winner)\n(positive = winner is better than Gaussian)')
        ax.set_ylabel('Layer')
        ax.set_title(model_name, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Best Model vs. Gaussian (BIC gain per layer)', fontsize=11)
    fig.tight_layout()
    save_fig(fig, 'bic_improvement_vs_gaussian')


def plot_anomaly_heatmap(df_tail):
    """Heatmap of log10(anomaly_ratio) across layers x models."""
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    models_order = [
        'GPT-2 Small (ReLU+L1)',
        'Gemma 2B (JumpReLU)',
        'Llama 3.1 8B (JumpReLU)',
    ]
    max_layers = int(df_tail['layer_idx'].max()) + 1

    matrix = np.full((len(models_order), max_layers), np.nan)
    for mi, model in enumerate(models_order):
        sub = df_tail[df_tail['model_display'] == model]
        for _, row in sub.iterrows():
            layer = int(row['layer_idx'])
            ratio = max(row['anomaly_ratio'], 0.01)  # avoid log(0)
            matrix[mi, layer] = ratio

    fig, ax = plt.subplots(figsize=(14, 3.5))
    vmin, vmax = 0.1, 1000
    im = ax.imshow(np.log10(np.clip(matrix, vmin, vmax)), aspect='auto',
                   cmap='RdYlGn', vmin=np.log10(vmin), vmax=np.log10(vmax))

    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels(['GPT-2 Small', 'Gemma 2B', 'Llama 3.1 8B'])
    ax.set_xlabel('Layer index')
    ax.set_title(r'Anomaly Ratio = Observed $6\sigma$ Features / Expected Under Best-Fit Null'
                 '\n(log₁₀ scale; green=anomalous, red=expected)', fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('log₁₀(anomaly ratio)')
    cbar.set_ticks([-1, 0, 1, 2, 3])
    cbar.set_ticklabels(['0.1×', '1×', '10×', '100×', '1000×'])

    fig.tight_layout()
    save_fig(fig, 'anomaly_ratio_heatmap')


# ── Model analysis ────────────────────────────────────────────────────────────
def analyze_model(release, sae_ids, model_key, model_display, n_layers_total,
                  all_fit_rows, all_tail_rows, all_thresh_rows, focus_layers=None):
    """Full distributional analysis for one model across all layers."""
    print(f'\n{"="*60}')
    print(f'{model_display}')
    print(f'{"="*60}')

    for layer_idx, sae_id in sae_ids:
        out_path = os.path.join(RESULTS1, f'cos_sim_{model_key}_L{layer_idx}.npy')

        result = extract_and_save_cos_sim(release, sae_id, out_path, model_key)
        if isinstance(result, tuple):
            cos_np, thresh_info = result
            if thresh_info:
                thresh_info.update({'layer_idx': layer_idx,
                                    'rel_depth': layer_idx / max(n_layers_total-1, 1)})
                all_thresh_rows.append(thresh_info)
        else:
            cos_np = result  # cached case

        n = len(cos_np)
        mu    = float(cos_np.mean())
        sigma = float(cos_np.std(ddof=1))
        skew  = float(stats.skew(cos_np))
        kurt  = float(stats.kurtosis(cos_np))

        # Fit all distributional models
        models, winner = fit_all_models(cos_np)

        # Tail p-values
        tail_stats = tail_pvalue(cos_np, winner, mu, sigma)

        rel_d = layer_idx / max(n_layers_total - 1, 1)

        for m in models:
            row = {
                'model_key': model_key,
                'model_display': model_display,
                'layer_idx': layer_idx,
                'rel_depth': rel_d,
                'fit_name': m['name'],
                'is_winner': m['name'] == winner['name'],
                'bic': m['bic'],
                'aic': m['aic'],
                'log_likelihood': m['ll'],
                'mu': mu, 'sigma': sigma, 'skewness': skew, 'excess_kurtosis': kurt,
                'n_features': n,
            }
            if m['name'] == 'Gaussian':
                row['delta_bic_vs_gaussian'] = 0.0
            else:
                gauss_bic = next(x['bic'] for x in models if x['name']=='Gaussian')
                row['delta_bic_vs_gaussian'] = gauss_bic - m['bic']
            all_fit_rows.append(row)

        tail_row = {
            'model_key': model_key,
            'model_display': model_display,
            'layer_idx': layer_idx,
            'rel_depth': rel_d,
            'mu': mu, 'sigma': sigma,
            'n_features': n,
            'winner_model': winner['name'],
            **tail_stats,
        }
        all_tail_rows.append(tail_row)

        # Flag anomalous
        flag = ('G1-YES' if tail_stats['anomaly_ratio'] > 10
                else 'G1-MARGINAL' if tail_stats['anomaly_ratio'] > 2
                else 'G1-NO')
        if tail_stats['observed_6sig'] > 0:
            print(f"  L{layer_idx:2d} | winner={winner['name']:10s} | "
                  f"obs={tail_stats['observed_6sig']:3d} exp={tail_stats['expected_6sig']:.2f} "
                  f"ratio={tail_stats['anomaly_ratio']:.1f} | {flag}")

        # Detailed plots for focus layers
        if focus_layers is None or layer_idx in focus_layers:
            count_6sig = int(np.sum(cos_np <= mu - 6*sigma))
            title = f'{model_display}\nLayer {layer_idx} (depth={rel_d:.2f})'
            stem_hist = f'hist_{model_key}_L{layer_idx}'
            plot_histogram_log(cos_np, mu, sigma, title, stem_hist,
                               COLORS[model_key], count_6sig)

            if count_6sig > 0:
                stem_qq = f'qq_{model_key}_L{layer_idx}'
                plot_qq(cos_np, winner, title, stem_qq, COLORS[model_key])


# ── Gate decision ──────────────────────────────────────────────────────────────
def evaluate_g1(df_tail, model_key, model_display, focus_note=''):
    """Write gate G1 decision for one model."""
    sub = df_tail[df_tail['model_key'] == model_key]
    if sub.empty:
        return

    layers_with_6sig = sub[sub['observed_6sig'] > 0].sort_values('anomaly_ratio', ascending=False)
    max_ratio = sub['anomaly_ratio'].max()
    n_g1_yes  = (sub['anomaly_ratio'] > 10).sum()
    n_g1_marg = ((sub['anomaly_ratio'] > 2) & (sub['anomaly_ratio'] <= 10)).sum()

    if n_g1_yes >= 1:
        decision = 'G1-YES'
    elif n_g1_marg >= 1:
        decision = 'G1-MARGINAL'
    else:
        decision = 'G1-NO'

    path = os.path.join(RESULTS1, f'G1_decision_{model_key}.md')
    with open(path, 'w') as f:
        f.write(f'# Gate G1: {model_display}\n\n')
        f.write(f'**Decision: {decision}**\n\n')
        f.write(f'{focus_note}\n\n')
        f.write(f'## Summary\n')
        f.write(f'Max anomaly ratio: {max_ratio:.1f}x\n')
        f.write(f'Layers with G1-YES (ratio >10x): {n_g1_yes}\n')
        f.write(f'Layers with G1-MARGINAL (ratio 2-10x): {n_g1_marg}\n\n')
        f.write(f'## Layers with 6sigma features\n')
        if layers_with_6sig.empty:
            f.write('None.\n')
        else:
            f.write(layers_with_6sig[['layer_idx','winner_model','observed_6sig',
                                       'expected_6sig','anomaly_ratio']].to_string(index=False))
        f.write(f'\n\n## Implications\n')
        if decision == 'G1-YES':
            f.write('Tail features are statistically anomalous. Mechanistic investigation warranted (Phases 2-4).\n')
        elif decision == 'G1-MARGINAL':
            f.write('Tail features partially explained by distributional shape. Proceed with caution.\n')
        else:
            f.write('Tail features explained by heavy-tailed distribution. Reframe as distributional study.\n')

    print(f'\n>>> G1 for {model_display}: {decision} (max ratio {max_ratio:.1f}x) <<<')
    return decision


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('='*60)
    print('PHASE 1: DISTRIBUTIONAL ANALYSIS')
    print('Geometry of SAE Artifacts Research Program')
    print('='*60)

    all_fit_rows   = []
    all_tail_rows  = []
    all_thresh_rows = []

    # ── GPT-2 Small (all 12 layers)
    gpt2_sae_ids = [(layer, f'blocks.{layer}.hook_resid_pre') for layer in range(12)]
    gpt2_focus = [4, 5, 8, 9, 10]  # key layers: spike, max 6sig, most extreme
    analyze_model(
        'gpt2-small-res-jb', gpt2_sae_ids,
        'gpt2', 'GPT-2 Small (ReLU+L1)', 12,
        all_fit_rows, all_tail_rows, all_thresh_rows,
        focus_layers=gpt2_focus
    )

    # ── Gemma 2B (all 26 layers, canonical)
    from sae_lens.saes.sae import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    gemma_info = d.get('gemma-scope-2b-pt-res-canonical')
    available = set(gemma_info.saes_map.keys()) if gemma_info else set()

    gemma_sae_ids = []
    for layer in range(26):
        id_16k = f'layer_{layer}/width_16k/canonical'
        id_65k = f'layer_{layer}/width_65k/canonical'
        if id_16k in available:
            gemma_sae_ids.append((layer, id_16k))
        elif id_65k in available:
            gemma_sae_ids.append((layer, id_65k))

    gemma_focus = [0, 1, 11, 25]  # early + high-6sigma layers
    analyze_model(
        'gemma-scope-2b-pt-res-canonical', gemma_sae_ids,
        'gemma', 'Gemma 2B (JumpReLU)', 26,
        all_fit_rows, all_tail_rows, all_thresh_rows,
        focus_layers=gemma_focus
    )

    # ── Llama 3.1 8B (all 32 layers)
    llama_sae_ids = [(layer, f'l{layer}r_8x') for layer in range(32)]
    llama_focus = [0, 1, 2, 7]  # early + anomalous layers
    analyze_model(
        'llama_scope_lxr_8x', llama_sae_ids,
        'llama', 'Llama 3.1 8B (JumpReLU)', 32,
        all_fit_rows, all_tail_rows, all_thresh_rows,
        focus_layers=llama_focus
    )

    # ── Save CSVs
    print('\n' + '='*60)
    print('SAVING RESULTS')
    print('='*60)

    df_fits = pd.DataFrame(all_fit_rows)
    df_fits.to_csv(os.path.join(RESULTS1, 'distributional_fits.csv'), index=False)
    print(f'Saved: distributional_fits.csv ({len(df_fits)} rows)')

    df_tail = pd.DataFrame(all_tail_rows)
    df_tail.to_csv(os.path.join(RESULTS1, 'tail_pvalues.csv'), index=False)
    print(f'Saved: tail_pvalues.csv ({len(df_tail)} rows)')

    if all_thresh_rows:
        df_thresh = pd.DataFrame(all_thresh_rows)
        df_thresh.to_csv(os.path.join(RESULTS1, 'threshold_stats.csv'), index=False)
        print(f'Saved: threshold_stats.csv ({len(df_thresh)} rows)')

    # ── Aggregate figures
    df_fits_winner = df_fits[df_fits['is_winner']].copy()
    plot_bic_comparison(df_fits_winner)

    plot_anomaly_heatmap(df_tail)

    # ── Gate G1 decisions
    print('\n' + '='*60)
    print('GATE G1 DECISIONS')
    print('='*60)

    g1_gpt2 = evaluate_g1(df_tail, 'gpt2', 'GPT-2 Small (ReLU+L1)',
        focus_note='Primary paper subject. Focus: late layers L4-L11.')
    g1_gemma = evaluate_g1(df_tail, 'gemma', 'Gemma 2B (JumpReLU)',
        focus_note='Note: 6sigma features have POSITIVE cosine (high-baseline model). '
                   'G1-YES here means "unusually low alignment for Gemma", not absolute anti-alignment.')
    g1_llama = evaluate_g1(df_tail, 'llama', 'Llama 3.1 8B (JumpReLU)',
        focus_note='Critical question: are 12-14 features at 6sigma in L0-L1 '
                   'anomalous given kurtosis=7.4? Or expected under Student-t null?')

    # ── Synthesis
    synthesis_path = os.path.join(RESULTS1, 'G1_synthesis.md')
    with open(synthesis_path, 'w') as f:
        f.write('# Phase 1 Gate G1 Synthesis\n\n')
        f.write(f'GPT-2 Small: {g1_gpt2}\n')
        f.write(f'Gemma 2B:    {g1_gemma}\n')
        f.write(f'Llama 3.1 8B: {g1_llama}\n\n')
        f.write('## Effect on program\n')
        if g1_gpt2 == 'G1-YES':
            f.write('- GPT-2 tail is anomalous: proceed with Phases 2-4 (toy model + co-activation)\n')
        if g1_llama in ('G1-YES', 'G1-MARGINAL'):
            f.write('- Llama anomaly is significant: expand Phase 4 to Llama L1\n')
        else:
            f.write('- Llama tail explained by kurtosis: focus analysis on GPT-2, Llama is context\n')

    print(f'\n>>> PHASE 1 COMPLETE <<<')
    print(f'Results in: {RESULTS1}')
    print(f'Figures in: {FIGURES1}')
    return df_fits, df_tail


if __name__ == '__main__':
    df_fits, df_tail = main()
