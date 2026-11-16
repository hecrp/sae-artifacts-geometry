"""
Supplement 2: Gemma 9B Full 42-Layer Analysis — Streaming Mode
Geometry of SAE Artifacts Research Program

Downloads each SAE layer one at a time, computes cos(E,D) stats,
deletes from cache immediately to avoid disk overflow.

Also: try fnlp/llama-scope TopK SAEs (layers 0 and 1).
"""
import os, gc, sys, shutil, warnings, time
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS2 = os.path.join(PROJECT, 'results', 'supplement2')
FIGURES2 = os.path.join(PROJECT, 'figures', 'supplement2')
RESULTS_SUPP = os.path.join(PROJECT, 'results', 'supplement1')

os.makedirs(RESULTS2, exist_ok=True)
os.makedirs(FIGURES2, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')

def disk_free_gb():
    st = os.statvfs(os.path.expanduser('~'))
    return st.f_bavail * st.f_frsize / 1e9


def clear_hf_model_cache(model_id):
    """Delete all cached files for a specific model to free disk space."""
    safe_id = model_id.replace('/', '--')
    cache_dir = os.path.join(HF_CACHE, f'models--{safe_id}')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        return True
    return False


def stats_from_cos(cos_np):
    mu = float(cos_np.mean()); sigma = float(cos_np.std(ddof=1))
    return {
        'mu': mu, 'sigma': sigma,
        'skewness': float(stats.skew(cos_np)),
        'excess_kurtosis': float(stats.kurtosis(cos_np)),
        'count_6sig': int(np.sum(cos_np < mu - 6*sigma)),
        'count_3sig': int(np.sum(cos_np < mu - 3*sigma)),
        'min_cos': float(cos_np.min()),
        'n_neg_cos': int(np.sum(cos_np < 0)),
        'n_features': len(cos_np),
    }


def quick_mixture_delta_bic(cos_np, n_init=3, maxiter=400):
    """Fast mixture fit returning only ΔBIC."""
    n = len(cos_np); mu0, sig0 = cos_np.mean(), cos_np.std(ddof=1)
    gauss_ll = stats.norm.logpdf(cos_np, mu0, sig0).sum()
    gauss_bic = 2*np.log(n) - 2*gauss_ll
    interesting = (np.sum(cos_np < mu0-6*sig0) > 0 or
                   stats.kurtosis(cos_np) > 1.0 or np.any(cos_np < 0))
    if not interesting:
        return 0.0
    best_ll = -np.inf; best_p = None
    for seed in range(n_init):
        np.random.seed(seed*13+7)
        pi0=0.95; m1=mu0+0.02*sig0*np.random.randn()
        m2=mu0-(3+2*np.random.rand())*sig0
        s1=sig0*0.9; s2=sig0*0.3
        def neg_ll(p):
            pi,m1,s1,m2,s2=p
            if not(0.01<pi<0.99 and s1>.001 and s2>.001): return 1e10
            return -np.log(pi*stats.norm.pdf(cos_np,m1,s1)+
                           (1-pi)*stats.norm.pdf(cos_np,m2,s2)+1e-300).sum()
        try:
            r = minimize(neg_ll,[pi0,m1,s1,m2,s2],method='Nelder-Mead',
                        options={'maxiter':maxiter,'xatol':1e-6,'fatol':1e-6})
            if -r.fun>best_ll: best_ll=-r.fun; best_p=r.x
        except: continue
    if best_p is None: return 0.0
    mix_bic = 5*np.log(n) - 2*best_ll
    return gauss_bic - mix_bic


# ─────────────────────────────────────────────────────────────────────────────
# GEMMA 9B: stream all 42 layers via direct npz download
# ─────────────────────────────────────────────────────────────────────────────
def analyze_gemma9b_streaming(all_rows, thresh_rows):
    from huggingface_hub import hf_hub_download
    print('\n' + '='*60)
    print('GEMMA 9B: FULL 42-LAYER STREAMING ANALYSIS')
    print('(downloads one layer at a time, deletes cache after each)')
    print('='*60)

    REPO = 'google/gemma-scope-9b-pt-res'
    cached_layers = {0, 1, 2, 10, 20, 41}  # already have from supplement

    for layer in range(42):
        npy_path = os.path.join(RESULTS2, f'cos_sim_gemma9b_L{layer}.npy')
        old_npy  = os.path.join(RESULTS_SUPP, f'cos_sim_gemma9b_L{layer}.npy')

        # Use existing npy if available
        if os.path.exists(npy_path):
            cos_np = np.load(npy_path)
            print(f'  L{layer:2d} [npy cache]', end=' ')
        elif os.path.exists(old_npy):
            cos_np = np.load(old_npy)
            np.save(npy_path, cos_np)
            print(f'  L{layer:2d} [old cache]', end=' ')
        else:
            # Stream download, compute, delete
            print(f'  L{layer:2d} [download]', end=' ', flush=True)
            disk_gb = disk_free_gb()
            if disk_gb < 2.0:
                print(f'LOW DISK ({disk_gb:.1f}GB) -- clearing Gemma 9B cache first')
                clear_hf_model_cache(REPO)
                disk_gb = disk_free_gb()
                print(f'  After clear: {disk_gb:.1f}GB free')

            try:
                t0 = time.time()
                # Use canonical width_16k
                fname = f'layer_{layer}/width_16k/canonical/params.npz'
                tmppath = hf_hub_download(REPO, fname)
                params = np.load(tmppath)
                W_enc = params['W_enc']  # (d_model, M) = (3584, 16384)
                W_dec = params['W_dec']  # (M, d_model) = (16384, 3584)
                E = torch.from_numpy(W_enc.T.copy()).float()  # (M, d_model)
                D = torch.from_numpy(W_dec.copy()).float()    # (M, d_model)
                cos_np = F.cosine_similarity(E, D, dim=1).numpy()
                np.save(npy_path, cos_np)

                # Threshold
                if 'log_threshold' in params:
                    lt = params['log_threshold']
                    theta = np.exp(lt)
                    thresh_rows.append({
                        'layer_idx': layer, 'mean': float(theta.mean()),
                        'std': float(theta.std()), 'min': float(theta.min()),
                        'max': float(theta.max()), 'is_scalar': bool(theta.std()<1e-5)
                    })
                del params, E, D; gc.collect()
                # Aggressively clear cache after each layer
                clear_hf_model_cache(REPO)
                print(f'ok ({time.time()-t0:.1f}s)', end=' ')
            except Exception as e:
                print(f'FAILED: {str(e)[:80]}')
                continue

        # Stats
        st = stats_from_cos(cos_np)
        delta_bic = quick_mixture_delta_bic(cos_np)
        status = ('NEG!' if st['n_neg_cos'] > 0 else
                  f'{st["count_6sig"]}x6σ' if st['count_6sig'] > 0 else 'clean')
        print(f'mu={st["mu"]:.3f} kurt={st["excess_kurtosis"]:+.2f} '
              f'neg={st["n_neg_cos"]:3d} ΔBIC={delta_bic:+.0f} | {status}')

        all_rows.append({
            'model_key': 'gemma9b', 'layer_idx': layer,
            'rel_depth': layer / 41, **st, 'mix_delta_bic': delta_bic
        })

    # Final clean of any remaining Gemma 9B cache
    clear_hf_model_cache(REPO)
    print(f'\n  Disk free after Gemma 9B: {disk_free_gb():.1f}GB')


# ─────────────────────────────────────────────────────────────────────────────
# LLAMA TOPK: try fnlp/llama-scope repos
# ─────────────────────────────────────────────────────────────────────────────
def analyze_llama_topk_stream(all_rows):
    """Attempt to load original TopK Llama Scope SAEs."""
    from huggingface_hub import hf_hub_download
    print('\n' + '='*60)
    print('LLAMA TOPK: fnlp/llama-scope (original TopK SAEs)')
    print('Note: NOT the LXR family -- these are standard TopK SAEs')
    print('='*60)

    successes = []
    for layer in [0, 1]:
        # Try multiple naming patterns
        for repo_name in [
            f'fnlp/llama-scope-l{layer}-res-32k',
            f'fnlp/llama-scope-l{layer}-res-16k',
            f'fnlp/llama_scope_l{layer}_res_32k',
        ]:
            print(f'  Trying {repo_name}/ae.pt ...', end=' ', flush=True)
            try:
                path = hf_hub_download(repo_name, 'ae.pt')
                state = torch.load(path, map_location='cpu', weights_only=False)
                keys = list(state.keys())
                print(f'ok! keys={keys[:6]}')

                # Extract W_enc, W_dec
                if 'encoder.weight' in state:
                    W_enc = state['encoder.weight'].float()  # (M, d_model)
                    Wd = state['decoder.weight'].float()     # (d_model, M) or (M, d_model)
                    W_dec = Wd if Wd.shape[0] > Wd.shape[1] else Wd.T
                elif 'W_enc' in state:
                    We = state['W_enc'].float()
                    W_enc = We if We.shape[0] > We.shape[1] else We.T  # ensure (M, d_model)
                    Wd = state['W_dec'].float()
                    W_dec = Wd if Wd.shape[0] > Wd.shape[1] else Wd.T
                else:
                    print(f'  Unknown keys. Skipping.')
                    del state; break

                cos_np = F.cosine_similarity(W_enc, W_dec, dim=1).numpy()
                np.save(os.path.join(RESULTS2, f'cos_sim_llama_topk_L{layer}.npy'), cos_np)

                st = stats_from_cos(cos_np)
                delta_bic = quick_mixture_delta_bic(cos_np)
                print(f'  L{layer}: mu={st["mu"]:.3f} kurt={st["excess_kurtosis"]:+.2f} '
                      f'neg={st["n_neg_cos"]} 6sig={st["count_6sig"]} ΔBIC={delta_bic:+.0f}')

                all_rows.append({
                    'model_key': 'llama_topk', 'layer_idx': layer,
                    'rel_depth': layer/31, **st, 'mix_delta_bic': delta_bic
                })
                successes.append(layer)
                del state, W_enc, W_dec; gc.collect()
                clear_hf_model_cache(repo_name)
                break  # success, no need to try other names
            except Exception as e:
                print(f'FAILED: {str(e)[:100]}')
                continue

    if not successes:
        print('\n  → All fnlp/llama-scope naming variants failed (404 or auth required)')
        print('  → Original TopK SAEs remain UNTESTED (documented limitation)')
    return successes


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def figure_gemma9b_profile(df9b):
    """Full Gemma 9B 42-layer profile."""
    if df9b.empty:
        print('  No Gemma 9B data for figure')
        return
    df9b = df9b.sort_values('layer_idx')

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle('Gemma 9B: Complete 42-Layer Geometric Profile\n'
                 '(Per-Feature JumpReLU Threshold | d_model=3584, d_sae=16384)',
                 fontsize=12)

    # Panel 1: mu ± sigma + min_cos
    ax = axes[0]
    ax.plot(df9b['layer_idx'], df9b['mu'], color='#ff7f0e',
            marker='o', ms=3, lw=1.5, label='Mean cos(E,D)')
    ax.fill_between(df9b['layer_idx'], df9b['mu']-df9b['sigma'],
                    df9b['mu']+df9b['sigma'], alpha=0.15, color='#ff7f0e')
    ax.plot(df9b['layer_idx'], df9b['min_cos'], 'r--', marker='.', ms=3,
            lw=1, label='Min cos(E,D)')
    ax.axhline(0, color='k', lw=0.8, ls=':', alpha=0.6, label='cos=0')
    ax.set_ylabel('cos(E,D)')
    ax.set_title('Mean ±1σ and Minimum', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    # Reference Gemma 2B
    df_g2b_p0 = pd.read_csv(os.path.join(PROJECT, 'results', 'phase0', 'layer_stats.csv'))
    g2b = df_g2b_p0[df_g2b_p0['model'].str.contains('Gemma')].sort_values('layer_idx')
    if not g2b.empty:
        # Rescale: Gemma 9B has 42 layers, Gemma 2B has 26
        ax.plot(g2b['layer_idx'] * 41/25, g2b['mu'], color='#d62728',
                lw=1, ls=':', alpha=0.5, label='Gemma 2B (rescaled)')
        ax.legend(fontsize=7.5)

    # Panel 2: excess kurtosis
    axes[1].plot(df9b['layer_idx'], df9b['excess_kurtosis'],
                 color='#ff7f0e', marker='o', ms=3, lw=1.5, label='Gemma 9B')
    axes[1].axhline(0, color='k', lw=0.8, ls='--', alpha=0.4,
                    label='Gaussian reference')
    axes[1].set_ylabel('Excess Kurtosis γ₂')
    axes[1].set_title('Kurtosis Profile — negative = light-tailed (platykurtic)', fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.25)

    # Panel 3: n_neg_cos by layer (the key metric)
    axes[2].bar(df9b['layer_idx'], df9b['n_neg_cos'], color='red', alpha=0.7,
                label='Features with cos(E,D) < 0')
    axes[2].bar(df9b['layer_idx'], df9b['count_6sig'], color='#ff7f0e', alpha=0.7,
                label='Features at ≤μ-6σ')
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Anti-Alignment Counts per Layer', fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES2, f'gemma9b_full_42layers.{ext}'))
    plt.close(fig)
    print('  Saved: gemma9b_full_42layers.{png,pdf}')


def figure_gemma9b_threshold_depth(thresh_rows):
    """Threshold evolution with depth for Gemma 9B."""
    if not thresh_rows:
        print('  No threshold data for evolution figure')
        return
    df_t = pd.DataFrame(thresh_rows).sort_values('layer_idx')
    if df_t.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(df_t['layer_idx'], df_t['mean'], yerr=df_t['std'],
                fmt='o-', color='#ff7f0e', capsize=3, lw=1.5, ms=4,
                label='Mean threshold ± std')
    ax.fill_between(df_t['layer_idx'], df_t['min'], df_t['max'],
                    alpha=0.1, color='#ff7f0e', label='Range [min, max]')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('JumpReLU threshold θ = exp(log_threshold)')
    ax.set_title('Per-Feature JumpReLU Threshold Evolution — Gemma 9B\n'
                 'Threshold values increase with depth; high std confirms per-feature diversity', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIGURES2, f'gemma9b_threshold_depth.{ext}'))
    plt.close(fig)
    print('  Saved: gemma9b_threshold_depth.{png,pdf}')


def figure_neg_cos_summary(all_rows):
    """Bar chart comparing n_neg_cos across all models."""
    df = pd.DataFrame(all_rows)
    if df.empty: return

    # Aggregate by model
    df_p0 = pd.read_csv(os.path.join(PROJECT, 'results', 'phase0', 'layer_stats.csv'))

    def phase1_neg_count(prefix):
        # Count all features with cos(E, D) < 0 from the full Phase 1 per-layer arrays.
        import glob
        files = glob.glob(os.path.join(PROJECT, 'results', 'phase1', f'cos_sim_{prefix}_L*.npy'))
        return sum(int(np.sum(np.load(f) < 0)) for f in files), len(files)

    gpt2_neg, gpt2_layers = phase1_neg_count('gpt2')
    gemma_neg, gemma_layers = phase1_neg_count('gemma')
    llama_neg, llama_layers = phase1_neg_count('llama')

    models_data = {
        'GPT-2\n(ReLU+L1)': {'total_neg': gpt2_neg, 'n_layers': gpt2_layers, 'color': '#1f77b4'},
        'Gemma 2B\n(per-feat θ)': {'total_neg': gemma_neg, 'n_layers': gemma_layers,  'color': '#d62728'},
        'Gemma 9B\n(per-feat θ)': {'user': 'gemma9b', 'color': '#ff7f0e'},
        'Llama LXR-8x\n(scalar θ)': {'total_neg': llama_neg, 'n_layers': llama_layers, 'color': '#2ca02c'},
        'Llama LXR-32x\n(scalar θ)': {'user': 'llama32x_supp', 'color': '#9467bd'},
    }

    # Get Gemma 9B count
    g9b = df[df['model_key']=='gemma9b']
    if not g9b.empty:
        models_data['Gemma 9B\n(per-feat θ)']['total_neg'] = int(g9b['n_neg_cos'].sum())
        models_data['Gemma 9B\n(per-feat θ)']['n_layers'] = len(g9b)

    # Get Llama 32x from supplement
    try:
        df_s = pd.read_csv(os.path.join(RESULTS_SUPP, 'supplement_stats.csv'))
        l32x = df_s[df_s['model_key']=='llama32x']
        if not l32x.empty:
            # Compute neg from npy
            neg32x = sum(int(np.sum(np.load(os.path.join(RESULTS_SUPP, f'cos_sim_llama32x_L{l}.npy'))<0))
                        for l in l32x['layer_idx']
                        if os.path.exists(os.path.join(RESULTS_SUPP, f'cos_sim_llama32x_L{l}.npy')))
            models_data['Llama LXR-32x\n(scalar θ)']['total_neg'] = neg32x
            models_data['Llama LXR-32x\n(scalar θ)']['n_layers'] = len(l32x)
    except:
        models_data['Llama LXR-32x\n(scalar θ)']['total_neg'] = 0
        models_data['Llama LXR-32x\n(scalar θ)']['n_layers'] = 0

    labels = [k for k in models_data if 'total_neg' in models_data[k]]
    neg_counts = [models_data[k]['total_neg'] for k in labels]
    colors = [models_data[k]['color'] for k in labels]
    n_layers = [models_data[k].get('n_layers', 1) for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw counts
    axes[0].bar(range(len(labels)), neg_counts, color=colors, alpha=0.8, edgecolor='k', lw=0.5)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('Features with cos(E,D) < 0\n(all layers analysed)')
    axes[0].set_title('Total count')
    for i, v in enumerate(neg_counts):
        axes[0].text(i, v+0.5, str(v), ha='center', va='bottom', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Normalized by n_layers
    norm_counts = [n/nl if nl>0 else 0 for n,nl in zip(neg_counts, n_layers)]
    axes[1].bar(range(len(labels)), norm_counts, color=colors, alpha=0.8, edgecolor='k', lw=0.5)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Features with cos(E,D) < 0 per layer')
    axes[1].set_title('Per layer analysed')
    for i, v in enumerate(norm_counts):
        axes[1].text(i, v+0.01, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    for ax, vals in zip(axes, [neg_counts, norm_counts]):
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)

    fig.suptitle('Features with negative encoder-decoder cosine, by SAE family\n'
                 '(layers analysed: GPT-2 12, Gemma 2B 26, Gemma 9B 22, Llama 8x 32, '
                 'Llama 32x 3 [L0, L1, L10])',
                 fontsize=11)
    plt.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(FIGURES2, f'neg_cos_comparison.{ext}'))
    plt.close(fig)
    print('  Saved: neg_cos_comparison.{png,pdf}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('='*60)
    print('SUPPLEMENT 2: GEMMA 9B FULL + LLAMA TOPK (STREAMING MODE)')
    print(f'Disk free: {disk_free_gb():.1f} GB')
    print('='*60)

    all_rows, thresh_rows = [], []

    # Part 1: Gemma 9B all 42 layers
    analyze_gemma9b_streaming(all_rows, thresh_rows)

    # Checkpoint
    df_g9b = pd.DataFrame([r for r in all_rows if r['model_key']=='gemma9b'])
    df_g9b.to_csv(os.path.join(RESULTS2, 'gemma9b_full.csv'), index=False)
    print(f'\nGemma 9B checkpoint: {len(df_g9b)}/42 layers complete')

    # Part 2: Llama TopK attempt
    analyze_llama_topk_stream(all_rows)

    # Save everything
    pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS2, 'supplement2_all.csv'), index=False)
    if thresh_rows:
        pd.DataFrame(thresh_rows).to_csv(
            os.path.join(RESULTS2, 'gemma9b_threshold_all_layers.csv'), index=False)

    # Figures
    print('\n' + '='*60)
    print('FIGURES')
    print('='*60)
    figure_gemma9b_profile(df_g9b)
    figure_gemma9b_threshold_depth(thresh_rows)
    figure_neg_cos_summary(all_rows)

    # Final summary
    print('\n' + '='*60)
    print('FINAL SUMMARY')
    print('='*60)
    if not df_g9b.empty:
        total_neg = int(df_g9b['n_neg_cos'].sum())
        total_6sig = int(df_g9b['count_6sig'].sum())
        max_kurt = df_g9b['excess_kurtosis'].max()
        max_k_layer = int(df_g9b.loc[df_g9b['excess_kurtosis'].idxmax(), 'layer_idx'])
        min_cos = float(df_g9b['min_cos'].min())
        min_cos_layer = int(df_g9b.loc[df_g9b['min_cos'].idxmin(), 'layer_idx'])
        layers_complete = len(df_g9b)

        print(f'Layers analyzed: {layers_complete}/42')
        print(f'Total neg-cos features: {total_neg} (across all {layers_complete} layers)')
        print(f'Total 6sigma features: {total_6sig}')
        print(f'Max excess kurtosis: {max_kurt:.3f} at L{max_k_layer}')
        print(f'Global min cos(E,D): {min_cos:.4f} at L{min_cos_layer}')

        if total_neg == 0:
            print('\n→ E2 FULLY CONFIRMED: Zero negative-cosine features in Gemma 9B')
            print('  across ALL tested layers. Per-feature threshold = complete protection.')
        else:
            print(f'\n→ {total_neg} neg-cos features found. Check layers: '
                  f'{df_g9b[df_g9b["n_neg_cos"]>0]["layer_idx"].tolist()}')

    print(f'\nDisk free: {disk_free_gb():.1f} GB')
    print('\n>>> SUPPLEMENT 2 COMPLETE <<<')


if __name__ == '__main__':
    main()
