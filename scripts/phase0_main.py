"""
Phase 0: Cross-Model Verification
Geometry of SAE Artifacts Research Program

Computes encoder-decoder cosine similarity statistics across:
  - GPT-2 Small (ReLU+L1, Bloom SAEs)
  - Gemma Scope 2B (JumpReLU)
  - Llama Scope / Llama 3.1 8B (JumpReLU via SAELens)

Key metrics per layer:
  - cos(E_i, D_i) distribution: mu, sigma, skewness, excess_kurtosis
  - Pearson r(||E||, cos(E,D))
  - Count of features at <=mu-3sigma and <=mu-6sigma
  - Most extreme anti-aligned features

Usage:
  # from the repository root
  python scripts/phase0_main.py

Output:
  results/phase0/layer_stats.csv
  results/phase0/extreme_features.csv
  results/phase0/sign_flip.csv
  figures/phase0/*.{png,pdf}
"""

import os
import sys
import gc           # Garbage collection, crucial for managing memory when loading massive SAE tensors
import warnings
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import torch
import torch.nn.functional as F

# Suppress warnings (like deprecation warnings from PyTorch/Pandas) for cleaner console output
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────────────────
# Define absolute paths based on the user's home directory to ensure the script runs anywhere
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT, 'results', 'phase0')
FIGURES_DIR = os.path.join(PROJECT, 'figures', 'phase0')

# Safely create directories if they don't already exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE  
# ────────────────────────────────────────────────────────────────────────────
# Set a globally consistent, publication-ready aesthetic for all generated plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,           # Good resolution for viewing on screen
    'savefig.dpi': 300,          # High resolution for saving to file
    'savefig.bbox': 'tight',     # Removes excess whitespace around saved plots
    'axes.spines.top': False,    # Clean up top/right borders
    'axes.spines.right': False,
})

# Dictionary mapping full descriptive names to specific colors for consistency across plots
COLOR_MAP = {
    'GPT-2 Small\n(ReLU+L1)':   '#1f77b4', # Standard Matplotlib Blue
    'Gemma 2B\n(JumpReLU)':     '#d62728', # Standard Matplotlib Red
    'Llama 3.1 8B\n(JumpReLU)': '#2ca02c', # Standard Matplotlib Green
}
# Shorter keys mapping to the same colors for easier code reference
COLORS = {
    'gpt2':  '#1f77b4',
    'gemma': '#d62728',
    'llama': '#2ca02c',
}

def save_figure(fig, stem):
    """Utility function to save a figure in both PNG (for web/quick view) and PDF (for papers)."""
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES_DIR, f'{stem}.{ext}'), dpi=300)
    plt.close(fig) # Frees up memory after saving
    print(f"  Saved: figures/phase0/{stem}.{{png,pdf}}")


# ────────────────────────────────────────────────────────────────────────────
# CORE STATISTICS FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def compute_cosine_stats(W_enc_tensor, W_dec_tensor):
    """
    Computes statistical relationships between the Encoder and Decoder weights.
    
    W_enc: (d_model, d_sae) -- SAELens convention for encoder weights
    W_dec: (d_sae, d_model) -- SAELens convention for decoder weights

    Returns: (stats_dict, cos_sim_np, enc_norms_np)
    """
    # Ensure tensors are float32 to prevent precision errors
    W_enc = W_enc_tensor.float()
    W_dec = W_dec_tensor.float()

    # Transpose encoder so both E and D have shape (d_sae, d_model).
    # This aligns the features so we can compare the i-th encoder feature to the i-th decoder feature.
    E = W_enc.T    
    D = W_dec      

    # Calculate Cosine Similarity per feature along the d_model dimension (dim=1)
    cos_sim = F.cosine_similarity(E, D, dim=1)   # Shape: (d_sae,)
    
    # Calculate the L2 Norm (length) of the encoder vectors
    enc_norms = E.norm(dim=1)                     # Shape: (d_sae,)

    # Move to CPU and convert to NumPy for use with SciPy and Pandas
    cos_np = cos_sim.cpu().numpy()
    norms_np = enc_norms.cpu().numpy()

    # Calculate standard distribution statistics
    mu    = float(np.mean(cos_np))
    sigma = float(np.std(cos_np, ddof=1)) # ddof=1 for sample standard deviation
    skew  = float(stats.skew(cos_np))     # Measures asymmetry of the distribution
    kurt  = float(stats.kurtosis(cos_np)) # Fisher's excess kurtosis (0 means perfectly Gaussian)

    # Calculate Pearson Correlation between Encoder Norms and Cosine Similarity.
    # We extract [0, 1] from the correlation matrix to get the actual r value.
    r_pearson = float(np.corrcoef(norms_np, cos_np)[0, 1])

    # Count how many features fall into extreme negative tails (anti-aligned features)
    count_3sig = int(np.sum(cos_np < mu - 3 * sigma))
    count_6sig = int(np.sum(cos_np < mu - 6 * sigma))

    # Return a dictionary of scalar stats, plus the raw arrays for plotting later
    return {
        'mu': mu, 'sigma': sigma,
        'skewness': skew, 'excess_kurtosis': kurt,
        'r_pearson': r_pearson,
        'count_3sig': count_3sig, 'count_6sig': count_6sig,
        'min_cos': float(np.min(cos_np)),
        'max_cos': float(np.max(cos_np)),
        'n_features': len(cos_np)
    }, cos_np, norms_np


def get_extreme_features(cos_np, norms_np, layer_idx, model_name, n_top=10):
    """
    Identifies the `n_top` features that have the lowest cosine similarity (most anti-aligned).
    Returns a list of dictionaries, one for each extreme feature.
    """
    mu    = np.mean(cos_np)
    sigma = np.std(cos_np, ddof=1)
    
    # np.argsort returns indices that would sort the array from lowest to highest.
    # So the first elements are the lowest (most negative/anti-aligned) cosine similarities.
    sorted_idx = np.argsort(cos_np)
    
    rows = []
    for rank, feat_idx in enumerate(sorted_idx[:n_top]):
        rows.append({
            'model': model_name, 
            'layer_idx': layer_idx,
            'feature_idx': int(feat_idx),
            'cos_ED': float(cos_np[feat_idx]),
            'sigma_deviation': float((cos_np[feat_idx] - mu) / sigma), # How many standard deviations away from the mean
            'enc_norm': float(norms_np[feat_idx]),
            'rank': rank + 1
        })
    return rows


# ────────────────────────────────────────────────────────────────────────────
# SAE LOADING WRAPPER
# ────────────────────────────────────────────────────────────────────────────
def load_and_analyze(release, sae_id, layer_idx, model_name, n_layers_total,
                     architecture, description=''):
    """
    Downloads/loads a specific SAE from HuggingFace via SAELens, computes its stats, 
    and handles memory cleanup immediately.
    """
    from sae_lens import SAE
    t0 = time.time()
    print(f"  Loading {sae_id} ...", end=' ', flush=True)
    try:
        # Load SAE to CPU to prevent overloading GPU VRAM (we only need to do math, not inference)
        sae = SAE.from_pretrained(release, sae_id, device='cpu')
        
        # Detach weights from the computation graph since we aren't training them
        W_enc = sae.W_enc.detach()
        W_dec = sae.W_dec.detach()
        d_model, d_sae = W_enc.shape
        
        # Get statistics and raw arrays
        s, cos_np, norms_np = compute_cosine_stats(W_enc, W_dec)

        # Build a row for our Pandas DataFrame
        row = {
            'model': model_name,
            'architecture': architecture,
            'n_layers_total': n_layers_total,
            'layer_idx': layer_idx,
            # relative_depth lets us compare models of different sizes on a 0-1 scale
            'relative_depth': round(layer_idx / max(n_layers_total - 1, 1), 4),
            'd_model': d_model, 'd_sae': d_sae,
            'sae_id': sae_id,
            **s # Unpack all the stats from the dictionary
        }
        
        elapsed = time.time() - t0
        print(f"ok ({elapsed:.1f}s) | mu={s['mu']:.3f} s={s['sigma']:.3f} "
              f"r={s['r_pearson']:+.3f} 6sig={s['count_6sig']}")
        
        # Critical: Delete large objects and force garbage collection to prevent RAM Out-Of-Memory errors
        del sae, W_enc, W_dec
        gc.collect()
        
        return row, cos_np, norms_np
    
    except Exception as e:
        print(f"FAILED: {e}")
        return None, None, None


# ────────────────────────────────────────────────────────────────────────────
# MODEL ANALYSIS FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def analyze_gpt2(all_rows, all_extreme, all_cos_data):
    """Analyzes all 12 layers of GPT-2 Small using Bloom's ReLU+L1 SAEs."""
    print("\n" + "="*60)
    print("GPT-2 SMALL (Bloom SAEs, ReLU+L1)")
    print("="*60)
    
    # Configuration metadata for SAELens
    release = 'gpt2-small-res-jb'
    model_name = 'GPT-2 Small\n(ReLU+L1)'
    architecture = 'ReLU+L1 (ghost gradients)'

    for layer in range(12):
        # The hook point 'resid_pre' refers to the residual stream before the attention layer
        sae_id = f'blocks.{layer}.hook_resid_pre'
        row, cos_np, norms_np = load_and_analyze(
            release, sae_id, layer, model_name, 12, architecture
        )
        if row is not None:
            all_rows.append(row)
            extreme = get_extreme_features(cos_np, norms_np, layer, model_name)
            all_extreme.extend(extreme)
            # Store raw data for generating histograms later
            all_cos_data[(model_name, layer)] = (cos_np, norms_np)


def analyze_gemma(all_rows, all_extreme, all_cos_data):
    """Analyzes all 26 layers of Gemma Scope 2B (JumpReLU)."""
    print("\n" + "="*60)
    print("GEMMA SCOPE 2B (JumpReLU, 26 layers)")
    print("="*60)
    
    release = 'gemma-scope-2b-pt-res-canonical'
    model_name = 'Gemma 2B\n(JumpReLU)'
    architecture = 'JumpReLU (learned threshold)'

    # We need to dynamically check which SAE widths are available because Gemma Scope 
    # uses 16k width for some layers and 65k for others.
    from sae_lens.saes.sae import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    info = d.get(release)
    if info is None:
        print(f"  ERROR: Release '{release}' not found in SAELens directory")
        return

    available_ids = set(info.saes_map.keys())
    
    for layer in range(26):
        # We prefer the 16k dictionary size, but fall back to 65k if necessary
        sae_id_16k = f'layer_{layer}/width_16k/canonical'
        sae_id_65k = f'layer_{layer}/width_65k/canonical'
        
        if sae_id_16k in available_ids:
            sae_id = sae_id_16k
        elif sae_id_65k in available_ids:
            sae_id = sae_id_65k
        else:
            # Absolute fallback: Find ANY canonical SAE for this layer
            layer_ids = [x for x in available_ids if f'layer_{layer}/' in x]
            if not layer_ids:
                print(f"  Layer {layer}: no SAE found, skipping")
                continue
            sae_id = layer_ids[0]

        row, cos_np, norms_np = load_and_analyze(
            release, sae_id, layer, model_name, 26, architecture
        )
        if row is not None:
            all_rows.append(row)
            extreme = get_extreme_features(cos_np, norms_np, layer, model_name)
            all_extreme.extend(extreme)
            all_cos_data[(model_name, layer)] = (cos_np, norms_np)


def analyze_llama(all_rows, all_extreme, all_cos_data):
    """Analyzes all 32 layers of Llama 3.1 8B using Llama Scope."""
    print("\n" + "="*60)
    print("LLAMA SCOPE / LLAMA 3.1 8B (JumpReLU via SAELens, 32 layers)")
    print("="*60)
    release = 'llama_scope_lxr_8x'
    model_name = 'Llama 3.1 8B\n(JumpReLU)'
    architecture = 'JumpReLU (Llama Scope)'

    for layer in range(32):
        # 'l{layer}r_8x' refers to Layer {layer}, Residual stream, 8x expansion factor
        sae_id = f'l{layer}r_8x'
        row, cos_np, norms_np = load_and_analyze(
            release, sae_id, layer, model_name, 32, architecture
        )
        if row is not None:
            all_rows.append(row)
            extreme = get_extreme_features(cos_np, norms_np, layer, model_name)
            all_extreme.extend(extreme)
            all_cos_data[(model_name, layer)] = (cos_np, norms_np)


# ────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ────────────────────────────────────────────────────────────────────────────

def figure_sign_flip_comparison(df):
    """
    Plots the main diagnostic figure: Correlation between Encoder Norms and Cosine Similarity 
    plotted against the relative depth of the model.
    A transition from >0 to <0 indicates the "sign flip" phenomenon.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define markers and labels manually for better legend presentation
    model_color = {k: COLORS[v] for k, v in zip(df['model'].unique(), ['gpt2', 'gemma', 'llama'])}
    model_marker = {
        'GPT-2 Small\n(ReLU+L1)':   'o', # Circle
        'Gemma 2B\n(JumpReLU)':     's', # Square
        'Llama 3.1 8B\n(JumpReLU)': '^', # Triangle
    }
    model_labels = {
        'GPT-2 Small\n(ReLU+L1)':   'GPT-2 Small (ReLU+L1, 124M)',
        'Gemma 2B\n(JumpReLU)':     'Gemma 2B (JumpReLU, 2.5B)',
        'Llama 3.1 8B\n(JumpReLU)': 'Llama 3.1 8B (JumpReLU, 8B)',
    }

    # Plot lines for each model
    for model in model_color:
        sub = df[df['model'] == model].sort_values('relative_depth')
        if sub.empty:
            continue
        ax.plot(sub['relative_depth'], sub['r_pearson'],
                color=model_color[model],
                marker=model_marker[model],
                linewidth=1.8, markersize=5, alpha=0.9,
                label=model_labels[model])

    # Draw a dashed black line at y=0 to clearly show the "flip"
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Relative Layer Depth (layer / total_layers)', fontsize=11)
    ax.set_ylabel(r'Pearson $r(\|E\|,\, \cos(E,D))$', fontsize=11)
    ax.set_title('Sign Flip in Encoder Norm vs.\ Encoder-Decoder Cosine Correlation\n'
                 r'Positive$\to$Negative transition indicates shift to superposition-dominated geometry',
                 fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'sign_flip_comparison')


def figure_mean_cosine_by_depth(df):
    """Plots how the average alignment (cosine similarity) degrades or improves over network depth."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    model_color = {k: COLORS[v] for k, v in zip(df['model'].unique(), ['gpt2', 'gemma', 'llama'])}
    model_labels = {
        'GPT-2 Small\n(ReLU+L1)':   'GPT-2 Small (ReLU+L1)',
        'Gemma 2B\n(JumpReLU)':     'Gemma 2B (JumpReLU)',
        'Llama 3.1 8B\n(JumpReLU)': 'Llama 3.1 8B (JumpReLU)',
    }

    for model in model_color:
        sub = df[df['model'] == model].sort_values('relative_depth')
        if sub.empty:
            continue
        # Main line represents the mean
        ax.plot(sub['relative_depth'], sub['mu'],
                color=model_color[model], linewidth=1.8,
                label=model_labels[model])
        
        # Shaded area represents +/- 1 Standard Deviation (sigma)
        ax.fill_between(sub['relative_depth'],
                        sub['mu'] - sub['sigma'],
                        sub['mu'] + sub['sigma'],
                        color=model_color[model], alpha=0.15)

    ax.set_xlabel('Relative Layer Depth')
    ax.set_ylabel(r'Mean $\cos(E, D)$ $\pm 1\sigma$', fontsize=11)
    ax.set_title(r'Encoder-Decoder Cosine Similarity Mean by Layer Depth', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'mean_cosine_by_depth')


def figure_6sigma_tail_by_depth(df):
    """Generates bar charts showing the count of severely anti-aligned features per layer."""
    # Create 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    model_list = [
        ('GPT-2 Small\n(ReLU+L1)',   COLORS['gpt2'],  'GPT-2 Small (ReLU+L1)'),
        ('Gemma 2B\n(JumpReLU)',     COLORS['gemma'], 'Gemma 2B (JumpReLU)'),
        ('Llama 3.1 8B\n(JumpReLU)', COLORS['llama'], 'Llama 3.1 8B (JumpReLU)'),
    ]

    for ax, (model, color, title) in zip(axes, model_list):
        sub = df[df['model'] == model].sort_values('relative_depth')
        if sub.empty:
            # Handle missing data gracefully
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='grey')
            ax.set_title(title, fontsize=10)
            continue
        
        # Bar chart: X is depth, Y is the raw count of features in the 6-sigma tail
        ax.bar(sub['relative_depth'], sub['count_6sig'],
               width=0.9/len(sub), color=color, alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.set_xlabel('Relative Depth')
        ax.set_ylabel('Features at $\\leq\\mu-6\\sigma$')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Count of Strongly Anti-Aligned Features ($\\geq 6\\sigma$) by Layer', fontsize=12)
    fig.tight_layout()
    save_figure(fig, '6sigma_tail_count')


def figure_heatmap(df):
    """Creates a visual heatmap of the Pearson Correlation to quickly spot where the sign flips occur."""
    from matplotlib.colors import TwoSlopeNorm

    models = [
        'GPT-2 Small\n(ReLU+L1)',
        'Gemma 2B\n(JumpReLU)',
        'Llama 3.1 8B\n(JumpReLU)',
    ]
    model_labels = ['GPT-2 Small', 'Gemma 2B', 'Llama 3.1 8B']

    # Because models have different layer counts, we group them into 10 normalized 'bins' (0% to 100% depth)
    bins = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    df['depth_bin'] = pd.cut(df['relative_depth'], bins=bins, labels=bin_centers)

    # Initialize a 3x10 matrix with NaNs (Not a Number)
    matrix = np.full((3, 10), np.nan)
    for mi, model in enumerate(models):
        sub = df[df['model'] == model].copy()
        if sub.empty:
            continue
        # Average the correlation values within each depth bin
        grouped = sub.groupby('depth_bin', observed=True)['r_pearson'].mean()
        for bi, bc in enumerate(bin_centers):
            if bc in grouped.index:
                matrix[mi, bi] = grouped[bc]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # TwoSlopeNorm ensures that 0 is perfectly white in our Red-Blue colormap
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', norm=norm,
                   extent=[-0.05, 1.05, -0.5, 2.5])

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(model_labels[::-1]) # Reverse to match Y-axis drawing direction
    ax.set_xlabel('Relative Layer Depth')
    ax.set_title(r'$r(\|E\|, \cos(E,D))$ Heatmap by Layer Depth and Model', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'Pearson $r$')

    # Add numeric text over the colored blocks for precise reading
    for mi in range(3):
        for bi in range(10):
            v = matrix[mi, bi]
            if not np.isnan(v):
                # Change text color to white if the background color is dark (abs(v) > 0.5)
                ax.text(bins[bi] + 0.05, 2 - mi, f'{v:+.2f}',
                        ha='center', va='center', fontsize=7.5,
                        color='white' if abs(v) > 0.5 else 'black')

    fig.tight_layout()
    save_figure(fig, 'heatmap_r_pearson')


def figure_distributions_comparison(all_cos_data, df):
    """
    Plots detailed Histograms of Cosine Similarity at Early, Middle, and Late layers.
    This lets us physically see the shape of the distribution changing over depth.
    """
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    fig.suptitle(r'Distribution of $\cos(E, D)$ at Early, Mid, and Late Layers', fontsize=12)

    models = [
        'GPT-2 Small\n(ReLU+L1)',
        'Gemma 2B\n(JumpReLU)',
        'Llama 3.1 8B\n(JumpReLU)',
    ]
    model_labels = ['GPT-2 Small (ReLU+L1)', 'Gemma 2B (JumpReLU)', 'Llama 3.1 8B (JumpReLU)']
    
    # We want to look at layers roughly at 0%, 50%, and 90% through the network
    depth_targets = [0.0, 0.5, 0.9]
    depth_labels = ['Early ($\\sim$0\\%)', 'Mid ($\\sim$50\\%)', 'Late ($\\sim$90\\%)']

    for mi, (model, label) in enumerate(zip(models, model_labels)):
        sub = df[df['model'] == model].sort_values('relative_depth')
        if sub.empty:
            for col in range(3):
                axes[mi, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[mi, col].transAxes, color='grey')
            continue

        for col, (dt, dl) in enumerate(zip(depth_targets, depth_labels)):
            ax = axes[mi, col]
            
            # Find the actual layer closest to our target depth (0, 0.5, or 0.9)
            closest_row = sub.iloc[(sub['relative_depth'] - dt).abs().argsort()[:1]]
            layer_idx = int(closest_row['layer_idx'].values[0])
            rel_d = float(closest_row['relative_depth'].values[0])

            key = (model, layer_idx)
            if key not in all_cos_data:
                continue

            cos_np, _ = all_cos_data[key]
            mu_v = float(np.mean(cos_np))
            sig_v = float(np.std(cos_np, ddof=1))
            r_v = float(closest_row['r_pearson'].values[0])
            n_6sig = int(closest_row['count_6sig'].values[0])

            # Draw the histogram
            ax.hist(cos_np, bins=80, color=list(COLORS.values())[mi],
                    alpha=0.75, edgecolor='none', density=True)
            
            # Draw lines indicating the Mean (mu) and the -6 Sigma boundary
            ax.axvline(mu_v, color='black', lw=1.5, linestyle='-', label=f'$\\mu$={mu_v:.3f}')
            ax.axvline(mu_v - 6*sig_v, color='red', lw=1.2, linestyle='--',
                       label=f'$-6\\sigma$ (n={n_6sig})')

            ax.set_title(f'{dl}, L{layer_idx} (d={rel_d:.2f})\n'
                         f'$r$={r_v:+.3f}, {n_6sig} feats $\\leq-6\\sigma$',
                         fontsize=9)
            ax.set_xlabel(r'$\cos(E_i, D_i)$', fontsize=9)
            if col == 0:
                ax.set_ylabel(label + '\n\nDensity', fontsize=9)
            ax.legend(fontsize=7.5, loc='upper left')
            ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_figure(fig, 'cosine_distributions')


def figure_skewness_kurtosis(df):
    """Plots higher-order statistical moments (Skewness and Kurtosis) to measure non-Gaussianity."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for model, color in [
        ('GPT-2 Small\n(ReLU+L1)',   COLORS['gpt2']),
        ('Gemma 2B\n(JumpReLU)',     COLORS['gemma']),
        ('Llama 3.1 8B\n(JumpReLU)', COLORS['llama']),
    ]:
        sub = df[df['model'] == model].sort_values('relative_depth')
        if sub.empty:
            continue
        label = model.replace('\n', ' ')
        
        # Plot Skewness (Top chart)
        axes[0].plot(sub['relative_depth'], sub['skewness'],
                     color=color, marker='o', markersize=4, linewidth=1.5, label=label)
        
        # Plot Excess Kurtosis (Bottom chart)
        axes[1].plot(sub['relative_depth'], sub['excess_kurtosis'],
                     color=color, marker='s', markersize=4, linewidth=1.5, label=label)

    # Reference lines for a perfect Gaussian distribution (0 for both)
    axes[0].axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    axes[1].axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5,
                    label='Gaussian (excess kurtosis=0)')

    axes[0].set_ylabel('Skewness $\\gamma_1$')
    axes[0].set_title(r'Skewness of $\cos(E,D)$ Distribution by Layer')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Excess Kurtosis $\\gamma_2$')
    axes[1].set_xlabel('Relative Layer Depth')
    axes[1].set_title(r'Excess Kurtosis of $\cos(E,D)$ Distribution by Layer')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'skewness_kurtosis')


# ────────────────────────────────────────────────────────────────────────────
# GATE G0 EVALUATION
# ────────────────────────────────────────────────────────────────────────────

def evaluate_gate_g0(df):
    """
    The "Gate G0" acts as an automated researcher decision point. 
    It checks if the anomalies discovered in GPT-2 (sign flip and fat tails) 
    actually appear in modern models (Gemma/Llama).
    
    G0-YES:    Generalizes to >=2 model families. Proceed with research.
    G0-PARTIAL: Only found in GPT-2, but deep tails exist elsewhere. Proceed carefully.
    G0-NO:     Total failure to generalize. Stop research.
    """
    print("\n" + "="*60)
    print("GATE G0 EVALUATION")
    print("="*60)

    model_sign_flip = {}
    model_has_6sig = {}

    for model in df['model'].unique():
        sub = df[df['model'] == model].sort_values('relative_depth')
        if len(sub) < 3:
            continue
        r_values = sub['r_pearson'].values
        
        # Analyze first 33% of layers vs last 33% of layers
        early = sub[sub['relative_depth'] <= 0.33]['r_pearson'].mean()
        late  = sub[sub['relative_depth'] >= 0.67]['r_pearson'].mean()
        
        # A "Sign Flip" is strictly defined as Early > 0.1 AND Late < -0.1
        has_flip = (early > 0.1) and (late < -0.1)
        max_neg_r = r_values.min()

        max_6sig = sub['count_6sig'].max()
        total_6sig = sub['count_6sig'].sum()

        model_sign_flip[model] = {
            'early_r': float(early), 'late_r': float(late),
            'has_flip': has_flip, 'max_neg_r': float(max_neg_r)
        }
        model_has_6sig[model] = {
            'max_6sig': int(max_6sig), 'total_6sig': int(total_6sig)
        }

        # Print logic trace to console
        flip_str = "YES" if has_flip else "NO"
        print(f"\n{model.replace(chr(10), ' ')}:")
        print(f"  Early r (depth<=0.33): {early:+.3f}")
        print(f"  Late r  (depth>=0.67): {late:+.3f}")
        print(f"  Sign flip: {flip_str}")
        print(f"  Max neg r: {max_neg_r:+.3f}")
        print(f"  Max 6sigma count: {max_6sig}, total across layers: {total_6sig}")

    models_with_flip = [m for m, v in model_sign_flip.items() if v['has_flip']]
    models_with_6sig = [m for m, v in model_has_6sig.items() if v['max_6sig'] > 0]

    # Gate logic conditionals
    if len(models_with_flip) >= 2:
        gate = 'G0-YES'
    elif len(models_with_flip) == 1 and 'GPT-2' in models_with_flip[0]:
        if len(models_with_6sig) > 1:
            gate = 'G0-PARTIAL'
        else:
            gate = 'G0-NO'
    else:
        gate = 'G0-NO'

    # Save the decision and rationale to a Markdown file as a persistent log
    decision_file = os.path.join(RESULTS_DIR, 'G0_decision.md')
    with open(decision_file, 'w') as f:
        f.write(f"# Gate G0 Decision\n\n")
        f.write(f"Decision: **{gate}**\n\n")
        f.write(f"## Model-by-Model Results\n\n")
        for model in model_sign_flip:
            v = model_sign_flip[model]
            s = model_has_6sig[model]
            f.write(f"### {model.replace(chr(10), ' ')}\n")
            f.write(f"- Early r (depth<=0.33): {v['early_r']:+.3f}\n")
            f.write(f"- Late r  (depth>=0.67): {v['late_r']:+.3f}\n")
            f.write(f"- Sign flip: {'YES' if v['has_flip'] else 'NO'}\n")
            f.write(f"- Max neg r: {v['max_neg_r']:+.3f}\n")
            f.write(f"- Max 6sigma count per layer: {s['max_6sig']}\n\n")
        f.write(f"## Rationale\n\n")
        f.write(f"Models with sign flip: {[m.replace(chr(10),' ') for m in models_with_flip]}\n\n")
        f.write(f"Models with 6sigma tail: {[m.replace(chr(10),' ') for m in models_with_6sig]}\n\n")
        
        # Write final recommendation string
        if gate == 'G0-YES':
            f.write("PROCEED with all phases. Sign flip and/or tail generalizes across model families.\n")
        elif gate == 'G0-PARTIAL':
            f.write("PROCEED but reframe. Sign flip is GPT-2 specific but tail found elsewhere.\n")
        else:
            f.write("TERMINATE. Neither phenomenon found outside GPT-2 Small + Bloom SAEs.\n")

    return gate, model_sign_flip, model_has_6sig


# ────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION PIPELINE
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("PHASE 0: CROSS-MODEL VERIFICATION")
    print("Geometry of SAE Artifacts Research Program")
    print("="*60)
    
    # Store aggregated data across all models
    all_rows = []
    all_extreme = []
    all_cos_data = {}  

    # ── Step 1: Process GPT-2
    analyze_gpt2(all_rows, all_extreme, all_cos_data)

    # Save a checkpoint! Because SAEs are huge, scripts like this often crash halfway through. 
    # Checkpoints ensure we don't lose the GPT-2 data if the Gemma download fails.
    df_checkpoint = pd.DataFrame(all_rows)
    df_checkpoint.to_csv(os.path.join(RESULTS_DIR, 'layer_stats_gpt2.csv'), index=False)
    print(f"\nCheckpoint saved: layer_stats_gpt2.csv ({len(df_checkpoint)} rows)")

    # ── Step 2: Process Gemma
    analyze_gemma(all_rows, all_extreme, all_cos_data)

    # Save second checkpoint
    df_checkpoint2 = pd.DataFrame(all_rows)
    df_checkpoint2.to_csv(os.path.join(RESULTS_DIR, 'layer_stats_gpt2_gemma.csv'), index=False)
    print(f"\nCheckpoint saved: layer_stats_gpt2_gemma.csv ({len(df_checkpoint2)} rows)")

    # ── Step 3: Process Llama
    analyze_llama(all_rows, all_extreme, all_cos_data)

    # ── Save Final CSVs
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Main stats CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'layer_stats.csv'), index=False)
    
    # Extract just the top 10 worst features per layer
    df_extreme = pd.DataFrame(all_extreme)
    df_extreme.to_csv(os.path.join(RESULTS_DIR, 'extreme_features.csv'), index=False)

    # Subset CSV specifically focusing on the sign flip variables
    sign_flip_rows = [{
        'model': row['model'],
        'architecture': row['architecture'],
        'layer_idx': row['layer_idx'],
        'relative_depth': row['relative_depth'],
        'r_pearson': row['r_pearson'],
        'n_features': row['n_features'],
        'd_sae': row['d_sae'],
        'sae_id': row['sae_id'],
    } for _, row in df.iterrows()]
    
    df_sf = pd.DataFrame(sign_flip_rows)
    df_sf.to_csv(os.path.join(RESULTS_DIR, 'sign_flip.csv'), index=False)

    # ── Trigger Matplotlib generations
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    figure_sign_flip_comparison(df)
    figure_mean_cosine_by_depth(df)
    figure_6sigma_tail_by_depth(df)
    figure_heatmap(df)
    figure_distributions_comparison(all_cos_data, df)
    figure_skewness_kurtosis(df)

    # ── Execute the programmatic Go/No-Go Decision
    gate, flip_info, sig_info = evaluate_gate_g0(df)

    # ── Print Final Output Console Table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    # Pandas grouping to create a nice, readable terminal summary
    summary = df.groupby(['model', 'architecture']).agg(
        n_layers=('layer_idx', 'count'),
        r_early=('r_pearson', lambda x: x[df.loc[x.index, 'relative_depth'] <= 0.33].mean()),
        r_late=('r_pearson', lambda x: x[df.loc[x.index, 'relative_depth'] >= 0.67].mean()),
        mean_mu=('mu', 'mean'),
        total_6sig=('count_6sig', 'sum'),
        max_6sig=('count_6sig', 'max'),
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\n>>> PHASE 0 COMPLETE <<<")
    print(f"Gate decision: {gate}")

    return df, df_extreme, gate

# Standard Python idiom to ensure code runs only when executed directly, not when imported
if __name__ == '__main__':
    df, df_extreme, gate = main()