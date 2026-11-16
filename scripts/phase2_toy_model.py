"""
Phase 2: Toy Model Validation
The Geometry of SAE Artifacts: Layer-Dependent Encoder-Decoder Misalignment

Sub-phases:
  2b: 10D numerical experiment (ReLU+L1)
  2c: TopK comparison
  2d: Phase boundary sweep (lambda x theta x p_dep)
  2e: JumpReLU scalar vs per-feature threshold (NEW from SUPL2 findings)

Analytical derivation (2a) is in results/phase2/analytical_proof.md

Usage:
  python scripts/phase2_toy_model.py --part 2b
  python scripts/phase2_toy_model.py --part 2c
  python scripts/phase2_toy_model.py --part 2d
  python scripts/phase2_toy_model.py --part 2e
  python scripts/phase2_toy_model.py --part all
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

# ---- Paths ----
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, 'results', 'phase2')
FIGURES = os.path.join(PROJECT, 'figures', 'phase2')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---- Publication style ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLOR_MAP = {
    'ReLU+L1':               '#1f77b4',
    'TopK':                   '#d62728',
    'JumpReLU (scalar)':     '#ff7f0e',
    'JumpReLU (per-feature)': '#2ca02c',
}


def save_figure(fig, name):
    fig.savefig(os.path.join(FIGURES, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figures/phase2/{name}.png + .pdf")


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(d_model=10, n_concepts=15, n_samples=100_000,
                             n_asymmetric_pairs=3, p_dep=0.3, noise_std=0.05,
                             seed=42, theta_deg=None):
    """
    Generate synthetic activation data with known asymmetric co-activation.

    Args:
        theta_deg: If given, force cos(parent, child) = cos(theta_deg) exactly.
                   Otherwise uses the 0.6/0.8 default (gives cos ~0.4-0.6).
    Returns:
        X:            (n_samples, d_model) tensor
        directions:   (n_concepts, d_model) ground-truth concept directions (unit norm)
        pair_indices: list of (parent_idx, child_idx) tuples
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    directions = torch.randn(n_concepts, d_model)
    directions = directions / directions.norm(dim=1, keepdim=True)

    pair_indices = []

    for i in range(n_asymmetric_pairs):
        parent_idx = i * 2
        child_idx  = i * 2 + 1
        parent = directions[parent_idx]

        if theta_deg is not None:
            theta_rad = np.deg2rad(theta_deg)
            cos_t = float(np.cos(theta_rad))
            sin_t = float(np.sin(theta_rad))
            # orthogonal complement to parent
            ortho = torch.randn(d_model)
            ortho = ortho - (ortho @ parent) * parent
            ortho = ortho / ortho.norm().clamp(min=1e-8)
            child = cos_t * parent + sin_t * ortho
        else:
            noise = torch.randn(d_model)
            child = 0.6 * parent + 0.8 * noise

        child = child / child.norm().clamp(min=1e-8)
        directions[child_idx] = child
        pair_indices.append((parent_idx, child_idx))

        cos_pc = float((parent * child).sum())
        print(f"    Pair {i}: parent={parent_idx}, child={child_idx},"
              f" cos(parent,child)={cos_pc:.3f}")

    # Generate co-activations
    X_list = []
    for _ in range(n_samples):
        active_mag = torch.zeros(n_concepts)
        for c in range(n_concepts):
            if rng.random() < 0.05:
                active_mag[c] = float(rng.uniform(0.5, 1.5))
        for parent_idx, child_idx in pair_indices:
            if active_mag[parent_idx] == 0:
                active_mag[child_idx] = 0
            elif rng.random() < p_dep:
                active_mag[child_idx] = float(rng.uniform(0.5, 1.5))
        x = (active_mag.unsqueeze(1) * directions).sum(0)
        x = x + noise_std * torch.randn(d_model)
        X_list.append(x)

    X = torch.stack(X_list)
    return X, directions, pair_indices


# =============================================================================
# SAE MODELS
# =============================================================================

class ReLUL1SAE(nn.Module):
    """Standard SAE: ReLU + L1 sparsity penalty."""

    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= norms

    def forward(self, x):
        z = torch.relu(x @ self.W_enc + self.b_enc)
        x_hat = z @ self.W_dec + self.b_dec
        return z, x_hat

    def loss(self, x, lambda_val=1e-3):
        z, x_hat = self.forward(x)
        recon   = (x - x_hat).pow(2).sum(1).mean()
        sparse  = lambda_val * z.sum(1).mean()
        total   = recon + sparse
        return total, float(recon), float(sparse)


class TopKSAE(nn.Module):
    """TopK SAE: hard sparsity, no L1 penalty."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k  = k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= norms

    def forward(self, x):
        pre_act = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre_act.topk(self.k, dim=1)
        z = torch.zeros_like(pre_act)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))
        x_hat = z @ self.W_dec + self.b_dec
        return z, x_hat

    def loss(self, x, lambda_val=0.0):
        z, x_hat = self.forward(x)
        recon = (x - x_hat).pow(2).sum(1).mean()
        return recon, float(recon), 0.0


class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE.
    per_feature_threshold=True  -> one learnable threshold per feature (Gemma-style)
    per_feature_threshold=False -> single scalar threshold shared by all (Llama LXR-style)

    Sparsity: L0 approximation via sigmoid straight-through estimator.
    """

    def __init__(self, d_model, d_sae, per_feature_threshold=True, beta=0.1):
        super().__init__()
        self.beta = beta
        self.per_feature = per_feature_threshold

        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Threshold parameterized as log for positivity
        if per_feature_threshold:
            self.log_theta = nn.Parameter(torch.zeros(d_sae))
        else:
            self.log_theta = nn.Parameter(torch.zeros(1))

    @property
    def theta(self):
        return torch.exp(self.log_theta)

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= norms

    def forward(self, x):
        pre_act = x @ self.W_enc + self.b_enc   # (batch, d_sae)
        theta   = self.theta                      # (d_sae,) or (1,)
        gate    = (pre_act > theta).float()
        z       = pre_act * gate
        x_hat   = z @ self.W_dec + self.b_dec
        return z, x_hat, pre_act

    def loss(self, x, lambda_val=1e-2):
        z, x_hat, pre_act = self.forward(x)
        recon = (x - x_hat).pow(2).sum(1).mean()
        # Smooth L0 approximation: differentiable w.r.t. both W_enc and log_theta
        theta    = self.theta
        l0_approx = torch.sigmoid((pre_act - theta) / self.beta).sum(1).mean()
        sparse   = lambda_val * l0_approx
        total    = recon + sparse
        return total, float(recon), float(sparse)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def compute_cosines(model):
    """Return cos(E_i, D_i) for each feature i as a numpy array."""
    with torch.no_grad():
        E = model.W_enc.T           # (d_sae, d_model)
        D = model.W_dec             # (d_sae, d_model)
        E_norm = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        D_norm = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (E_norm * D_norm).sum(1)
    return cosines.numpy()


def train_sae(model, X, n_steps=15_000, batch_size=512, lr=1e-3,
              lambda_val=1e-3, log_every=3000):
    """Generic training loop. Returns (losses, cosines_history)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    losses = []
    cosines_history = []   # list of (step, cosines_array)

    for step in range(n_steps):
        model.normalize_decoder()
        idx     = torch.randperm(n)[:batch_size]
        x_batch = X[idx]

        optimizer.zero_grad()
        total, recon, sparse = model.loss(x_batch, lambda_val)
        total.backward()
        optimizer.step()

        losses.append(float(total))

        if step % log_every == 0 or step == n_steps - 1:
            cos = compute_cosines(model)
            cosines_history.append((step, cos.copy()))
            n_neg = int((cos < 0).sum())
            print(f"    step {step:5d} | total={float(total):.4f}"
                  f" recon={recon:.4f} sparse={sparse:.4f}"
                  f" | cos mean={cos.mean():.3f} min={cos.min():.3f}"
                  f" n_neg={n_neg}")

    return losses, cosines_history


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_cosines(cosines, model_name, pair_indices, d_sae):
    """Print and return a dict of cosine distribution stats."""
    mu     = float(cosines.mean())
    sigma  = float(cosines.std())
    n_neg  = int((cosines < 0).sum())
    n_3sig = int((cosines < mu - 3 * sigma).sum())
    n_6sig = int((cosines < mu - 6 * sigma).sum())
    min_c  = float(cosines.min())
    max_c  = float(cosines.max())

    child_cos = [float(cosines[ci]) for _, ci in pair_indices]
    print(f"\n  [{model_name}] cos(E,D):"
          f" mu={mu:.3f} sigma={sigma:.3f} min={min_c:.3f}"
          f" n_neg={n_neg} n_3sig={n_3sig} n_6sig={n_6sig}")
    print(f"    -> designed child features: {[f'{c:.3f}' for c in child_cos]}")

    return dict(
        model=model_name,
        mu=mu, sigma=sigma, n_neg=n_neg, n_3sig=n_3sig, n_6sig=n_6sig,
        min_cos=min_c, max_cos=max_c,
        child_cosines=child_cos,
        mean_child_cos=float(np.mean(child_cos)),
        any_child_neg=any(c < 0 for c in child_cos),
    )


# =============================================================================
# PHASE 2b
# =============================================================================

def run_phase2b(seeds=(42, 1337, 7)):
    print("\n" + "="*60)
    print("PHASE 2b: 10D Numerical Experiment  (ReLU+L1)")
    print("="*60)

    D_MODEL, N_CONCEPTS, D_SAE = 10, 15, 30
    N_SAMPLES, N_PAIRS, P_DEP  = 100_000, 3, 0.3
    LAMBDA, N_STEPS             = 1e-3, 15_000

    all_results = []
    cosines_per_seed = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print("  Generating data...")
        X, directions, pair_indices = generate_synthetic_data(
            d_model=D_MODEL, n_concepts=N_CONCEPTS, n_samples=N_SAMPLES,
            n_asymmetric_pairs=N_PAIRS, p_dep=P_DEP, seed=seed)

        print(f"  Training ReLU+L1 (lambda={LAMBDA})...")
        torch.manual_seed(seed)
        model = ReLUL1SAE(D_MODEL, D_SAE)
        losses, cosines_hist = train_sae(model, X, n_steps=N_STEPS,
                                         lambda_val=LAMBDA, log_every=3000)

        cos = compute_cosines(model)
        cosines_per_seed[seed] = cos

        stats = analyze_cosines(cos, f'ReLU+L1 seed={seed}', pair_indices, D_SAE)
        stats.update(seed=seed, lambda_val=LAMBDA, architecture='ReLU+L1')
        all_results.append(stats)

        if seed == seeds[0]:
            _plot_training_evolution(cosines_hist, pair_indices,
                                     'Phase 2b: ReLU+L1 Training Dynamics',
                                     'phase2b_cosine_evolution')

    # Distribution figure (3 seeds side by side)
    _, _, pi = generate_synthetic_data(n_asymmetric_pairs=N_PAIRS, seed=seeds[0])
    child_idxs = [ci for _, ci in pi]

    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4))
    for ax, seed in zip(axes, seeds):
        cos = cosines_per_seed[seed]
        ax.hist(cos, bins=20, color=COLOR_MAP['ReLU+L1'], alpha=0.75, edgecolor='k')
        ax.axvline(0, color='red', lw=2, ls='--', label='cos=0')
        mu, sig = cos.mean(), cos.std()
        ax.axvline(mu - 3*sig, color='orange', lw=1.5, ls=':', label='mu-3σ')
        _, _, pi_s = generate_synthetic_data(n_asymmetric_pairs=N_PAIRS, seed=seed)
        for _, ci in pi_s:
            ax.axvline(cos[ci], color='purple', lw=2, alpha=0.85)
        n_neg = int((cos < 0).sum())
        ax.set_title(f'ReLU+L1  seed={seed}\nn_neg={n_neg}')
        ax.set_xlabel('cos(E, D)')
        ax.set_ylabel('Count')
        if seed == seeds[0]:
            ax.legend(fontsize=8)

    fig.suptitle('Phase 2b: cos(E,D) Distribution -- ReLU+L1 Toy Model\n'
                 '(Purple lines = designed child features)', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase2b_relu_l1_distributions')

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS, 'experiment_2b_results.csv'), index=False)
    print(f"\nSaved results/phase2/experiment_2b_results.csv")

    return all_results, cosines_per_seed, pair_indices


def _plot_training_evolution(cosines_hist, pair_indices, title, fname):
    steps  = [s for s, _ in cosines_hist]
    means  = [c.mean() for _, c in cosines_hist]
    mins   = [c.min()  for _, c in cosines_hist]
    child_idxs = [ci for _, ci in pair_indices]
    child_traces = {ci: [c[ci] for _, c in cosines_hist] for ci in child_idxs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, means, 'b-o', ms=4, label='mean cos(E,D)')
    ax1.plot(steps, mins,  'r-s', ms=4, label='min cos(E,D)')
    ax1.axhline(0, color='k', lw=1, ls='--')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Cosine')
    ax1.set_title('Population statistics')
    ax1.legend()
    ax1.grid(alpha=0.3)

    colors = ['purple', 'darkviolet', 'indigo']
    for i, (ci, tr) in enumerate(child_traces.items()):
        ax2.plot(steps, tr, color=colors[i], marker='o', ms=4,
                 label=f'Child {ci}')
    ax2.axhline(0, color='k', lw=1.5, ls='--')
    ax2.set_xlabel('Training step')
    ax2.set_ylabel('cos(E, D)')
    ax2.set_title('Designed child features')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, fname)


# =============================================================================
# PHASE 2c
# =============================================================================

def run_phase2c(seeds=(42, 1337, 7)):
    print("\n" + "="*60)
    print("PHASE 2c: TopK SAE Comparison")
    print("="*60)

    D_MODEL, N_CONCEPTS, D_SAE = 10, 15, 30
    N_SAMPLES, N_PAIRS, P_DEP  = 100_000, 3, 0.3
    K, N_STEPS                  = 5, 15_000
    LAMBDA                      = 1e-3

    all_results = []
    cos_relu = {}
    cos_topk = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        X, _, pair_indices = generate_synthetic_data(
            n_asymmetric_pairs=N_PAIRS, p_dep=P_DEP, seed=seed)

        # ReLU+L1
        print("  Training ReLU+L1...")
        torch.manual_seed(seed * 10 + 1)
        m_relu = ReLUL1SAE(D_MODEL, D_SAE)
        train_sae(m_relu, X, n_steps=N_STEPS, lambda_val=LAMBDA, log_every=5000)
        c_relu = compute_cosines(m_relu)
        cos_relu[seed] = c_relu
        stats = analyze_cosines(c_relu, f'ReLU+L1 seed={seed}', pair_indices, D_SAE)
        stats.update(seed=seed, architecture='ReLU+L1')
        all_results.append(stats)

        # TopK
        print(f"  Training TopK (k={K})...")
        torch.manual_seed(seed * 10 + 2)
        m_topk = TopKSAE(D_MODEL, D_SAE, k=K)
        train_sae(m_topk, X, n_steps=N_STEPS, lambda_val=0.0, log_every=5000)
        c_topk = compute_cosines(m_topk)
        cos_topk[seed] = c_topk
        stats_t = analyze_cosines(c_topk, f'TopK k={K} seed={seed}', pair_indices, D_SAE)
        stats_t.update(seed=seed, architecture='TopK')
        all_results.append(stats_t)

    # Comparison figure
    fig, axes = plt.subplots(2, len(seeds), figsize=(5*len(seeds), 8))
    for col, seed in enumerate(seeds):
        _, _, pi = generate_synthetic_data(n_asymmetric_pairs=N_PAIRS,
                                            p_dep=P_DEP, seed=seed)
        for row, (cos_dict, arch) in enumerate([
            (cos_relu, 'ReLU+L1'),
            (cos_topk, 'TopK'),
        ]):
            ax  = axes[row, col]
            cos = cos_dict[seed]
            col_c = COLOR_MAP[arch]
            ax.hist(cos, bins=20, color=col_c, alpha=0.75, edgecolor='k')
            ax.axvline(0, color='red', lw=2, ls='--')
            mu, sig = cos.mean(), cos.std()
            ax.axvline(mu - 3*sig, color='orange', lw=1.5, ls=':')
            for _, ci in pi:
                ax.axvline(cos[ci], color='purple', lw=2, alpha=0.8)
            n_neg = int((cos < 0).sum())
            ax.set_title(f'{arch} | seed={seed}\nn_neg={n_neg}  min={cos.min():.3f}')
            ax.set_xlabel('cos(E, D)')
            ax.set_ylabel('Count')

    fig.suptitle('Phase 2c: ReLU+L1 vs TopK cos(E,D)\n'
                 '(Purple = designed child features)', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase2c_topk_comparison')

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS, 'experiment_2c_topk_results.csv'), index=False)
    print(f"\nSaved results/phase2/experiment_2c_topk_results.csv")

    return all_results


# =============================================================================
# PHASE 2d: PHASE BOUNDARY SWEEP  (fast version using Adam)
# =============================================================================

def run_phase2d():
    print("\n" + "="*60)
    print("PHASE 2d: Phase Boundary Sweep  (lambda x theta x p_dep)")
    print("="*60)

    D_MODEL, N_CONCEPTS, D_SAE = 10, 15, 30
    N_SAMPLES = 30_000   # smaller for speed in sweep
    N_PAIRS   = 3
    N_STEPS   = 6_000    # fewer steps for speed
    SEEDS     = [42, 1337, 7]

    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    thetas  = [5, 15, 30, 45, 60, 75]      # angle parent-child in degrees
    p_deps  = [0.1, 0.3, 0.5, 0.7, 0.9]

    total_runs = len(lambdas) * len(thetas) * len(p_deps)
    print(f"Grid: {len(lambdas)} x {len(thetas)} x {len(p_deps)} = {total_runs} combos"
          f" x {len(SEEDS)} seeds = {total_runs * len(SEEDS)} runs")
    print("ETA: ~30-60 minutes")

    records = []
    done    = 0
    t0      = time.time()

    for lam, theta_deg, p_dep in product(lambdas, thetas, p_deps):
        seed_n_neg    = []
        seed_min_cos  = []
        seed_child_cos = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            X, _, pair_indices = generate_synthetic_data(
                d_model=D_MODEL, n_concepts=N_CONCEPTS, n_samples=N_SAMPLES,
                n_asymmetric_pairs=N_PAIRS, p_dep=p_dep, seed=seed,
                theta_deg=theta_deg)

            model = ReLUL1SAE(D_MODEL, D_SAE)
            torch.manual_seed(seed + 1)
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
            n     = X.shape[0]

            for step in range(N_STEPS):
                model.normalize_decoder()
                idx     = torch.randperm(n)[:512]
                x_batch = X[idx]
                opt.zero_grad()
                total, _, _ = model.loss(x_batch, lam)
                total.backward()
                opt.step()

            cos = compute_cosines(model)
            seed_n_neg.append(int((cos < 0).sum()))
            seed_min_cos.append(float(cos.min()))
            child_cos = [float(cos[ci]) for _, ci in pair_indices]
            seed_child_cos.append(float(np.mean(child_cos)))

        records.append(dict(
            lambda_val=lam,
            theta_deg=theta_deg,
            p_dep=p_dep,
            mean_n_neg=float(np.mean(seed_n_neg)),
            std_n_neg=float(np.std(seed_n_neg)),
            mean_min_cos=float(np.mean(seed_min_cos)),
            std_min_cos=float(np.std(seed_min_cos)),
            mean_child_cos=float(np.mean(seed_child_cos)),
            std_child_cos=float(np.std(seed_child_cos)),
            any_neg=float(np.mean(seed_n_neg)) > 0.5,
        ))

        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total_runs - done) if done > 0 else 0
        print(f"  [{done:3d}/{total_runs}] lam={lam:.0e} theta={theta_deg:2d}deg"
              f" p={p_dep:.1f} | n_neg={np.mean(seed_n_neg):.1f}"
              f" child_cos={np.mean(seed_child_cos):.3f}"
              f" | ETA={eta/60:.0f}min")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, 'phase_diagram.csv'), index=False)
    print(f"\nSaved results/phase2/phase_diagram.csv")

    _plot_phase_diagrams(df, lambdas, thetas, p_deps)

    return records


def _plot_phase_diagrams(df, lambdas, thetas, p_deps):
    thetas_sorted  = sorted(thetas)
    lambdas_sorted = sorted(lambdas)
    p_deps_sorted  = sorted(p_deps)

    # ---- Figure 1: (lambda, theta) at p_dep=0.3 ----
    p_fixed = 0.3
    df_sub = df[np.abs(df['p_dep'] - p_fixed) < 0.01]

    Z_neg   = np.zeros((len(thetas_sorted), len(lambdas_sorted)))
    Z_child = np.zeros((len(thetas_sorted), len(lambdas_sorted)))

    for row in df_sub.itertuples():
        ti = thetas_sorted.index(row.theta_deg)
        li = lambdas_sorted.index(row.lambda_val)
        Z_neg[ti, li]   = row.mean_n_neg
        Z_child[ti, li] = row.mean_child_cos

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(Z_neg, aspect='auto', origin='lower',
                     cmap='YlOrRd', interpolation='nearest')
    ax1.set_xticks(range(len(lambdas_sorted)))
    ax1.set_xticklabels([f'{l:.0e}' for l in lambdas_sorted], rotation=45)
    ax1.set_yticks(range(len(thetas_sorted)))
    ax1.set_yticklabels([f'{t}deg' for t in thetas_sorted])
    plt.colorbar(im1, ax=ax1, label='Mean n_neg features')
    ax1.set_xlabel('L1 lambda')
    ax1.set_ylabel('theta (angle parent-child, degrees)')
    ax1.set_title(f'Mean n_neg features  (p_dep={p_fixed})')

    im2 = ax2.imshow(Z_child, aspect='auto', origin='lower',
                     cmap='RdYlGn_r', interpolation='nearest',
                     vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(len(lambdas_sorted)))
    ax2.set_xticklabels([f'{l:.0e}' for l in lambdas_sorted], rotation=45)
    ax2.set_yticks(range(len(thetas_sorted)))
    ax2.set_yticklabels([f'{t}deg' for t in thetas_sorted])
    plt.colorbar(im2, ax=ax2, label='Mean cos(E,D) for child features')
    ax2.set_xlabel('L1 lambda')
    ax2.set_ylabel('theta (angle parent-child, degrees)')
    ax2.set_title(f'Mean child-feature cos(E,D)  (p_dep={p_fixed})')

    fig.suptitle('Phase Diagram: Anti-Alignment Region in (lambda, theta) Space\n'
                 'Anti-alignment strongest at large lambda & small theta',
                 fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase2d_phase_diagram_lambda_theta')

    # ---- Figure 2: (lambda, p_dep) at theta=30 ----
    theta_fixed = 30
    df_t30 = df[df['theta_deg'] == theta_fixed]

    Z_neg2   = np.zeros((len(p_deps_sorted), len(lambdas_sorted)))
    Z_child2 = np.zeros((len(p_deps_sorted), len(lambdas_sorted)))

    for row in df_t30.itertuples():
        pi = p_deps_sorted.index(row.p_dep)
        li = lambdas_sorted.index(row.lambda_val)
        Z_neg2[pi, li]   = row.mean_n_neg
        Z_child2[pi, li] = row.mean_child_cos

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    im3 = ax3.imshow(Z_neg2, aspect='auto', origin='lower',
                     cmap='YlOrRd', interpolation='nearest')
    ax3.set_xticks(range(len(lambdas_sorted)))
    ax3.set_xticklabels([f'{l:.0e}' for l in lambdas_sorted], rotation=45)
    ax3.set_yticks(range(len(p_deps_sorted)))
    ax3.set_yticklabels([f'{p}' for p in p_deps_sorted])
    plt.colorbar(im3, ax=ax3, label='Mean n_neg features')
    ax3.set_xlabel('L1 lambda')
    ax3.set_ylabel('p_dep  P(child fires | parent fires)')
    ax3.set_title(f'Mean n_neg features  (theta={theta_fixed}deg)')

    im4 = ax4.imshow(Z_child2, aspect='auto', origin='lower',
                     cmap='RdYlGn_r', interpolation='nearest',
                     vmin=-0.5, vmax=0.5)
    ax4.set_xticks(range(len(lambdas_sorted)))
    ax4.set_xticklabels([f'{l:.0e}' for l in lambdas_sorted], rotation=45)
    ax4.set_yticks(range(len(p_deps_sorted)))
    ax4.set_yticklabels([f'{p}' for p in p_deps_sorted])
    plt.colorbar(im4, ax=ax4, label='Mean cos(E,D) for child features')
    ax4.set_xlabel('L1 lambda')
    ax4.set_ylabel('p_dep  P(child fires | parent fires)')
    ax4.set_title(f'Mean child-feature cos(E,D)  (theta={theta_fixed}deg)')

    fig2.suptitle('Phase Diagram: Anti-Alignment Region in (lambda, p_dep) Space\n'
                  'Anti-alignment strengthens at high p_dep (strong co-activation)',
                  fontweight='bold')
    plt.tight_layout()
    save_figure(fig2, 'phase2d_phase_diagram_lambda_pdep')


# =============================================================================
# PHASE 2e: JumpReLU SCALAR vs PER-FEATURE THRESHOLD
# =============================================================================

def run_phase2e(seeds=(42, 1337, 7)):
    print("\n" + "="*60)
    print("PHASE 2e: JumpReLU Scalar vs Per-Feature Threshold")
    print("  Motivated by Supplement 2:")
    print("    Scalar threshold (Llama LXR) -> 232 features with cos<0 (29 at <= mu-6sigma)")
    print("    Per-feature threshold (Gemma) -> ZERO anti-aligned features")
    print("="*60)

    D_MODEL, N_CONCEPTS, D_SAE = 10, 15, 30
    N_SAMPLES, N_PAIRS, P_DEP  = 100_000, 3, 0.3
    LAMBDA_L0, N_STEPS          = 1e-2, 15_000

    all_results = []
    cos_scalar  = {}
    cos_perfeat = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        X, _, pair_indices = generate_synthetic_data(
            n_asymmetric_pairs=N_PAIRS, p_dep=P_DEP, seed=seed)

        for per_feat, name, cos_store in [
            (False, 'JumpReLU (scalar)',      cos_scalar),
            (True,  'JumpReLU (per-feature)', cos_perfeat),
        ]:
            print(f"  Training {name}  (lambda_l0={LAMBDA_L0})...")
            torch.manual_seed(seed * 100 + (0 if not per_feat else 1))
            model = JumpReLUSAE(D_MODEL, D_SAE,
                                 per_feature_threshold=per_feat, beta=0.1)
            train_sae(model, X, n_steps=N_STEPS, lambda_val=LAMBDA_L0,
                      log_every=5000)

            cos = compute_cosines(model)
            cos_store[seed] = cos

            with torch.no_grad():
                th = model.theta.detach().numpy()
            print(f"    Threshold: shape={th.shape}, "
                  f"mean={th.mean():.4f}, std={th.std():.6f}")

            stats = analyze_cosines(cos, f'{name} seed={seed}', pair_indices, D_SAE)
            stats.update(
                architecture=name, seed=seed,
                lambda_l0=LAMBDA_L0,
                per_feature_threshold=per_feat,
                theta_mean=float(th.mean()),
                theta_std=float(th.std()),
            )
            all_results.append(stats)

    # Comparison figure
    fig, axes = plt.subplots(2, len(seeds), figsize=(5*len(seeds), 8))

    names = ['JumpReLU (scalar)', 'JumpReLU (per-feature)']
    stores = [cos_scalar, cos_perfeat]

    for col, seed in enumerate(seeds):
        _, _, pi = generate_synthetic_data(n_asymmetric_pairs=N_PAIRS,
                                            p_dep=P_DEP, seed=seed)
        for row, (name, store) in enumerate(zip(names, stores)):
            ax  = axes[row, col]
            cos = store[seed]
            ax.hist(cos, bins=20, color=COLOR_MAP[name], alpha=0.75, edgecolor='k')
            ax.axvline(0, color='red', lw=2, ls='--')
            mu, sig = cos.mean(), cos.std()
            ax.axvline(mu - 3*sig, color='orange', lw=1.5, ls=':')
            for _, ci in pi:
                ax.axvline(cos[ci], color='purple', lw=2, alpha=0.8)
            n_neg = int((cos < 0).sum())
            label_s = 'Scalar' if 'scalar' in name else 'Per-feature'
            ax.set_title(f'{label_s} JumpReLU | seed={seed}\nn_neg={n_neg}')
            ax.set_xlabel('cos(E, D)')
            ax.set_ylabel('Count')

    fig.suptitle(
        'Phase 2e: JumpReLU Scalar vs Per-Feature Threshold\n'
        '(Empirical: Llama=scalar=232 neg, Gemma=per-feature=1 neg)',
        fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase2e_jumprelu_comparison')

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS, 'jumprelu_comparison.csv'), index=False)
    print(f"\nSaved results/phase2/jumprelu_comparison.csv")

    return all_results


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def make_summary_figure(results_2b, results_2c, results_2e):
    """Single figure comparing all four architectures."""
    all_r = results_2b + [r for r in results_2c if r['architecture'] != 'ReLU+L1'] + results_2e
    df    = pd.DataFrame(all_r)

    archs = ['ReLU+L1', 'TopK', 'JumpReLU (scalar)', 'JumpReLU (per-feature)']
    means_neg   = []
    stds_neg    = []
    means_min   = []
    stds_min    = []
    means_child = []
    stds_child  = []

    for a in archs:
        sub = df[df['architecture'] == a]
        if len(sub) == 0:
            means_neg.append(0); stds_neg.append(0)
            means_min.append(1); stds_min.append(0)
            means_child.append(1); stds_child.append(0)
        else:
            means_neg.append(float(sub['n_neg'].mean()))
            stds_neg.append(float(sub['n_neg'].std()))
            means_min.append(float(sub['min_cos'].mean()))
            stds_min.append(float(sub['min_cos'].std()))
            means_child.append(float(sub['mean_child_cos'].mean()))
            stds_child.append(float(sub['mean_child_cos'].std()))

    colors = [COLOR_MAP.get(a, 'grey') for a in archs]
    arch_labels = ['ReLU+L1', 'TopK', 'JumpReLU\n(scalar)', 'JumpReLU\n(per-feat)']
    x = range(len(archs))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def bplot(ax, means, stds, ylabel, title, ref=None):
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
               edgecolor='k', capsize=5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(arch_labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ref is not None:
            ax.axhline(ref, color='red', lw=1.5, ls='--')

    bplot(axes[0], means_neg, stds_neg,
          'Mean n_neg features (out of 30)',
          'Anti-aligned Feature Count', ref=0)
    bplot(axes[1], means_min, stds_min,
          'Mean min cos(E,D)',
          'Most Anti-Aligned Feature', ref=0)
    bplot(axes[2], means_child, stds_child,
          'Mean cos(E,D) for child features',
          'Designed Child Feature Alignment', ref=0)

    fig.suptitle(
        'Phase 2 Summary: cos(E,D) Statistics Across Architectures\n'
        '(Toy Model: 10D, 30 features, 3 designed asymmetric pairs, 3 seeds)',
        fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'phase2_summary_comparison')
    print("Saved figures/phase2/phase2_summary_comparison.png")

    return dict(zip(archs, zip(means_neg, means_min, means_child)))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Toy Model Validation')
    parser.add_argument('--part', default='all',
                        choices=['2b', '2c', '2d', '2e', 'all'])
    args = parser.parse_args()

    print("\nPhase 2: Toy Model Validation")
    print(f"Part: {args.part}")
    print(f"Results -> {RESULTS}")
    print(f"Figures -> {FIGURES}\n")

    t0 = time.time()
    results_2b = results_2c = results_2e = []

    if args.part in ('2b', 'all'):
        results_2b, _, pair_indices = run_phase2b(seeds=[42, 1337, 7])

    if args.part in ('2c', 'all'):
        results_2c = run_phase2c(seeds=[42, 1337, 7])

    if args.part in ('2d', 'all'):
        run_phase2d()

    if args.part in ('2e', 'all'):
        results_2e = run_phase2e(seeds=[42, 1337, 7])

    if args.part == 'all' and results_2b and results_2c and results_2e:
        make_summary_figure(results_2b, results_2c, results_2e)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")
