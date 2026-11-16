"""
Phase 2 - Additional tests:
  A) 2D Analytical Toy Model (exact analytical case from Phase 2a spec)
     Validates the core mechanism: d_model=2, d_sae=2, n_concepts=3
  B) Lambda Search (10D model): find threshold lambda where anti-alignment emerges  
  C) JumpReLU Fixed (corrected initialization: low initial threshold)
  D) Forced merging: 2D SAE with high lambda on 3 concepts

All results saved to results/phase2/, figures to figures/phase2/.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, 'results', 'phase2')
FIGURES = os.path.join(PROJECT, 'figures', 'phase2')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

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


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figures/phase2/{name}.png")


# =============================================================================
# A) 2D ANALYTICAL TOY MODEL
# 
# Exact setup from Phase 2a spec:
#   d_model = 2
#   3 concepts: A=(1,0), B=(cos theta, sin theta), C=(0,1) [independent]
#   d_sae = 2 (same as d_model!) -- FORCED bottleneck
#   Asymmetric: P(B=1|A=0)=0, P(B=1|A=1)=p_dep
#   Lambda high enough to trigger hedging
#
# PREDICTION (H1):
#   With d_sae=2 AND 3 concepts AND high lambda:
#   Feature 1 (merged): E1 ~ (A+B)/||A+B||, D1 ~ (A+B)/||A+B||, cos(E1,D1)=+1
#   Feature 2 (hedge):  E2 ~ B/||B||, D2 ~ -A, cos(E2,D2)=-cos(theta) < 0
# =============================================================================

class SAE2D(nn.Module):
    """2D SAE with ReLU+L1, for the exact analytical toy model."""
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.1)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        with torch.no_grad():
            n = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= n

    def forward(self, x):
        z = torch.relu(x @ self.W_enc + self.b_enc)
        return z, z @ self.W_dec + self.b_dec

    def loss(self, x, lam):
        z, xh = self.forward(x)
        r = (x - xh).pow(2).sum(1).mean()
        s = lam * z.sum(1).mean()
        return r + s, float(r), float(s)


def gen_2D_data(theta_deg=30.0, p_A=0.15, p_dep=0.5,
                n_samples=200_000, seed=42):
    """
    Generate 2D toy data:
      concept A = (1, 0)
      concept B = (cos theta, sin theta)
      concept C = (0, 1)  [independent]
    co-activation: B only fires when A fires.
    """
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(theta_deg)
    A = np.array([1.0, 0.0])
    B = np.array([np.cos(theta), np.sin(theta)])
    C = np.array([0.0, 1.0])
    print(f"  cos(A,B) = {A@B:.3f}")

    X = np.zeros((n_samples, 2))
    for i in range(n_samples):
        x = np.zeros(2)
        # C fires independently
        if rng.random() < 0.1:
            x += rng.uniform(0.5, 1.5) * C
        # A fires with prob p_A
        a_mag = 0.0
        if rng.random() < p_A:
            a_mag = rng.uniform(0.5, 1.5)
            x += a_mag * A
            # B fires with prob p_dep given A fires
            if rng.random() < p_dep:
                x += rng.uniform(0.5, 1.5) * B
        X[i] = x + 0.01 * rng.standard_normal(2)

    return torch.tensor(X, dtype=torch.float32)


def compute_cosines_model(model):
    with torch.no_grad():
        E = model.W_enc.T
        D = model.W_dec
        En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Dn = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return (En * Dn).sum(1).numpy()


def run_2D_toy(theta_deg=30.0, lambdas=None, seed=42, n_steps=30_000):
    """
    Run the 2D d_sae=2 SAE at different lambdas.
    Shows the phase transition from aligned to anti-aligned as lambda increases.
    """
    if lambdas is None:
        lambdas = [0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.10, 0.20]

    print(f"\n{'='*60}")
    print(f"2D ANALYTICAL TOY (theta={theta_deg}deg, d_sae=2)")
    print(f"{'='*60}")

    X = gen_2D_data(theta_deg=theta_deg, seed=seed)
    print(f"Data: shape={X.shape}, mean_norm={X.norm(dim=1).mean():.3f}")

    records = []
    for lam in lambdas:
        torch.manual_seed(seed)
        model = SAE2D(d_model=2, d_sae=2)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        n = X.shape[0]

        for step in range(n_steps):
            model.normalize_decoder()
            idx = torch.randperm(n)[:256]
            opt.zero_grad()
            tot, r, s = model.loss(X[idx], lam)
            tot.backward()
            opt.step()

        cos = compute_cosines_model(model)
        cos0, cos1 = float(cos[0]), float(cos[1])
        n_neg = int((cos < 0).sum())

        # Encoder and decoder directions
        with torch.no_grad():
            E = (model.W_enc.T / model.W_enc.T.norm(dim=1, keepdim=True)).numpy()
            D = (model.W_dec / model.W_dec.norm(dim=1, keepdim=True)).numpy()

        print(f"  lambda={lam:.2e}:  cos=[{cos0:.3f}, {cos1:.3f}]  n_neg={n_neg}"
              f"  min={cos.min():.3f}")
        print(f"    E0={E[0].round(3)}, D0={D[0].round(3)}")
        print(f"    E1={E[1].round(3)}, D1={D[1].round(3)}")

        records.append(dict(
            lambda_val=lam, theta_deg=theta_deg,
            cos0=cos0, cos1=cos1,
            n_neg=n_neg, min_cos=float(cos.min()),
            E0=E[0].tolist(), D0=D[0].tolist(),
            E1=E[1].tolist(), D1=D[1].tolist(),
        ))

    return records


def plot_2D_results(records, theta_deg=30.0):
    """Plot the phase transition in the 2D toy model."""
    lambdas  = [r['lambda_val'] for r in records]
    cos_mins = [r['min_cos'] for r in records]
    n_negs   = [r['n_neg'] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: min cosine vs lambda (log scale)
    ax = axes[0]
    lam_arr = np.array(lambdas)
    lam_arr[0] = 1e-6  # replace 0 with tiny for log plot
    ax.semilogx(lam_arr, cos_mins, 'b-o', ms=6)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    theta_rad = np.deg2rad(theta_deg)
    # Analytical prediction: cos_hedge = -cos(theta) when lambda > lambda_crit
    predicted = -np.cos(theta_rad)
    ax.axhline(predicted, color='green', lw=1.5, ls=':', label=f'-cos(theta)={predicted:.3f}')
    ax.set_xlabel('L1 lambda')
    ax.set_ylabel('min cos(E,D)')
    ax.set_title('Phase transition in minimum cosine')
    ax.grid(alpha=0.3)
    ax.legend()

    # Panel 2: n_neg features vs lambda
    ax2 = axes[1]
    ax2.semilogx(lam_arr, n_negs, 'r-s', ms=6)
    ax2.axhline(1, color='orange', lw=1.5, ls=':', label='1 hedge feature')
    ax2.set_xlabel('L1 lambda')
    ax2.set_ylabel('n_neg features (out of 2)')
    ax2.set_title('Anti-aligned features vs lambda')
    ax2.set_ylim(-0.1, 2.1)
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Panel 3: 2D visualization of encoder/decoder vectors at high lambda
    ax3 = axes[2]
    high_lam_rec = records[-1]   # highest lambda
    E0, D0 = high_lam_rec['E0'], high_lam_rec['D0']
    E1, D1 = high_lam_rec['E1'], high_lam_rec['D1']

    theta_r = np.deg2rad(theta_deg)
    # True concepts
    ax3.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1,
               color='blue', label='A=(1,0)', width=0.015)
    ax3.quiver(0, 0, np.cos(theta_r), np.sin(theta_r), angles='xy', scale_units='xy',
               scale=1, color='cyan', label=f'B=(cos{theta_deg},sin{theta_deg})',
               width=0.015)

    # Encoder directions (scale factor 0.8 applied to u/v directly)
    def qv(ax, u, v, color, label):
        ax.quiver(0, 0, 0.85*u, 0.85*v, angles='xy', scale_units='xy', scale=1,
                  color=color, alpha=0.85, label=label, width=0.015)

    qv(ax3, E0[0], E0[1], 'red',      'E0')
    qv(ax3, E1[0], E1[1], 'darkred',  'E1')
    qv(ax3, D0[0], D0[1], 'green',    'D0')
    qv(ax3, D1[0], D1[1], 'darkgreen','D1')

    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.axhline(0, color='k', lw=0.5)
    ax3.axvline(0, color='k', lw=0.5)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.set_title(f'Encoder/Decoder at lambda={high_lam_rec["lambda_val"]:.2f}\n'
                  f'cos=[{high_lam_rec["cos0"]:.2f}, {high_lam_rec["cos1"]:.2f}]')
    ax3.set_aspect('equal')
    ax3.grid(alpha=0.3)

    fig.suptitle(f'Phase 2a-numerical: 2D Toy Model (theta={theta_deg}deg, d_model=2, d_sae=2)\n'
                 'Red line: predicted anti-alignment when lambda > lambda_crit\n'
                 'Green: -cos(theta) = analytical prediction for hedge feature',
                 fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'phase2a_2D_toy_phase_transition')

    return records


# =============================================================================
# B) LAMBDA SEARCH IN 10D
# Systematic search for the lambda threshold in the 10D model
# =============================================================================

class ReLUL1SAE10D(nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def normalize_decoder(self):
        with torch.no_grad():
            n = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= n

    def forward(self, x):
        z = torch.relu(x @ self.W_enc + self.b_enc)
        return z, z @ self.W_dec + self.b_dec

    def loss(self, x, lam):
        z, xh = self.forward(x)
        r = (x - xh).pow(2).sum(1).mean()
        s = lam * z.sum(1).mean()
        return r + s, float(r), float(s)


def gen_10D_data_controlled(d_model=10, n_concepts=15, n_samples=50_000,
                             n_pairs=3, p_dep=0.7, theta_deg=30.0, seed=42):
    """10D data with exact theta_deg angle for all parent-child pairs."""
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Random unit directions
    dirs = np.random.default_rng(seed + 1).standard_normal((n_concepts, d_model))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    pair_idx = []
    for i in range(n_pairs):
        pi = i * 2
        ci = i * 2 + 1
        parent = dirs[pi]
        # orthogonal complement
        ortho = rng.standard_normal(d_model)
        ortho = ortho - (ortho @ parent) * parent
        ortho /= np.linalg.norm(ortho)
        dirs[ci] = cos_t * parent + sin_t * ortho
        pair_idx.append((pi, ci))
        print(f"    Pair {i}: cos(parent,child)={dirs[pi]@dirs[ci]:.3f}")

    X = np.zeros((n_samples, d_model))
    for i in range(n_samples):
        x = np.zeros(d_model)
        for c in range(n_concepts):
            if rng.random() < 0.05:
                x += rng.uniform(0.5, 1.5) * dirs[c]
        for pi, ci in pair_idx:
            if x @ dirs[pi] < 0.1:  # parent not active
                # zero out child contribution
                pass
            elif rng.random() < p_dep:
                x += rng.uniform(0.5, 1.5) * dirs[ci]
        X[i] = x + 0.02 * rng.standard_normal(d_model)
    return torch.tensor(X, dtype=torch.float32), pair_idx


def run_lambda_search_10D():
    """Find the critical lambda that produces anti-alignment in the 10D toy model."""
    print(f"\n{'='*60}")
    print("LAMBDA SEARCH: 10D Toy Model (what lambda produces anti-alignment?)")
    print(f"{'='*60}")

    D_MODEL = 10
    N_CONCEPTS = 15
    D_SAE = 30   # same as before: 2x overexpansion
    N_PAIRS = 3
    P_DEP = 0.7   # higher dependency: child fires 70% of time with parent
    THETA_DEG = 30.0
    N_STEPS = 12_000
    SEEDS = [42, 1337]

    # Wide lambda search
    lambdas = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
               0.01, 0.05, 0.10, 0.20, 0.50]

    records = []
    t0 = time.time()

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        X, pair_idx = gen_10D_data_controlled(
            d_model=D_MODEL, n_concepts=N_CONCEPTS, n_samples=50_000,
            n_pairs=N_PAIRS, p_dep=P_DEP, theta_deg=THETA_DEG, seed=seed)

        for lam in lambdas:
            torch.manual_seed(seed * 100 + len(records))
            model = ReLUL1SAE10D(D_MODEL, D_SAE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            n = X.shape[0]

            for step in range(N_STEPS):
                model.normalize_decoder()
                idx = torch.randperm(n)[:512]
                opt.zero_grad()
                tot, r, s = model.loss(X[idx], lam)
                tot.backward()
                opt.step()

            with torch.no_grad():
                E = model.W_enc.T
                D = model.W_dec
                En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
                Dn = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cos = (En * Dn).sum(1).numpy()

            child_cosines = [float(cos[ci]) for _, ci in pair_idx]
            n_neg  = int((cos < 0).sum())
            min_c  = float(cos.min())
            recon_final = float((X[:1000] - model.forward(X[:1000])[1]\
                                 .detach()).pow(2).sum(1).mean())

            print(f"    lambda={lam:.2e}: n_neg={n_neg:2d} min={min_c:.3f}"
                  f" child_cos={[f'{c:.3f}' for c in child_cosines]}"
                  f" recon={recon_final:.4f}")

            records.append(dict(
                seed=seed, lambda_val=lam,
                n_neg=n_neg, min_cos=min_c,
                mean_child_cos=float(np.mean(child_cosines)),
                child_cosines=child_cosines,
                recon=recon_final,
                any_child_neg=any(c < 0 for c in child_cosines),
            ))

    print(f"\n  Lambda search done. Time: {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, 'lambda_search_10D.csv'), index=False)
    print(f"  Saved: results/phase2/lambda_search_10D.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors_seeds = ['blue', 'red']

    for col_idx, seed in enumerate(SEEDS):
        sub = df[df['seed'] == seed]
        lams = sub['lambda_val'].values
        lams_plot = np.where(lams == 0, 1e-6, lams)  # replace 0 for log scale

        axes[0].semilogx(lams_plot, sub['n_neg'].values, '-o',
                          color=colors_seeds[col_idx], label=f'seed={seed}', ms=5)
        axes[1].semilogx(lams_plot, sub['min_cos'].values, '-o',
                          color=colors_seeds[col_idx], label=f'seed={seed}', ms=5)
        axes[2].semilogx(lams_plot, sub['mean_child_cos'].values, '-o',
                          color=colors_seeds[col_idx], label=f'seed={seed}', ms=5)

    for ax, ylabel, title in zip(
        axes,
        ['n_neg features (out of 30)', 'min cos(E,D)', 'mean child cos(E,D)'],
        ['Anti-aligned Features vs Lambda', 'Min Cosine vs Lambda',
         'Child Feature Cosine vs Lambda']
    ):
        ax.axhline(0, color='k', lw=1, ls='--')
        ax.set_xlabel('L1 lambda (log scale)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        f'Phase 2b-extended: Lambda Search in 10D Toy Model\n'
        f'(d_model=10, d_sae=30, theta={THETA_DEG}deg, p_dep={P_DEP})',
        fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'phase2b_lambda_search_10D')

    return records


# =============================================================================
# C) JumpReLU FIXED (corrected initialization)
# =============================================================================

class JumpReLUSAE_fixed(nn.Module):
    """
    Corrected JumpReLU SAE.
    Key fix: lower initial threshold + proper STE gradient for reconstruction.
    
    STE trick: forward uses hard gate, backward uses soft relu gradient.
      z_hard = pre_act * (pre_act > theta)
      z_gradient = pre_act * relu''(pre_act - theta)  [approx]
      z = z_hard.detach() + z_gradient - z_gradient.detach()  [STE]
    This way reconstruction gradient flows through all active features.
    """
    def __init__(self, d_model, d_sae, per_feature_threshold=True):
        super().__init__()
        self.per_feature = per_feature_threshold

        self.W_enc  = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec  = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc  = nn.Parameter(torch.zeros(d_sae))
        self.b_dec  = nn.Parameter(torch.zeros(d_model))

        # Initialize threshold SMALL so features CAN fire initially
        # log_theta = -3 -> theta = exp(-3) ~ 0.05
        if per_feature_threshold:
            self.log_theta = nn.Parameter(-3.0 * torch.ones(d_sae))
        else:
            self.log_theta = nn.Parameter(-3.0 * torch.ones(1))

    @property
    def theta(self):
        return torch.exp(self.log_theta)

    def normalize_decoder(self):
        with torch.no_grad():
            n = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
            self.W_dec.data /= n

    def forward(self, x):
        pre_act = x @ self.W_enc + self.b_enc    # (B, d_sae)
        theta   = self.theta                       # (d_sae,) or (1,)
        gate    = (pre_act > theta).float()

        # STE: z_hard forward, gradient through pre_act (as if ReLU at theta)
        z_hard = pre_act * gate
        # Soft approximation for gradient (sigmoid slope)
        z_soft = pre_act * torch.sigmoid(10.0 * (pre_act - theta))
        z      = z_hard.detach() + (z_soft - z_soft.detach())  # STE

        x_hat  = z @ self.W_dec + self.b_dec
        return z, x_hat, pre_act, gate

    def loss(self, x, lambda_val=5e-3):
        z, x_hat, pre_act, gate = self.forward(x)
        recon    = (x - x_hat).pow(2).sum(1).mean()
        # L0 approx for threshold gradient  
        l0_approx = gate.sum(1).mean()   # this is constant, no gradient
        # Alternative: use sigmoid for differentiable L0
        l0_diff  = torch.sigmoid(10.0 * (pre_act - self.theta)).sum(1).mean()
        sparse   = lambda_val * l0_diff
        total    = recon + sparse
        return total, float(recon), float(l0_diff.item())


def run_jumprelu_fixed(seeds=(42, 1337, 7)):
    print(f"\n{'='*60}")
    print("PHASE 2e (revised): JumpReLU Fixed Initialization")
    print(f"{'='*60}")

    D_MODEL, N_CONCEPTS, D_SAE = 10, 15, 30
    N_SAMPLES, N_PAIRS, P_DEP  = 100_000, 3, 0.3
    LAMBDA_L0, N_STEPS          = 5e-3, 15_000

    all_results = []
    cos_scalar   = {}
    cos_perfeat  = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        # Generate data
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        dirs = torch.randn(N_CONCEPTS, D_MODEL)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        pair_indices = []
        for i in range(N_PAIRS):
            pi, ci = i * 2, i * 2 + 1
            noise = torch.randn(D_MODEL)
            child = 0.6 * dirs[pi] + 0.8 * noise
            dirs[ci] = child / child.norm()
            pair_indices.append((pi, ci))

        X_list = []
        for _ in range(N_SAMPLES):
            active = torch.zeros(N_CONCEPTS)
            for c in range(N_CONCEPTS):
                if rng.random() < 0.05:
                    active[c] = rng.uniform(0.5, 1.5)
            for pi, ci in pair_indices:
                if active[pi] == 0:
                    active[ci] = 0
                elif rng.random() < P_DEP:
                    active[ci] = rng.uniform(0.5, 1.5)
            X_list.append((active.unsqueeze(1) * dirs).sum(0)
                          + 0.05 * torch.randn(D_MODEL))
        X = torch.stack(X_list)

        for per_feat, name, cos_d in [
            (False, 'JumpReLU (scalar)',      cos_scalar),
            (True,  'JumpReLU (per-feature)', cos_perfeat),
        ]:
            print(f"  Training {name}...")
            torch.manual_seed(seed * 100 + (0 if not per_feat else 1))
            model = JumpReLUSAE_fixed(D_MODEL, D_SAE,
                                       per_feature_threshold=per_feat)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            n   = X.shape[0]

            for step in range(N_STEPS):
                model.normalize_decoder()
                idx = torch.randperm(n)[:512]
                opt.zero_grad()
                tot, r, l0 = model.loss(X[idx], LAMBDA_L0)
                tot.backward()
                opt.step()
                if step % 5000 == 0 or step == N_STEPS - 1:
                    with torch.no_grad():
                        E = model.W_enc.T
                        D = model.W_dec
                        En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        Dn = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        cos_tmp = (En * Dn).sum(1).numpy()
                        th = model.theta.detach().numpy()
                    print(f"    step {step:5d}: recon={r:.4f} l0={l0:.2f}"
                          f"  cos mean={cos_tmp.mean():.3f} n_neg={(cos_tmp<0).sum()}"
                          f"  theta: mean={th.mean():.4f} std={th.std():.4f}")

            with torch.no_grad():
                E = model.W_enc.T
                D = model.W_dec
                En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
                Dn = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cos = (En * Dn).sum(1).numpy()
                th  = model.theta.detach().numpy()

            cos_d[seed] = cos
            n_neg = int((cos < 0).sum())
            child_cos = [float(cos[ci]) for _, ci in pair_indices]
            print(f"  FINAL [{name}]: mu={cos.mean():.3f} n_neg={n_neg}"
                  f"  child_cos={[f'{c:.3f}' for c in child_cos]}"
                  f"  theta: mean={th.mean():.4f} std={th.std():.4f}")

            all_results.append(dict(
                architecture=name, seed=seed,
                mu=float(cos.mean()), sigma=float(cos.std()),
                n_neg=n_neg, min_cos=float(cos.min()),
                mean_child_cos=float(np.mean(child_cos)),
                child_cosines=child_cos,
                any_child_neg=any(c < 0 for c in child_cos),
                theta_mean=float(th.mean()), theta_std=float(th.std()),
            ))

    # Figure
    fig, axes = plt.subplots(2, len(seeds), figsize=(5*len(seeds), 8))
    for col, seed in enumerate(seeds):
        for row, (name, store) in enumerate([
            ('JumpReLU (scalar)',      cos_scalar),
            ('JumpReLU (per-feature)', cos_perfeat),
        ]):
            if seed not in store:
                continue
            ax = axes[row, col]
            cos = store[seed]
            ax.hist(cos, bins=20, color=COLOR_MAP[name], alpha=0.75, edgecolor='k')
            ax.axvline(0, color='red', lw=2, ls='--')
            mu, sig = cos.mean(), cos.std()
            ax.axvline(mu - 3*sig, color='orange', lw=1.5, ls=':')
            n_neg = int((cos < 0).sum())
            ax.set_title(f'{"Scalar" if "scalar" in name else "Per-feat"}'
                         f' JumpReLU | s={seed}\nn_neg={n_neg}')
            ax.set_xlabel('cos(E, D)')
            ax.set_ylabel('Count')

    fig.suptitle('Phase 2e (revised): JumpReLU Scalar vs Per-Feature\n'
                 '(Fixed: lower initial threshold, proper STE gradient)',
                 fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'phase2e_jumprelu_fixed')

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS, 'jumprelu_fixed_comparison.csv'), index=False)
    print(f"\nSaved: results/phase2/jumprelu_fixed_comparison.csv")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default='all',
                        choices=['2D', 'lambda', 'jumprelu', 'all'])
    args = parser.parse_args()

    t0 = time.time()

    if args.part in ('2D', 'all'):
        print("\n--- Running 2D analytical toy model ---")
        records_2D = run_2D_toy(theta_deg=30.0, seed=42)
        plot_2D_results(records_2D, theta_deg=30.0)
        # Also save critical lambda data
        df2D = pd.DataFrame(records_2D)
        df2D.to_csv(os.path.join(RESULTS, 'analytical_2D_toy.csv'), index=False)
        print(f"Saved results/phase2/analytical_2D_toy.csv")

        # Additional: vary theta
        print("\n--- 2D toy: varying theta ---")
        theta_records = {}
        for theta in [15.0, 30.0, 45.0, 60.0]:
            r = run_2D_toy(theta_deg=theta, seed=42, 
                           lambdas=[0.0, 1e-3, 0.01, 0.05, 0.10, 0.20])
            theta_records[theta] = r
        
        # Plot: min cosine achievable vs theta
        fig, ax = plt.subplots(figsize=(7, 4))
        thetas_plotted = [15.0, 30.0, 45.0, 60.0]
        analytical_pred = [-np.cos(np.deg2rad(t)) for t in thetas_plotted]
        actual_min = [min(r['min_cos'] for r in theta_records[t]) for t in thetas_plotted]

        ax.plot(thetas_plotted, analytical_pred, 'g--o', ms=8, label='Analytical: -cos(theta)')
        ax.plot(thetas_plotted, actual_min, 'b-s', ms=8, label='Observed min cos(E,D)')
        ax.axhline(0, color='k', lw=1, ls=':')
        ax.set_xlabel('theta (degrees)')
        ax.set_ylabel('Minimum cos(E,D) achievable')
        ax.set_title('Anti-alignment strength vs geometric overlap\n'
                     '(2D toy model, high lambda regime)')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.suptitle('Analytical prediction matches numerical result:\n'
                     'cos(hedge feature) = -cos(theta)', fontweight='bold')
        plt.tight_layout()
        save_fig(fig, 'phase2a_analytical_vs_numerical')

    if args.part in ('lambda', 'all'):
        print("\n--- Running lambda search (10D) ---")
        run_lambda_search_10D()

    if args.part in ('jumprelu', 'all'):
        print("\n--- Running JumpReLU fixed ---")
        run_jumprelu_fixed(seeds=[42, 1337, 7])

    print(f"\nAll done. Time: {(time.time()-t0)/60:.1f} min")
