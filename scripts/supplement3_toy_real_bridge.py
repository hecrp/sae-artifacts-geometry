"""
Supplement 3: Connecting Toy Model to Real SAEs
  S3A: Bisector test on real GPT-2 L8 anti-aligned features
  S3B: Expansion ratio sweep (Missing Experiment 2D)
  S3C: Seed distribution statistics (100 seeds)
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from itertools import product

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, 'results', 'supplement3')
FIGURES = os.path.join(PROJECT, 'figures', 'supplement3')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
COLOR_BLIND = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']

def save_fig(fig, stem):
    fig.savefig(os.path.join(FIGURES, f'{stem}.png'), dpi=300)
    fig.savefig(os.path.join(FIGURES, f'{stem}.pdf'))
    plt.close(fig)
    print(f"  Saved: figures/supplement3/{stem}.png")

# ══════════════════════════════════════════════════════════════════════════════
# S3A: BISECTOR TEST ON REAL GPT-2 L8 SAE
# ══════════════════════════════════════════════════════════════════════════════

def run_s3a():
    print("\n" + "="*60)
    print("S3A: BISECTOR TEST ON REAL GPT-2 L8 SAE")
    print("="*60)
    t0 = time.time()

    # Load the Bloom GPT-2 L8 SAE
    try:
        from sae_lens import SAE
        sae_result = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device="cpu"
        )
        if isinstance(sae_result, tuple):
            sae = sae_result[0]
        else:
            sae = sae_result
        print(f"  SAE loaded: d_model={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    except Exception as e:
        print(f"  ERROR loading SAE: {e}")
        return

    W_enc = sae.W_enc.detach().cpu()   # d_model x d_sae
    W_dec = sae.W_dec.detach().cpu()   # d_sae x d_model

    d_model, d_sae = W_enc.shape

    # Normalize encoder cols and decoder rows for cosine calculations
    W_enc_norm = F.normalize(W_enc, dim=0)   # d_model x d_sae (each col unit-norm)
    W_dec_norm = F.normalize(W_dec, dim=1)   # d_sae x d_model (each row unit-norm)

    # Compute full cos(E_i, D_i) distribution
    # E_i = W_enc[:, i] (column i of encoder)
    # D_i = W_dec[i, :] (row i of decoder)
    cos_ED = (W_enc_norm * W_dec_norm.T).sum(dim=0)  # d_sae
    mu = cos_ED.mean().item()
    sigma = cos_ED.std().item()
    print(f"  cos(E,D) distribution: mu={mu:.4f}, sigma={sigma:.4f}")

    # Reference anti-aligned features from Phase 0 analysis
    # (L8 Bloom SAE features)
    ANTI_FEATURES = {
        19139: {'cos_expected': -0.525, 'sigma_expected': -11.38, 'enc_norm_expected': 0.727},
        18573: {'cos_expected': -0.421, 'sigma_expected': -10.05, 'enc_norm_expected': 0.233},
        11969: {'cos_expected': -0.286, 'sigma_expected':  -7.5,  'enc_norm_expected': 0.47},
        20804: {'cos_expected': -0.130, 'sigma_expected':  -5.0,  'enc_norm_expected': 0.47},
    }

    print(f"\n  Pre-registered bisector prediction:")
    print(f"  For each anti-aligned f: j* = argmax_j cos(E_f, D_j)")
    print(f"  Should have cos(D_f, D_{{j*}}) < 0 (decoder of f opposes j* decoder)")
    print(f"  This would confirm: E_f detects A, D_f writes -A (hedge structure)")

    results = []

    for feat_idx, info in ANTI_FEATURES.items():
        print(f"\n  --- Feature {feat_idx} ---")

        # Get feature vectors
        E_f = W_enc[:, feat_idx]          # d_model
        D_f = W_dec[feat_idx, :]          # d_model
        E_f_norm = F.normalize(E_f, dim=0)
        D_f_norm = F.normalize(D_f, dim=0)

        # Empirical cos(E,D) for this feature
        cos_ed = cos_ED[feat_idx].item()
        sigma_dev = (cos_ed - mu) / sigma
        enc_norm = E_f.norm().item()
        dec_norm = D_f.norm().item()

        print(f"    cos(E,D) = {cos_ed:.4f} ({sigma_dev:.2f} sigma)")
        print(f"    ||E|| = {enc_norm:.4f}, ||D|| = {dec_norm:.4f}")

        # ── Test 1: Encoder detection test ─────────────────────────────────
        # j_enc = argmax_j cos(E_f, D_j)
        # cos(E_f, D_j) for all j, using unit-norm E_f and unit-norm W_dec rows
        cos_Ef_Dj = W_dec_norm @ E_f_norm  # d_sae
        cos_Ef_Dj[feat_idx] = -2.0  # exclude self

        # Exclude anti-aligned features (cos(E,D)<0) to find "clean aligned" parents
        anti_mask = cos_ED < (mu - 3 * sigma)
        cos_Ef_Dj_aligned = cos_Ef_Dj.clone()
        cos_Ef_Dj_aligned[anti_mask] = -2.0

        j_enc_idx = cos_Ef_Dj_aligned.argmax().item()
        cos_Ef_Djenc = cos_Ef_Dj_aligned[j_enc_idx].item()

        # ── Test 2: Decoder opposition test ────────────────────────────────
        # j_dec = argmin_j cos(D_f, D_j) (which feature decoder does D_f oppose most?)
        cos_Df_Dj = W_dec_norm @ D_f_norm  # d_sae
        cos_Df_Dj[feat_idx] = 2.0  # exclude self (want argmin)
        cos_Df_Dj_aligned = cos_Df_Dj.clone()
        cos_Df_Dj_aligned[anti_mask] = 2.0  # exclude anti-aligned

        j_dec_idx = cos_Df_Dj_aligned.argmin().item()
        cos_Df_Djdec = cos_Df_Dj_aligned[j_dec_idx].item()

        # ── Test 3: Cross test ──────────────────────────────────────────────
        # If j_enc is the "parent A" (most detected by E_f):
        #   does D_f also oppose D_{j_enc}? (i.e., cos(D_f, D_{j_enc}) < 0?)
        cos_Df_with_jenc = (W_dec_norm[j_enc_idx] * D_f_norm).sum().item()
        # If j_dec is the "parent B" (most opposed by D_f):
        #   does E_f also detect D_{j_dec}? (i.e., cos(E_f, D_{j_dec}) > 0?)
        cos_Ef_with_jdec = (W_dec_norm[j_dec_idx] * E_f_norm).sum().item()

        # ── Bisector score ──────────────────────────────────────────────────
        # Combined score: how well does f satisfy the hedge/absorption structure?
        # Ideal: E_f detects A (cos_Ef_Djenc > 0) AND D_f opposes A (cos_Df_with_jenc < 0)
        # Score = cos_Ef_Djenc * (-cos_Df_with_jenc) in [0,1] if structure holds

        bisector_score = cos_Ef_Djenc * (-cos_Df_with_jenc)
        hedge_structure = (cos_Ef_Djenc > 0.1) and (cos_Df_with_jenc < -0.1)

        # ── Top-5 alignments for the encoder ───────────────────────────────
        top5_enc = cos_Ef_Dj_aligned.topk(5).indices.tolist()
        top5_enc_cos = [cos_Ef_Dj_aligned[i].item() for i in top5_enc]
        # Check if j_enc == j_dec
        same_parent = (j_enc_idx == j_dec_idx)

        print(f"    j_enc={j_enc_idx}: cos(E_f, D_{{j_enc}}) = {cos_Ef_Djenc:+.4f}")
        print(f"    j_dec={j_dec_idx}: cos(D_f, D_{{j_dec}}) = {cos_Df_Djdec:+.4f}")
        print(f"    Cross: cos(D_f, D_{{j_enc}}) = {cos_Df_with_jenc:+.4f}  [<0 = hedge]")
        print(f"    Cross: cos(E_f, D_{{j_dec}}) = {cos_Ef_with_jdec:+.4f}  [>0 = hedge]")
        print(f"    Same parent? j_enc==j_dec: {same_parent}")
        print(f"    Bisector score: {bisector_score:.4f}")
        print(f"    Hedge structure CONFIRMED: {hedge_structure}")
        print(f"    Top-5 encoder detections: {[(i, f'{c:.3f}') for i,c in zip(top5_enc, top5_enc_cos)]}")

        results.append({
            'feat_idx': feat_idx,
            'cos_ed': cos_ed,
            'sigma_dev': sigma_dev,
            'enc_norm': enc_norm,
            'dec_norm': dec_norm,
            'j_enc': j_enc_idx,
            'cos_Ef_Djenc': cos_Ef_Djenc,
            'j_dec': j_dec_idx,
            'cos_Df_Djdec': cos_Df_Djdec,
            'cos_Df_with_jenc': cos_Df_with_jenc,
            'cos_Ef_with_jdec': cos_Ef_with_jdec,
            'bisector_score': bisector_score,
            'same_parent': same_parent,
            'hedge_confirmed': hedge_structure,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS, 'bisector_test_results.csv'), index=False)
    print(f"\n  Saved: results/supplement3/bisector_test_results.csv")

    n_confirmed = df['hedge_confirmed'].sum()
    print(f"\n  SUMMARY: {n_confirmed}/4 features confirm the hedge/bisector structure")
    print(f"  Pre-registered prediction: F19139 YES, F18573 NO (based on enc_norm)")

    # ── Figure: 2x2 grid, one panel per feature ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("S3A: Bisector Test — Real GPT-2 L8 Anti-Aligned Features", fontsize=13)

    for ax, row in zip(axes.flatten(), results):
        feat = row['feat_idx']
        j_enc = row['j_enc']

        # Get feature vectors in the 2D subspace {D_{j_enc}, perp}
        E_f_n = F.normalize(W_enc[:, feat], dim=0).numpy()
        D_f_n = F.normalize(W_dec[feat], dim=0).numpy()
        D_jenc_n = W_dec_norm[j_enc].numpy()

        # Gram-Schmidt: build 2D basis
        b1 = D_jenc_n / np.linalg.norm(D_jenc_n)
        b2 = E_f_n - np.dot(E_f_n, b1) * b1
        b2_norm = np.linalg.norm(b2)
        if b2_norm < 1e-8:
            b2 = np.random.randn(len(b1)); b2 -= np.dot(b2, b1)*b1; b2 /= np.linalg.norm(b2)
        else:
            b2 = b2 / b2_norm

        def proj2d(v):
            return np.dot(v, b1), np.dot(v, b2)

        # Project key vectors
        e1, e2 = proj2d(E_f_n)
        d1, d2 = proj2d(D_f_n)
        djenc1, djenc2 = proj2d(D_jenc_n)

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')

        # Plot arrows from origin
        origin = np.array([0., 0.])
        ax.annotate('', xy=(e1, e2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLOR_BLIND[0], lw=2))
        ax.annotate('', xy=(d1, d2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLOR_BLIND[1], lw=2))
        ax.annotate('', xy=(djenc1, djenc2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLOR_BLIND[2], lw=2, ls='--'))

        ax.text(e1*1.05, e2*1.05, f'E_f', color=COLOR_BLIND[0], fontsize=9)
        ax.text(d1*1.05, d2*1.05, f'D_f', color=COLOR_BLIND[1], fontsize=9)
        ax.text(djenc1*1.05, djenc2*1.05, f'D_{{j*={j_enc}}}', color=COLOR_BLIND[2], fontsize=9)

        hedge_str = "HEDGE" if row['hedge_confirmed'] else "no hedge"
        lbl = (f"F{feat}: cos(E,D)={row['cos_ed']:.3f}\n"
               f"||E||={row['enc_norm']:.3f}  {hedge_str}\n"
               f"cos(E_f,D_{{j*}})={row['cos_Ef_Djenc']:+.3f}  "
               f"cos(D_f,D_{{j*}})={row['cos_Df_with_jenc']:+.3f}")
        ax.set_title(lbl, fontsize=9)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('$D_{j^*}$  axis', fontsize=8)
        ax.set_ylabel('Perp axis', fontsize=8)

    plt.tight_layout()
    save_fig(fig, 'supplement3a_bisector_test')
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# S3B: EXPANSION RATIO SWEEP (Missing Experiment 2D)
# ══════════════════════════════════════════════════════════════════════════════

class ToyModelExpRatio(torch.nn.Module):
    """ReLU+L1 toy SAE for expansion ratio sweep."""
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, d_sae) * 0.1)
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(F.normalize(torch.randn(d_sae, d_model), dim=1))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.d_sae = d_sae
        self.d_model = d_model

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def forward(self, x):
        h = F.relu(x @ self.W_enc + self.b_enc)
        x_hat = h @ self.W_dec + self.b_dec
        return x_hat, h

    def loss(self, x, lam):
        x_hat, h = self(x)
        recon = ((x - x_hat)**2).sum(-1).mean()
        sparsity = h.abs().sum(-1).mean()
        return recon + lam * sparsity, recon, sparsity


def make_toy_data_for_expratio(d_model, n_concepts, n_samples, seed):
    """Simple sparse data: n_concepts features with uniform p=0.05 activation."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_concepts, d_model))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    acts = rng.random((n_samples, n_concepts)) < 0.05
    mags = rng.exponential(1.0, size=(n_samples, n_concepts)) * acts

    data = (mags[:, :, None] * directions[None, :, :]).sum(axis=1)
    return torch.tensor(data, dtype=torch.float32)


def get_cos_ED(model):
    """Compute cos(E_i, D_i) for all SAE features."""
    E = F.normalize(model.W_enc.detach().T, dim=1)   # d_sae x d_model
    D = F.normalize(model.W_dec.detach(), dim=1)      # d_sae x d_model
    return (E * D).sum(dim=1)


def run_s3b():
    print("\n" + "="*60)
    print("S3B: EXPANSION RATIO SWEEP (Missing Experiment 2D)")
    print("="*60)
    t0 = time.time()

    torch.set_num_threads(4)

    D_MODEL = 10
    N_CONCEPTS = 15
    N_SAMPLES = 50000
    N_STEPS = 8000
    LAMBDA = 5e-3
    N_SEEDS = 20
    LR = 3e-4
    BATCH = 1024

    # Expansion ratios to test (d_SAE / d_model)
    # Include 32 to match real GPT-2 L8 Bloom SAE
    EXP_RATIOS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

    records = []

    for exp_ratio in EXP_RATIOS:
        d_sae = D_MODEL * exp_ratio
        n_neg_list = []
        min_cos_list = []

        for seed in range(N_SEEDS):
            torch.manual_seed(seed * 100 + exp_ratio)
            data = make_toy_data_for_expratio(D_MODEL, N_CONCEPTS, N_SAMPLES, seed=seed)

            model = ToyModelExpRatio(D_MODEL, d_sae)
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            for step in range(N_STEPS):
                idx = torch.randint(0, N_SAMPLES, (BATCH,))
                x = data[idx]
                loss, _, _ = model.loss(x, LAMBDA)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 100 == 0:
                    model.normalize_decoder()

            model.normalize_decoder()
            cos_ED = get_cos_ED(model)
            mu = cos_ED.mean().item()
            sg = cos_ED.std().item()
            n_neg = (cos_ED < 0).sum().item()
            min_cos = cos_ED.min().item()
            n_3sig = (cos_ED < mu - 3*sg).sum().item()

            n_neg_list.append(n_neg)
            min_cos_list.append(min_cos)

            records.append({
                'exp_ratio': exp_ratio,
                'd_sae': d_sae,
                'seed': seed,
                'n_neg': n_neg,
                'min_cos': min_cos,
                'mu_cos': mu,
                'sigma_cos': sg,
                'n_3sig': n_3sig,
            })

        mean_n_neg = np.mean(n_neg_list)
        p_any_neg  = np.mean([x > 0 for x in n_neg_list])
        mean_min   = np.mean(min_cos_list)
        print(f"  exp_ratio={exp_ratio:3d}x (d_sae={d_sae:3d}): "
              f"n_neg={mean_n_neg:.2f}+/-{np.std(n_neg_list):.2f}  "
              f"P(any_neg)={p_any_neg:.2f}  min_cos={mean_min:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, 'expansion_ratio_sweep.csv'), index=False)
    print(f"\n  Saved: results/supplement3/expansion_ratio_sweep.csv")

    # Summary by exp_ratio
    summary = df.groupby('exp_ratio').agg(
        mean_n_neg=('n_neg', 'mean'),
        std_n_neg=('n_neg', 'std'),
        p_any_neg=('n_neg', lambda x: (x > 0).mean()),
        mean_min_cos=('min_cos', 'mean'),
        std_min_cos=('min_cos', 'std'),
    ).reset_index()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("S3B: Expansion Ratio Sweep — Does More Overcomplete = Less Anti-Aligned?", fontsize=12)

    err_kw = dict(fmt='o-', capsize=4, lw=2)

    ax = axes[0]
    ax.errorbar(summary['exp_ratio'], summary['mean_n_neg'], yerr=summary['std_n_neg'],
                color=COLOR_BLIND[0], **err_kw)
    ax.axvline(32, color='red', ls='--', lw=1.5, label='Real GPT-2 (32x)')
    ax.set_title('Mean n_neg vs expansion ratio')
    ax.set_xlabel('d_SAE / d_model')
    ax.set_ylabel('Mean features with cos(E,D) < 0')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(summary['exp_ratio'], summary['p_any_neg'], 'o-', color=COLOR_BLIND[1], lw=2)
    ax.axvline(32, color='red', ls='--', lw=1.5)
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.set_title('P(any anti-aligned) vs expansion ratio')
    ax.set_xlabel('d_SAE / d_model')
    ax.set_ylabel('Fraction of seeds with n_neg ≥ 1')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 1.05)

    ax = axes[2]
    ax.errorbar(summary['exp_ratio'], summary['mean_min_cos'], yerr=summary['std_min_cos'],
                color=COLOR_BLIND[2], **err_kw)
    ax.axvline(32, color='red', ls='--', lw=1.5)
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_title('Mean min cos(E,D) vs expansion ratio')
    ax.set_xlabel('d_SAE / d_model')
    ax.set_ylabel('Mean minimum cos(E_i, D_i)')
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    save_fig(fig, 'supplement3b_expansion_ratio')
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# S3C: SEED DISTRIBUTION STATISTICS (100 seeds)
# ══════════════════════════════════════════════════════════════════════════════

class ToyModelSeedDist(torch.nn.Module):
    """Same as Phase 2b toy model, used for seed distribution."""
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, d_sae) * 0.1)
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(F.normalize(torch.randn(d_sae, d_model), dim=1))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def forward(self, x):
        h = F.relu(x @ self.W_enc + self.b_enc)
        return h @ self.W_dec + self.b_dec, h

    def loss(self, x, lam):
        x_hat, h = self(x)
        recon = ((x - x_hat) ** 2).sum(-1).mean()
        sparse = h.abs().sum(-1).mean()
        return recon + lam * sparse


def make_10d_data(seed, n=50000):
    """10D toy data: 15 concepts, 3 asymmetric pairs."""
    import numpy as np
    rng = np.random.default_rng(seed)
    d, n_c = 10, 15

    # Random directions
    dirs = rng.standard_normal((n_c, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # 3 asymmetric (parent, child) pairs: parent+child correlated
    theta = np.radians(30)
    for i in range(3):
        parent_idx = i * 2
        child_idx  = i * 2 + 1
        parent_dir = dirs[parent_idx]
        child_dir  = np.cos(theta) * parent_dir + np.sin(theta) * rng.standard_normal(d)
        child_dir -= np.dot(child_dir, parent_dir) * parent_dir * (1 - np.cos(theta))
        child_dir /= np.linalg.norm(child_dir)
        dirs[child_idx] = child_dir

    acts = rng.random((n, n_c)) < 0.1
    mags = rng.exponential(1.0, size=(n, n_c)) * acts
    data = (mags[:, :, None] * dirs[None]).sum(axis=1)
    return torch.tensor(data, dtype=torch.float32)


def train_and_eval(seed, lam, d_model=10, d_sae=30, n_steps=8000, lr=3e-4, batch=1024):
    torch.manual_seed(seed * 999 + int(lam * 1e5))
    data = make_10d_data(seed)
    n = data.shape[0]

    model = ToyModelSeedDist(d_model, d_sae)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        idx = torch.randint(0, n, (batch,))
        L = model.loss(data[idx], lam)
        opt.zero_grad(); L.backward(); opt.step()
        if step % 100 == 0:
            model.normalize_decoder()

    model.normalize_decoder()
    E = F.normalize(model.W_enc.detach().T, dim=1)
    D = F.normalize(model.W_dec.detach(), dim=1)
    cos_ED = (E * D).sum(dim=1)
    mu = cos_ED.mean().item()
    sg = cos_ED.std().item()
    n_neg = int((cos_ED < 0).sum().item())
    min_c = cos_ED.min().item()
    n_3s  = int((cos_ED < mu - 3*sg).sum().item())
    return {
        'seed': seed, 'lambda': lam,
        'n_neg': n_neg, 'min_cos': min_c,
        'mu_cos': mu, 'sigma_cos': sg, 'n_3sig': n_3s
    }


def run_s3c():
    print("\n" + "="*60)
    print("S3C: SEED DISTRIBUTION STATISTICS (100 seeds)")
    print("="*60)
    t0 = time.time()

    torch.set_num_threads(4)
    N_SEEDS = 100
    LAMBDAS = [0.0, 5e-3]   # no sparsity vs standard L1

    records = []
    for lam in LAMBDAS:
        n_neg_list = []
        for seed in range(N_SEEDS):
            r = train_and_eval(seed, lam)
            records.append(r)
            n_neg_list.append(r['n_neg'])
            if seed % 20 == 19:
                print(f"  lambda={lam:.0e}, seed {seed+1:3d}/100: "
                      f"P(n_neg>=1 so far)={np.mean([x>0 for x in n_neg_list]):.2f}  "
                      f"mean_n_neg={np.mean(n_neg_list):.2f}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, 'seed_distribution.csv'), index=False)
    print(f"\n  Saved: results/supplement3/seed_distribution.csv")

    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        p_any = (sub['n_neg'] > 0).mean()
        print(f"\n  lambda={lam:.0e}: P(n_neg>=1)={p_any:.3f} | "
              f"mean_n_neg={sub['n_neg'].mean():.2f} +/- {sub['n_neg'].std():.2f} | "
              f"mean_min={sub['min_cos'].mean():.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("S3C: Seed Distribution Statistics (100 seeds, 10D toy)", fontsize=12)

    lam_labels = {0.0: 'lambda=0 (no L1)', 5e-3: 'lambda=5e-3 (standard L1)'}
    colors_lam  = {0.0: COLOR_BLIND[0], 5e-3: COLOR_BLIND[1]}

    for row_idx, lam in enumerate(LAMBDAS):
        sub = df[df['lambda'] == lam]
        col = colors_lam[lam]
        lbl = lam_labels[lam]

        # Panel 1: histogram of n_neg
        ax = axes[row_idx, 0]
        vals = sub['n_neg'].values
        bins = np.arange(-0.5, vals.max() + 1.5, 1)
        ax.hist(vals, bins=bins, color=col, edgecolor='white', alpha=0.8)
        ax.axvline(vals.mean(), color='red', ls='--', label=f'mean={vals.mean():.2f}')
        ax.set_title(f'{lbl}\nn_neg distribution')
        ax.set_xlabel('n_neg (features with cos < 0)')
        ax.set_ylabel('Count / 100 seeds')
        ax.legend(fontsize=9)

        # Panel 2: histogram of min_cos
        ax = axes[row_idx, 1]
        mins = sub['min_cos'].values
        ax.hist(mins, bins=20, color=col, edgecolor='white', alpha=0.8)
        ax.axvline(0, color='gray', ls=':', lw=1)
        ax.axvline(mins.mean(), color='red', ls='--', label=f'mean={mins.mean():.3f}')
        ax.set_title(f'{lbl}\nmin cos(E,D) distribution')
        ax.set_xlabel('min cos(E_i, D_i)')
        ax.set_ylabel('Count / 100 seeds')
        ax.legend(fontsize=9)

        # Panel 3: scatter n_neg vs min_cos
        ax = axes[row_idx, 2]
        ax.scatter(sub['n_neg'], sub['min_cos'], c=col, alpha=0.5, s=20)
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_title(f'{lbl}\nn_neg vs min_cos')
        ax.set_xlabel('n_neg')
        ax.set_ylabel('min cos(E_i, D_i)')

    plt.tight_layout()
    save_fig(fig, 'supplement3c_seed_distribution')
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['S3A', 'S3B', 'S3C', 'all'], default='all')
    args = parser.parse_args()

    t_global = time.time()
    results = {}

    if args.part in ('S3A', 'all'):
        results['S3A'] = run_s3a()

    if args.part in ('S3B', 'all'):
        results['S3B'] = run_s3b()

    if args.part in ('S3C', 'all'):
        results['S3C'] = run_s3c()

    print(f"\n{'='*60}")
    print(f"SUPPLEMENT 3 COMPLETE. Total time: {(time.time()-t_global)/60:.1f} min")

if __name__ == '__main__':
    main()
