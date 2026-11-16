# Phase 2: Toy Model Validation
# Priority: P2 | Duration: ~1 week | Compute: NEGLIGIBLE | Gate: G2
# REVISED POST PHASE 0 (March 2026)

## Objective -- REVISED
Phase 0 established that anti-aligned features exist in JumpReLU SAEs (Llama 3.1 8B),
which CHALLENGES the original toy model (which was only about L1 pressure).

The revised objective is DUAL:
  2A. Validate that L1 pressure CAN produce anti-aligned features (original hypothesis).
      This addresses GPT-2 LATE-layer features.
  2B. Demonstrate a SECOND mechanism that produces anti-aligned features WITHOUT L1.
      This addresses Llama EARLY-layer features and architecture-independent phenomena.

The toy model must now show that cos(E,D) < 0 arises from at least two distinct pathways:
  (a) L1 magnitude shrinkage + asymmetric co-activation (H1, for GPT-2)
  (b) Superposition interference or representation compression WITHOUT L1 (H3/H6, for Llama)

## Original Objective (still valid for sub-task 2A)
Formally validate the toy model from Section 3 of the project proposal.
Show analytically AND numerically that L1 pressure under asymmetric co-activation
produces anti-aligned features (cos(E,D) < 0).

---

## Sub-phases

### Phase 2a: Formal Analytical Derivation

Complete the derivation of Proposition 1 (Section 3.3 of paper) rigorously.

SETUP (from paper):
  d_model = 2, 3 ground-truth concepts A, B, C
  a = (1, 0), b = (cos(theta), sin(theta)) for theta in (0, pi/2)
  Asymmetric co-activation: P(A=1,B=1) = p_AB, P(A=1,B=0) = p_A, P(A=0,B=1) = 0
  (B always implies A, but A can occur without B)
  SAE: d_sae = 2, ReLU + L1 penalty lambda

WHAT TO DERIVE:
  1. The optimal SAE solution without L1 (lambda=0) -- should give cos(E,D) >= 0 for both.
  2. The "merged" solution under sufficient L1:
     Feature 1 (merged): E1 prop to a+b, D1 prop to a+b  [cos(E1,D1) = 1]
     Feature 2 (hedge): E2 prop to b, D2 prop to -a
     -> cos(E2, D2) = <b, -a> / (||b|| * ||a||) = -cos(theta) < 0 for theta < pi/2
  3. CRITICAL: Show the merged solution has LOWER total loss than the clean solution
     when: lambda * p_AB > (reconstruction_cost_of_merging / p_A)
     Derive the exact threshold lambda_crit as function of (theta, p_A, p_AB).
  4. Show that cos(E2, D2) = -cos(theta) and anti-alignment is STRONGER for smaller theta
     (more geometric overlap between concepts A and B).

APPENDIX: The full derivation goes into Appendix B of the paper (currently placeholder).

OUTPUTS:
  - Complete analytical proof (for the project proposal, Appendix B)
  - Expression for lambda_crit(theta, p_A, p_AB)
  - Phase diagram: (theta, lambda) space showing anti-aligned region

### Phase 2b: 10D Numerical Experiment

SETUP:
  d_model = 10 (not 2: more realistic geometry, can test in higher dimensions)
  n_concepts = 15 (overcomplete: 15 concepts in 10D space)
  d_sae = 30 (2x expansion)
  
  Synthetic dataset construction:
    1. Sample 15 random "concept directions" as unit vectors in R^10
       (use orthogonalization to ensure most are nearly orthogonal)
    2. Create 3 pairs of (parent, child) concepts with:
       - High geometric overlap (dot product ~ 0.3-0.5)
       - Asymmetric co-activation: P(child=1|parent=0) = 0, P(child=1|parent=1) = p_dep
    3. Generate 100K synthetic activation vectors as linear combinations
       of active concepts
    4. Add Gaussian noise

```python
import torch
import torch.nn as nn
import numpy as np

def generate_synthetic_data(d_model=10, n_concepts=15, n_samples=100000,
                            n_asymmetric_pairs=3, p_dep=0.3, lambda_l1=1e-3):
    """Generate synthetic data with known asymmetric co-activation structure"""
    
    # Random concept directions (unit vectors)
    directions = torch.randn(n_concepts, d_model)
    directions = directions / directions.norm(dim=1, keepdim=True)
    
    # Create 3 (parent, child) pairs with forced geometric overlap
    for i in range(n_asymmetric_pairs):
        parent_idx = i * 2
        child_idx = i * 2 + 1
        # Make child similar to parent: cos(parent, child) ~ 0.4
        noise = torch.randn(d_model)
        directions[child_idx] = 0.6 * directions[parent_idx] + 0.8 * noise
        directions[child_idx] = directions[child_idx] / directions[child_idx].norm()
    
    # Generate co-activation patterns
    activations_list = []
    for _ in range(n_samples):
        active_mask = torch.zeros(n_concepts)
        
        # Each concept independently active with low probability
        for c in range(n_concepts):
            if torch.rand(1) < 0.05:
                active_mask[c] = torch.rand(1).item() + 0.5  # magnitude 0.5-1.5
        
        # Enforce asymmetric structure: child only active when parent active
        for i in range(n_asymmetric_pairs):
            parent_idx = i * 2
            child_idx = i * 2 + 1
            if active_mask[parent_idx] == 0:
                active_mask[child_idx] = 0  # child can't fire without parent
            elif torch.rand(1) < p_dep:
                # Child fires WITH parent with probability p_dep
                active_mask[child_idx] = torch.rand(1).item() + 0.5
        
        x = (active_mask.unsqueeze(1) * directions).sum(0)
        x += 0.05 * torch.randn(d_model)  # noise
        activations_list.append(x)
    
    return torch.stack(activations_list), directions, active_mask

# SAE model
class SimpleSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
    
    def normalize_decoder(self):
        """Enforce unit norm on decoder row vectors"""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True)
            self.W_dec.data /= norms.clamp(min=1e-6)
    
    def forward(self, x):
        z = torch.relu(x @ self.W_enc + self.b_enc)
        x_hat = z @ self.W_dec + self.b_dec
        return z, x_hat
    
    def loss(self, x, lambda_l1=1e-3):
        z, x_hat = self.forward(x)
        recon_loss = (x - x_hat).pow(2).sum(1).mean()
        sparsity_loss = lambda_l1 * z.abs().sum(1).mean()
        return recon_loss + sparsity_loss
```

TRAINING:
  - Adam optimizer, lr=1e-3, 10000 steps
  - Normalize decoder every step (unit norm constraint)
  - Track cos(E_i, D_i) distribution every 1000 steps

WHAT TO MEASURE:
  - Final cos(E,D) distribution: are there anti-aligned features?
  - Which features are anti-aligned? Do they correspond to the 3 designed child features?
  - What are their activation patterns? Do they fire only when parent is active?

EXPECTED RESULT (if H1 is correct):
  - The 3 (parent, child) pairs should produce anti-aligned hedge features
  - The hedge features should have: cos(E,D) < 0, fire only when parent fires

### Phase 2c: TopK comparison

SAME setup as 2b, but replace ReLU + L1 with TopK (k = d_sae * sparsity_level):

```python
class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.k = k
    
    def forward(self, x):
        pre_act = x @ self.W_enc + self.b_enc
        # TopK: keep top-k activations, zero rest
        topk_vals, topk_idx = pre_act.topk(self.k, dim=1)
        z = torch.zeros_like(pre_act)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))
        x_hat = z @ self.W_dec + self.b_dec
        return z, x_hat
    
    def loss(self, x, lambda_l1=0):  # No L1 penalty
        z, x_hat = self.forward(x)
        recon_loss = (x - x_hat).pow(2).sum(1).mean()
        return recon_loss  # Pure reconstruction, no sparsity penalty
```

PREDICTION (from H1): TopK SAE should have NO anti-aligned features because:
  1. No magnitude-shrinkage pressure from L1
  2. Once a feature fires (in TopK), its magnitude is unconstrained
  3. The hedging mechanism (merged feature + correction term) only arises when
     L1 creates pressure to merge co-activating features

If TopK DOES produce anti-aligned features:
  -> H3 or H5 may be correct (architecture-independent mechanism)
  -> Reframe paper focus toward superposition interference

### Phase 2d: Phase boundary characterization

Sweep the (lambda, theta) space to find where anti-alignment first appears:

PARAMETERS TO SWEEP:
  - lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
  - theta: [5, 15, 30, 45, 60, 75] degrees (angle between concept A and B)
  - p_dep: [0.1, 0.3, 0.5, 0.7, 0.9] (probability child fires when parent fires)

For each combination (lambda, theta, p_dep):
  - Train 3 SAEs (different seeds)
  - Measure: count of features with cos(E,D) < 0, mean of minimum cos(E,D)

BUILD the 2D phase diagrams:
  - (lambda, theta) for fixed p_dep = 0.3
  - (lambda, p_dep) for fixed theta = 30 degrees

---

## Expected Outputs (save to results/phase2/)

- results/phase2/analytical_proof.md  (or the project proposal, Appendix B)
- results/phase2/lambda_crit_formula.md  (the derived threshold expression)
- results/phase2/experiment_2b_results.csv  (cos(E,D) stats per run)
- results/phase2/experiment_2c_topk_results.csv  (TopK comparison)
- results/phase2/phase_diagram.csv  (sweep results)
- results/phase2/G2_decision.md
- figures/phase2/cosine_evolution.png  (cos(E,D) distribution during training)
- figures/phase2/phase_diagram.png  (2D phase diagram: anti-alignment region)
- figures/phase2/topk_comparison.png  (ReLU+L1 vs TopK distributions)

---

## Gate G2 Decision Criteria

G2-YES:  Toy model (10D, 3 asymmetric pairs) reliably produces cos(E,D) < 0 for the
GATE G2-YES (STRONG):
  2A: ReLU+L1 toy model reliably produces cos(E,D)<0 under designed conditions.
  2B: JumpReLU toy model ALSO produces cos(E,D)<0 via a different mechanism (e.g.,
      superposition overload or early-layer compression).
  -> Both mechanisms formally validated. Proceed with full empirical program.

GATE G2-YES (PARTIAL -- L1 only):
  2A succeeds: ReLU+L1 produces anti-alignment.
  2B fails: cannot produce anti-alignment in JumpReLU toy model.
  -> H1 validated for GPT-2 late layers. The Llama early-layer phenomenon remains
     mechanistically unexplained. May suggest H6 requires a non-toy-model approach
     (e.g., the model's actual early-layer representation structure matters).
  Proceed with Phase 3 (seed stability), document the open question.

GATE G2-YES (PARTIAL -- non-L1 only):
  2B succeeds: JumpReLU toy model produces anti-alignment.
  2A fails: original L1 mechanism cannot be validated in toy setting.
  -> This is unexpected. Suggests even L1-based anti-alignment operates through
     superposition interference, not direct magnitude shrinkage.
  Revise paper narrative: H3/H6 dominant across ALL architectures.

GATE G2-NO:  Neither mechanism produces anti-alignment in toy setting.
  -> Toy model too simplistic. Mechanism requires actual LLM representation geometry.
     Pivot: focus Phase 3 and 4 on empirical characterization without toy model support.

---

## Script Template

See: scripts/phase2_toy_model.py (to be created when implementation begins)
See: scripts/phase2_sweep.py (for the phase boundary sweep)

Key imports: torch, numpy, matplotlib, itertools, scipy
Runtime: 2b experiment ~15 min, 2c comparison ~15 min, 2d sweep ~2-4 hours
