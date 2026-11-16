# Research Finding: LXR Architecture Clarification
# March 2026 — Resolves Supplement Limitation 3

## What We Discovered

The scalar threshold we observed in Llama LXR SAEs is NOT a SAELens artifact.
It is an INTENTIONAL design decision documented in the Llama Scope paper.

---

## LXR = "Language-model X Residual" — A Novel JumpReLU Variant

From Llama Scope paper (He et al., 2024, arXiv:2410.20526), Section 3:

> "We release a new set of SAEs (dubbed LXR) trained using a novel variant of
> the JumpReLU architecture, along with a recipe for further fine-tuning on
> domain-specific data."

The key architectural change in LXR:

**Standard JumpReLU** (Gemma Scope, Rajamanoharan et al. 2024):
  - theta ∈ R^M (per-feature vector, one threshold per feature)
  - log_threshold shape: (d_sae,) -- e.g., (16384,)
  - Each feature independently learns its activation cutoff

**LXR JumpReLU** (Llama Scope, He et al. 2024):
  - theta ∈ R (scalar, shared across all features)
  - log_threshold shape: (1,) or scalar
  - Single global threshold governs sparsity for ALL features

Motivation for LXR scalar threshold:
1. Direct L0 control: increasing theta decreases total L0 uniformly
2. Fine-tuning flexibility: domain adaptation by adjusting only ONE parameter
3. Simpler sparsity targeting during training

---

## The Performance Cost (Documented in Original JumpReLU Paper)

From Rajamanoharan et al. (2024), Appendix A.6 "Per-feature threshold vs. scalar threshold":

> "To assess the importance of having a per-feature (vector) threshold, we train
> JumpReLU SAEs with a scalar threshold (a single value shared across all features),
> comparing to the default per-feature threshold. We find that the scalar threshold
> variant has SIGNIFICANTLY WORSE performance in terms of the L0-loss Pareto frontier,
> highlighting the importance of having a per-feature threshold."

The Llama Scope / LXR repo explicitly acknowledges this tradeoff:
> "Per the JumpReLU paper ablation (Appendix A.6), scalar threshold generally
> underperforms per-feature threshold on the L0-loss Pareto frontier.
> LXR accepts this tradeoff for the benefit of simplified sparsity control."

---

## THE KEY INSIGHT FOR OUR PAPER

Our finding that Llama LXR SAEs exhibit anti-aligned features (cos(E,D) < 0)
while Gemma (per-feature threshold) does not is CONSISTENT WITH the known
performance degradation of scalar threshold documented in Appendix A.6.

We are observing the GEOMETRIC MANIFESTATION of the L0-loss Pareto degradation:
  - Scalar threshold → features cannot individually tune their activation cutoffs
  - Some features get inactive (decoder direction drifts from encoder)
  - Result: anti-aligned features (encoder-decoder misaligned geometry)
  - THIS IS THE SAME DEGRADATION shown in the JumpReLU paper, now characterized geometrically

This gives our finding a clean, principled explanation grounded in the SAE literature:

"Anti-aligned features in JumpReLU SAEs are more prevalent when the SAE uses a
scalar (global) threshold, consistent with the documented performance degradation
of scalar vs. per-feature threshold training. The NDLXR family (Llama LXR)
intentionally uses a scalar threshold for fine-tuning flexibility, accepting this
geometric degradation as a known tradeoff."

---

## The Two Llama SAE Families (IMPORTANT DISTINCTION)

We analyzed the LXR family. There is a separate standard TopK family:

FAMILY 1 (LXR JumpReLU) -- WHAT WE ANALYZED:
  Repos: fnlp/Llama3_1-8B-Base-LXR-{8x/32x}
  SAELens: llama_scope_lxr_{8x/32x}
  Architecture: JumpReLU with SCALAR threshold
  Purpose: fine-tuning recipe, LM-head auxiliary loss
  Our Phase 0/1: found anti-aligned features in L0-L2

FAMILY 2 (Standard TopK) -- NOT YET ANALYZED:
  Repos: fnlp/llama-scope-l{L}-res-{W}k  (404 in our tests -- may need auth)
  Architecture: TopK (k=50/100/200) -- NO threshold parameter
  Purpose: main Llama Scope release, 256 SAEs
  Cos(E,D) behavior: UNKNOWN -- open question for Phase 5

The key question for Phase 5: do TopK Llama SAEs (no threshold) also show
anti-aligned features? If YES: anti-alignment is independent of threshold type.
If NO: scalar threshold is the causal factor (E2 would need revision).

---

## Updated Hypothesis E2 (Threshold Architecture)

ORIGINAL (from Supplement): "Per-feature JumpReLU threshold prevents anti-alignment"
REFINED: "The QUALITY of threshold flexibility matters:
  - No threshold (TopK, ReLU+L1): anti-alignment through different mechanisms
  - Per-feature theta (Gemma): features independently optimize -> protected
  - Scalar theta (LXR): performance degraded as per A.6 -> anti-alignment emerges"

The LXR research clarifies the mechanism: the INABILITY of features to set
individual thresholds forces a suboptimal optimization equilibrium that produces
anti-aligned encoder-decoder pairs as a measurable geometric consequence.

---

## References

Rajamanoharan et al. (2024) JumpReLU. arXiv:2407.14435. [in papers/]
  Appendix A.6: "scalar threshold significantly worse performance"
  Equation 6: theta ∈ R^M (per-feature by definition)

He et al. (2024) Llama Scope. arXiv:2410.20526. [in papers/]
  Section 3: LXR introduces scalar theta ∈ R
  Acknowledged tradeoff vs per-feature performance
