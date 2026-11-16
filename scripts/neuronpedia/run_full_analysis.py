#!/usr/bin/env python3
"""
Complete Neuronpedia analysis for the Geometry SAE project.
Queries ALL key features, parents, extremes, and runs searches.

Output: results/full_analysis_YYYYMMDD.json
"""

import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from neuronpedia_client_v3 import NeuronpediaV3, FeatureProfile, sae_id_for

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "supplement4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ALL FEATURES WE WANT TO QUERY
# ============================================================

# GPT-2 Small: Anti-aligned reference features (Hedge Family L8)
GPT2_HEDGE_FAMILY = {
    "8-res-jb": [19139, 18573, 11969, 20804],
}

# GPT-2 Small: Hedge Family PARENT features (from bisector test)
GPT2_PARENTS = {
    "8-res-jb": [17487, 21174, 16152],
}

# GPT-2 Small: features queried as "extreme". Only 18298, 19679 and 3268 are anti-aligned;
# 4855, 18458, 7685 and 21750 have cos(E, D) near their layer mean (see docs/ERRATA.md, E6).
GPT2_EXTREME = {
    "9-res-jb":  [18298],   # cos=-0.736, -13.94sigma, most extreme in GPT-2
    "10-res-jb": [19679, 3268],   # cos=-0.684, cos=-0.573
    "11-res-jb": [4855],    # most extreme in L11
    "4-res-jb":  [18458],   # early layer extreme
    "6-res-jb":  [7685],    # mid layer extreme
    "7-res-jb":  [21750],   # mid layer extreme
}

# GPT-2 Small: Normal (aligned) features for COMPARISON
GPT2_CONTROLS = {
    "8-res-jb": [1803, 1442, 10666],  # temporal, Paris, positivity
}

# Random layer-8 features used as an additional comparison group. They are NOT anti-aligned.
GPT2_EXTRA_L8 = {
    "8-res-jb": [
        5977, 22561, 6165, 3696,  # random selection across sigma range
    ],
}

# Llama 3.1 8B: Most extreme anti-aligned features (EARLY layers)
# Note: These use different SAE ID format: l{layer}r_8x
LLAMA_EXTREME = {
    # Most extreme per layer
    "l1r_8x": [27745, 24381],   # cos=-0.646, -0.614 (very large encoder norms)
    "l0r_8x": [15649],          # not anti-aligned (cos=+0.24)
    "l2r_8x": [30420],          # not anti-aligned (cos=+0.38)
}

# Gemma 2B: The SINGLE anti-aligned feature
GEMMA_SINGLE = {
    "1-gemmascope-res-16k": [3378],  # cos=-0.465, the only one
}

# ============================================================
# SEARCHES TO RUN
# ============================================================

EXPLANATION_SEARCHES = [
    "hedge",
    "suppression",
    "inhibition",
    "correction",
    "subtraction",
    "negative",
    "opposite direction",
]

TEXT_SEARCHES = [
    # Texts designed to possibly trigger anti-aligned features
    "The defendant was found guilty of criminal charges in a landmark legal case.",
    "The company's stock price dropped significantly after the announcement.",
    "This is a normal sentence about everyday things like cooking and gardening.",
]

TOPK_TEXTS = [
    # Short texts to see which features fire
    "The court ruled against the defendant in a criminal trial.",
    "The year is 2024 and everything has changed dramatically.",
]


def query_feature_group(client, model, features_by_sae, group_name, registry=None):
    """Query a group of features and return profiles."""
    print(f"\n{'='*60}")
    print(f"  QUERYING: {group_name} ({model})")
    print(f"{'='*60}")
    
    all_profiles = []
    for sae_id, indices in features_by_sae.items():
        print(f"\n  SAE: {sae_id} — {len(indices)} features")
        for idx in indices:
            p = client.get_feature(model, sae_id, idx)
            
            # Enrich from registry if available
            if registry and (sae_id, idx) in registry:
                reg = registry[(sae_id, idx)]
                p.cos_ed = reg.get("cos_ed")
                p.sigma = reg.get("sigma")
                p.enc_norm = reg.get("enc_norm")
            
            status = "DEAD" if p.is_dead() else f"frac={p.frac_nonzero:.2e}"
            expl = p.explanation_text[:60]
            print(f"    F{idx:>5d}: {status:>18s} | maxAct={p.max_act_approx:.4f} | {expl}")
            all_profiles.append(p)
    
    return all_profiles


def run_searches(client, model):
    """Run explanation and text searches."""
    results = {"explanation_searches": {}, "text_searches": {}, "topk_searches": {}}
    
    print(f"\n{'='*60}")
    print(f"  EXPLANATION SEARCHES")
    print(f"{'='*60}")
    
    for query in EXPLANATION_SEARCHES:
        print(f"\n  Searching: '{query}'...")
        data = client.search_explanations(query, model)
        if data:
            # Handle different response formats
            items = data if isinstance(data, list) else data.get("results", [])
            results["explanation_searches"][query] = items[:20]  # top 20
            print(f"    Found {len(items)} results")
            for item in items[:3]:
                if isinstance(item, dict):
                    desc = item.get("description", item.get("explanation", "?"))
                    fidx = item.get("index", item.get("featureIndex", "?"))
                    print(f"      F{fidx}: {str(desc)[:80]}")
        else:
            print(f"    No results")
            results["explanation_searches"][query] = []
    
    print(f"\n{'='*60}")
    print(f"  TEXT SEARCHES (search-all)")
    print(f"{'='*60}")
    
    for text in TEXT_SEARCHES:
        print(f"\n  Text: '{text[:60]}...'")
        data = client.search_features_by_text(text, model)
        if data:
            results["text_searches"][text[:60]] = data[:10]
            print(f"    Found {len(data)} activating features")
        else:
            print(f"    No results")
            results["text_searches"][text[:60]] = []
    
    print(f"\n{'='*60}")
    print(f"  TOP-K BY TOKEN")
    print(f"{'='*60}")
    
    for text in TOPK_TEXTS:
        print(f"\n  Text: '{text[:60]}...'")
        data = client.search_topk_by_token(text, model, "8-res-jb", k=5)
        if data:
            results["topk_searches"][text[:60]] = data
            # Try to show some info
            if isinstance(data, dict):
                tokens = data.get("tokens", [])
                print(f"    Tokens: {tokens[:10]}")
            elif isinstance(data, list):
                print(f"    Got {len(data)} token results")
        else:
            print(f"    No results")
            results["topk_searches"][text[:60]] = {}
    
    return results


def analyze_profiles(profiles, group_name):
    """Quick analysis of a group of profiles."""
    print(f"\n{'='*60}")
    print(f"  ANALYSIS: {group_name}")
    print(f"{'='*60}")
    
    dead = [p for p in profiles if p.is_dead()]
    rare = [p for p in profiles if p.is_rare()]
    active = [p for p in profiles if not p.is_dead() and not p.is_rare()]
    has_expl = [p for p in profiles if p.explanations]
    
    print(f"  Total:       {len(profiles)}")
    print(f"  Dead:        {len(dead)} ({100*len(dead)/len(profiles):.0f}%)")
    print(f"  Rare:        {len(rare)} ({100*len(rare)/len(profiles):.0f}%)")
    print(f"  Active:      {len(active)} ({100*len(active)/len(profiles):.0f}%)")
    print(f"  Has explain: {len(has_expl)} ({100*len(has_expl)/len(profiles):.0f}%)")
    
    if active:
        print(f"\n  Active features:")
        for p in sorted(active, key=lambda x: x.frac_nonzero, reverse=True):
            print(f"    {p.summary_line()}")
    
    # Decoder similarity network
    print(f"\n  Decoder cosine similarity network:")
    our_indices = {p.index for p in profiles}
    for p in profiles:
        neighbors_in_group = []
        for idx, val in zip(p.top_similar_indices[:10], p.top_similar_values[:10]):
            if idx != p.index and idx in our_indices:
                neighbors_in_group.append((idx, val))
        if neighbors_in_group:
            print(f"    F{p.index} connected to: {[(f'F{i}', f'{v:.3f}') for i, v in neighbors_in_group]}")
    
    # Shared parent log tokens
    print(f"\n  Logit token patterns:")
    for p in profiles:
        top3_pos = list(zip(p.pos_logit_tokens[:3], p.pos_logit_values[:3]))
        top3_neg = list(zip(p.neg_logit_tokens[:3], p.neg_logit_values[:3]))
        print(f"    F{p.index}: promotes={top3_pos}  suppresses={top3_neg}")


def main():
    client = NeuronpediaV3(delay=0.4)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "timestamp": timestamp,
        "profiles": {},
        "searches": {},
        "analysis": {},
    }
    
    # --- GPT-2 Small queries ---
    
    # 1. Hedge Family (anti-aligned reference)
    hedge_profiles = query_feature_group(
        client, "gpt2-small", GPT2_HEDGE_FAMILY, "GPT-2 Hedge Family (anti-aligned)")
    all_results["profiles"]["gpt2_hedge_family"] = [p.to_dict() for p in hedge_profiles]
    
    # 2. Parent features
    parent_profiles = query_feature_group(
        client, "gpt2-small", GPT2_PARENTS, "GPT-2 Hedge Parents")
    all_results["profiles"]["gpt2_parents"] = [p.to_dict() for p in parent_profiles]
    
    # 3. Most extreme features across layers
    extreme_profiles = query_feature_group(
        client, "gpt2-small", GPT2_EXTREME, "GPT-2 Most Extreme Anti-Aligned")
    all_results["profiles"]["gpt2_extreme"] = [p.to_dict() for p in extreme_profiles]
    
    # 4. Control (normal) features
    control_profiles = query_feature_group(
        client, "gpt2-small", GPT2_CONTROLS, "GPT-2 Controls (aligned)")
    all_results["profiles"]["gpt2_controls"] = [p.to_dict() for p in control_profiles]
    
    # 5. Extra L8 anti-aligned
    extra_profiles = query_feature_group(
        client, "gpt2-small", GPT2_EXTRA_L8, "GPT-2 Extra L8 Anti-Aligned")
    all_results["profiles"]["gpt2_extra_l8"] = [p.to_dict() for p in extra_profiles]
    
    # 6. Llama extreme features
    llama_profiles = query_feature_group(
        client, "llama3.1-8b", LLAMA_EXTREME, "Llama 3.1 8B Extreme Anti-Aligned")
    all_results["profiles"]["llama_extreme"] = [p.to_dict() for p in llama_profiles]
    
    # 7. Gemma single feature
    gemma_profiles = query_feature_group(
        client, "gemma-2-2b", GEMMA_SINGLE, "Gemma 2B Single Anti-Aligned")
    all_results["profiles"]["gemma_single"] = [p.to_dict() for p in gemma_profiles]
    
    # --- Searches ---
    search_results = run_searches(client, "gpt2-small")
    all_results["searches"] = search_results
    
    # --- Analysis ---
    print("\n\n" + "#"*60)
    print("  ANALYSIS SECTION")
    print("#"*60)
    
    analyze_profiles(hedge_profiles, "Hedge Family")
    analyze_profiles(parent_profiles, "Hedge Parents")  
    analyze_profiles(hedge_profiles + parent_profiles, "Hedge Family + Parents Combined")
    analyze_profiles(extreme_profiles, "Most Extreme Anti-Aligned")
    analyze_profiles(control_profiles, "Controls (Aligned)")
    
    if any(p.api_success for p in llama_profiles):
        analyze_profiles(llama_profiles, "Llama Extreme")
    
    if any(p.api_success for p in gemma_profiles):
        analyze_profiles(gemma_profiles, "Gemma Single")
    
    # --- Compare anti-aligned vs controls ---
    print(f"\n{'='*60}")
    print(f"  ANTI-ALIGNED vs CONTROLS COMPARISON")
    print(f"{'='*60}")
    
    anti_all = hedge_profiles + extreme_profiles + extra_profiles
    anti_dead = sum(1 for p in anti_all if p.is_dead())
    ctrl_dead = sum(1 for p in control_profiles if p.is_dead())
    
    anti_frac = [p.frac_nonzero for p in anti_all if not p.is_dead()]
    ctrl_frac = [p.frac_nonzero for p in control_profiles if not p.is_dead()]
    
    print(f"  Anti-aligned: {len(anti_all)} features, {anti_dead} dead ({100*anti_dead/max(len(anti_all),1):.0f}%)")
    print(f"  Controls:     {len(control_profiles)} features, {ctrl_dead} dead ({100*ctrl_dead/max(len(control_profiles),1):.0f}%)")
    
    if anti_frac:
        import statistics
        print(f"  Anti-aligned frac_nonzero: mean={statistics.mean(anti_frac):.2e}, median={statistics.median(anti_frac):.2e}")
    if ctrl_frac:
        print(f"  Controls frac_nonzero:     mean={statistics.mean(ctrl_frac):.2e}, median={statistics.median(ctrl_frac):.2e}")
    
    # --- Save everything ---
    out_file = RESULTS_DIR / f"full_analysis_{timestamp}.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {out_file}")
    print(f"\n{client.stats_summary()}")


if __name__ == "__main__":
    main()
