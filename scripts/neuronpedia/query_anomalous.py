#!/usr/bin/env python3
"""
Query Anomalous Features Script - COMPLETE VERSION

Specific script for querying the anti-aligned features identified in the
Geometry of SAE Artifacts project. Retrieves feature information, explanations,
and activation patterns from Neuronpedia.

Features documented:
- GPT-2 Small: 86 negative-cosine features from the top-10-per-layer tables (layers 0-11)
- Gemma 2B: 1 anti-aligned feature (F3378, L1)
- Llama 3.1 8B: 86 negative-cosine features from the top-10-per-layer tables (layers 0-29)

Reference features (primary research focus):
- GPT-2 L8: F19139, F18573, F11969, F20804 (Hedge Family)
- GPT-2 L9: F18298 (most extreme, cos=-0.736)
- GPT-2 L10: F19679, F3268
- Hedge Family Parents: F17487, F21174, F16152

Usage:
    export NEURONPEDIA_API_KEY="your_key"
    python query_anomalous.py --model gpt2-small --layer 8
    python query_anomalous.py --model gpt2-small --reference-features
    python query_anomalous.py --model llama-3.1-8b --layer 0
    python query_anomalous.py --sigma-threshold 6.0 --export results.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from neuronpedia_client import NeuronpediaClient
from feature_registry import (
    GPT2_SMALL_FEATURES, GEMMA_2B_FEATURES, LLAMA_31_8B_FEATURES,
    GPT2_HEDGE_FAMILY_PARENTS, FeatureSpec, get_all_features,
    get_features_by_sigma_threshold, MODEL_STATS
)


def query_feature_set(
    client: NeuronpediaClient,
    features: List[FeatureSpec],
    include_activations: bool = False,
    test_text: Optional[str] = None
) -> Dict:
    """Query a set of features and return results."""
    if not features:
        return {"error": "No features provided"}
    
    model = features[0].model
    results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total_features": len(features),
        "successful": 0,
        "failed": 0,
        "features": {}
    }
    
    for feat in features:
        # Determine SAE ID based on model
        if model == "gpt2-small":
            sae_id = f"{feat.layer}-res-jb"
        elif model == "gemma-2b":
            sae_id = f"{feat.layer}-gemmascope-res-16k"
        elif model == "llama-3.1-8b":
            sae_id = f"l{feat.layer}r_8x"
        else:
            sae_id = str(feat.layer)
        
        try:
            feature_info = client.get_feature(model, sae_id, feat.index)
            
            feat_result = {
                "success": True,
                "layer": feat.layer,
                "index": feat.index,
                "cos_ed": feat.cos_ed,
                "sigma": feat.sigma,
                "encoder_norm": feat.encoder_norm,
                "hypothesis": feat.hypothesis,
                "notes": feat.notes,
                "neuronpedia_data": {
                    "max_activation": feature_info.max_activation,
                    "top_tokens": feature_info.top_tokens,
                    "explanations": feature_info.explanations,
                }
            }
            
            # Get activations if requested
            if include_activations and test_text:
                act_data = client.get_activations(model, sae_id, feat.index, test_text)
                feat_result["activations"] = act_data.get("activations", {})
            
            results["features"][feat.index] = feat_result
            results["successful"] += 1
            print(f"  F{feat.index} (L{feat.layer}): {feat.hypothesis or 'Unknown'}")
            
        except Exception as e:
            results["features"][feat.index] = {
                "success": False,
                "layer": feat.layer,
                "index": feat.index,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"  FAILED F{feat.index} (L{feat.layer}): {e}")
    
    return results


def query_reference_features(client: NeuronpediaClient, include_parents: bool = False) -> Dict:
    """Query the primary reference features (GPT-2 L8)."""
    print("\n" + "="*60)
    print("QUERYING REFERENCE FEATURES (GPT-2 Small Layer 8)")
    print("="*60)
    
    features = GPT2_SMALL_FEATURES[8]
    results = query_feature_set(client, features)
    
    if include_parents:
        print("\nQuerying Hedge Family Parent Features...")
        parent_features = list(GPT2_HEDGE_FAMILY_PARENTS.values())
        parent_results = query_feature_set(client, parent_features)
        results["parent_features"] = parent_results
    
    return results


def query_most_extreme(client: NeuronpediaClient, model: str, top_n: int = 5) -> Dict:
    """Query the most extreme features for a model."""
    print(f"\nQuerying top {top_n} most extreme features for {model}...")
    
    all_features = []
    for layer_feats in get_all_features(model).values():
        all_features.extend(layer_feats)
    
    # Sort by cos_ed (most negative first)
    all_features.sort(key=lambda x: x.cos_ed)
    top_features = all_features[:top_n]
    
    return query_feature_set(client, top_features)


def export_results(results: Dict, output_path: Path):
    """Export results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults exported to: {output_path}")


def print_summary():
    """Print summary of all documented features."""
    print("\n" + "="*70)
    print("GEOMETRY OF SAE ARTIFACTS - DOCUMENTED ANTI-ALIGNED FEATURES")
    print("="*70)
    
    for model, stats in MODEL_STATS.items():
        print(f"\n{model.upper()}")
        print(f"  Architecture: {stats['architecture']}")
        print(f"  Total features with cos(E,D) < 0: {stats['total_features']}")
        print(f"  Depth pattern: {stats['depth_pattern']}")
        print(f"  Primary hypothesis: {stats['primary_hypothesis']}")
        
        if 'most_extreme' in stats:
            me = stats['most_extreme']
            print(f"  Most extreme: F{me.index} (L{me.layer}, cos={me.cos_ed:.3f}, σ={me.sigma:.2f})")
    
    print("\n" + "="*70)
    print("REFERENCE FEATURES (Primary Research Focus)")
    print("="*70)
    print("\nGPT-2 Small Layer 8 (Hedge Family):")
    for feat in GPT2_SMALL_FEATURES[8]:
        print(f"  F{feat.index}: cos={feat.cos_ed:.3f}, σ={feat.sigma:.2f}, {feat.hypothesis}")
    
    print("\nHedge Family Parents:")
    for idx, feat in GPT2_HEDGE_FAMILY_PARENTS.items():
        print(f"  F{idx}: {feat.notes}")
    
    print("\nGPT-2 Small Layer 9 (Most Extreme):")
    print(f"  F18298: cos=-0.736, σ=-13.94 (Most extreme in GPT-2)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Query anomalous features from Neuronpedia API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query reference features (GPT-2 L8)
  python query_anomalous.py --reference-features
  
  # Query all features for a specific layer
  python query_anomalous.py --model gpt2-small --layer 8
  
  # Query all features with |sigma| >= 6.0
  python query_anomalous.py --model gpt2-small --sigma-threshold 6.0
  
  # Query top 5 most extreme features
  python query_anomalous.py --model gpt2-small --most-extreme 5
  
  # Query specific feature with activation analysis
  python query_anomalous.py --model gpt2-small --layer 8 --feature 19139 \\
      --test-text "The year is 2024"
  
  # Export results
  python query_anomalous.py --reference-features --export results/reference_features.json
  
  # Print summary only (no API calls)
  python query_anomalous.py --summary
        """
    )
    
    parser.add_argument("--model", 
                       choices=["gpt2-small", "gemma-2b", "llama-3.1-8b"],
                       help="Model to query")
    parser.add_argument("--layer", type=int,
                       help="Specific layer to query")
    parser.add_argument("--feature", type=int,
                       help="Specific feature index to query")
    parser.add_argument("--reference-features", action="store_true",
                       help="Query the primary reference features (GPT-2 L8)")
    parser.add_argument("--include-parents", action="store_true",
                       help="Include hedge family parent features")
    parser.add_argument("--sigma-threshold", type=float,
                       help="Query features with |sigma| >= threshold")
    parser.add_argument("--most-extreme", type=int, metavar="N",
                       help="Query top N most extreme features")
    parser.add_argument("--test-text",
                       help="Test text for activation analysis")
    parser.add_argument("--export",
                       help="Export results to JSON file")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary of documented features and exit")
    
    args = parser.parse_args()
    
    # Print summary and exit if requested
    if args.summary:
        print_summary()
        return 0
    
    # Initialize client
    try:
        client = NeuronpediaClient()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    results = {}
    
    # Handle different query modes
    if args.reference_features:
        results = query_reference_features(client, args.include_parents)
    
    elif args.most_extreme and args.model:
        results = query_most_extreme(client, args.model, args.most_extreme)
    
    elif args.sigma_threshold and args.model:
        print(f"\nQuerying features with |sigma| >= {args.sigma_threshold} for {args.model}...")
        features = get_features_by_sigma_threshold(args.model, args.sigma_threshold)
        print(f"Found {len(features)} features meeting threshold")
        results = query_feature_set(client, features, test_text=args.test_text)
    
    elif args.feature is not None and args.layer is not None and args.model:
        # Query specific feature
        print(f"\nQuerying {args.model} Layer {args.layer} Feature {args.feature}...")
        feat = get_all_features(args.model).get(args.layer, [])
        matching = [f for f in feat if f.index == args.feature]
        if matching:
            results = query_feature_set(client, matching, test_text=args.test_text)
        else:
            print(f"Feature F{args.feature} not found in registry for {args.model} L{args.layer}")
            return 1
    
    elif args.layer is not None and args.model:
        # Query all features for a layer
        print(f"\nQuerying {args.model} Layer {args.layer}...")
        features = get_all_features(args.model).get(args.layer, [])
        if features:
            results = query_feature_set(client, features, test_text=args.test_text)
        else:
            print(f"No documented anti-aligned features for {args.model} L{args.layer}")
            return 1
    
    else:
        parser.print_help()
        print("\n\nUse --summary to see all documented features")
        return 0
    
    # Export if requested
    if args.export:
        export_results(results, Path(args.export))
    else:
        # Print results
        print("\n" + "="*60)
        print(json.dumps(results, indent=2, default=str))
    
    return 0


if __name__ == "__main__":
    exit(main())
