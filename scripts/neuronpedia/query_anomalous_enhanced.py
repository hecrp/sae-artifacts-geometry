#!/usr/bin/env python3
"""
Query Anomalous Features Script - ENHANCED VERSION v2

Complete feature retrieval with:
- Comprehensive logging for monitoring
- Exponential backoff retries
- Multiple API endpoints for maximum information
- Robust error handling with partial results
- Progress tracking and statistics

Features covered: the 173 entries of feature_registry.py (see docs/ERRATA.md, E1)
Usage:
    # Test connection
    python query_anomalous_enhanced.py --test-connection
    
    # Query reference features with full data
    python query_anomalous_enhanced.py --reference-features
    
    # Query specific feature with custom text
    python query_anomalous_enhanced.py --model gpt2-small --layer 8 --feature 19139 \\
        --custom-text "The year is 2024 and the date is January 15"
    
    # Query all reference features with progress tracking
    python query_anomalous_enhanced.py --reference-features --export results/ref_features.json
    
    # Query by sigma threshold
    python query_anomalous_enhanced.py --model gpt2-small --sigma-threshold 10.0
"""

import json
import sys
import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)

# Import the enhanced client
from neuronpedia_client_v2 import NeuronpediaClientV2, FeatureCompleteInfo, logger as client_logger
from feature_registry import (
    GPT2_SMALL_FEATURES, GEMMA_2B_FEATURES, LLAMA_31_8B_FEATURES,
    GPT2_HEDGE_FAMILY_PARENTS, FeatureSpec, get_all_features,
    get_features_by_sigma_threshold, MODEL_STATS
)


# Configure root logger for the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("QueryAnomalous")


class QueryManager:
    """Manages feature queries with progress tracking and error handling."""
    
    def __init__(self, client: NeuronpediaClientV2):
        self.client = client
        self.results = []
        self.errors = []
        self.start_time = time.time()
    
    def query_feature_safe(self, model: str, layer: str, index: int,
                          custom_text: Optional[str] = None,
                          include_connected: bool = True) -> Optional[FeatureCompleteInfo]:
        """
        Query a single feature with comprehensive error handling.
        """
        try:
            logger.info(f"\n{'─'*60}")
            logger.info(f"Querying {model}/{layer}/F{index}")
            logger.info(f"{'─'*60}")
            
            info = self.client.get_feature_complete(
                model, layer, index,
                custom_text=custom_text,
                include_connected=include_connected
            )
            
            self.results.append(info)
            return info
            
        except Exception as e:
            logger.error(f"Failed to query {model}/{layer}/F{index}: {e}")
            self.errors.append({
                "model": model,
                "layer": layer,
                "index": index,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def query_feature_set(self, features: List[FeatureSpec],
                         custom_text: Optional[str] = None,
                         include_connected: bool = True) -> Dict:
        """
        Query a set of features with progress tracking.
        """
        if not features:
            logger.warning("No features provided to query")
            return {"error": "No features provided"}
        
        model = features[0].model
        total = len(features)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERYING {total} FEATURES FOR {model.upper()}")
        logger.info(f"{'='*60}\n")
        
        results_data = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "total_features": total,
            "custom_text": custom_text,
            "include_connected": include_connected,
            "features": {},
            "failed": [],
            "statistics": {}
        }
        
        for i, feat in enumerate(features, 1):
            logger.info(f"\n[{i}/{total}] Processing F{feat.index} (L{feat.layer})")
            
            # Determine SAE ID
            if model == "gpt2-small":
                sae_id = f"{feat.layer}-res-jb"
            elif model == "gemma-2b":
                sae_id = f"{feat.layer}-gemmascope-res-16k"
            elif model == "llama-3.1-8b":
                sae_id = f"l{feat.layer}r_8x"
            else:
                sae_id = str(feat.layer)
            
            info = self.query_feature_safe(
                model, sae_id, feat.index,
                custom_text=custom_text,
                include_connected=include_connected
            )
            
            if info:
                # Add our registry metadata
                feature_data = info.to_dict()
                feature_data["registry_metadata"] = {
                    "cos_ed": feat.cos_ed,
                    "sigma": feat.sigma,
                    "encoder_norm": feat.encoder_norm,
                    "hypothesis": feat.hypothesis,
                    "notes": feat.notes
                }
                results_data["features"][feat.index] = feature_data
            else:
                results_data["failed"].append({
                    "index": feat.index,
                    "layer": feat.layer,
                    "sae_id": sae_id
                })
            
            # Progress update every 5 features
            if i % 5 == 0:
                elapsed = time.time() - self.start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(f"\nProgress: {i}/{total} ({i/total*100:.1f}%) | "
                           f"Rate: {rate:.2f} features/sec | "
                           f"Elapsed: {elapsed:.1f}s")
        
        # Final statistics
        success_count = len(results_data["features"])
        results_data["statistics"] = {
            "success_count": success_count,
            "failed_count": len(results_data["failed"]),
            "success_rate": success_count / total * 100 if total > 0 else 0,
            "total_time_seconds": time.time() - self.start_time,
            "api_stats": self.client.get_stats()
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETED: {success_count}/{total} successful "
                   f"({results_data['statistics']['success_rate']:.1f}%)")
        logger.info(f"{'='*60}\n")
        
        return results_data
    
    def get_summary(self) -> Dict:
        """Get execution summary."""
        return {
            "total_results": len(self.results),
            "total_errors": len(self.errors),
            "elapsed_time": time.time() - self.start_time,
        }


def test_connection(client: NeuronpediaClientV2) -> bool:
    """Test API connection with a simple query."""
    logger.info("\n" + "="*60)
    logger.info("TESTING API CONNECTION")
    logger.info("="*60)
    
    try:
        # Try to get a known feature
        result = client.get_feature_basic("gpt2-small", "8-res-jb", 19139)
        logger.info("Connection successful!")
        logger.info(f"   Retrieved feature data: {list(result.keys())}")
        return True
    except Exception as e:
        logger.error(f"ERROR: Connection failed: {e}")
        return False


def export_results(results: Dict, output_path: Path):
    """Export results to JSON file with error handling."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults exported to: {output_path}")
        logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"ERROR: Failed to export results: {e}")


def print_feature_summary(info: FeatureCompleteInfo):
    """Print a formatted summary of feature data."""
    print("\n" + "="*60)
    print(f"FEATURE SUMMARY: {info.model_id}/{info.layer}/F{info.index}")
    print("="*60)
    
    print(f"\nBasic Information:")
    print(f"   Max Activation: {info.max_activation}")
    print(f"   Top Tokens: {', '.join(info.top_tokens[:5]) if info.top_tokens else 'N/A'}")
    print(f"   Explanations: {len(info.explanations)}")
    
    if info.explanations:
        print(f"\nExplanations:")
        for i, exp in enumerate(info.explanations[:3], 1):
            desc = exp.get("description", "N/A")
            print(f"   {i}. {desc[:60]}{'...' if len(desc) > 60 else ''}")
    
    print(f"\nAPI Endpoint Status:")
    for endpoint, status in info.api_status.items():
        icon = "ok  " if status == "success" else "err " if "error" in status else "skip"
        print(f"   {icon} {endpoint}: {status}")
    
    if info.errors:
        print(f"\nWARNING: Errors ({len(info.errors)}):")
        for err in info.errors:
            print(f"   - {err['endpoint']}: {err['error']}")
    
    print("="*60)


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
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Query anomalous features from Neuronpedia API (Enhanced v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test API connection
  python query_anomalous_enhanced.py --test-connection
  
  # Query reference features (GPT-2 L8 Hedge Family)
  python query_anomalous_enhanced.py --reference-features
  
  # Query with custom text for activation analysis
  python query_anomalous_enhanced.py --reference-features \\
      --custom-text "The year is 2024"
  
  # Query specific feature
  python query_anomalous_enhanced.py --model gpt2-small --layer 8 --feature 19139
  
  # Query by sigma threshold
  python query_anomalous_enhanced.py --model gpt2-small --sigma-threshold 10.0
  
  # Query and export results
  python query_anomalous_enhanced.py --reference-features \\
      --export results/reference_features.json
  
  # Print summary only
  python query_anomalous_enhanced.py --summary
        """
    )
    
    parser.add_argument("--test-connection", action="store_true",
                       help="Test API connection and exit")
    parser.add_argument("--model", 
                       choices=["gpt2-small", "gemma-2b", "llama-3.1-8b"],
                       help="Model to query")
    parser.add_argument("--layer", type=str,
                       help="Specific layer to query (e.g., '8-res-jb' or '8')")
    parser.add_argument("--feature", type=int,
                       help="Specific feature index to query")
    parser.add_argument("--reference-features", action="store_true",
                       help="Query the primary reference features (GPT-2 L8)")
    parser.add_argument("--include-parents", action="store_true",
                       help="Include hedge family parent features")
    parser.add_argument("--sigma-threshold", type=float,
                       help="Query features with |sigma| >= threshold")
    parser.add_argument("--custom-text",
                       help="Custom text for activation analysis")
    parser.add_argument("--no-connected", action="store_true",
                       help="Skip connected neurons query (faster)")
    parser.add_argument("--export",
                       help="Export results to JSON file")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary of documented features and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        client_logger.setLevel(logging.DEBUG)
    
    # Print summary and exit if requested
    if args.summary:
        print_summary()
        return 0
    
    # Initialize client
    try:
        logger.info("\n" + "="*60)
        logger.info("INITIALIZING NEURONPEDIA CLIENT v2")
        logger.info("="*60)
        client = NeuronpediaClientV2(rate_limit_delay=0.5)
    except ValueError as e:
        logger.error(f"Failed to initialize client: {e}")
        return 1
    
    # Test connection if requested
    if args.test_connection:
        success = test_connection(client)
        client.print_stats()
        return 0 if success else 1
    
    # Create query manager
    manager = QueryManager(client)
    results = None
    
    # Handle different query modes
    if args.reference_features:
        # Query GPT-2 L8 reference features
        features = GPT2_SMALL_FEATURES[8]
        if args.include_parents:
            features = features + list(GPT2_HEDGE_FAMILY_PARENTS.values())
        
        results = manager.query_feature_set(
            features,
            custom_text=args.custom_text,
            include_connected=not args.no_connected
        )
    
    elif args.sigma_threshold and args.model:
        # Query by sigma threshold
        logger.info(f"\nQuerying features with |sigma| >= {args.sigma_threshold}")
        features = get_features_by_sigma_threshold(args.model, args.sigma_threshold)
        logger.info(f"Found {len(features)} features meeting threshold")
        
        if features:
            results = manager.query_feature_set(
                features,
                custom_text=args.custom_text,
                include_connected=not args.no_connected
            )
        else:
            logger.warning("No features found meeting the threshold")
            return 0
    
    elif args.feature is not None and args.layer and args.model:
        # Query single feature
        logger.info(f"\nQuerying single feature: {args.model}/{args.layer}/F{args.feature}")
        info = manager.query_feature_safe(
            args.model, args.layer, args.feature,
            custom_text=args.custom_text,
            include_connected=not args.no_connected
        )
        
        if info:
            print_feature_summary(info)
            results = info.to_dict()
        else:
            logger.error("Failed to retrieve feature")
            return 1
    
    elif args.layer and args.model:
        # Query all features for a layer
        layer_num = int(args.layer.split('-')[0]) if '-' in args.layer else int(args.layer)
        features = get_all_features(args.model).get(layer_num, [])
        
        if features:
            logger.info(f"Querying {len(features)} features from {args.model} L{layer_num}")
            results = manager.query_feature_set(
                features,
                custom_text=args.custom_text,
                include_connected=not args.no_connected
            )
        else:
            logger.warning(f"No documented features for {args.model} L{layer_num}")
            return 0
    
    else:
        parser.print_help()
        print("\n\nUse --summary to see all documented features")
        print("Use --test-connection to verify API access")
        return 0
    
    # Export if requested
    if args.export and results:
        export_results(results, Path(args.export))
    
    # Print final stats
    client.print_stats()
    
    # Print manager summary
    summary = manager.get_summary()
    logger.info(f"\nQuery Summary:")
    logger.info(f"  Total results: {summary['total_results']}")
    logger.info(f"  Total errors: {summary['total_errors']}")
    logger.info(f"  Elapsed time: {summary['elapsed_time']:.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
