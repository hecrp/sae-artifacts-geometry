#!/usr/bin/env python3
"""
Batch Analysis Script for Anomalous Features

Performs comprehensive analysis of anti-aligned features across multiple models.
Generates comparison tables and statistics for the research project.

Usage:
    export NEURONPEDIA_API_KEY="your_key"
    python batch_analyze.py --output-dir ../../results/neuronpedia/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from neuronpedia_client import NeuronpediaClient


class BatchAnalyzer:
    """Batch analyzer for anomalous features across models."""
    
    def __init__(self, client: NeuronpediaClient):
        self.client = client
        self.results = defaultdict(dict)
    
    def analyze_feature_set(
        self,
        model_id: str,
        layer: int,
        features: List[int],
        sae_id: Optional[str] = None
    ) -> Dict:
        """Analyze a set of features and return statistics."""
        if sae_id is None:
            sae_id = f"{layer}-res-jb" if model_id == "gpt2-small" else str(layer)
        
        feature_data = []
        
        for idx in features:
            try:
                info = self.client.get_feature(model_id, sae_id, idx)
                feature_data.append({
                    "index": idx,
                    "max_activation": info.max_activation,
                    "num_explanations": len(info.explanations) if info.explanations else 0,
                    "top_tokens": info.top_tokens[:5] if info.top_tokens else [],
                    "success": True
                })
            except Exception as e:
                feature_data.append({
                    "index": idx,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate statistics
        successful = [f for f in feature_data if f.get("success")]
        activations = [f["max_activation"] for f in successful if f.get("max_activation")]
        
        return {
            "model": model_id,
            "layer": layer,
            "sae_id": sae_id,
            "total_features": len(features),
            "successful_queries": len(successful),
            "failed_queries": len(features) - len(successful),
            "statistics": {
                "mean_max_activation": sum(activations) / len(activations) if activations else None,
                "min_max_activation": min(activations) if activations else None,
                "max_max_activation": max(activations) if activations else None,
            },
            "features": feature_data
        }
    
    def generate_comparison_table(self, results_by_model: Dict) -> str:
        """Generate a markdown comparison table."""
        lines = [
            "# Anomalous Features Comparison Across Models",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "| Model | Layer | Features | Success Rate | Mean Max Act | Min Max Act | Max Max Act |",
            "|-------|-------|----------|--------------|--------------|-------------|-------------|",
        ]
        
        for model_id, layers in sorted(results_by_model.items()):
            for layer_num, result in sorted(layers.items()):
                stats = result.get("statistics", {})
                line = (
                    f"| {model_id} | {layer_num} | "
                    f"{result['total_features']} | "
                    f"{result['successful_queries']}/{result['total_features']} | "
                    f"{stats.get('mean_max_activation', 'N/A'):.4f} | "
                    f"{stats.get('min_max_activation', 'N/A'):.4f} | "
                    f"{stats.get('max_max_activation', 'N/A'):.4f} |"
                )
                lines.append(line)
        
        return "\n".join(lines)
    
    def generate_feature_report(self, result: Dict) -> str:
        """Generate detailed report for a feature set."""
        lines = [
            f"# Feature Analysis: {result['model']} Layer {result['layer']}",
            "",
            f"SAE ID: {result['sae_id']}",
            f"Total features analyzed: {result['total_features']}",
            f"Successful queries: {result['successful_queries']}",
            "",
            "## Statistics",
            "",
        ]
        
        stats = result.get("statistics", {})
        for key, value in stats.items():
            if value is not None:
                lines.append(f"- **{key}**: {value:.6f}")
        
        lines.extend([
            "",
            "## Individual Features",
            "",
        ])
        
        for feat in result.get("features", []):
            if feat.get("success"):
                lines.append(f"### Feature {feat['index']}")
                lines.append(f"- Max Activation: {feat.get('max_activation', 'N/A')}")
                lines.append(f"- Explanations: {feat.get('num_explanations', 0)}")
                if feat.get("top_tokens"):
                    lines.append(f"- Top Tokens: {', '.join(feat['top_tokens'])}")
                lines.append("")
            else:
                lines.append(f"### Feature {feat['index']} - ERROR")
                lines.append(f"- Error: {feat.get('error', 'Unknown')}")
                lines.append("")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Batch analysis of anomalous features")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"),
                       help="Output directory for reports")
    parser.add_argument("--models", nargs="+", default=["gpt2-small"],
                       choices=["gpt2-small", "gemma-2-2b", "gemma-2-9b", "llama-3.1-8b"],
                       help="Models to analyze")
    
    args = parser.parse_args()
    
    # Initialize
    try:
        client = NeuronpediaClient()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    analyzer = BatchAnalyzer(client)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define features to analyze (from query_anomalous.py)
    GPT2_FEATURES = {
        8: [19139, 18573, 11969, 20804],
        9: [18298],
        6: [17432, 19845, 11234, 15678, 18765, 20341],
        10: [21987, 17654, 14523],
    }
    
    all_results = {}
    
    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"Analyzing model: {model_id}")
        print(f"{'='*60}")
        
        all_results[model_id] = {}
        
        if model_id == "gpt2-small":
            for layer, features in GPT2_FEATURES.items():
                print(f"\nLayer {layer}: {len(features)} features...")
                result = analyzer.analyze_feature_set(model_id, layer, features)
                all_results[model_id][layer] = result
                
                # Save individual report
                report_path = output_dir / f"{model_id}_L{layer}_report.md"
                report_path.write_text(analyzer.generate_feature_report(result))
                print(f"  Saved report to {report_path}")
    
    # Generate comparison table
    comparison_md = analyzer.generate_comparison_table(all_results)
    comparison_path = output_dir / "comparison_table.md"
    comparison_path.write_text(comparison_md)
    print(f"\nComparison table saved to {comparison_path}")
    
    # Save raw JSON
    json_path = output_dir / "batch_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved to {json_path}")
    
    print(f"\n{'='*60}")
    print("Batch analysis complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
