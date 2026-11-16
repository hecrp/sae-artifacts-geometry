#!/usr/bin/env python3
"""
Neuronpedia API Client for Geometry SAE Artifacts Project

This module provides a client for querying feature data from Neuronpedia API.
Focused on retrieving information about anti-aligned features identified in the project.

Usage:
    from neuronpedia_client import NeuronpediaClient
    
    client = NeuronpediaClient(api_key="your_key")
    feature = client.get_feature("gpt2-small", "8-res-jb", 19139)
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class FeatureInfo:
    """Structured container for feature information."""
    model_id: str
    layer: str
    index: int
    max_activation: Optional[float] = None
    top_tokens: Optional[List[str]] = None
    explanations: Optional[List[Dict]] = None
    raw_data: Optional[Dict] = None


class NeuronpediaClient:
    """
    Client for interacting with the Neuronpedia API.
    
    API Documentation: https://www.neuronpedia.org/api-doc
    Authentication: API Key via x-api-key header
    """
    
    BASE_URL = "https://www.neuronpedia.org"
    
    # Models and SAEs relevant to the Geometry SAE project
    SUPPORTED_MODELS = {
        "gpt2-small": {
            "name": "GPT-2 Small",
            "sae_prefix": "res-jb",
            "architecture": "ReLU+L1",
            "layers": list(range(12)),
            "features_per_layer": 24576,
        },
        "gemma-2-2b": {
            "name": "Gemma 2B",
            "sae_sets": ["gemmascope-res-16k", "gemmascope-res-131k"],
            "architecture": "JumpReLU",
            "layers": list(range(26)),
        },
        "gemma-2-9b": {
            "name": "Gemma 9B", 
            "sae_sets": ["gemmascope-res-16k", "gemmascope-res-131k"],
            "architecture": "JumpReLU (per-feature threshold)",
            "layers": list(range(42)),
        },
        "llama-3.1-8b": {
            "name": "Llama 3.1 8B",
            "sae_prefix": "llamascope",
            "architecture": "JumpReLU (scalar threshold)",
            "layers": list(range(32)),
        }
    }
    
    # Reference anti-aligned features from the project
    ANOMALOUS_FEATURES = {
        "gpt2-small": {
            8: {  # Layer 8
                19139: {"cos": -0.526, "sigma": -11.38, "norm": 0.727, "type": "active_hedge"},
                18573: {"cos": -0.421, "sigma": -10.05, "norm": 0.233, "type": "dead_candidate"},
                11969: {"cos": -0.286, "sigma": -8.33, "norm": 0.258, "type": "dead_candidate"},
                20804: {"cos": -0.130, "sigma": -6.35, "norm": 0.667, "type": "active_hedge"},
            },
            9: {  # Layer 9 - most extreme
                18298: {"cos": -0.736, "sigma": -13.94, "norm": None, "type": "extreme"},
            }
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Neuronpedia client.
        
        Args:
            api_key: Neuronpedia API key. If not provided, reads from NEURONPEDIA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("NEURONPEDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide via api_key parameter or NEURONPEDIA_API_KEY env var. "
                "Get your key at https://www.neuronpedia.org/account"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to the API."""
        url = urljoin(self.BASE_URL, endpoint)
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_feature(self, model_id: str, layer: str, index: int) -> FeatureInfo:
        """
        Get detailed information about a specific feature.
        
        Endpoint: GET /api/feature/{modelId}/{layer}/{index}
        
        Args:
            model_id: Model ID (e.g., "gpt2-small")
            layer: SAE ID / layer (e.g., "8-res-jb")
            index: Feature index
            
        Returns:
            FeatureInfo object with feature data
        """
        endpoint = f"/api/feature/{model_id}/{layer}/{index}"
        data = self._request("GET", endpoint)
        
        return FeatureInfo(
            model_id=model_id,
            layer=layer,
            index=index,
            max_activation=data.get("maxActivation"),
            top_tokens=data.get("topTokens", []),
            explanations=data.get("explanations", []),
            raw_data=data
        )
    
    def get_activations(
        self, 
        model_id: str, 
        source: str, 
        index: int,
        custom_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get activations for a feature.
        
        If custom_text is provided, uses /api/activation/new to get activations
        for specific input text. Otherwise uses /api/activation/get for default
        activation data.
        
        Args:
            model_id: Model ID
            source: SAE ID / source
            index: Feature index
            custom_text: Optional custom text to analyze
            
        Returns:
            Activation data dictionary
        """
        if custom_text:
            # Get activations for custom text
            endpoint = "/api/activation/new"
            payload = {
                "feature": {
                    "modelId": model_id,
                    "source": source,
                    "index": str(index)
                },
                "customText": custom_text
            }
            return self._request("POST", endpoint, json=payload)
        else:
            # Get default activation data
            endpoint = "/api/activation/get"
            payload = {
                "modelId": model_id,
                "source": source,
                "index": str(index)
            }
            return self._request("POST", endpoint, json=payload)
    
    def search_explanations(
        self,
        query: str,
        model_id: Optional[str] = None,
        layers: Optional[List[str]] = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search for features by explanation text.
        
        Args:
            query: Search query (e.g., "numbers", "dates")
            model_id: Optional model to search within
            layers: Optional list of SAE IDs to search
            offset: Pagination offset
            
        Returns:
            Search results with matching features
        """
        if model_id and layers:
            # Search specific SAEs
            endpoint = "/api/explanation/search"
            payload = {
                "modelId": model_id,
                "layers": layers,
                "query": query,
                "offset": offset
            }
        elif model_id:
            # Search entire model
            endpoint = "/api/explanation/search-model"
            payload = {
                "modelId": model_id,
                "query": query,
                "offset": offset
            }
        else:
            # Search all features
            endpoint = "/api/explanation/search-all"
            payload = {
                "query": query,
                "offset": offset
            }
        
        return self._request("POST", endpoint, json=payload)
    
    def get_source_activations(
        self,
        model_id: str,
        source: str,
        custom_text: str
    ) -> Dict[str, Any]:
        """
        Get activations for ALL features in a source/SAE for given text.
        
        Endpoint: POST /api/activation/source
        
        Args:
            model_id: Model ID
            source: SAE ID / source
            custom_text: Text to analyze
            
        Returns:
            Activations for all features in the SAE
        """
        endpoint = "/api/activation/source"
        payload = {
            "modelId": model_id,
            "source": source,
            "customText": custom_text
        }
        return self._request("POST", endpoint, json=payload)
    
    def get_anomalous_feature_info(
        self,
        model_id: str = "gpt2-small",
        layer: Optional[int] = None
    ) -> Dict[int, FeatureInfo]:
        """
        Retrieve information for all known anomalous (anti-aligned) features.
        
        Args:
            model_id: Model to query (default: gpt2-small)
            layer: Optional specific layer, otherwise queries all known layers
            
        Returns:
            Dictionary mapping feature indices to FeatureInfo objects
        """
        results = {}
        
        if model_id not in self.ANOMALOUS_FEATURES:
            print(f"No anomalous features recorded for model {model_id}")
            return results
        
        layers_to_query = [layer] if layer is not None else self.ANOMALOUS_FEATURES[model_id].keys()
        
        for l in layers_to_query:
            if l not in self.ANOMALOUS_FEATURES[model_id]:
                continue
                
            layer_str = f"{l}-res-jb" if model_id == "gpt2-small" else str(l)
            
            for feat_idx in self.ANOMALOUS_FEATURES[model_id][l].keys():
                try:
                    info = self.get_feature(model_id, layer_str, feat_idx)
                    results[feat_idx] = info
                    print(f"  Retrieved F{feat_idx} (Layer {l})")
                except Exception as e:
                    print(f"  FAILED F{feat_idx} (Layer {l}): {e}")
        
        return results
    
    def format_feature_summary(self, feature: FeatureInfo) -> str:
        """Format feature information as a readable summary."""
        lines = [
            f"Feature {feature.index} in {feature.model_id}/{feature.layer}",
            f"  Max Activation: {feature.max_activation}",
            f"  Top Tokens: {', '.join(feature.top_tokens[:10]) if feature.top_tokens else 'N/A'}",
        ]
        
        if feature.explanations:
            lines.append("  Explanations:")
            for exp in feature.explanations[:3]:
                desc = exp.get("description", "N/A")
                lines.append(f"    - {desc[:80]}..." if len(desc) > 80 else f"    - {desc}")
        
        return "\n".join(lines)


def main():
    """CLI demonstration of the Neuronpedia client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Neuronpedia API for feature data")
    parser.add_argument("--model", default="gpt2-small", help="Model ID")
    parser.add_argument("--layer", default="8-res-jb", help="SAE layer ID")
    parser.add_argument("--index", type=int, help="Feature index")
    parser.add_argument("--anomalous", action="store_true", help="Query all known anomalous features")
    parser.add_argument("--text", help="Custom text for activation analysis")
    
    args = parser.parse_args()
    
    # Initialize client (requires NEURONPEDIA_API_KEY env var)
    try:
        client = NeuronpediaClient()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.anomalous:
        # Query all anomalous features
        layer_num = int(args.layer.split("-")[0]) if "-" in args.layer else None
        print(f"\nQuerying anomalous features for {args.model}...")
        results = client.get_anomalous_feature_info(args.model, layer_num)
        
        print(f"\n{'='*60}")
        print(f"Retrieved {len(results)} anomalous features")
        print(f"{'='*60}\n")
        
        for idx, feat_info in sorted(results.items()):
            print(client.format_feature_summary(feat_info))
            print()
    
    elif args.index is not None:
        # Query specific feature
        print(f"\nQuerying feature {args.index} from {args.model}/{args.layer}...")
        
        feature = client.get_feature(args.model, args.layer, args.index)
        print("\n" + client.format_feature_summary(feature))
        
        if args.text:
            print(f"\n--- Activation analysis for text: '{args.text}' ---")
            activations = client.get_activations(args.model, args.layer, args.index, args.text)
            print(json.dumps(activations, indent=2))
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
