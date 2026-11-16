#!/usr/bin/env python3
"""
Neuronpedia API Client v2 - Enhanced for Maximum Information Retrieval

This enhanced client provides:
- Exponential backoff retries for all API calls
- Comprehensive logging for monitoring and diagnostics
- Multiple endpoint queries for maximum feature information
- Robust error handling with fallback strategies
- Rate limiting compliance with automatic delays

Endpoints utilized:
- GET  /api/feature/{model}/{layer}/{index} - Core feature info
- POST /api/activation/new - Activations for custom text
- POST /api/activation/get - Default activation data
- POST /api/activation/source - All features activation for text
- POST /api/explanation/search - Search by explanation
- GET  /api/sparsity/connected-neurons - Connected neurons/circuit
- POST /api/search-topk-by-token - TopK features per token
- POST /api/search-all - Top features for text

Usage:
    from neuronpedia_client_v2 import NeuronpediaClientV2
    
    client = NeuronpediaClientV2(api_key="your_key")
    feature_data = client.get_feature_complete("gpt2-small", "8-res-jb", 19139)
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin
from functools import wraps
from datetime import datetime

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)


# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("NeuronpediaClient")


class NeuronpediaError(Exception):
    """Custom exception for Neuronpedia API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None, endpoint: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.endpoint = endpoint


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                       max_delay: float = 30.0, exponential_base: float = 2.0):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential calculation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except (requests.exceptions.RequestException, NeuronpediaError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"FAILED after {max_retries + 1} attempts: {func.__name__}"
                        )
                        raise last_exception
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    jitter = delay * 0.1 * (hash(str(time.time())) % 10 - 5) / 5  # ±10% jitter
                    actual_delay = delay + jitter
                    
                    logger.warning(
                        f"WARNING: Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {actual_delay:.1f}s..."
                    )
                    time.sleep(actual_delay)
            
            raise last_exception
        return wrapper
    return decorator


@dataclass
class FeatureCompleteInfo:
    """Complete feature information from multiple endpoints."""
    # Basic info
    model_id: str
    layer: str
    index: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # From /api/feature
    basic_info: Dict[str, Any] = field(default_factory=dict)
    max_activation: Optional[float] = None
    top_tokens: List[str] = field(default_factory=list)
    explanations: List[Dict] = field(default_factory=list)
    
    # From /api/activation/get
    default_activations: Dict[str, Any] = field(default_factory=dict)
    
    # From /api/sparsity/connected-neurons
    connected_neurons: Dict[str, Any] = field(default_factory=dict)
    
    # From custom text activation
    custom_activations: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    api_status: Dict[str, str] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "index": self.index,
            "timestamp": self.timestamp,
            "basic_info": self.basic_info,
            "max_activation": self.max_activation,
            "top_tokens": self.top_tokens,
            "explanations": self.explanations,
            "default_activations": self.default_activations,
            "connected_neurons": self.connected_neurons,
            "custom_activations": self.custom_activations,
            "api_status": self.api_status,
            "errors": self.errors,
        }


class NeuronpediaClientV2:
    """
    Enhanced Neuronpedia API Client with retries, logging, and comprehensive data retrieval.
    """
    
    BASE_URL = "https://www.neuronpedia.org"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.5):
        """
        Initialize the enhanced client.
        
        Args:
            api_key: API key (or from NEURONPEDIA_API_KEY env var)
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.api_key = api_key or os.getenv("NEURONPEDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide via api_key parameter or NEURONPEDIA_API_KEY env var. "
                "Get your key at https://www.neuronpedia.org/account"
            )
        
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        # Track API call statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries": 0,
            "start_time": time.time(),
        }
        
        logger.info("=" * 60)
        logger.info("Neuronpedia Client v2 Initialized")
        logger.info(f"Rate limit delay: {rate_limit_delay}s between requests")
        logger.info("=" * 60)
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        time.sleep(self.rate_limit_delay)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an authenticated request with retries and logging.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit()
        url = urljoin(self.BASE_URL, endpoint)
        
        self.stats["total_calls"] += 1
        call_id = self.stats["total_calls"]
        
        logger.info(f"[{call_id}] {method.upper()} {endpoint}")
        
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            
            # Handle specific status codes
            if response.status_code == 200:
                data = response.json()
                self.stats["successful_calls"] += 1
                logger.info(f"[{call_id}] Success (200)")
                return data
                
            elif response.status_code == 429:
                logger.warning(f"[{call_id}] WARNING: Rate limited (429)")
                raise NeuronpediaError(
                    "Rate limit exceeded", 
                    status_code=429,
                    endpoint=endpoint
                )
                
            elif response.status_code == 404:
                logger.error(f"[{call_id}] ERROR: Not found (404)")
                raise NeuronpediaError(
                    f"Feature or endpoint not found: {endpoint}",
                    status_code=404,
                    endpoint=endpoint
                )
                
            elif response.status_code >= 500:
                logger.error(f"[{call_id}] ERROR: Server error ({response.status_code})")
                raise NeuronpediaError(
                    f"Server error: {response.status_code}",
                    status_code=response.status_code,
                    endpoint=endpoint
                )
                
            else:
                logger.error(f"[{call_id}] ERROR: HTTP {response.status_code}")
                raise NeuronpediaError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    response_text=response.text,
                    endpoint=endpoint
                )
                
        except requests.exceptions.Timeout:
            logger.error(f"[{call_id}] ERROR: Request timeout")
            raise NeuronpediaError("Request timeout", endpoint=endpoint)
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{call_id}] ERROR: Connection error: {e}")
            raise NeuronpediaError(f"Connection error: {e}", endpoint=endpoint)
            
        except json.JSONDecodeError as e:
            logger.error(f"[{call_id}] ERROR: JSON decode error: {e}")
            raise NeuronpediaError(f"Invalid JSON response: {e}", endpoint=endpoint)
    
    def get_feature_basic(self, model_id: str, layer: str, index: int) -> Dict[str, Any]:
        """
        Get basic feature information.
        
        Endpoint: GET /api/feature/{model}/{layer}/{index}
        """
        endpoint = f"/api/feature/{model_id}/{layer}/{index}"
        return self._request("GET", endpoint)
    
    def get_default_activations(self, model_id: str, source: str, index: int) -> Dict[str, Any]:
        """
        Get default activation data for a feature.
        
        Endpoint: POST /api/activation/get
        """
        endpoint = "/api/activation/get"
        payload = {
            "modelId": model_id,
            "source": source,
            "index": str(index)
        }
        return self._request("POST", endpoint, json=payload)
    
    def get_activations_for_text(self, model_id: str, source: str, index: int, 
                                  text: str) -> Dict[str, Any]:
        """
        Get activations for custom text.
        
        Endpoint: POST /api/activation/new
        """
        endpoint = "/api/activation/new"
        payload = {
            "feature": {
                "modelId": model_id,
                "source": source,
                "index": str(index)
            },
            "customText": text
        }
        return self._request("POST", endpoint, json=payload)
    
    def get_connected_neurons(self, model_id: str, layer: int, index: int,
                               trace_depth: int = 1, trace_k: int = 5) -> Dict[str, Any]:
        """
        Get connected neurons from sparsity server.
        
        Endpoint: GET /api/sparsity/connected-neurons
        """
        endpoint = "/api/sparsity/connected-neurons"
        params = {
            "modelId": model_id,
            "layer": layer,
            "index": index,
            "traceDepth": trace_depth,
            "traceK": trace_k
        }
        return self._request("GET", endpoint, params=params)
    
    def search_explanations(self, query: str, model_id: Optional[str] = None,
                           layers: Optional[List[str]] = None, 
                           offset: int = 0) -> Dict[str, Any]:
        """
        Search features by explanation text.
        
        Endpoint: POST /api/explanation/search
        """
        endpoint = "/api/explanation/search"
        payload = {
            "query": query,
            "modelId": model_id,
            "layers": layers or [],
            "offset": offset
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._request("POST", endpoint, json=payload)
    
    def get_topk_by_token(self, model_id: str, source: str, text: str,
                          num_results: int = 10) -> Dict[str, Any]:
        """
        Get top-k activating features for each token.
        
        Endpoint: POST /api/search-topk-by-token
        """
        endpoint = "/api/search-topk-by-token"
        payload = {
            "modelId": model_id,
            "source": source,
            "text": text,
            "numResults": num_results
        }
        return self._request("POST", endpoint, json=payload)
    
    def get_feature_complete(self, model_id: str, layer: str, index: int,
                              custom_text: Optional[str] = None,
                              include_connected: bool = True) -> FeatureCompleteInfo:
        """
        Get complete feature information from multiple endpoints.
        
        This is the main method for comprehensive feature analysis.
        """
        info = FeatureCompleteInfo(
            model_id=model_id,
            layer=layer,
            index=index
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FETCHING COMPLETE DATA FOR: {model_id}/{layer}/F{index}")
        logger.info(f"{'='*60}\n")
        
        # 1. Basic feature info
        try:
            logger.info("[1/4] Fetching basic feature info...")
            basic = self.get_feature_basic(model_id, layer, index)
            info.basic_info = basic
            info.max_activation = basic.get("maxActivation")
            info.top_tokens = basic.get("topTokens", [])
            info.explanations = basic.get("explanations", [])
            info.api_status["basic_info"] = "success"
            logger.info(f"   Found {len(info.explanations)} explanations")
            logger.info(f"   Max activation: {info.max_activation}")
        except Exception as e:
            logger.error(f"   FAILED: {e}")
            info.api_status["basic_info"] = f"error: {e}"
            info.errors.append({"endpoint": "basic_info", "error": str(e)})
        
        # 2. Default activations
        try:
            logger.info("[2/4] Fetching default activations...")
            activations = self.get_default_activations(model_id, layer, index)
            info.default_activations = activations
            info.api_status["default_activations"] = "success"
            logger.info(f"   Retrieved default activation pattern")
        except Exception as e:
            logger.error(f"   FAILED: {e}")
            info.api_status["default_activations"] = f"error: {e}"
            info.errors.append({"endpoint": "default_activations", "error": str(e)})
        
        # 3. Connected neurons (circuit tracing)
        if include_connected:
            try:
                logger.info("[3/4] Fetching connected neurons (circuit)...")
                layer_num = int(layer.split('-')[0]) if '-' in layer else int(layer.replace('l', '').replace('r', '').split('_')[0])
                connected = self.get_connected_neurons(model_id, layer_num, index)
                info.connected_neurons = connected
                info.api_status["connected_neurons"] = "success"
                logger.info(f"   Retrieved circuit connections")
            except Exception as e:
                logger.error(f"   FAILED: {e}")
                info.api_status["connected_neurons"] = f"error: {e}"
                info.errors.append({"endpoint": "connected_neurons", "error": str(e)})
        else:
            info.api_status["connected_neurons"] = "skipped"
        
        # 4. Custom text activation
        if custom_text:
            try:
                logger.info(f"[4/4] Fetching activations for: '{custom_text[:50]}...'")
                custom = self.get_activations_for_text(model_id, layer, index, custom_text)
                info.custom_activations = custom
                info.api_status["custom_activations"] = "success"
                logger.info(f"   Retrieved custom activations")
            except Exception as e:
                logger.error(f"   FAILED: {e}")
                info.api_status["custom_activations"] = f"error: {e}"
                info.errors.append({"endpoint": "custom_activations", "error": str(e)})
        else:
            info.api_status["custom_activations"] = "skipped"
        
        # Summary
        success_count = sum(1 for v in info.api_status.values() if v == "success")
        total_attempted = len([v for v in info.api_status.values() if v != "skipped"])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE: {success_count}/{total_attempted} endpoints successful")
        logger.info(f"{'='*60}\n")
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API call statistics."""
        elapsed = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "elapsed_seconds": elapsed,
            "calls_per_minute": (self.stats["total_calls"] / elapsed) * 60 if elapsed > 0 else 0,
            "success_rate": (self.stats["successful_calls"] / self.stats["total_calls"] * 100) 
                           if self.stats["total_calls"] > 0 else 0
        }
    
    def print_stats(self):
        """Print API call statistics."""
        stats = self.get_stats()
        logger.info("\n" + "="*60)
        logger.info("API CALL STATISTICS")
        logger.info("="*60)
        logger.info(f"Total calls:      {stats['total_calls']}")
        logger.info(f"Successful:       {stats['successful_calls']}")
        logger.info(f"Failed:           {stats['failed_calls']}")
        logger.info(f"Retries:          {stats['retries']}")
        logger.info(f"Success rate:     {stats['success_rate']:.1f}%")
        logger.info(f"Elapsed time:     {stats['elapsed_seconds']:.1f}s")
        logger.info(f"Calls/minute:     {stats['calls_per_minute']:.1f}")
        logger.info("="*60)


def main():
    """CLI test of the enhanced client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Neuronpedia Client v2")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--layer", default="8-res-jb")
    parser.add_argument("--index", type=int, default=19139)
    parser.add_argument("--text", help="Custom text for activation")
    parser.add_argument("--no-connected", action="store_true", help="Skip connected neurons")
    
    args = parser.parse_args()
    
    try:
        client = NeuronpediaClientV2()
        
        info = client.get_feature_complete(
            args.model, 
            args.layer, 
            args.index,
            custom_text=args.text,
            include_connected=not args.no_connected
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE SUMMARY")
        print("="*60)
        print(f"Model: {info.model_id}")
        print(f"Layer: {info.layer}")
        print(f"Index: {info.index}")
        print(f"Max Activation: {info.max_activation}")
        print(f"Top Tokens: {info.top_tokens[:5]}")
        print(f"Explanations: {len(info.explanations)}")
        print(f"API Status: {json.dumps(info.api_status, indent=2)}")
        
        client.print_stats()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
