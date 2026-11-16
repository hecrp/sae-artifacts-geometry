#!/usr/bin/env python3
"""
Neuronpedia API Client v3 — Geometry SAE Project
=================================================
Complete rewrite focused on extracting MAXIMUM useful data for 
anti-aligned feature analysis. Changes from v2:

1. Single GET /api/feature call gets ALL data (explanations, activations,
   logit weights, cosine similarity, histograms, UMAP, etc.) — no need
   for separate /api/activation/get calls.
2. Proper extraction of ALL fields the API returns (v2 missed many).
3. New endpoints: search-all, explanation/search, steer (experimental).
4. S3 bulk download support for full-SAE analysis.
5. Structured output with FeatureProfile dataclass for analysis.
6. Cleaner code, less noise in logs.

Author: Geometry SAE Project
Date: March 2026
"""

import os
import sys
import json
import time
import gzip
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# --- Load .env ---
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# --- Logging ---
log = logging.getLogger("neuronpedia")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)-5s %(message)s', '%H:%M:%S'))
    log.addHandler(h)
    log.setLevel(logging.INFO)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ExplanationInfo:
    """A single auto-interp explanation for a feature."""
    description: str
    model_name: str = ""
    type_name: str = ""
    scores: List[Dict] = field(default_factory=list)
    
    @property
    def best_score(self) -> Optional[float]:
        if not self.scores:
            return None
        return max(s.get("value", 0) for s in self.scores)


@dataclass 
class ActivationExample:
    """A single activation example from Neuronpedia."""
    tokens: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    max_value: float = 0.0
    max_token_idx: int = 0
    dfa_values: List[float] = field(default_factory=list)
    loss_values: Any = None
    bin_min: float = 0.0
    bin_max: float = 0.0


@dataclass
class FeatureProfile:
    """
    Complete profile of a Neuronpedia feature — everything we can extract.
    This is what we use for analysis.
    """
    # Identity
    model_id: str = ""
    sae_id: str = ""       # e.g. "8-res-jb"
    index: int = 0
    layer_num: int = -1    # parsed from sae_id
    
    # Our geometry project data (filled from registry)
    cos_ed: Optional[float] = None    # encoder-decoder cosine
    sigma: Optional[float] = None     # sigma deviation
    enc_norm: Optional[float] = None  # encoder norm
    
    # Explanations
    explanations: List[ExplanationInfo] = field(default_factory=list)
    
    # Activation statistics
    frac_nonzero: float = 0.0       # what fraction of tokens activate this feature
    max_act_approx: float = 0.0     # approximate maximum activation
    
    # Logit weight analysis (decoder direction projected onto unembedding)
    pos_logit_tokens: List[str] = field(default_factory=list)   # top 10 promoted tokens
    pos_logit_values: List[float] = field(default_factory=list)
    neg_logit_tokens: List[str] = field(default_factory=list)   # top 10 suppressed tokens
    neg_logit_values: List[float] = field(default_factory=list)
    
    # Decoder cosine similarity to other features
    top_similar_indices: List[int] = field(default_factory=list)   # top 25 by cos(D_i, D_j)
    top_similar_values: List[float] = field(default_factory=list)
    
    # Correlated features (by co-activation)
    corr_feature_indices: List[int] = field(default_factory=list)
    corr_feature_pearson: List[float] = field(default_factory=list)
    
    # Correlated neurons (MLP neurons)
    corr_neuron_indices: List[int] = field(default_factory=list)
    corr_neuron_pearson: List[float] = field(default_factory=list)
    
    # Neuron alignment
    neuron_align_indices: List[int] = field(default_factory=list)
    neuron_align_values: List[float] = field(default_factory=list)
    
    # Histograms
    freq_hist_heights: List[float] = field(default_factory=list)
    freq_hist_values: List[float] = field(default_factory=list)
    logit_hist_heights: List[float] = field(default_factory=list)
    logit_hist_values: List[float] = field(default_factory=list)
    
    # UMAP embedding
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    umap_cluster: Optional[int] = None
    
    # Activation examples (full detail)
    activation_examples: List[ActivationExample] = field(default_factory=list)
    n_examples: int = 0
    
    # Steering
    has_vector: bool = False
    
    # Metadata  
    query_timestamp: str = ""
    api_success: bool = False
    error: str = ""
    
    def is_dead(self) -> bool:
        """Feature never fires in the dataset."""
        return self.frac_nonzero == 0
    
    def is_rare(self, threshold=1e-5) -> bool:
        """Feature fires but extremely rarely."""
        return 0 < self.frac_nonzero < threshold
    
    @property
    def explanation_text(self) -> str:
        """First explanation text, or 'NO EXPLANATION'."""
        if self.explanations:
            return self.explanations[0].description
        return "NO EXPLANATION"
    
    @property
    def top_similar_summary(self) -> List[Tuple[int, float]]:
        """Top 5 most similar features by decoder cosine."""
        return list(zip(self.top_similar_indices[:5], self.top_similar_values[:5]))
    
    def to_dict(self) -> Dict:
        """Serialize to dict (for JSON export)."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list) and v and isinstance(v[0], (ExplanationInfo, ActivationExample)):
                d[k] = [asdict(x) if hasattr(x, '__dataclass_fields__') else x for x in v]
            else:
                d[k] = v
        return d
    
    def summary_line(self) -> str:
        """One-line summary for tables."""
        dead = "DEAD" if self.is_dead() else f"frac={self.frac_nonzero:.2e}"
        expl = self.explanation_text[:50]
        cos = f"cos={self.cos_ed:.3f}" if self.cos_ed is not None else "cos=?"
        return f"F{self.index:>5d} L{self.layer_num:>2d} {cos} {dead:>15s} | {expl}"


# ============================================================
# API CLIENT
# ============================================================

class NeuronpediaV3:
    """
    Lean Neuronpedia client. One GET call per feature gives us everything.
    """
    
    BASE = "https://www.neuronpedia.org"
    
    # SAE ID mapping per model
    # Neuronpedia SAE ID mappings (verified March 2026)
    # NOTE: Llama SAEs on Neuronpedia are LlamaScope (Chen Delong et al.),
    # NOT the LXR SAEs (Rajamanoharan et al.) used in our geometry project.
    # The LXR feature indices do NOT correspond to LlamaScope indices.
    SAE_IDS = {
        "gpt2-small": lambda layer: f"{layer}-res-jb",
        "gemma-2-2b": lambda layer: f"{layer}-gemmascope-res-16k",
        # LlamaScope variants (4 types per layer, layers 0-31):
        "llama-3.1-8b-res-32x": lambda layer: f"{layer}-llamascope-res-32x",
        "llama-3.1-8b-res-8x": lambda layer: f"{layer}-llamascope-slimscope-res-8x",
        "llama-3.1-8b-attn-32x": lambda layer: f"{layer}-llamascope-attn-32x",
        "llama-3.1-8b-mlp-32x": lambda layer: f"{layer}-llamascope-mlp-32x",
    }
    
    def __init__(self, api_key: str = None, delay: float = 0.3):
        self.api_key = api_key or os.getenv("NEURONPEDIA_API_KEY", "")
        self.delay = delay
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
        self.session.headers["Content-Type"] = "application/json"
        self.session.headers["Accept"] = "application/json"
        self._call_count = 0
        self._errors = 0
        self._start = time.time()
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET request with retry."""
        time.sleep(self.delay)
        url = f"{self.BASE}{endpoint}"
        self._call_count += 1
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    log.debug(f"GET {endpoint} -> {r.status_code}")
                    self._errors += 1
                    return {}
            except Exception as e:
                if attempt == 2:
                    log.error(f"GET {endpoint} failed: {e}")
                    self._errors += 1
                    return {}
                time.sleep(2 ** attempt)
        return {}
    
    def _post(self, endpoint: str, payload: dict) -> dict:
        """POST request with retry."""
        time.sleep(self.delay)
        url = f"{self.BASE}{endpoint}"
        self._call_count += 1
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    log.debug(f"POST {endpoint} -> {r.status_code}: {r.text[:200]}")
                    self._errors += 1
                    return {}
            except Exception as e:
                if attempt == 2:
                    log.error(f"POST {endpoint} failed: {e}")
                    self._errors += 1
                    return {}
                time.sleep(2 ** attempt)
        return {}
    
    # ----------------------------------------------------------
    # CORE: Get complete feature profile (single API call!)
    # ----------------------------------------------------------
    
    def get_feature(self, model: str, sae_id: str, index: int) -> FeatureProfile:
        """
        Query a single feature. Returns a FeatureProfile with ALL available data.
        
        The GET /api/feature/{model}/{sae_id}/{index} endpoint returns everything:
        explanations, activation examples, logit weights, cosine similarities,
        histograms, UMAP coordinates, etc.
        """
        data = self._get(f"/api/feature/{model}/{sae_id}/{index}")
        if not data:
            return FeatureProfile(
                model_id=model, sae_id=sae_id, index=index,
                error="API returned empty", query_timestamp=datetime.now().isoformat()
            )
        
        # Parse layer number from sae_id
        layer_num = -1
        try:
            if sae_id[0].isdigit():
                layer_num = int(sae_id.split('-')[0])
            elif sae_id.startswith('l'):
                layer_num = int(sae_id[1:].split('r')[0])
        except:
            pass
        
        # Parse explanations
        explanations = []
        for exp in data.get("explanations", []):
            explanations.append(ExplanationInfo(
                description=exp.get("description", ""),
                model_name=exp.get("explanationModelName", ""),
                type_name=exp.get("typeName", ""),
                scores=exp.get("scores", [])
            ))
        
        # Parse activation examples
        act_examples = []
        for act in data.get("activations", []):
            act_examples.append(ActivationExample(
                tokens=act.get("tokens", []),
                values=act.get("values", []),
                max_value=act.get("maxValue", 0),
                max_token_idx=act.get("maxValueTokenIndex", 0),
                dfa_values=act.get("dfaValues", []),
                loss_values=act.get("lossValues"),
                bin_min=act.get("binMin", 0),
                bin_max=act.get("binMax", 0),
            ))
        
        profile = FeatureProfile(
            model_id=model,
            sae_id=sae_id,
            index=index,
            layer_num=layer_num,
            
            explanations=explanations,
            
            frac_nonzero=data.get("frac_nonzero", 0) or 0,
            max_act_approx=data.get("maxActApprox", 0) or data.get("maxActivation", 0) or 0,
            
            pos_logit_tokens=data.get("pos_str", []),
            pos_logit_values=data.get("pos_values", []),
            neg_logit_tokens=data.get("neg_str", []),
            neg_logit_values=data.get("neg_values", []),
            
            top_similar_indices=data.get("topkCosSimIndices", []),
            top_similar_values=data.get("topkCosSimValues", []),
            
            corr_feature_indices=data.get("correlated_features_indices", []),
            corr_feature_pearson=data.get("correlated_features_pearson", []),
            
            corr_neuron_indices=data.get("correlated_neurons_indices", []),
            corr_neuron_pearson=data.get("correlated_neurons_pearson", []),
            
            neuron_align_indices=data.get("neuron_alignment_indices", []),
            neuron_align_values=data.get("neuron_alignment_values", []),
            
            freq_hist_heights=data.get("freq_hist_data_bar_heights", []),
            freq_hist_values=data.get("freq_hist_data_bar_values", []),
            logit_hist_heights=data.get("logits_hist_data_bar_heights", []),
            logit_hist_values=data.get("logits_hist_data_bar_values", []),
            
            umap_x=data.get("umap_x"),
            umap_y=data.get("umap_y"),
            umap_cluster=data.get("umap_cluster"),
            
            activation_examples=act_examples,
            n_examples=len(act_examples),
            
            has_vector=data.get("hasVector", False),
            
            query_timestamp=datetime.now().isoformat(),
            api_success=True,
        )
        
        return profile
    
    # ----------------------------------------------------------
    # SEARCH: Find features by text / explanation
    # ----------------------------------------------------------
    
    def search_features_by_text(self, text: str, model: str = "gpt2-small",
                                 source: str = None, num_results: int = 10) -> List[Dict]:
        """
        POST /api/search-all — Find top activating features for input text.
        Like the Neuronpedia search UI.
        """
        payload = {
            "modelId": model,
            "text": text,
            "selectedLayers": [],
        }
        if source:
            payload["sourceSetName"] = source
        
        data = self._post("/api/search-all", payload)
        if not data:
            return []
        
        # The response has a "results" array with feature activations
        results = data if isinstance(data, list) else data.get("results", [])
        return results[:num_results]
    
    def search_topk_by_token(self, text: str, model: str = "gpt2-small",
                              source: str = "8-res-jb", k: int = 10) -> Dict:
        """
        POST /api/search-topk-by-token — Top K features per token.
        Returns per-token breakdown of which features fire.
        """
        payload = {
            "modelId": model,
            "source": source,  
            "text": text,
            "numResults": k,
        }
        return self._post("/api/search-topk-by-token", payload)
    
    def search_explanations(self, query: str, model: str = "gpt2-small",
                            sae_ids: List[str] = None) -> List[Dict]:
        """
        POST /api/explanation/search — Find features by explanation text.
        Useful for finding semantically similar features.
        """
        payload = {"query": query}
        if model:
            payload["modelId"] = model
        if sae_ids:
            payload["layers"] = sae_ids
            
        return self._post("/api/explanation/search", payload)
    
    def search_explanations_all(self, query: str) -> List[Dict]:
        """
        POST /api/explanation/search-all — Search across ALL models.
        """
        return self._post("/api/explanation/search-all", {"query": query})
    
    # ----------------------------------------------------------
    # CUSTOM TEXT ACTIVATION
    # ----------------------------------------------------------
    
    def activate_on_text(self, model: str, sae_id: str, index: int, 
                          text: str) -> Dict:
        """
        POST /api/activation/new — Run custom text through the model and
        get activation values for a specific feature.
        """
        payload = {
            "feature": {
                "modelId": model,
                "source": sae_id,
                "index": str(index),
            },
            "customText": text,
        }
        return self._post("/api/activation/new", payload)
    
    # ----------------------------------------------------------
    # BATCH: Query multiple features efficiently
    # ----------------------------------------------------------
    
    def get_features_batch(self, model: str, sae_id: str, 
                           indices: List[int],
                           registry: Dict = None) -> List[FeatureProfile]:
        """
        Query multiple features sequentially with progress reporting.
        Optionally enrich with registry data (cos_ed, sigma, enc_norm).
        """
        profiles = []
        total = len(indices)
        
        for i, idx in enumerate(indices):
            if (i + 1) % 5 == 0 or i == 0:
                log.info(f"  [{i+1}/{total}] Querying F{idx}...")
            
            p = self.get_feature(model, sae_id, idx)
            
            # Enrich from registry if available
            if registry and idx in registry:
                reg = registry[idx]
                p.cos_ed = reg.get("cos_ed")
                p.sigma = reg.get("sigma")
                p.enc_norm = reg.get("enc_norm")
            
            profiles.append(p)
        
        success = sum(1 for p in profiles if p.api_success)
        log.info(f"  Batch complete: {success}/{total} successful")
        return profiles
    
    # ----------------------------------------------------------
    # S3 BULK DATA
    # ----------------------------------------------------------
    
    def download_s3_explanations(self, model: str, sae_id: str, 
                                  output_dir: Path) -> Path:
        """
        Download bulk explanation data from S3.
        URL pattern: neuronpedia-datasets.s3.amazonaws.com/v1/{model}/{sae_id}/explanations/
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = f"https://neuronpedia-datasets.s3.amazonaws.com/v1/{model}/{sae_id}"
        all_data = []
        
        # Try downloading batches
        for batch_num in range(20):  # Usually < 10 batches
            url = f"{base_url}/explanations/batch-{batch_num}.jsonl.gz"
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    content = gzip.decompress(r.content).decode('utf-8')
                    lines = [json.loads(l) for l in content.strip().split('\n') if l.strip()]
                    all_data.extend(lines)
                    log.info(f"  S3 batch-{batch_num}: {len(lines)} explanations")
                elif r.status_code == 404 or r.status_code == 403:
                    break  # No more batches
                else:
                    log.warning(f"  S3 batch-{batch_num}: HTTP {r.status_code}")
                    break
            except Exception as e:
                log.error(f"  S3 batch-{batch_num} error: {e}")
                break
        
        if all_data:
            out_file = output_dir / f"explanations_{model}_{sae_id}.jsonl"
            with open(out_file, 'w') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
            log.info(f"  Saved {len(all_data)} explanations to {out_file}")
            return out_file
        
        return None
    
    def download_s3_features(self, model: str, sae_id: str,
                              output_dir: Path) -> Path:
        """Download bulk feature metadata from S3."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = f"https://neuronpedia-datasets.s3.amazonaws.com/v1/{model}/{sae_id}"
        all_data = []
        
        for batch_num in range(20):
            url = f"{base_url}/features/batch-{batch_num}.jsonl.gz"
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200:
                    content = gzip.decompress(r.content).decode('utf-8')
                    lines = [json.loads(l) for l in content.strip().split('\n') if l.strip()]
                    all_data.extend(lines)
                    log.info(f"  S3 features batch-{batch_num}: {len(lines)} features")
                elif r.status_code in (404, 403):
                    break
                else:
                    break
            except Exception as e:
                log.error(f"  S3 features batch-{batch_num} error: {e}")
                break
        
        if all_data:
            out_file = output_dir / f"features_{model}_{sae_id}.jsonl"
            with open(out_file, 'w') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
            log.info(f"  Saved {len(all_data)} features to {out_file}")
            return out_file
        
        return None
    
    # ----------------------------------------------------------
    # STATS
    # ----------------------------------------------------------
    
    def stats_summary(self) -> str:
        elapsed = time.time() - self._start
        rate = (self._call_count / elapsed * 60) if elapsed > 0 else 0
        return (f"API calls: {self._call_count} | errors: {self._errors} | "
                f"elapsed: {elapsed:.0f}s | rate: {rate:.0f}/min")


# ============================================================
# CONVENIENCE: Get SAE ID for a model+layer
# ============================================================

def sae_id_for(model: str, layer: int) -> str:
    """Get SAE source ID for a model and layer number."""
    if model == "gpt2-small":
        return f"{layer}-res-jb"
    elif model in ("gemma-2-2b", "gemma-2b"):
        return f"{layer}-gemmascope-res-16k"
    elif model in ("llama3.1-8b", "llama-3.1-8b"):
        return f"l{layer}r_8x"
    else:
        raise ValueError(f"Unknown model: {model}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neuronpedia Client v3")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--sae", default="8-res-jb", help="SAE source ID")
    parser.add_argument("--index", type=int, default=19139)
    parser.add_argument("--search", help="Search features by text")
    parser.add_argument("--explain-search", help="Search by explanation")
    parser.add_argument("--topk", help="Top-K features per token for text")
    parser.add_argument("--activate", help="Custom text activation")
    args = parser.parse_args()
    
    client = NeuronpediaV3()
    
    if args.search:
        results = client.search_features_by_text(args.search, args.model)
        print(json.dumps(results, indent=2)[:3000])
    elif args.explain_search:
        results = client.search_explanations(args.explain_search, args.model)
        print(json.dumps(results, indent=2)[:3000])
    elif args.topk:
        results = client.search_topk_by_token(args.topk, args.model, args.sae)
        print(json.dumps(results, indent=2)[:3000])
    elif args.activate:
        results = client.activate_on_text(args.model, args.sae, args.index, args.activate)
        print(json.dumps(results, indent=2)[:3000])
    else:
        profile = client.get_feature(args.model, args.sae, args.index)
        print(f"\n{'='*60}")
        print(f"FEATURE PROFILE: {profile.model_id} / {profile.sae_id} / F{profile.index}")
        print(f"{'='*60}")
        print(f"  Explanation:   {profile.explanation_text}")
        print(f"  Dead:          {profile.is_dead()}")
        print(f"  frac_nonzero:  {profile.frac_nonzero:.2e}")
        print(f"  maxAct:        {profile.max_act_approx}")
        print(f"  #Examples:     {profile.n_examples}")
        print(f"  #Explanations: {len(profile.explanations)}")
        print(f"  Has vector:    {profile.has_vector}")
        print(f"  UMAP:          ({profile.umap_x}, {profile.umap_y}) cluster={profile.umap_cluster}")
        print(f"\n  Top 10 PROMOTED tokens (decoder -> logits):")
        for t, v in zip(profile.pos_logit_tokens[:10], profile.pos_logit_values[:10]):
            print(f"    {v:+.4f}  '{t}'")
        print(f"\n  Top 10 SUPPRESSED tokens:")
        for t, v in zip(profile.neg_logit_tokens[:10], profile.neg_logit_values[:10]):
            print(f"    {v:+.4f}  '{t}'")
        print(f"\n  Top 5 similar features (decoder cosine):")
        for idx, val in profile.top_similar_summary:
            print(f"    F{idx}: cos={val:.4f}")
        print(f"\n  Correlated neurons: {list(zip(profile.corr_neuron_indices, profile.corr_neuron_pearson))}")
        print(f"  Neuron alignment:  {list(zip(profile.neuron_align_indices, profile.neuron_align_values))}")
        print(f"\n  {client.stats_summary()}")
