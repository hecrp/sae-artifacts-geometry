#!/usr/bin/env python3
"""
Download bulk explanations from S3 for GPT-2 Small layer 8 (res-jb SAE),
analyze anti-aligned and parent features.
"""

import json
import gzip
import os
import sys
import urllib.request
import io
from collections import Counter

# Configuration
BASE_URL = "https://neuronpedia-datasets.s3.amazonaws.com/v1/gpt2-small/8-res-jb/explanations/batch-{N}.jsonl.gz"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "supplement4")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "s3_explanations_analysis.json")

# Feature indices of interest. All must be layer-8 indices, because BASE_URL points at the
# layer-8 export. (An earlier version also listed features from layers 4-11 here, which
# retrieved explanations for unrelated layer-8 features; see docs/ERRATA.md, E6.)
ANTI_ALIGNED = [19139, 18573, 11969, 20804]   # cos(E, D) < 0 at <= mu - 6 sigma in layer 8
RANDOM_L8 = [5977, 22561, 6165, 3696]        # random layer-8 features (cos near the layer mean)
PARENTS = [17487, 21174, 16152]
ALL_TARGETS = set(ANTI_ALIGNED + RANDOM_L8 + PARENTS)

def download_batch(n):
    """Download and decompress a single batch file. Returns list of parsed JSON objects."""
    url = BASE_URL.format(N=n)
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=60) as response:
            compressed = response.read()
        decompressed = gzip.decompress(compressed)
        lines = decompressed.decode('utf-8').strip().split('\n')
        records = []
        for line in lines:
            if line.strip():
                records.append(json.loads(line))
        return records
    except urllib.error.HTTPError as e:
        if e.code == 404 or e.code == 403:
            return None  # No more batches
        print(f"  HTTP Error {e.code}: {e}")
        return None
    except Exception as e:
        print(f"  Error downloading batch {n}: {e}")
        return None

def extract_index_and_explanation(record):
    """Extract feature index and explanation from a record, handling various formats."""
    # Try various keys for index
    index = None
    for key in ['index', 'feature_index', 'feature', 'idx']:
        if key in record:
            val = record[key]
            if isinstance(val, (int, float)):
                index = int(val)
                break
            elif isinstance(val, str) and val.isdigit():
                index = int(val)
                break
            elif isinstance(val, dict):
                for subkey in ['index', 'feature_index', 'idx']:
                    if subkey in val:
                        index = int(val[subkey])
                        break
    
    # Try various keys for explanation
    explanation = None
    for key in ['description', 'explanation', 'text', 'label']:
        if key in record:
            val = record[key]
            if isinstance(val, str):
                explanation = val
                break
            elif isinstance(val, dict):
                for subkey in ['description', 'text', 'explanation']:
                    if subkey in val:
                        explanation = val[subkey]
                        break
    
    return index, explanation

def main():
    print("=" * 70)
    print("S3 Explanations Analysis for GPT-2 Small Layer 8 (res-jb SAE)")
    print("=" * 70)
    
    # Store explanations for target features
    target_explanations = {}
    
    total_features = 0
    total_with_explanation = 0
    total_without_explanation = 0
    
    # Let's first peek at the structure of batch-0
    print("\nStep 1: Downloading and examining data structure...")
    
    batch_n = 0
    consecutive_failures = 0
    max_consecutive_failures = 3
    sample_record_printed = False
    
    while consecutive_failures < max_consecutive_failures:
        print(f"  Downloading batch-{batch_n}.jsonl.gz ...", end=" ", flush=True)
        records = download_batch(batch_n)
        
        if records is None:
            print("NOT FOUND / ERROR")
            consecutive_failures += 1
            batch_n += 1
            continue
        
        consecutive_failures = 0
        print(f"OK ({len(records)} records)")
        
        # Print sample on first successful batch
        if not sample_record_printed and records:
            print(f"\n  Sample record keys: {list(records[0].keys())}")
            sample_json = json.dumps(records[0], indent=2, default=str)
            # Truncate if too long
            if len(sample_json) > 500:
                sample_json = sample_json[:500] + "..."
            print(f"  Sample record:\n{sample_json}\n")
            sample_record_printed = True
        
        for record in records:
            index, explanation = extract_index_and_explanation(record)
            
            total_features += 1
            has_explanation = bool(explanation and explanation.strip())
            
            if has_explanation:
                total_with_explanation += 1
            else:
                total_without_explanation += 1
            
            if index is not None and index in ALL_TARGETS:
                group = "anti-aligned" if index in ANTI_ALIGNED else ("random_l8" if index in RANDOM_L8 else "parent")
                target_explanations[index] = {
                    'index': index,
                    'group': group,
                    'explanation': explanation or '',
                    'has_explanation': has_explanation,
                    'raw_record': record
                }
                print(f"    >>> FOUND target feature {index} ({group}): {(explanation or 'NO EXPLANATION')[:120]}")
        
        batch_n += 1
    
    total_batches = batch_n - consecutive_failures
    
    print(f"\n{'=' * 70}")
    print(f"Download complete. {total_batches} batches downloaded, {total_features} total records.")
    print(f"{'=' * 70}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS OF TARGET FEATURES")
    print("=" * 70)
    
    # Anti-aligned features
    print("\n--- ANTI-ALIGNED FEATURES ---")
    anti_aligned_found = []
    anti_aligned_missing = []
    anti_aligned_explanations = {}
    
    for idx in sorted(ANTI_ALIGNED):
        if idx in target_explanations:
            entry = target_explanations[idx]
            expl = entry['explanation']
            has = entry['has_explanation']
            anti_aligned_explanations[idx] = expl
            if has:
                anti_aligned_found.append(idx)
                print(f"  Feature {idx}: \"{expl}\"")
            else:
                anti_aligned_missing.append(idx)
                print(f"  Feature {idx}: [EMPTY/NO EXPLANATION]")
        else:
            anti_aligned_missing.append(idx)
            print(f"  Feature {idx}: [NOT FOUND IN DATA]")
    
    # Parent features
    print("\n--- PARENT FEATURES ---")
    parent_found = []
    parent_missing = []
    parent_explanations = {}
    
    for idx in sorted(PARENTS):
        if idx in target_explanations:
            entry = target_explanations[idx]
            expl = entry['explanation']
            has = entry['has_explanation']
            parent_explanations[idx] = expl
            if has:
                parent_found.append(idx)
                print(f"  Feature {idx}: \"{expl}\"")
            else:
                parent_missing.append(idx)
                print(f"  Feature {idx}: [EMPTY/NO EXPLANATION]")
        else:
            parent_missing.append(idx)
            print(f"  Feature {idx}: [NOT FOUND IN DATA]")
    
    # Pattern analysis
    print("\n--- PATTERN ANALYSIS ---")
    
    all_anti_expls = [e for e in anti_aligned_explanations.values() if e and e.strip()]
    all_parent_expls = [e for e in parent_explanations.values() if e and e.strip()]
    
    if all_anti_expls:
        # Word frequency analysis
        all_anti_words = []
        for expl in all_anti_expls:
            words = expl.lower().replace(',', ' ').replace('.', ' ').replace('"', ' ').replace("'", ' ').split()
            all_anti_words.extend(words)
        
        stop_words = {'the', 'a', 'an', 'in', 'of', 'to', 'and', 'or', 'is', 'it', 'for', 'on', 'with', 'at', 'by', 
                      'from', 'as', 'that', 'this', 'are', 'was', 'be', 'has', 'have', 'had', 'do', 'does', 'did', 
                      'will', 'would', 'could', 'should', 'may', 'might', 'can', 'not', 'no', 'but', 'if', 'when', 
                      'where', 'which', 'who', 'whom', 'whose', 'how', 'what', 'why', 'its', 'their', 'they', 'them',
                      'these', 'those', 'such', 'than', 'other', 'more', 'also', 'been', 'being', 'were', 'about'}
        
        word_counts = Counter(all_anti_words)
        filtered_counts = {w: c for w, c in word_counts.items() if w not in stop_words and len(w) > 2}
        top_words = sorted(filtered_counts.items(), key=lambda x: -x[1])[:25]
        
        print(f"\n  Found {len(all_anti_expls)} anti-aligned features with explanations")
        print(f"  Top recurring words in anti-aligned explanations:")
        for word, count in top_words:
            print(f"    '{word}': {count}")
        
        # Look for common themes
        print(f"\n  All anti-aligned explanations:")
        for idx in sorted(anti_aligned_explanations.keys()):
            expl = anti_aligned_explanations[idx]
            if expl:
                print(f"    [{idx}] {expl}")
    else:
        print("  No anti-aligned features had explanations to analyze.")
    
    if all_parent_expls:
        print(f"\n  All parent explanations:")
        for idx in sorted(parent_explanations.keys()):
            expl = parent_explanations[idx]
            if expl:
                print(f"    [{idx}] {expl}")
    
    # Summary stats
    print(f"\n--- SUMMARY STATS ---")
    print(f"  Total features in dataset: {total_features}")
    print(f"  Features with explanations: {total_with_explanation} ({100*total_with_explanation/max(total_features,1):.1f}%)")
    print(f"  Features without explanations: {total_without_explanation} ({100*total_without_explanation/max(total_features,1):.1f}%)")
    print(f"  Anti-aligned with explanation: {len(anti_aligned_found)}/{len(ANTI_ALIGNED)}")
    print(f"  Anti-aligned missing/empty: {len(anti_aligned_missing)}/{len(ANTI_ALIGNED)}")
    print(f"  Parent with explanation: {len(parent_found)}/{len(PARENTS)}")
    print(f"  Parent missing/empty: {len(parent_missing)}/{len(PARENTS)}")
    
    # Build output JSON
    result = {
        "metadata": {
            "model": "gpt2-small",
            "layer": 8,
            "sae": "res-jb",
            "source": "S3 bulk explanations",
            "base_url": BASE_URL,
            "total_batches_downloaded": total_batches,
            "total_features_in_dataset": total_features,
            "total_with_explanations": total_with_explanation,
            "total_without_explanations": total_without_explanation,
            "explanation_coverage_pct": round(100 * total_with_explanation / max(total_features, 1), 2)
        },
        "anti_aligned_features": {
            "indices": sorted(ANTI_ALIGNED),
            "found_with_explanation": sorted(anti_aligned_found),
            "missing_or_empty": sorted(anti_aligned_missing),
            "explanations": {str(k): v for k, v in sorted(anti_aligned_explanations.items())},
            "coverage": f"{len(anti_aligned_found)}/{len(ANTI_ALIGNED)}"
        },
        "parent_features": {
            "indices": sorted(PARENTS),
            "found_with_explanation": sorted(parent_found),
            "missing_or_empty": sorted(parent_missing),
            "explanations": {str(k): v for k, v in sorted(parent_explanations.items())},
            "coverage": f"{len(parent_found)}/{len(PARENTS)}"
        },
        "detailed_records": {},
        "pattern_analysis": {
            "anti_aligned_common_words": [],
            "observations": []
        }
    }
    
    # Add top words
    if all_anti_expls:
        result["pattern_analysis"]["anti_aligned_common_words"] = [
            {"word": w, "count": c} for w, c in top_words
        ]
    
    # Add observations
    observations = []
    if len(anti_aligned_found) == 0:
        observations.append("None of the anti-aligned features have explanations in the S3 bulk data.")
    elif len(anti_aligned_found) < len(ANTI_ALIGNED):
        observations.append(f"Only {len(anti_aligned_found)}/{len(ANTI_ALIGNED)} anti-aligned features have explanations.")
    else:
        observations.append(f"All {len(ANTI_ALIGNED)} anti-aligned features have explanations.")
    
    if len(parent_found) == 0:
        observations.append("None of the parent features have explanations in the S3 bulk data.")
    elif len(parent_found) < len(PARENTS):
        observations.append(f"Only {len(parent_found)}/{len(PARENTS)} parent features have explanations.")
    else:
        observations.append(f"All {len(PARENTS)} parent features have explanations.")
    
    result["pattern_analysis"]["observations"] = observations
    
    # Full records for target features
    for idx in sorted(ALL_TARGETS):
        if idx in target_explanations:
            entry = target_explanations[idx]
            # Make raw_record serializable
            raw = entry['raw_record']
            result["detailed_records"][str(idx)] = {
                "index": idx,
                "group": entry['group'],
                "explanation": entry['explanation'],
                "has_explanation": entry['has_explanation'],
                "raw_record": raw
            }
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("Done!")

if __name__ == "__main__":
    main()
