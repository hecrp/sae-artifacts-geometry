#!/usr/bin/env python3
"""
Feature registry for the Neuronpedia queries (Supplement 4).

Lists the negative-cosine features (cos(E, D) < 0) that appear in
results/phase0/extreme_features.csv. That table keeps only the 10 most extreme
features per layer, so this registry is a subset of all negative-cosine features
(see docs/ERRATA.md, E1):

- GPT-2 Small (ReLU+L1): 86 listed here; 196 in total, 35 at <= mu - 6 sigma
- Gemma 2B (JumpReLU): 1 listed here; 1 in total
- Llama 3.1 8B (JumpReLU): 86 listed here; 232 in total, 29 at <= mu - 6 sigma

Last updated: March 2026 (docstring revised November 2026)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    """Specification for an anti-aligned feature."""
    index: int
    cos_ed: float
    sigma: float
    encoder_norm: float
    layer: int
    model: str
    hypothesis: Optional[str] = None
    notes: Optional[str] = None


# ============================================================================
# GPT-2 SMALL (ReLU+L1 SAEs by Joseph Bloom)
# Source: jbloom/GPT2-Small-SAEs (gpt2-small-res-jb)
# SAE Pattern: {layer}-res-jb
# Negative-cosine features in the top-10-per-layer table: 86
# ============================================================================

GPT2_SMALL_FEATURES: Dict[int, List[FeatureSpec]] = {
    0: [  # 4 features
        FeatureSpec(6523, -0.0609, -2.58, 0.210, 0, "gpt2-small", "?", "Low sigma"),
        FeatureSpec(16746, -0.0070, -2.36, 0.205, 0, "gpt2-small", "?", "Low sigma"),
        FeatureSpec(7436, -0.0034, -2.34, 0.204, 0, "gpt2-small", "?", "Low sigma"),
        FeatureSpec(3198, -0.0001, -2.33, 0.216, 0, "gpt2-small", "?", "Low sigma"),
    ],
    1: [  # 10 features
        FeatureSpec(158, -0.0726, -3.12, 0.194, 1, "gpt2-small", "?", ""),
        FeatureSpec(8397, -0.0668, -3.08, 0.194, 1, "gpt2-small", "?", ""),
        FeatureSpec(15054, -0.0623, -3.05, 0.165, 1, "gpt2-small", "?", ""),
        FeatureSpec(5233, -0.0587, -3.02, 0.179, 1, "gpt2-small", "?", ""),
        FeatureSpec(24012, -0.0583, -3.02, 0.293, 1, "gpt2-small", "?", ""),
        FeatureSpec(15143, -0.0509, -2.96, 0.284, 1, "gpt2-small", "?", ""),
        FeatureSpec(20067, -0.0480, -2.94, 0.352, 1, "gpt2-small", "?", ""),
        FeatureSpec(15232, -0.0449, -2.92, 0.283, 1, "gpt2-small", "?", ""),
        FeatureSpec(13935, -0.0417, -2.90, 0.159, 1, "gpt2-small", "?", ""),
        FeatureSpec(20599, -0.0358, -2.86, 0.325, 1, "gpt2-small", "?", ""),
    ],
    2: [  # 10 features
        FeatureSpec(15228, -0.0746, -3.26, 0.149, 2, "gpt2-small", "?", ""),
        FeatureSpec(24473, -0.0648, -3.18, 0.331, 2, "gpt2-small", "?", ""),
        FeatureSpec(21416, -0.0586, -3.13, 0.209, 2, "gpt2-small", "?", ""),
        FeatureSpec(14628, -0.0403, -2.98, 0.343, 2, "gpt2-small", "?", ""),
        FeatureSpec(11964, -0.0403, -2.98, 0.343, 2, "gpt2-small", "?", ""),
        FeatureSpec(21071, -0.0364, -2.95, 0.156, 2, "gpt2-small", "?", ""),
        FeatureSpec(19787, -0.0325, -2.92, 0.138, 2, "gpt2-small", "?", ""),
        FeatureSpec(23257, -0.0267, -2.87, 0.244, 2, "gpt2-small", "?", ""),
        FeatureSpec(2924, -0.0252, -2.86, 0.176, 2, "gpt2-small", "?", ""),
        FeatureSpec(14332, -0.0240, -2.85, 0.167, 2, "gpt2-small", "?", ""),
    ],
    3: [  # 10 features
        FeatureSpec(14772, -0.0481, -3.68, 0.248, 3, "gpt2-small", "?", ""),
        FeatureSpec(10753, -0.0463, -3.67, 0.323, 3, "gpt2-small", "?", ""),
        FeatureSpec(15417, -0.0427, -3.63, 0.291, 3, "gpt2-small", "?", ""),
        FeatureSpec(9449, -0.0398, -3.60, 0.580, 3, "gpt2-small", "?", ""),
        FeatureSpec(11716, -0.0260, -3.47, 0.472, 3, "gpt2-small", "?", ""),
        FeatureSpec(9501, -0.0226, -3.44, 0.414, 3, "gpt2-small", "?", ""),
        FeatureSpec(14935, -0.0208, -3.42, 0.276, 3, "gpt2-small", "?", ""),
        FeatureSpec(1119, -0.0189, -3.40, 0.347, 3, "gpt2-small", "?", ""),
        FeatureSpec(10328, -0.0160, -3.37, 0.461, 3, "gpt2-small", "?", ""),
        FeatureSpec(2954, -0.0139, -3.35, 0.519, 3, "gpt2-small", "?", ""),
    ],
    4: [  # 10 features
        FeatureSpec(23261, -0.3607, -9.48, 0.294, 4, "gpt2-small", "H1?", "Strong anti-alignment"),
        FeatureSpec(18419, -0.1063, -6.08, 0.267, 4, "gpt2-small", "?", ""),
        FeatureSpec(19371, -0.0607, -5.47, 0.668, 4, "gpt2-small", "?", ""),
        FeatureSpec(14273, -0.0562, -5.41, 0.339, 4, "gpt2-small", "?", ""),
        FeatureSpec(18838, -0.0462, -5.27, 0.739, 4, "gpt2-small", "?", ""),
        FeatureSpec(19083, -0.0397, -5.19, 0.595, 4, "gpt2-small", "?", ""),
        FeatureSpec(13892, -0.0334, -5.10, 0.527, 4, "gpt2-small", "?", ""),
        FeatureSpec(20429, -0.0309, -5.07, 0.168, 4, "gpt2-small", "?", ""),
        FeatureSpec(22817, -0.0157, -4.87, 0.359, 4, "gpt2-small", "?", ""),
        FeatureSpec(17261, -0.0137, -4.84, 0.603, 4, "gpt2-small", "?", ""),
    ],
    5: [  # 10 features
        FeatureSpec(12688, -0.1707, -7.42, 0.679, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(10691, -0.1510, -7.14, 0.851, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(6597, -0.1427, -7.02, 0.756, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(7307, -0.1318, -6.87, 0.563, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(12519, -0.1187, -6.69, 0.496, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(7258, -0.1051, -6.49, 0.321, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(3376, -0.0945, -6.35, 0.447, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(21369, -0.0907, -6.29, 0.485, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(13115, -0.0881, -6.25, 0.385, 5, "gpt2-small", "H1?", ""),
        FeatureSpec(8839, -0.0810, -6.15, 0.839, 5, "gpt2-small", "H1?", ""),
    ],
    6: [  # 8 features
        FeatureSpec(5295, -0.3534, -9.57, 0.263, 6, "gpt2-small", "H1?", "Strong anti-alignment"),
        FeatureSpec(13063, -0.2423, -8.10, 0.611, 6, "gpt2-small", "H1?", ""),
        FeatureSpec(17733, -0.1937, -7.46, 0.312, 6, "gpt2-small", "H1?", ""),
        FeatureSpec(22578, -0.1617, -7.03, 0.710, 6, "gpt2-small", "H1?", ""),
        FeatureSpec(18854, -0.0995, -6.21, 0.626, 6, "gpt2-small", "H1?", ""),
        FeatureSpec(23870, -0.0905, -6.09, 0.274, 6, "gpt2-small", "H1?", ""),
        FeatureSpec(13180, -0.0726, -5.85, 0.464, 6, "gpt2-small", "?", ""),
        FeatureSpec(13800, -0.0133, -5.07, 0.321, 6, "gpt2-small", "?", ""),
    ],
    7: [  # 4 features
        FeatureSpec(7052, -0.3589, -9.41, 0.217, 7, "gpt2-small", "H1?", "Strong anti-alignment"),
        FeatureSpec(99, -0.2801, -8.40, 0.960, 7, "gpt2-small", "H1?", "Large encoder norm"),
        FeatureSpec(6837, -0.1739, -7.05, 0.155, 7, "gpt2-small", "?", ""),
        FeatureSpec(18530, -0.0734, -5.77, 0.225, 7, "gpt2-small", "?", ""),
    ],
    8: [  # 4 features - REFERENCE LAYER (most studied)
        FeatureSpec(19139, -0.5255, -11.38, 0.727, 8, "gpt2-small", "H1", 
                    "Primary reference feature. Large norm, active hedge. Parent: F17487/F21174/F16152"),
        FeatureSpec(18573, -0.4209, -10.05, 0.233, 8, "gpt2-small", "H1",
                    "Small norm but active hedge. Parent: F17487. Bisector test PASS"),
        FeatureSpec(11969, -0.2857, -8.33, 0.258, 8, "gpt2-small", "H1",
                    "Active hedge. Parent: F17487. Bisector test PASS"),
        FeatureSpec(20804, -0.1301, -6.35, 0.667, 8, "gpt2-small", "H2?",
                    "No hedge structure (bisector test FAIL). Possible dead feature or different mechanism"),
    ],
    9: [  # 3 features - MOST EXTREME
        FeatureSpec(18298, -0.7360, -13.94, 0.165, 9, "gpt2-small", "H1",
                    "MOST EXTREME anti-aligned feature in GPT-2 Small. Highest sigma deviation."),
        FeatureSpec(13180, -0.0308, -5.03, 0.762, 9, "gpt2-small", "?", ""),
        FeatureSpec(21059, -0.0116, -4.79, 0.876, 9, "gpt2-small", "?", ""),
    ],
    10: [  # 3 features
        FeatureSpec(19679, -0.6841, -13.92, 0.931, 10, "gpt2-small", "H1",
                    "Second most extreme feature in GPT-2. Large encoder norm."),
        FeatureSpec(3268, -0.5729, -12.45, 0.863, 10, "gpt2-small", "H1",
                    "Third most extreme feature."),
        FeatureSpec(5223, -0.1121, -6.35, 1.173, 10, "gpt2-small", "?", "Very large encoder norm"),
    ],
    11: [  # 10 features
        FeatureSpec(16366, -0.2019, -7.40, 0.802, 11, "gpt2-small", "H1?", ""),
        FeatureSpec(1327, -0.1598, -6.83, 0.183, 11, "gpt2-small", "?", ""),
        FeatureSpec(18262, -0.1303, -6.44, 0.793, 11, "gpt2-small", "H1?", ""),
        FeatureSpec(1412, -0.1247, -6.36, 1.104, 11, "gpt2-small", "?", "Large encoder norm"),
        FeatureSpec(7184, -0.1088, -6.15, 0.082, 11, "gpt2-small", "?", "Very small encoder norm"),
        FeatureSpec(3736, -0.1043, -6.09, 0.195, 11, "gpt2-small", "?", ""),
        FeatureSpec(24041, -0.0961, -5.98, 0.088, 11, "gpt2-small", "?", ""),
        FeatureSpec(21372, -0.0578, -5.46, 0.153, 11, "gpt2-small", "?", ""),
        FeatureSpec(15079, -0.0537, -5.41, 0.050, 11, "gpt2-small", "?", "Very small encoder norm"),
        FeatureSpec(4181, -0.0443, -5.28, 0.805, 11, "gpt2-small", "?", ""),
    ],
}

# Hedge Family Parent Features (from Supplement 3)
# These are the parent features that the anti-aligned features detect
GPT2_HEDGE_FAMILY_PARENTS: Dict[int, FeatureSpec] = {
    17487: FeatureSpec(17487, 0.0, 0.0, 0.0, 8, "gpt2-small", "PARENT",
                       "Parent cluster member for F19139, F18573, F11969, F20804"),
    21174: FeatureSpec(21174, 0.0, 0.0, 0.0, 8, "gpt2-small", "PARENT",
                       "Parent cluster member. F19139 decoder opposes this feature."),
    16152: FeatureSpec(16152, 0.0, 0.0, 0.0, 8, "gpt2-small", "PARENT",
                       "Parent cluster member. F11969 decoder opposes this feature."),
}


# ============================================================================
# GEMMA 2B (JumpReLU SAEs)
# Source: google/gemma-scope-2b-pt-res-canonical
# SAE Sets: gemmascope-res-16k, gemmascope-res-131k
# Total anti-aligned features (cos<0): 1
# ============================================================================

GEMMA_2B_FEATURES: Dict[int, List[FeatureSpec]] = {
    1: [  # Only 1 feature with cos<0 in entire model
        FeatureSpec(3378, -0.4652, -8.50, 2.020, 1, "gemma-2b", "H1/H3?",
                    "ONLY absolutely anti-aligned feature in Gemma 2B. High alignment baseline (mu~0.75)"),
    ],
}


# ============================================================================
# LLAMA 3.1 8B (JumpReLU SAEs)
# Source: llama_scope_lxr_8x (SAELens)
# SAE Pattern: l{L}r_8x
# Negative-cosine features in the top-10-per-layer table: 86
# Note: Concentrated in early layers (L0-L2), opposite pattern from GPT-2
# ============================================================================

LLAMA_31_8B_FEATURES: Dict[int, List[FeatureSpec]] = {
    0: [  # 10 features - early layer concentration (depth-inverted pattern)
        FeatureSpec(5326, -0.4260, -7.94, 118.653, 0, "llama-3.1-8b", "H3?",
                    "Early layer anti-alignment. Very large encoder norm (normalization artifact)"),
        FeatureSpec(4296, -0.4019, -7.70, 147.566, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(6331, -0.3724, -7.40, 118.028, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(23747, -0.3557, -7.23, 121.596, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(13460, -0.3150, -6.82, 111.861, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(1686, -0.3104, -6.78, 144.446, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(27807, -0.2985, -6.66, 102.692, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(18933, -0.2957, -6.63, 147.927, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(29772, -0.2694, -6.37, 127.068, 0, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(29366, -0.2478, -6.15, 92.373, 0, "llama-3.1-8b", "H3?", ""),
    ],
    1: [  # 10 features - most extreme in Llama
        FeatureSpec(27745, -0.6458, -8.66, 98.983, 1, "llama-3.1-8b", "H3?",
                    "MOST EXTREME anti-aligned feature in Llama 3.1 8B"),
        FeatureSpec(24381, -0.6143, -8.40, 78.904, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(13765, -0.5915, -8.21, 112.815, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(4138, -0.5825, -8.13, 111.274, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(7061, -0.5752, -8.07, 108.399, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(23364, -0.5667, -8.00, 84.950, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(19442, -0.5642, -7.98, 117.984, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(7712, -0.5419, -7.79, 94.055, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(23372, -0.5365, -7.74, 109.092, 1, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(22083, -0.4635, -7.13, 121.668, 1, "llama-3.1-8b", "H3?", ""),
    ],
    2: [  # 10 features
        FeatureSpec(22038, -0.5853, -7.45, 65.634, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(21834, -0.5405, -7.12, 55.867, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(8927, -0.2739, -5.13, 35.208, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(25905, -0.2403, -4.88, 30.158, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(9325, -0.1845, -4.46, 49.927, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(6944, -0.1588, -4.27, 39.058, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(27797, -0.0728, -3.63, 38.919, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(16033, -0.0529, -3.48, 38.290, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(28058, -0.0193, -3.23, 41.672, 2, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(3824, -0.0175, -3.22, 45.182, 2, "llama-3.1-8b", "H3?", ""),
    ],
    3: [  # 3 features
        FeatureSpec(20808, -0.0554, -3.23, 33.541, 3, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(13997, -0.0375, -3.12, 31.474, 3, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(30720, -0.0075, -2.92, 29.855, 3, "llama-3.1-8b", "H3?", ""),
    ],
    4: [  # 7 features
        FeatureSpec(1324, -0.2624, -4.64, 25.366, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(6611, -0.0708, -3.44, 20.714, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(22066, -0.0461, -3.29, 22.895, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(29741, -0.0430, -3.27, 18.337, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(30893, -0.0374, -3.23, 22.957, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(14918, -0.0209, -3.13, 25.097, 4, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(25183, -0.0151, -3.10, 31.211, 4, "llama-3.1-8b", "H3?", ""),
    ],
    5: [  # 4 features
        FeatureSpec(16817, -0.2893, -4.76, 21.259, 5, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(1962, -0.1660, -4.00, 21.984, 5, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(3633, -0.1245, -3.75, 14.625, 5, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(30067, -0.0216, -3.12, 19.142, 5, "llama-3.1-8b", "H3?", ""),
    ],
    6: [  # 2 features
        FeatureSpec(31422, -0.0254, -3.11, 18.533, 6, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(27272, -0.0252, -3.11, 19.957, 6, "llama-3.1-8b", "H3?", ""),
    ],
    7: [  # 1 feature
        FeatureSpec(1168, -0.5470, -6.43, 18.630, 7, "llama-3.1-8b", "H3?", "Strong anti-alignment"),
    ],
    8: [  # 4 features
        FeatureSpec(26862, -0.4911, -6.00, 13.133, 8, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(8055, -0.2220, -4.37, 11.574, 8, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(29360, -0.0647, -3.42, 15.503, 8, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(362, -0.0081, -3.07, 13.086, 8, "llama-3.1-8b", "H3?", ""),
    ],
    9: [  # 2 features
        FeatureSpec(8442, -0.4370, -5.68, 10.298, 9, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(26380, -0.2355, -4.47, 16.166, 9, "llama-3.1-8b", "H3?", ""),
    ],
    11: [  # 1 feature
        FeatureSpec(8929, -0.1312, -3.84, 9.670, 11, "llama-3.1-8b", "H3?", ""),
    ],
    14: [  # 1 feature
        FeatureSpec(29212, -0.0118, -3.17, 10.134, 14, "llama-3.1-8b", "H3?", ""),
    ],
    15: [  # 1 feature
        FeatureSpec(6225, -0.1494, -4.09, 6.920, 15, "llama-3.1-8b", "H3?", ""),
    ],
    16: [  # 10 features
        FeatureSpec(23804, -0.4121, -5.92, 7.349, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(2376, -0.3198, -5.32, 6.072, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(15199, -0.2784, -5.05, 8.020, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(30563, -0.2716, -5.00, 6.744, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(30751, -0.2306, -4.74, 4.814, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(7861, -0.1933, -4.49, 6.699, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(17654, -0.1143, -3.98, 8.357, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(19551, -0.1135, -3.97, 6.017, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(22074, -0.0987, -3.88, 4.022, 16, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(15654, -0.0851, -3.79, 4.809, 16, "llama-3.1-8b", "H3?", ""),
    ],
    17: [  # 5 features
        FeatureSpec(32030, -0.0954, -3.94, 4.195, 17, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(19976, -0.0727, -3.78, 3.507, 17, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(17570, -0.0316, -3.50, 5.885, 17, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(2463, -0.0158, -3.39, 4.184, 17, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(31315, -0.0124, -3.37, 3.415, 17, "llama-3.1-8b", "H3?", ""),
    ],
    18: [  # 2 features
        FeatureSpec(11002, -0.1595, -4.45, 5.443, 18, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(1245, -0.0578, -3.75, 6.068, 18, "llama-3.1-8b", "H3?", ""),
    ],
    19: [  # 4 features
        FeatureSpec(31463, -0.1813, -4.68, 6.161, 19, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(20716, -0.0492, -3.76, 5.154, 19, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(14314, -0.0378, -3.68, 5.237, 19, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(19106, -0.0019, -3.43, 4.330, 19, "llama-3.1-8b", "H3?", ""),
    ],
    20: [  # 1 feature
        FeatureSpec(25713, -0.0313, -3.66, 4.668, 20, "llama-3.1-8b", "H3?", ""),
    ],
    21: [  # 1 feature
        FeatureSpec(13832, -0.1043, -4.16, 4.627, 21, "llama-3.1-8b", "H3?", ""),
    ],
    23: [  # 2 features
        FeatureSpec(17593, -0.1866, -4.66, 3.428, 23, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(15362, -0.0339, -3.59, 3.260, 23, "llama-3.1-8b", "H3?", ""),
    ],
    26: [  # 1 feature
        FeatureSpec(23432, -0.0765, -3.61, 2.145, 26, "llama-3.1-8b", "H3?", ""),
    ],
    27: [  # 2 features
        FeatureSpec(30187, -0.1349, -3.84, 1.888, 27, "llama-3.1-8b", "H3?", ""),
        FeatureSpec(14996, -0.0386, -3.25, 1.703, 27, "llama-3.1-8b", "H3?", ""),
    ],
    28: [  # 1 feature
        FeatureSpec(29179, -0.0086, -3.07, 2.587, 28, "llama-3.1-8b", "H3?", ""),
    ],
    29: [  # 1 feature
        FeatureSpec(5684, -0.2343, -4.37, 2.629, 29, "llama-3.1-8b", "H3?", ""),
    ],
}


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

MODEL_STATS = {
    "gpt2-small": {
        "total_features": 86,
        "architecture": "ReLU+L1",
        "sae_pattern": "{layer}-res-jb",
        "layers_with_cos_neg": list(GPT2_SMALL_FEATURES.keys()),
        "primary_hypothesis": "H1 (Subtractive Hedges)",
        "depth_pattern": "Late layers (L4-L11)",
        "sign_flip": True,
        "most_extreme": FeatureSpec(18298, -0.736, -13.94, 0.165, 9, "gpt2-small", "H1", "Most extreme"),
        "reference_layer": 8,
        "reference_features": [19139, 18573, 11969, 20804],
    },
    "gemma-2b": {
        "total_features": 1,
        "architecture": "JumpReLU (per-feature threshold)",
        "sae_sets": ["gemmascope-res-16k", "gemmascope-res-131k"],
        "layers_with_cos_neg": [1],
        "primary_hypothesis": "H3 (Superposition Interference)",
        "depth_pattern": "Minimal (L1 only)",
        "sign_flip": False,
        "mean_cosine": 0.75,
        "most_extreme": FeatureSpec(3378, -0.465, -8.50, 2.020, 1, "gemma-2b", "H3?", "Only cos<0"),
    },
    "llama-3.1-8b": {
        "total_features": 86,
        "architecture": "JumpReLU (scalar threshold)",
        "sae_pattern": "l{layer}r_8x",
        "layers_with_cos_neg": list(LLAMA_31_8B_FEATURES.keys()),
        "primary_hypothesis": "H3 (Superposition Interference)",
        "depth_pattern": "Early layers (L0-L2 concentrated)",
        "sign_flip": False,
        "most_extreme": FeatureSpec(27745, -0.646, -8.66, 98.983, 1, "llama-3.1-8b", "H3?", "Most extreme"),
        "mean_cosine": 0.48,
        "extreme_kurtosis": [0, 1],  # L0, L1 have gamma_2 > 5.9
    },
}


def get_all_features(model: str) -> Dict[int, List[FeatureSpec]]:
    """Get all anti-aligned features for a model."""
    if model == "gpt2-small":
        return GPT2_SMALL_FEATURES
    elif model == "gemma-2b":
        return GEMMA_2B_FEATURES
    elif model == "llama-3.1-8b":
        return LLAMA_31_8B_FEATURES
    else:
        raise ValueError(f"Unknown model: {model}")


def get_feature(model: str, layer: int, index: int) -> Optional[FeatureSpec]:
    """Get a specific feature by model, layer, and index."""
    features = get_all_features(model)
    if layer not in features:
        return None
    for feat in features[layer]:
        if feat.index == index:
            return feat
    return None


def get_features_by_sigma_threshold(model: str, min_sigma: float = 6.0) -> List[FeatureSpec]:
    """Get features with sigma deviation >= threshold (in absolute value)."""
    all_features = []
    for layer_features in get_all_features(model).values():
        for feat in layer_features:
            if abs(feat.sigma) >= min_sigma:
                all_features.append(feat)
    return sorted(all_features, key=lambda x: x.cos_ed)


if __name__ == "__main__":
    # Print summary
    print("=" * 70)
    print("GEOMETRY OF SAE ARTIFACTS - COMPLETE FEATURE REGISTRY")
    print("=" * 70)
    
    for model, stats in MODEL_STATS.items():
        print(f"\n{model.upper()}")
        print(f"  Architecture: {stats['architecture']}")
        print(f"  Total anti-aligned features: {stats['total_features']}")
        print(f"  Depth pattern: {stats['depth_pattern']}")
        print(f"  Primary hypothesis: {stats['primary_hypothesis']}")
        print(f"  Sign flip: {stats.get('sign_flip', 'N/A')}")
        if 'most_extreme' in stats:
            me = stats['most_extreme']
            print(f"  Most extreme: F{me.index} (L{me.layer}, cos={me.cos_ed:.3f}, sigma={me.sigma:.2f})")
