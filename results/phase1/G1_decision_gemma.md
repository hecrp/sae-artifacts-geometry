# Gate G1: Gemma 2B (JumpReLU)

**Decision: G1-YES**

Note: 6sigma features have POSITIVE cosine (high-baseline model). G1-YES here means "unusually low alignment for Gemma", not absolute anti-alignment.

## Summary
Max anomaly ratio: 61864.9x
Layers with G1-YES (ratio >10x): 9
Layers with G1-MARGINAL (ratio 2-10x): 0

## Layers with 6sigma features
 layer_idx winner_model  observed_6sig  expected_6sig  anomaly_ratio
         1     Gaussian              1       0.000016   61864.910388
        17      Mixture              1       0.000217    4609.368167
        16      Mixture              1       0.001042     959.704728
        20      Mixture              1       0.001954     511.813304
        25      Mixture              4       0.009261     431.912603
        18      Mixture              1       0.002637     379.227721
        11         Beta              3       0.079385      37.790458
        12         Beta              3       0.101420      29.580039
        13         Beta              2       0.083533      23.942686

## Implications
Tail features are statistically anomalous. Mechanistic investigation warranted (Phases 2-4).
