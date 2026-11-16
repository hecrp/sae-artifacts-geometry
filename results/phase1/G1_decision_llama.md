# Gate G1: Llama 3.1 8B (JumpReLU)

**Decision: G1-YES**

Critical question: are 12-14 features at 6sigma in L0-L1 anomalous given kurtosis=7.4? Or expected under Student-t null?

## Summary
Max anomaly ratio: 336065.1x
Layers with G1-YES (ratio >10x): 3
Layers with G1-MARGINAL (ratio 2-10x): 1

## Layers with 6sigma features
 layer_idx winner_model  observed_6sig  expected_6sig  anomaly_ratio
         2      Mixture              2       0.000006  336065.051046
         7     Gaussian              1       0.000032   30932.455194
         1      Mixture             14       0.286966      48.786274
         0      Mixture             12       2.625634       4.570324

## Implications
Tail features are statistically anomalous. Mechanistic investigation warranted (Phases 2-4).
