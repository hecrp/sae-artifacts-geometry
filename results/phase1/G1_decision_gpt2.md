# Gate G1: GPT-2 Small (ReLU+L1)

**Decision: G1-YES**

Primary paper subject. Focus: late layers L4-L11.

## Summary
Max anomaly ratio: 10000000000.0x
Layers with G1-YES (ratio >10x): 7
Layers with G1-MARGINAL (ratio 2-10x): 1

## Layers with 6sigma features
 layer_idx winner_model  observed_6sig  expected_6sig  anomaly_ratio
         9      Mixture              1   5.659187e-12   1.000000e+10
         8      Mixture              4   8.607740e-06   4.646981e+05
         7      Mixture              3   1.179383e-05   2.543704e+05
         6      Mixture              6   9.711052e-05   6.178527e+04
        10      Mixture              3   7.842143e-05   3.825485e+04
        11      Mixture              6   1.809209e-03   3.316366e+03
         5      Mixture             10   5.144524e-03   1.943814e+03
         4      Mixture              2   5.498790e-01   3.637164e+00

## Implications
Tail features are statistically anomalous. Mechanistic investigation warranted (Phases 2-4).
