# Gate G0 Decision

Decision: **G0-PARTIAL**

## Model-by-Model Results

### GPT-2 Small (ReLU+L1)
- Early r (depth<=0.33): +0.486
- Late r  (depth>=0.67): -0.713
- Sign flip: YES
- Max neg r: -0.756
- Max 6sigma count per layer: 10

### Gemma 2B (JumpReLU)
- Early r (depth<=0.33): -0.876
- Late r  (depth>=0.67): -0.768
- Sign flip: NO
- Max neg r: -0.941
- Max 6sigma count per layer: 4

### Llama 3.1 8B (JumpReLU)
- Early r (depth<=0.33): -0.835
- Late r  (depth>=0.67): -0.756
- Sign flip: NO
- Max neg r: -0.895
- Max 6sigma count per layer: 14

## Rationale

Models with sign flip: ['GPT-2 Small (ReLU+L1)']

Models with 6sigma tail: ['GPT-2 Small (ReLU+L1)', 'Gemma 2B (JumpReLU)', 'Llama 3.1 8B (JumpReLU)']

PROCEED but reframe. Sign flip is GPT-2 specific but tail found elsewhere.
