# Results — EXP 028: Chatbot Tool vs API Parity

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Dialogue parity

| Dialogue | max Δp | Latency (s) | Disclaimer |
|----------|--------|-------------|------------|
| dialogue_01 | 0.00e+00 | 0.037 | yes |
| dialogue_02 | 0.00e+00 | 0.033 | yes |
| dialogue_03 | 0.00e+00 | 0.032 | yes |
| dialogue_04 | 0.00e+00 | 0.033 | yes |
| dialogue_05 | 0.00e+00 | 0.033 | yes |
| dialogue_06 | 0.00e+00 | 0.033 | yes |
| dialogue_07 | 0.00e+00 | 0.033 | yes |
| dialogue_08 | 0.00e+00 | 0.037 | yes |
| dialogue_09 | 0.00e+00 | 0.033 | yes |
| dialogue_10 | 0.00e+00 | 0.032 | yes |

## Verdict
**accepted** — max |Δp|=0.00e+00; max latency=0.037s.

## Conclusion
Chatbot tool adapter matches REST API predictions within numerical tolerance.

## Limitations
- Scripted golden dialogues; no live Ollama routing in this experiment.
- Research prototype only — not a clinical deployment claim.
