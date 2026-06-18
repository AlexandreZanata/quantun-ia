# Hypothesis — EXP 028: Chatbot Tool Path vs API Parity

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

For **10 golden chatbot dialogues** (each with exactly 30 Wisconsin Breast Cancer features), the
`score_breast_cancer` tool adapter (`src/application/chatbot_tool.py`) will return probabilities
**within 1e-5** of `POST /api/v1/predictions` on the same checkpoint bundle, and every assistant
message will include the research disclaimer.

## Why I expect this

Both paths call `predict_nanomodel.execute` with identical DTOs. exp_026 proved API training parity;
exp_028 extends that to the chatbot tool surface (Ollama-compatible schema + NL wrapper).

## What would prove me wrong

- Any dialogue yields max \|Δp\| ≥ 1e-5 between tool adapter and API
- Tool accepts ≠30 features without `INVALID_FEATURES` error
- Assistant message missing the research disclaimer
- p99 tool+API latency > 2 s per dialogue on RTX 4060 (local gate)

## Metrics I will measure

- [x] max \|Δp\| per dialogue (tool vs API)
- [x] Feature count validation (30 per row)
- [x] Disclaimer present in formatted message
- [x] Wall-clock latency per dialogue on CUDA

## Success criteria

- 10/10 dialogues pass max \|Δp\| < 1e-5
- All messages contain disclaimer text
- `make check-real` stays green after merge

## Known limitations

- Scripted dialogues (not live Ollama LLM routing)
- Single dataset (breast cancer); not a clinical deployment claim
