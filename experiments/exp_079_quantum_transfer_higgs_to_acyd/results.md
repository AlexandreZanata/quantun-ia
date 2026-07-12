# Results — EXP 079: Quantum head transfer HIGGS → ACYD (H-Q13)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (ROC-AUC)

- Train rows: **50,107** · Val rows: **5,830**
- Trainable head params: **289**
- Scratch head (frozen C4): **0.6749**
- Transfer head (HIGGS init → fine-tune): **0.6785**
- Transfer advantage: **+0.35 pp**
- Hypothesis: advantage **< +0.5 pp**
- Source checkpoint: `/data/dev/projects/webstorm/quantun-ia/artifacts/exp_037/large_nano_hybrid/seed_42/best.pt`
- Elapsed: **30.2s**

## Verdict
**honest negative confirmed** — cross-domain QNN head transfer on frozen C4.

## Limitations
- Head-only transfer; C1 backbone shapes incompatible with ACYD input_dim.
- PennyLane QNN on CPU; frozen C4 backbone on CUDA.
