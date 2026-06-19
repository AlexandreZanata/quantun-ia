# Nano Model Factory

Ship **downloadable, reproducible nanomodels** with one command. Each bundle includes weights,
scaler, config, optional calibration, exports (ONNX / TorchScript), and a CPU inference script.

## Quick start

```bash
pip install -e .

# Ship from existing GPU-trained checkpoint (RTX 4060)
QML_DEVICE=cuda qml-ship --model large_nano_mlp_synthea --skip-train --skip-gate

# Install bundle into artifacts/ for API + Streamlit
qml-download --model large_nano_mlp_synthea

# Makefile shortcuts
make ship-all-p0
make ship-hybrid-higgs   # frozen MLP + QNN head (exp_037, publication gate)
make ship MODEL=large_nano_mlp_higgs SKIP_TRAIN=1 SKIP_GATE=1
make download-model MODEL=large_nano_mlp_synthea
```

## Full pipeline (train + gate + ship)

```bash
source .local/env.sh   # workstation only
make health-gpu
qml-ship --model large_nano_mlp_synthea --profile publication
```

Stages: registry lookup → train (if needed) → real gate test → serve publish → export →
`dist/serve_models/{registry_key}/` + `MANIFEST.sha256`.

Publication profile **requires CUDA** and **cannot skip gates**.

## Registry

All shippable models live in `config/nanomodel_registry.yaml`.

| Registry key | Dataset | Gate |
|--------------|---------|------|
| `large_nano_mlp_synthea` | Synthea CV | exp_034 real gate |
| `large_nano_mlp_higgs` | HIGGS | exp_032 real gate |
| `quantum_nano_bc` | Breast cancer | nanotrainer serve |
| `large_nano_mlp_synthea_calibrated` | Synthea + isotonic | exp_043 |
| `large_nano_hybrid_higgs` | HIGGS QNN head | exp_037 |

## Bundle layout

```
dist/serve_models/large_nano_mlp_synthea/
├── best.pt
├── config.json
├── scaler.joblib
├── calibration_isotonic.json   # when applicable
├── model_card.md
├── metrics.json
├── MANIFEST.sha256
├── exports/
│   ├── native/
│   ├── onnx/model.onnx
│   └── torchscript/model.pt
└── inference/
    ├── predict.py
    └── schema.json
```

## CPU inference

```bash
python dist/serve_models/large_nano_mlp_synthea/inference/predict.py \
  --input tests/fixtures/synthea_patient_row.json
```

## Export formats

| Format | Models | Use case |
|--------|--------|----------|
| Native | All | quantun-ia API, Streamlit, PennyLane QNN |
| ONNX | Classical MLP | ONNX Runtime, edge deploy |
| TorchScript | PyTorch | Portable torch inference |
| Hugging Face folder | Planned upload | Community download |

Quantum hybrids do not export to ONNX (PennyLane layers).

## Release integration

After `make ship-all-p0`, run `make release` to include `serve_models/` in the Zenodo bundle
with SHA-256 checksums in `MANIFEST.txt`.

## Tests

```bash
pytest tests/contracts/test_nanomodel_registry.py -q
pytest tests/unit/test_nanomodel_ship.py -q
pytest tests/integration/test_download_and_predict.py -q
QML_DEVICE=cuda pytest tests/real/test_nanomodel_ship_gate.py -m real -v
```
