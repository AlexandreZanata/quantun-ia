# Docker

## Images

The `Dockerfile` uses multi-stage builds:

| Stage | Purpose | Command |
|-------|---------|---------|
| `base` | Python 3.12 slim + build tools | — |
| `deps` | Install CPU PyTorch + all requirements | — |
| `dev` | Interactive development shell | `make dev` |
| `test` | Run pytest with coverage | `make test` |
| `dashboard` | Streamlit on port 8501 | `make dashboard` |
| `experiment` | Run EXP 001 | `make experiment` |

CPU-only PyTorch is used in Docker to keep images lean (~2 GB vs ~8 GB with CUDA).

## Services

### docker-compose.yml

```yaml
services:
  app        # Interactive dev shell with volume mount
  test       # Runs pytest suite
  dashboard  # Streamlit on :8501
  experiment # Runs EXP 001 by default
```

### docker-compose.test.yml

Used by CI and `make docker-test`:
- `test` — full pytest suite
- `lint` — ruff check

## Common Commands

```bash
# Build all images
make docker-build

# Run tests
make test
make docker-test

# Start dashboard (http://localhost:8501)
make dashboard

# Run first experiment
make experiment

# Interactive shell
make dev
# then inside container:
python experiments/exp_001_quantum_vs_classical/run.py

# Lint
make lint
make docker-lint
```

## Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `.` | `/app` | Source code (dev mode) |
| `./logs` | `/app/logs` | Experiment results persist on host |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Set in all services |
| `PYTHONDONTWRITEBYTECODE` | `1` | No .pyc files in container |

## Troubleshooting

### Build fails on torch install
The Dockerfile installs CPU PyTorch from the official index. If it fails:
```bash
docker compose build --no-cache test
```

### Permission errors on logs/
```bash
mkdir -p logs && chmod 777 logs
```

### Dashboard not loading
Ensure port 8501 is not in use:
```bash
docker compose down
make dashboard
```

### Out of memory during quantum experiments
Reduce qubit count or batch size in `run.py`. Quantum simulators scale as O(2^n) in memory.

## Production Notes

For GPU support, replace the CPU PyTorch install line in `Dockerfile`:
```dockerfile
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Requires `nvidia-container-toolkit` and `deploy.resources.reservations.devices` in compose.
