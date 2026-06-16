# syntax=docker/dockerfile:1

# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt requirements-dev.txt ./

# Install CPU-only PyTorch to keep the image lean
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

# ── Development ───────────────────────────────────────────────────────────────
FROM deps AS dev

COPY . .

CMD ["bash"]

# ── Test runner ───────────────────────────────────────────────────────────────
FROM deps AS test

COPY . .

RUN mkdir -p logs

CMD ["pytest", "tests/", "--cov=src", "--cov-report=term-missing", "--cov-fail-under=70"]

# ── Dashboard ─────────────────────────────────────────────────────────────────
FROM deps AS dashboard

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# ── Experiment runner ─────────────────────────────────────────────────────────
FROM deps AS experiment

COPY . .

ENTRYPOINT ["python"]
CMD ["experiments/exp_001_quantum_vs_classical/run.py"]
