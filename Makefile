.PHONY: dev test test-watch lint lint-fix typecheck coverage dashboard dashboard-local experiment experiment-large repro export-results hpo figures latex-tables release release-check paper-sync paper-build paper-build-publication arxiv-bundle replay-publication replay-publication-artifacts repro-publication-ci open-science-preflight power-analysis microqml-bench publish-leaderboard publish-leaderboard-check check health docker-build docker-test docker-lint clean install train-demo nano-parity-bench nano-parity-download nano-parity-publication api api-demo e2e reviewer-repro citation-ready citation-ready-full finalize-citation dvc-check dvc-setup dvc-push model-card

PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

install:
	$(PYTHON) -m pip install -e .

health:
	$(PYTHON) scripts/health_check.py

check: lint-local typecheck
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/ --cov=src --cov-fail-under=80 -q
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/integration/ tests/contracts/ -q
	MLFLOW_DISABLE=1 QML_DEVICE=cuda $(PYTHON) -m pytest tests/e2e/ -q

e2e:
	MLFLOW_DISABLE=1 $(PYTHON) -m pytest tests/e2e/ -v

typecheck:
	$(PYTHON) -m mypy src/training src/quantum

dev:
	docker compose up app

test:
	docker compose run --rm test

test-watch:
	docker compose run --rm app pytest tests/ -v --tb=short -x

lint:
	docker compose -f docker-compose.test.yml run --rm lint

lint-local:
	ruff check src/ tests/ experiments/ scripts/

lint-fix:
	docker compose run --rm app ruff check --fix src/ tests/ experiments/

coverage:
	docker compose run --rm test pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

dashboard:
	docker compose up dashboard

dashboard-local:
	@$(PYTHON) dashboard/terminal_report.py
	@$(PYTHON) -m streamlit run dashboard/app.py --server.headless true --server.port 8501

experiment:
	docker compose run --rm experiment

experiment-large:
	QML_PROFILE=publication_large $(PYTHON) scripts/run_publication_large.py

repro:
	MLFLOW_DISABLE=1 $(PYTHON) -m pytest tests/integration/test_exp_001_smoke.py -v

export-results:
	$(PYTHON) scripts/export_results.py

hpo:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/run_hpo.py --exp exp_011_uci_tabular_qml --trials 5 --model perceptron

nas:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_016_hybrid_nas/run.py

poison-topology:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_017_poison_topology/run.py

fusion:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_018_feature_fusion/run.py

train-demo:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_train --model perceptron --dataset breast_cancer --profile ci

nano-parity-bench:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_parity_bench --suite --profile ci

nano-parity-download:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.nano_parity_bench --download-only

nano-parity-publication:
	MLFLOW_DISABLE=1 QML_PROFILE=publication $(PYTHON) experiments/exp_022_nano_quantum_parity/run.py --profile publication --write-results

api:
	MLFLOW_DISABLE=1 $(PYTHON) -m scripts.api_server --host 127.0.0.1 --port 8000

api-demo:
	MLFLOW_DISABLE=1 $(PYTHON) experiments/exp_020_api_smoke/run.py

figures:
	$(PYTHON) scripts/generate_figures.py

latex-tables:
	$(PYTHON) scripts/export_latex_tables.py

release:
	$(PYTHON) scripts/prepare_release.py

release-check:
	$(PYTHON) scripts/prepare_release.py --verify-only dist/release

open-science-preflight:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/open_science_preflight.py

paper-sync:
	$(PYTHON) scripts/build_paper.py --sync-only

paper-artifacts: latex-tables figures paper-sync
	@echo "Paper artifacts ready (tables + figures synced to paper/)"

paper-build: paper-artifacts
	$(PYTHON) scripts/build_paper.py

paper-build-publication:
	@mkdir -p logs
	cp tests/contracts/fixtures/publication_experiments.jsonl logs/experiments.jsonl
	$(MAKE) paper-artifacts

repro-publication-ci:
	MLFLOW_DISABLE=1 bash scripts/repro_publication_ci.sh

arxiv-bundle: paper-build
	$(PYTHON) scripts/prepare_arxiv_submission.py

arxiv-bundle-sources:
	$(PYTHON) scripts/prepare_arxiv_submission.py --skip-pdf-check

reviewer-repro:
	MLFLOW_DISABLE=1 bash scripts/reviewer_repro.sh

citation-ready:
	$(PYTHON) scripts/validate_citation_ready.py --skip-release

citation-ready-full: release release-check
	$(PYTHON) scripts/validate_citation_ready.py

finalize-citation:
	@test -n "$(DOI)" || (echo "Usage: make finalize-citation DOI=10.5281/zenodo.XXXXXXX [ARXIV_ID=2606.XXXXX]" && exit 1)
	$(PYTHON) scripts/finalize_citation.py --doi "$(DOI)" $(if $(ARXIV_ID),--arxiv-id "$(ARXIV_ID)",)

dvc-check:
	$(PYTHON) scripts/validate_dvc.py

dvc-setup:
	$(PYTHON) -m pip install -q dvc
	$(PYTHON) scripts/dvc_remote_setup.py

dvc-push:
	$(PYTHON) scripts/dvc_push.py --setup-remote

dvc-push-full:
	$(PYTHON) scripts/dvc_push.py --setup-remote --replay

replay-publication:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/replay_publication.py

replay-publication-artifacts:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/replay_publication.py --artifacts-only

power-analysis:
	$(PYTHON) scripts/power_analysis.py --table

microqml-bench:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/export_microqml_bench.py

publish-leaderboard:
	$(PYTHON) scripts/publish_leaderboard.py

publish-leaderboard-check:
	$(PYTHON) scripts/publish_leaderboard.py --verify-only

export-reference-datasets:
	$(PYTHON) scripts/export_reference_datasets.py

model-card:
	$(PYTHON) scripts/generate_model_card.py

experiments-new:
	MLFLOW_DISABLE=1 $(PYTHON) scripts/run_exp_011_015.py --profile publication

results-new:
	$(PYTHON) scripts/generate_results_md.py

docker-build:
	docker compose build

docker-test:
	docker compose -f docker-compose.test.yml run --rm test

docker-lint:
	docker compose -f docker-compose.test.yml run --rm lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
