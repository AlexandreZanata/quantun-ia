.PHONY: dev test test-watch lint lint-fix coverage dashboard dashboard-local experiment experiment-large repro export-results docker-build docker-test docker-lint clean

PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

dev:
	docker compose up app

test:
	docker compose run --rm test

test-watch:
	docker compose run --rm app pytest tests/ -v --tb=short -x

lint:
	docker compose -f docker-compose.test.yml run --rm lint

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
