.PHONY: dev test test-watch lint lint-fix coverage dashboard experiment docker-build docker-test docker-lint clean

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

experiment:
	docker compose run --rm experiment

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
