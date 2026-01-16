.PHONY: help install dev test lint format type-check clean docker-up docker-down api worker

# Default target
help:
	@echo "Blockchain Insider Detection System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run linter (ruff)"
	@echo "  format       Format code (ruff)"
	@echo "  type-check   Run type checker (mypy)"
	@echo "  clean        Clean build artifacts"
	@echo "  docker-up    Start Docker containers"
	@echo "  docker-down  Stop Docker containers"
	@echo "  api          Start API server"
	@echo "  worker       Start Celery worker"

# Dependencies
install:
	poetry install --only main

dev:
	poetry install
	poetry run pre-commit install

# Testing
test:
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-e2e:
	poetry run pytest tests/e2e/ -v

# Code quality
lint:
	poetry run ruff check src tests dags etl

format:
	poetry run ruff format src tests dags etl
	poetry run ruff check --fix src tests dags etl

type-check:
	poetry run mypy src

check: lint type-check test

# Cleanup
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker
docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

docker-ps:
	cd docker && docker-compose ps

# Development servers
api:
	poetry run uvicorn src.api.main:app --reload --port 8080

worker:
	poetry run celery -A src.worker worker --loglevel=info

# Database management
db-init:
	poetry run python scripts/setup_databases.py

db-migrate:
	@echo "Running database migrations..."
	# Add migration commands here

db-seed:
	poetry run python scripts/load_sample_data.py

# Airflow
airflow-init:
	poetry run airflow db init
	poetry run airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin

airflow-webserver:
	poetry run airflow webserver --port 8081

airflow-scheduler:
	poetry run airflow scheduler

# Embedding model
download-models:
	poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Benchmarks
benchmark-search:
	poetry run python scripts/benchmark_search.py

benchmark-embedding:
	poetry run python scripts/benchmark_embedding.py

# Documentation
docs-serve:
	cd docs && mkdocs serve

docs-build:
	cd docs && mkdocs build
