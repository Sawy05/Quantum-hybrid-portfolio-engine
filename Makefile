.PHONY: help install dev-install test lint format bench barren serve clean docker-build docker-up

PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn

help:
	@echo "Quantum-Classical Hybrid Portfolio Engine Build Commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install production + dev testing dependencies"
	@echo "  make test           Run full pytest suite"
	@echo "  make lint           Check code quality with ruff"
	@echo "  make format         Auto-format code with ruff"
	@echo "  make bench          Run all quantum-classical benchmarks"
	@echo "  make barren         Generate Barren Plateau scaling diagnostics"
	@echo "  make serve          Launch local FastAPI server"
	@echo "  make clean          Clean cache, build, and bytecode artifacts"
	@echo "  make docker-build   Build container image"
	@echo "  make docker-up      Start container service via docker-compose"

install:
	$(PIP) install --upgrade pip
	$(PIP) install .

dev-install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	pytest -v --durations=10

lint:
	ruff check .

format:
	ruff format .

bench:
	$(PYTHON) scripts/run_benchmarks.py

barren:
	$(PYTHON) scripts/plot_barren_plateaus.py

serve:
	$(UVICORN) src.api.server:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

docker-build:
	docker build -t quantum-hybrid-portfolio-engine:latest .

docker-up:
	docker-compose up -d --build
