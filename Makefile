.PHONY: help install install-dev test test-unit test-integration benchmark clean format lint docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,docs,benchmark]"
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v --cov=perceptual_interdependence --cov-report=html --cov-report=term-missing

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

benchmark:  ## Run performance benchmarks
	pytest tests/benchmarks/ -v
	python -c "from src.perceptual_interdependence.algorithms.cpu_math import get_cpu_math; get_cpu_math().benchmark_performance((2048, 2048))"

clean:  ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:  ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:  ## Run linting checks
	flake8 src/ tests/ scripts/
	mypy src/
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

docs:  ## Build documentation
	cd docs && make html

migrate:  ## Migrate from legacy structure
	python scripts/migrate_legacy.py

demo:  ## Run quick demo
	perceptual-bind bind --albedo data/samples/original_albedo.png --normal data/samples/original_normal.png --user-id 42 --output-dir data/results

gui:  ## Launch GUI
	perceptual-bind gui

validate:  ## Validate system integrity
	python -c "from src.perceptual_interdependence.utils.validation import ValidationSuite; v = ValidationSuite(); r = v.validate_system_integrity(); print('System:', 'VALID' if r['valid'] else 'INVALID')"