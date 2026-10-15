.PHONY: requirements clean test help

PROJECT_NAME = tir_jde
PYTHON_INTERPRETER = python

## Install Python dependencies (pip; environment.yml/conda remains primary)
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete compiled Python files and caches
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

## Run the test suite
test:
	$(PYTHON_INTERPRETER) -m pytest

## Show this help
help:
	@echo "Available targets:"
	@echo "  requirements  install the package (pip install -e .)"
	@echo "  clean         remove compiled Python files and __pycache__"
	@echo "  test          run pytest"

.DEFAULT_GOAL := help
