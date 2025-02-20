PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_ACTIVATE:= $(VENV_BIN)/activate
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
FLAKE8 := $(VENV_BIN)/flake8
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
MYPY := $(VENV_BIN)/mypy

DIST_DIR := dist
VERSION := $(shell python setup.py --version)
PACKAGE_NAME := $(shell python setup.py --name)

.PHONY: help clean install dev-install test lint format check coverage build \
        update-deps pre-commit setup test-file branch

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup        - Initial setup (venv + deps)"
	@echo "  make install      - Install production dependencies"
	@echo "  make upload       - Upload package to PyPI"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code"
	@echo "  make check        - Run all checks (lint, type check, test)"
	@echo "  make coverage     - Run tests with coverage report"
	@echo "  make build        - Build package distribution"
	@echo "  make update-deps  - Update frozen dependencies"
	@echo "  make pre-commit   - Run pre-commit hooks"
	@echo "  make release-local- Create a local release"
	@echo "  make test-file    - Run specific test file (usage: make test-file file=path/to/test.py)"
	@echo "  make branch       - Create new git branch (usage: make branch name=feature/name)"


define run_in_venv
	bash -c 'source $(VENV_ACTIVATE) && $(1)'
endef

all: venv

$(VENV):
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created."
	@$(PIP) install --upgrade pip

clean:
	@echo "Cleaning up..."
	@rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

install: $(VENV)
	@echo "Installing production dependencies..."
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .
	@echo "Installation complete."

dev-install: install
	@echo "Installing development dependencies..."
	@$(PIP) install -r requirements-dev.txt
	@pre-commit install
	@echo "Dev installation complete."

test: $(VENV)
	@echo "Running tests..."
	@$(PYTEST)

lint: $(VENV)
	@echo "Running linting checks..."
	@echo "Running flake8..."
	@$(FLAKE8) src tests
	@echo "Running black..."
	@$(BLACK) --check src tests
	@echo "Running isort..."
	@$(ISORT) --check-only src tests
	@echo "Running MYPY..."
	@$(MYPY) src
	@echo "Linting checks complete."

lint-mypy: $(VENV)
	@echo "Running mypy..."
	@$(MYPY) src

format: $(VENV)
	@echo "Formatting code..."
	@$(BLACK) src tests
	@$(ISORT) src tests
	@echo "Formatting complete."

check: lint test

coverage: $(VENV)
	@echo "Running tests with coverage..."
	@$(PYTEST) --cov=src --cov-report=html --cov-report=xml

build: clean
	@echo "Building package..."
	@$(PYTHON) setup.py sdist bdist_wheel
	@echo "Build complete."


check-build: build
	@echo "Checking built distribution..."
	@$(call run_in_venv, pip install --upgrade twine)
	@$(call run_in_venv, twine check $(DIST_DIR)/*)

upload-test: check-build
	@echo "Preparing to upload $(PACKAGE_NAME) version $(VERSION) to TestPyPI..."
	@read -p "Are you sure? [y/N] " confirm && [ $$confirm = "y" ]
	@$(call run_in_venv, twine upload --repository testpypi $(DIST_DIR)/*)
	@echo "Upload to TestPyPI complete. Check https://test.pypi.org/project/$(PACKAGE_NAME)"
	@echo "To install from TestPyPI: pip install --index-url https://test.pypi.org/simple/ $(PACKAGE_NAME)"

upload: check-build
	@echo "Preparing to upload $(PACKAGE_NAME) version $(VERSION) to PyPI..."
	@echo "Warning: This will make the package publicly available!"
	@read -p "Are you sure? [y/N] " confirm && [ $$confirm = "y" ]
	@$(call run_in_venv, twine upload $(DIST_DIR)/*)
	@echo "Upload to PyPI complete. Check https://pypi.org/project/$(PACKAGE_NAME)"
	@echo "To install: pip install $(PACKAGE_NAME)"

release:
	@if [ "$(new_version)" = "" ]; then \
		echo "Usage: make release new_version=X.Y.Z"; \
		exit 1; \
	fi
	@echo "Creating new release version $(new_version)..."
	@python scripts/update_version.py $(new_version)
	@echo "Version $(new_version) has been created and tagged"
	@make upload

release-local:
	@if [ "$(new_version)" = "" ]; then \
		echo "Usage: make release new_version=X.Y.Z"; \
		exit 1; \
	fi
	@echo "Creating new release version $(new_version)..."
	@python scripts/update_version.py $(new_version)
	git add setup.py
	git commit -am "Release version $(new_version)"
	git tag -a v$(new_version) -m "Version $(new_version)"
	git push origin v$(new_version)
	@echo "Version $(new_version) has been created and tagged"
	@make upload

update-deps: $(VENV)
	@echo "Updating frozen dependencies..."
	@$(PIP) freeze > requirements.frozen.txt
	@echo "Dependencies updated."

pre-commit:
	@pre-commit run --all-files

setup:
	@if [ ! -f "setup.sh" ]; then \
		echo "Error: setup.sh not found!"; \
		exit 1; \
	fi
	@chmod +x setup.sh
	@./setup.sh

test-file:
	@if [ "$(file)" = "" ]; then \
		echo "Usage: make test-file file=path/to/test_file.py"; \
	else \
		$(PYTEST) $(file) -v; \
	fi

branch:
	@if [ "$(name)" = "" ]; then \
		echo "Usage: make branch name=feature/new-feature"; \
	else \
		git checkout -b $(name); \
	fi
