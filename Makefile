# Cascade-XID — reproducible entry points.
# Every pipeline script must be reachable from a target in this file and read its
# parameters from configs/config.yaml.

VENV ?= .venv

ifeq ($(OS),Windows_NT)
SHELL := /usr/bin/bash
VENV_PY := $(VENV)/Scripts/python.exe
PYTHON ?= py -3.11
else
VENV_PY := $(VENV)/bin/python
PYTHON ?= python3.11
endif

CONFIG ?= configs/config.yaml

.PHONY: setup venv install dirs test lint clean

setup: venv install dirs
	@$(VENV_PY) -c "from src.config import load_config; c = load_config('$(CONFIG)'); print('config OK:', c.project_name, '| seed', c.seed, '| splits', c.split.train, c.split.val, c.split.test)"

venv:
	@test -x $(VENV_PY) || $(PYTHON) -m venv $(VENV)

install: venv
	@$(VENV_PY) -m pip install --upgrade pip
	@$(VENV_PY) -m pip install -r requirements-dev.txt

dirs:
	@$(VENV_PY) -c "from src.config import load_config, ensure_directories; ensure_directories(load_config('$(CONFIG)'))"

test:
	@$(VENV_PY) -m pytest

lint:
	@$(VENV_PY) -m ruff check src tests

clean:
	@rm -rf $(VENV) .pytest_cache .ruff_cache
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
