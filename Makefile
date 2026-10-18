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

.PHONY: setup venv install dirs test lint clean load-cicids2017 load-ciciot2023 eda-cicids2017 eda-ciciot2023 preprocess-cicids2017 preprocess-ciciot2023 tier1-benchmark calibrate-cicids2017 calibrate-ciciot2023 gate-cicids2017 gate-ciciot2023

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

# Phase 1: data loading and EDA
# Optionally pass DEV_SAMPLE=50000 for fast development.
DEV_SAMPLE ?=

load-cicids2017:
	@$(VENV_PY) -m src.data.loader --dataset cicids2017 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

load-ciciot2023:
	@$(VENV_PY) -m src.data.loader --dataset ciciot2023 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

eda-cicids2017:
	@$(VENV_PY) -m src.data.eda --dataset cicids2017 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

eda-ciciot2023:
	@$(VENV_PY) -m src.data.eda --dataset ciciot2023 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

# Phase 2: leakage-safe preprocessing and stratified splits.
preprocess-cicids2017:
	@$(VENV_PY) -m src.data.preprocess --dataset cicids2017 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

preprocess-ciciot2023:
	@$(VENV_PY) -m src.data.preprocess --dataset ciciot2023 --config $(CONFIG) \
		$(if $(DEV_SAMPLE),--dev-sample $(DEV_SAMPLE),)

# Phase 3: train and benchmark Tier-1 candidates on the processed splits.
# DATASET ?= cicids2017; override with `make tier1-benchmark DATASET=ciciot2023`.
DATASET ?= cicids2017

tier1-benchmark:
	@$(VENV_PY) -m src.tier1.train --dataset $(DATASET) --config $(CONFIG)

# Phase 4: probability calibration and the confidence gate.
calibrate-cicids2017:
	@$(VENV_PY) -m src.tier1.calibrate --dataset cicids2017 --config $(CONFIG)

calibrate-ciciot2023:
	@$(VENV_PY) -m src.tier1.calibrate --dataset ciciot2023 --config $(CONFIG)

gate-cicids2017:
	@$(VENV_PY) -m src.gate.gate --dataset cicids2017 --config $(CONFIG)

gate-ciciot2023:
	@$(VENV_PY) -m src.gate.gate --dataset ciciot2023 --config $(CONFIG)

clean:
	@rm -rf $(VENV) .pytest_cache .ruff_cache
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
