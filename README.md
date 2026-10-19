# Cascade-XID

Cascade-XID is a research prototype for a confidence-gated intrusion-detection
cascade. Tier 1 scores each flow; uncertain flows may be escalated to a local
LLM with SHAP context and retrieved ATT&CK/CVE passages. It is **not a production
IDS**.

## Status and evidence

The empirical E1–E7 suite is currently blocked: the real datasets, processed
splits, trained benchmark artifacts, and a local Ollama service are not present.
`reports/results.md` marks those experiment results as **not measured**. Do not
interpret mock tests or the provisional retrieval query set as real detection
results. The known-claims audit is in `reports/expectations_audit.md`; a source
copy of the original report is not in this repository.

## Setup

Requirements are pinned for Python 3.11. From a clean clone:

```bash
make setup
make test
make lint
```

`make setup` creates `.venv`, installs `requirements-dev.txt` (including runtime
requirements), and creates configured directories. On Windows, GNU Make and
the Python 3.11 launcher used by the Makefile must be available.

## Data

### CICIDS2017

1. Register at the [official UNB CIC IDS 2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Download the **MachineLearningCVE** archive and extract its eight CSV files
   into `data/raw/cicids2017/`.
3. Record the source/version and licence for your experiment; do not edit the
   labels to improve results.

The loader also documents the current archive mirror and expected filenames in
`src/data/loader.py`. CICIoT2023 is optional and its official download
instructions are in the same loader.

## Reproduce the implemented data/model pipeline

After placing real CSVs in the raw-data directory:

```bash
make load-cicids2017
make eda-cicids2017
```

Review measured EDA counts before choosing zero-day classes. Set
`preprocess.zeroday_classes` in `configs/config.yaml`, then run:

```bash
make preprocess-cicids2017
make tier1-benchmark DATASET=cicids2017
make calibrate-cicids2017
make gate-cicids2017
```

Only use thresholds and model selections emitted by those real runs. For a
development sample, `DEV_SAMPLE=<rows>` is available on load/EDA/preprocess
targets; sample results are not test-set evidence.

To build the local knowledge base, download the versioned Enterprise STIX bundle
from the official [`mitre-attack/attack-stix-data` release](https://github.com/mitre-attack/attack-stix-data/releases)
recorded in `data/kb/attack_source.json`, place it at
`data/kb/enterprise-attack.json`, then run:

```bash
mkdir -p data/kb
curl -L "https://github.com/mitre-attack/attack-stix-data/releases/download/v20.0/enterprise-attack.json" -o data/kb/enterprise-attack.json
```

Check the release/version and checksum against the source metadata before
rebuilding if you choose a different ATT&CK release.

```bash
make kb-build
```

The build writes the FAISS index under `models/kb_index/` and a provisional
dense/hybrid report under `reports/retrieval_eval.*`. Review the rows in
`data/kb/retrieval_eval.csv` and mappings in `configs/attack_lookup.yaml` before
using their labels as ground truth. Add only CVE descriptions curated from
cited sources to `data/kb/cve_curated.csv`; its example rows are placeholders.

## Run the cascade and demo

After dataset preprocessing, benchmark, calibration, gate fitting, KB build,
and local Ollama setup, a CSV can be scored with:

```bash
python -m src.cli run --input flows.csv --gate-profile p10 --out out.jsonl
```

Set `data.dataset` in config or pass `--dataset`. The run writes per-flow
`OutputRecord` JSONL and `reports/run_stats.json`.

Launch the Streamlit review UI with:

```bash
make demo
```

The app accepts an uploaded CSV and lets you select rows from that file. It
does not include fabricated built-in sample flows. Until the benchmark/model
artifacts and Ollama are available, it reports the missing prerequisites rather
than producing placeholder predictions. Cost estimates require user-entered
token prices and are explicitly assumptions, not billing measurements.

## Experiments and report draft

The requested experiments are described in `reports/results.md`, but **E1–E7
Make targets and empirical experiment runners are not yet implemented**. The
report currently records why each result is not measured. Do not treat the
existing retrieval evaluation as a substitute for detection, ablation,
zero-day, cost, or human-rating experiments.

`reports/draft_report.md` contains the requested report sections and cites the
available report files. Unsupported claims are marked TODO. The known
expectations-versus-evidence audit is in `reports/expectations_audit.md`.

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/` | Project configuration, label map, feature dictionary and draft ATT&CK lookup |
| `data/raw/` | User-downloaded source flow datasets |
| `data/interim/`, `data/processed/` | Loaded data and leakage-safe splits |
| `data/kb/` | ATT&CK source metadata, CVE curation template and retrieval queries |
| `models/` | Fitted Tier-1 models, calibrators and generated KB index (local artifacts) |
| `src/data/` | Data loading, EDA and preprocessing |
| `src/tier1/`, `src/gate/` | Tier-1 training/calibration and escalation gate |
| `src/explain/`, `src/rag/`, `src/tier2/` | SHAP, retrieval, local LLM and validation |
| `src/pipeline/`, `src/cli.py`, `app/` | Cascade orchestration, CLI and Streamlit demo |
| `src/eval/`, `tests/` | Evaluation modules and tests |
| `reports/` | Experiment status/results, figures and run artifacts |
| `docs/PROGRESS.md` | Append-only implementation and verification log |
