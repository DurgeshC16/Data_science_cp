# Cascade-XID — Progress Log

Append-only. New entries go at the **bottom**. Never edit or delete earlier
entries. Metrics are only recorded here once they have actually been measured;
unmeasured items are marked "not measured".

---

## Phase 0 — Repository scaffold (2026-10-18)

### Done

- Created the directory layout: `data/{raw,interim,processed,kb}`, `models/`,
  `reports/figures/`, `configs/`, `src/{data,tier1,gate,explain,rag,tier2,pipeline,eval}`,
  `tests/`, `notebooks/`, `docs/`.
- `requirements.txt`: Python 3.11 dependencies pinned with `==` (numpy, pandas,
  pyarrow, scikit-learn, xgboost, shap, torch, optuna, sentence-transformers,
  faiss-cpu, pydantic, pyyaml, matplotlib, pytest, typer, tqdm, requests,
  streamlit, plus pinned transitive pins for transformers/tokenizers/
  huggingface-hub).
- `requirements-dev.txt`: `ruff` pinned, installed by `make setup` for `make lint`.
- `configs/config.yaml`: paths, seed 42, 70/15/15 split, Tier-1 settings, gate
  thresholds (placeholders), SHAP, RAG, local LLM backend, evaluation cap.
- `src/config.py`: typed pydantic loader with validation (split sums to 1, gate
  range ordering) and `ensure_directories`.
- `src/schema.py`: `OutputRecord` with the nine fixed output fields; ATT&CK ids
  validated against `T####` / `T####.###`.
- `Makefile`: `setup`, `test`, `lint` (+ `venv`, `install`, `dirs`, `clean`);
  overridable `PYTHON`, `VENV`, `CONFIG`.
- `tests/test_smoke.py`: layout, config, output schema, requirements pinning and
  Makefile target checks. `make setup` → PASS, `make test` → 100 % PASS (see
  verification note below).
- `AGENTS.md` with project summary, architecture, output-record contract and
  rules (seeds, leakage, configurability, JSON metrics, typing, no fabrication,
  stop-instead-of-fake, append-only log).

### Decisions

- **`numpy==1.26.4` pinned explicitly.** `faiss-cpu==1.8.0.post1` and
  `shap==0.46.0` are built against the NumPy 1.x ABI on this platform; allowing
  pip to resolve NumPy 2.x risks import failures.
- **Output schema enforces ATT&CK id shape and tier/LLM consistency** rather than
  trusting the LLM. `tier_reached == 1` may not cite retrieved evidence, and
  `llm_invoked` implies Tier 2. This makes fabricated/invalid ids fail loudly.
- **Gate thresholds are placeholders.** `max_prob_threshold: 0.75` and
  `entropy_threshold: 1.0` are starting points only and are *not* reported as
  calibrated; Phase 3 must fit them on the validation split and record the
  resulting escalation rate.
- **LLM backend = Ollama** (`http://localhost:11434`, `llama3.1:8b`) as the
  local open-weight default. No hosted API is used.
- **Config paths resolve from the project root**, not the current working
  directory, so scripts run identically from `make` and from notebooks.
- **`src/config.py` and `src/schema.py` live at the package root** (outside the
  eight pipeline subpackages) because they are shared by every stage.
- **`py.project` is deliberately absent** for now; the repo is not packaged, and
  `pytest.ini` supplies `pythonpath = .`. Revisit if an installable distribution
  is needed.

### Environment / setup notes

- Host: Windows 10/11, Git Bash. `make` was not installed; GNU Make 3.82.90
  (`mingw32-make`) was already present in `C:\MinGW\bin` and was exposed as
  `make.exe` in that directory (outside the repo).
- Python 3.11 is available via the `py -3.11` launcher; the venv is `.venv`
  (git-ignored).
- Setup/install log: `reports/setup.log`.

### Metrics

- None yet — no dataset has been downloaded and no model has been trained.
  Nothing in this phase is a measured result.

### Open issues

- `data/raw/` is empty: no dataset selected (`data.dataset` is `null` in
  `config.yaml`). Phase 1 must record the exact dataset, version/checksum and
  licence before any modelling.
- `data/kb/` is empty: ATT&CK and CVE passages still need to be sourced with
  provenance. No ATT&CK/CVE ids may be written into the KB until then.
- Tier-1 model, calibrator and gate thresholds are uncalibrated placeholders.
- No LLM backend has been contacted; Ollama availability and the model tag are
  unverified.
- `torch` installs the default PyPI wheel for this platform; GPU acceleration is
  not configured.

### Verification

- `make setup` and `make test` were executed on 2026-10-18; results are appended
  below in the Phase 0 verification entry once the install finished.

---

## Phase 1 — Data loading and EDA (2026-10-18)

_This entry is written after the fact: the Phase 1 code was committed as
`986b661` ("Phase 1 implemented") without a log entry. It records only what
actually exists; no metric below is invented._

### Done

- `src/data/loader.py`: loads every `*.csv` in `data/raw/cicids2017/` or
  `data/raw/ciciot2023/` into one DataFrame, normalises column names (strip
  whitespace, snake_case), and handles the documented CICIDS2017 quirks (leading
  spaces in names, the duplicated `Fwd Header Length` column, `±inf` -> `NaN`,
  en/em-dash and casing inconsistencies in labels, UTF-8 -> latin-1 fallback).
  Cleansed frames are written to `data/interim/<dataset>.parquet`; `data.dev_sample`
  / `--dev-sample N` samples with the configured seed.
- Missing or empty dataset folders print download instructions and raise
  `DataNotFoundError`; nothing is synthesised.
- `src/data/eda.py`: `reports/eda_<dataset>.md` plus figures for shape, dtypes,
  class counts/percentages (log-scale bar chart), missing/inf per column,
  constant columns, duplicate-row percentage, correlated pairs above 0.95 and
  ten key feature distributions per class.
- `configs/label_map.yaml`: unified 10-class taxonomy plus binary Normal/Attack
  view. Coverage verified by script: 15/15 CICIDS2017 and 35/35 CICIoT2023 known
  raw labels mapped. Uncertain entries are flagged, not guessed: `HEARTBLEED`
  -> `Other`, `DDoS-SlowLoris` -> `DDoS`, `Backdoor_Malware` -> `Infiltration`.
- `tests/test_loader.py`: 40 tests covering column normalisation, the CICIDS2017
  quirks, `inf`/`NaN` handling, label normalisation, `DataNotFoundError` and
  label-map coverage/taxonomy validity.
- `Makefile`: `load-*` and `eda-*` targets with optional `DEV_SAMPLE`.

### Decisions

- CICIDS2017 labels are upper-cased and dash-normalised for matching; CICIoT2023
  labels keep their original casing. The label map matches case-insensitively
  with whitespace collapsed, so loader casing changes cannot silently drop rows.
- Duplicate handling, identifier removal and the shortcut documentation were
  deliberately deferred to Phase 2 so that they happen in one place before the
  split.

### Metrics

- **Not measured.** `data/raw/` is still empty (only `.gitkeep`), so no loader
  or EDA run has ever seen real data. Class counts, imbalance ratios, missing/inf
  counts, duplicate percentages and correlated pairs are all "not measured".
- Consequently `reports/eda_cicids2017.md` and `reports/eda_ciciot2023.md` do
  **not** exist. They will be generated by `make eda-*` once the CSVs are in place.

### Open issues

- Datasets are not downloaded; every data-dependent acceptance item is blocked.
- `make lint` was not run in Phase 1 because `ruff` (declared in
  `requirements-dev.txt`) was not installed in `.venv`. Phase 2 installed it and
  fixed the resulting findings.

### Verification

- `python -m pytest tests/test_loader.py tests/test_smoke.py` -> PASS on
  2026-10-18.
- Both loaders were invoked against the empty raw directories; each printed
  download instructions and raised `DataNotFoundError` (exit code 1 through the
  CLI), i.e. the stop-instead-of-fake rule holds.

---

## Phase 2 — Leak-free preprocessing and splits (2026-10-18)

### Done

- `src/data/preprocess.py`:
  - structural cleaning before any split: `±inf` -> `NaN`, exact duplicate rows
    dropped, identifier columns dropped by configurable regexes (IPs, MACs, host
    addresses, `flow_id`, timestamps), constant columns dropped;
  - label unification through `configs/label_map.yaml` with a hard failure on
    unmapped labels;
  - zero-day holdout taken **before** splitting and written to
    `data/processed/zeroday_test.parquet` (plus a per-dataset copy);
  - stratified 60/10/10/20 train/val/calibration/test split on the unified
    multiclass label;
  - train-only imputation and scaling; fitted `FittedPreprocessor` (imputer
    values + sklearn scaler) saved to `models/<dataset>_preprocessor.joblib`,
    feature list and dtypes to `models/<dataset>_features.json`;
  - balanced class weights computed from train only and saved to
    `models/<dataset>_class_weights.json`; no resampling anywhere;
  - `reports/split_summary.md` (per-class counts and percentages per split,
    cleaning stats, retained shortcut risks) and `reports/split_summary.json`
    (same numbers plus config, seed and git commit).
  - `--propose-zeroday` prints the rarest attack classes as candidates, sorted by
    measured counts.
- `src/config.py` + `configs/config.yaml`: `split.calibration` added and the
  fractions changed to 60/10/10/20 as specified; new `preprocess` block
  (impute strategy, missing-fraction cutoff, dedupe/constant/identifier
  switches, scaling, class weighting, zero-day classes and their cap).
- `tests/test_preprocess.py`: 20 tests on a synthetic fixture proving split
  disjointness, class-proportion preservation (absolute tolerance 0.02),
  holdout absence from every modelling split, train-only scaler/imputer
  statistics, duplicate removal before splitting, identifier/constant column
  removal, weights-only imbalance handling, artifact round-trip and the report
  contents.
- `Makefile`: `preprocess-cicids2017`, `preprocess-ciciot2023` targets.
- Installed the already-pinned `scikit-learn==1.5.2` and `ruff==0.6.8` into
  `.venv` (they were declared but absent), then fixed four pre-existing lint
  findings in `src/data/eda.py` and `tests/test_smoke.py`.

### Decisions

- **60/10/10/20 replaces the earlier 70/15/15 contract.** The calibration split
  is a first-class, disjoint frame used only for the probability calibrator and
  the gate thresholds; `SplitConfig` now requires all four fractions and
  `test_smoke.py` asserts the new contract (the old assertion summed three
  fractions and would otherwise have been silently wrong).
- **Median imputation is the default.** Flow-feature distributions are
  heavy-tailed; dropping rows would delete whole attack classes. `mean` and
  `drop_rows` remain selectable, and `max_column_missing_fraction` (0.30) drops
  columns that are mostly missing *in train* rather than imputing them.
- **Missingness cutoff and scaling are fitted on train only.** Column removal
  and imputer values are computed from the train frame; val/calibration/test
  inherit them unchanged. Only the label-free structural steps (dedupe,
  constant columns, identifier regexes) run over the whole pool, and those do
  not use the target.
- **Identifier detection is regex-based and deliberately does not match
  `destination_port`.** That column is retained and documented as a shortcut
  risk in both the module docstring and `reports/split_summary.md`; an ablation
  is required before any generalisation claim.
- **Zero-day classes may be given as a unified class or as a raw dataset label.**
  Raw labels are resolved through the label map, so an EDA-derived name works
  directly; the resolved unified classes are logged. Unknown names raise.
- **Rows are never resampled.** Imbalance handling is a class-weight option
  only, computed from train counts and handed to Tier 1 later.
- **The split summary merges per dataset.** Processing a second dataset updates
  only its own section, so both datasets' counts stay in one file.

### Metrics

- **Not measured on real data.** `data/raw/` is still empty, so no split,
  imbalance ratio or class-weight vector exists for CICIDS2017 or CICIoT2023.
  `reports/split_summary.md` and `reports/split_summary.json` are therefore not
  yet present for the real datasets; running `make preprocess-*` will write them.
- The numbers exercised by `tests/test_preprocess.py` come from a synthetic
  fixture and are **not** dataset results.

### Zero-day candidates (proposed, counts not measured)

- CICIDS2017: `HEARTBLEED` and `WEB ATTACK - SQL INJECTION`.
- CICIoT2023: `Backdoor_Malware` and `Uploading_Attack`.
- Rationale: semantic rarity of the attack type, not measured counts. The EDA
  that would confirm the rare classes cannot run while `data/raw/` is empty, so
  `preprocess.zeroday_classes` is left empty until measured class counts exist.
  `--propose-zeroday` will rank real candidates from the loaded data.

### Open issues

- Datasets still absent: the end-to-end preprocessing acceptance artifact
  (`reports/split_summary.md` with real class counts) is blocked.
- Zero-day class choice must be confirmed against measured EDA counts.
- A class with fewer than 4 rows cannot be stratified across four splits;
  preprocessing raises and asks for it to be moved to the zero-day holdout.
- `destination_port` shortcut risk is documented but not yet ablated.

### Verification

- `python -m pytest` -> PASS (all suites).
- `python -m ruff check src tests` -> PASS.
- `python -m src.data.preprocess --dataset cicids2017` against the empty raw
  directory -> download instructions printed, exit code 1 (no fabricated split).
