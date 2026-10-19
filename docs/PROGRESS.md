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
---

## Phase 3 — Tier-1 training and benchmark (2026-10-19)

### Done

- `src/tier1/train.py` (new):
  - loads the Phase-2 parquet splits plus `models/<dataset>_features.json`,
    verifies every feature is present and that no `flow_id` appears in two
    splits (`load_processed_splits`);
  - three model families: `RandomForestClassifier`, `XGBClassifier`
    (`tree_method=hist`) and `TorchMLP`, a feed-forward network with 2-3 ReLU
    hidden layers and dropout;
  - both tasks: `binary` (Normal/Attack) and `multiclass` (10-class unified
    taxonomy), taken from the Phase-2 label columns;
  - a bounded Optuna search (`tier1.optuna.n_trials` + `time_budget_seconds`,
    TPE sampler seeded from `seed`) that fits on **train** and maximises
    **validation** macro-F1; the test split is not a parameter of
    `tune_hyperparameters`;
  - final fits for every configured seed, then one test evaluation per seed
    (`compute_metrics`: accuracy, macro/weighted precision, recall and F1,
    per-class precision/recall/F1/support, confusion matrix);
  - `measure_latency` (single-flow `predict_proba`, median/p95/mean over
    `benchmark.latency_calls` calls after `warmup_calls`), and
    `measure_batch_throughput` (flows/sec on up to
    `benchmark.throughput_max_rows` test rows);
  - `save_model` / `load_tier1_model` for `.joblib` (sklearn) and `.pt`
    (state dict + architecture) artifacts, returning the on-disk size;
  - `aggregate_runs` (mean +/- sample std across seeds) and
    `select_default_model` (configurable latency-constrained rule);
  - report writers: `reports/tier1_benchmark.json`,
    `reports/tier1_benchmark.md` (comparison table) and
    `reports/figures/tier1_confusion_<task>_<family>.png`;
  - CLI (`python -m src.tier1.train --dataset ...`) with overrides for tasks,
    families, seeds, trial count, time budget and directories.
- `src/config.py` + `configs/config.yaml`: `tier1` reworked into `families`,
  `tasks`, `seeds` (3 seeds), `n_jobs`, `inference_n_jobs`,
  `optuna{n_trials,time_budget_seconds}`, `selection{metric,max_p95_latency_ms,
  tie_breaker}`, `benchmark{latency_calls,warmup_calls,batch_size,
  throughput_max_rows,torch_threads}`, `random_forest`, `xgboost` and `mlp`
  search bounds, with validators (2-3 MLP layers, unique seeds, non-empty
  lists, ordered XGBoost learning-rate bounds).
- `Makefile`: `tier1-benchmark` target (`DATASET ?= cicids2017`).
- `src/tier1/__init__.py`: lazy exports via a module-level `__getattr__` so that
  `python -m src.tier1.train` does not import the module twice (eager imports
  triggered a runpy warning on the `make tier1-benchmark` path).
- `tests/test_tier1.py`: 22 tests on a synthetic fixture in the real on-disk
  format - disjoint splits and duplicate-`flow_id` rejection, hand-computed
  metric values, per-family fit/predict/save/load round trips (RF, XGBoost,
  MLP), latency/throughput measurement, validation-only tuning objective
  (test labels are corrupted so a test-peeking tuner would score differently),
  trial/budget bounds, mean +/- std aggregation, both selection-rule branches,
  inference-thread pinning, full benchmark artifacts, empty-data CLI exit code,
  and config/Makefile contracts.
- Installed the already-pinned `xgboost==2.1.1`, `optuna==4.0.0` and
  `torch==2.4.1` (CPU) into `.venv`; they were declared in `requirements.txt`
  but absent.

### Decisions

- **Model selection uses the validation split, never the test split.** Choosing
  the champion by test macro-F1 would turn the test split into a selection
  signal, which the phase constraints forbid. Latency is a hardware property
  measured on test rows and carries no label information. Rule:
  `highest mean val.macro_f1 subject to mean p95 latency <= 25 ms`, with
  `accuracy` as tie-breaker; when nothing meets the budget the report records
  the lowest-p95 model and sets `selection.fallback_used = true`.
- **Inference runs with a fixed thread count (`tier1.inference_n_jobs = 1`);
  training keeps `n_jobs = -1`.** `RandomForestClassifier.predict_proba`
  parallelises *every* call, so a single-flow measurement with `n_jobs=-1`
  mostly measures worker start-up. On the synthetic smoke fixture, pinning
  moved RF median single-flow latency from 28.9 ms to 3.8 ms (p95 40.3 -> 10.4
  ms) with identical predictions. The setting is recorded in the report and
  stored in the saved estimator.
- **Hyper-parameters are tuned once per (task, family) with `config.seed`, then
  refit per seed.** Running the full search per seed would triple the budget;
  refitting the winning configuration for each of the three seeds still yields
  the required seed spread. Refits use the **train** split only (not
  train+val), because the validation split has already been used for selection.
- **Test is touched only after every task/family finished tuning.**
  `run_benchmark` runs Phase A (all studies, train/val only) then Phase B
  (final fits, metrics, timing, artifacts), so test metrics exist once per seed
  and are labelled "reported, not used for selection".
- **`accuracy_per_ms` is defined as test accuracy divided by median single-flow
  latency (ms)** and written into the JSON `definitions` block so the ratio is
  unambiguous.
- **Class weights, not resampling.** Each family uses balanced class weights
  computed from the training split (`class_weight` dict for RF, `sample_weight`
  for XGBoost, weighted cross-entropy for the MLP). No row is duplicated.
- **MLP early stopping uses the validation split** (the designated tuning
  split); the test split is never used for stopping.
- **Saved artifacts carry the served configuration**: the `.joblib` payload
  holds the estimator with `n_jobs` pinned to the inference value plus labels,
  feature names, seed and hyper-parameters; the `.pt` payload holds the state
  dict plus architecture, so later phases reload without guessing.

### Metrics

- **Not measured on real data.** `data/raw/` is still empty (only `.gitkeep`),
  so no parquet split exists and `reports/tier1_benchmark.json`,
  `reports/tier1_benchmark.md` and the confusion-matrix figures are **not**
  present for CICIDS2017 or CICIoT2023. Training cannot start without data, and
  per AGENTS.md rule 7 nothing was synthesised in its place.
- The only numbers this phase produced come from a **throw-away synthetic
  fixture** (600 rows, 6 features, 3 classes, 3 trials, 3 seeds) used to prove
  the code paths run. They are **not** dataset results and were deliberately
  written outside `reports/`. Observed there: all six task/family combinations
  completed, 18 runs -> 6 aggregates, 6 confusion-matrix PNGs, a markdown table
  with mean +/- std, and the rule selected the MLP for `binary`.
- Latency, throughput, model size, training time and macro-F1 for
  CICIDS2017/CICIoT2023: **not measured**.

### Open issues

- Datasets still absent: every dataset-dependent acceptance item (benchmark
  JSON, comparison table, confusion-matrix figures, recorded default model) is
  blocked until CSVs land in `data/raw/<dataset>/` and
  `make preprocess-<dataset>` runs.
- `tier1.model` in `configs/config.yaml` is still the Phase-0 placeholder
  (`random_forest`); once a real run exists it must be set to whatever
  `reports/tier1_benchmark.json` records as `chosen_default`. Nothing overrides
  it silently at runtime.
- The default search budget (20 trials / 600 s per task+family, 6 combinations)
  has never run against real data; wall-clock on CICIDS2017 (~2.8M flows, 78
  features) is **unknown**. If RF trials exhaust RAM, lower
  `tier1.random_forest.n_estimators_max` or run with `DEV_SAMPLE`.
- MLP throughput depends on `tier1.benchmark.torch_threads` (default 1); other
  values are not comparable across runs.
- The `destination_port` shortcut risk from Phase 2 is still not ablated, so
  benchmark scores must not be read as generalisation claims.

### Verification

- `python -m pytest -q` -> 110 passed (22 new in `tests/test_tier1.py`).
- `python -m ruff check src tests` -> All checks passed.
- `make test` and `make lint` -> exit 0 (2026-10-19).
- Synthetic fixture end to end through the CLI (`--tasks binary,multiclass
  --families random_forest,xgboost,mlp --seeds 42,43,44 --n-trials 3`) -> exit 0;
  wrote `tier1_benchmark.json` (tuning entries, 18 runs, 6 aggregates with
  mean/std, selection block, environment versions, git commit),
  `tier1_benchmark.md` and 6 confusion-matrix PNGs; all 18 saved models reload
  through `load_tier1_model`.
- Real-data path in this environment: `make tier1-benchmark` -> prints
  "feature list not found ... Run `make preprocess-cicids2017` first." and exits
  non-zero, leaving `reports/` without a benchmark artifact
  (stop-instead-of-fake behaviour holds).
---

## Phase 4 — Probability calibration and the confidence gate (2026-10-19)

### Done

- `src/tier1/calibrate.py` (new): fits a probability calibrator per benchmarked
  Tier-1 run on `gate.calibrator_fit_split` and evaluates it before/after with
  expected calibration error (ECE), the multiclass Brier score, log loss and a
  reliability diagram.
  - methods: `none` (identity), `isotonic` (monotone piecewise-constant per
    class), `sigmoid` (Platt scaling on the class score);
  - multiclass = one-vs-rest per class followed by row renormalisation (an
    approximation, stated in the report);
  - a class with a single label value in the fit split is left uncalibrated and
    recorded under `identity_classes` instead of fitting an arbitrary map;
  - artifacts `models/tier1/<dataset>_<task>_<family>_seed<seed>_calibrator.joblib`,
    `reports/calibration_report.json`, `reports/calibration_report.md` and
    `reports/figures/tier1_reliability_<task>_<family>.png`;
  - CLI `python -m src.tier1.calibrate --dataset ...` with task/family/seed and
    directory overrides.
- `src/gate/gate.py` (new): turns probabilities into the two gate scores
  (`max_probability`, predictive `entropy` in nats or bits) and applies the OR
  policy `escalate if max_prob < max_prob_threshold OR entropy >
  entropy_threshold`.
  - thresholds are **fitted** on `gate.threshold_fit_split` (`val`) by sweeping
    quantile levels: `max_prob_threshold = Q_t(max_prob)` and
    `entropy_threshold = Q_{1-t}(entropy)`, selecting the pair whose escalation
    rate lies in `[target_escalation_rate_min, target_escalation_rate_max]` and
    is closest to the band midpoint;
  - if no candidate reaches the band the run raises with the closest achievable
    rate in the message (no threshold is reported as if it had hit the target);
  - reports escalation rate, Tier-1-only accuracy, escalated accuracy, error
    capture rate, per-class escalation rates and the config placeholders side by
    side with the fitted values;
  - scores the zero-day holdout when present, which measures whether the gate
    escalates classes the Tier-1 model never saw;
  - artifacts `reports/gate_report.json`, `reports/gate_report.md`,
    `reports/figures/gate_sweep_<task>_<family>.png` and
    `reports/figures/gate_entropy_<task>_<family>.png`.
- `src/tier1/train.py`: added `trained_model_from_artifact`, `read_benchmark_runs`
  and `resolve_report_path`, so calibration and the gate reuse the benchmark's
  model artifacts instead of re-training.
- `src/config.py` + `configs/config.yaml`: `gate` gains
  `use_calibrated_probabilities`, `calibrator_fit_split`, `threshold_fit_split`,
  `grid_steps` and `entropy_base`; `eval` gains `calibration_bins`. The
  `max_prob_threshold` / `entropy_threshold` values stay PLACEHOLDERS.
- `Makefile`: `calibrate-cicids2017`, `calibrate-ciciot2023`, `gate-cicids2017`,
  `gate-ciciot2023`.
- `tests/conftest.py` (new): shared synthetic processed-split builder and a tiny
  benchmark runner used by the new test modules.
- `tests/test_calibrate.py` + `tests/test_gate.py`: 35 tests — hand-computed ECE,
  Brier and log loss, isotonic improvement on overconfident scores, identity
  handling for degenerate classes, artifact round-trip **and cross-process
  load**, the calibrator being independent of the test split, threshold search
  landing in band, the unreachable-band failure, deterministic thresholds,
  escalation/OR semantics, error-capture accounting, zero-day scoring, raw vs
  calibrated probability modes, missing-artifact exit codes and config
  validators.

### Decisions

- **The calibrator is fitted on the dedicated `calibration` split by default,
  not on validation.** Phase 2 created that split for exactly this purpose, but
  the architecture diagram in AGENTS.md says "calibrator fitted on the
  validation split only". Both readings are disjoint from train and test, so the
  choice is one config line (`gate.calibrator_fit_split: calibration | val`);
  the default follows the dedicated split, and the report records which split
  was used. Gate thresholds are always fitted on `val`.
- **The calibrator artifact is a plain dict, not the pickled dataclass.** The
  first end-to-end smoke run failed with
  `AttributeError: Can't get attribute 'ProbabilityCalibrator' on <module
  'src.gate.gate'>`: a class defined in a module that a CLI executes as
  `__main__` is pickled as `__main__.ProbabilityCalibrator`, so the artifact
  written by `make calibrate-*` could not be loaded by `make gate-*`. Fixed by
  storing `{"metadata": ..., "estimators": ...}` and rebuilding on load, plus a
  regression test that loads the artifact from a separate interpreter.
- **Model selection is not involved here: the gate is per run.** Every
  benchmarked (task, family, seed) gets its own calibrator and its own
  thresholds, so the report shows the spread instead of a single number. The
  default Tier-1 family still comes from the Phase-3 selection rule.
- **Test is scored, never fitted on.** Thresholds and calibrators are fitted on
  `val` / `calibration`; the test split is only scored with the frozen
  thresholds, and a test asserts that corrupting the test labels leaves both the
  calibrators and the thresholds bit-identical.
- **A threshold pair that misses the 5–15 % band is a failure, not a footnote.**
  `fit_thresholds` raises with the closest achievable rate, and the CLI exits 1.
- **The config placeholders are reported, not silently replaced.** Each gate
  record keeps a `placeholder_thresholds` block labelled "config placeholders,
  not FITTED thresholds", so the placeholder values can never be mistaken for
  calibrated ones.
- **Entropy unit is explicit** (`gate.entropy_base: natural | bits`); the
  placeholder `entropy_threshold: 1.0` is only meaningful once the unit is known,
  so the unit is recorded in every report.
- **Zero-day scoring is defined on the multiclass taxonomy.** Phase 2 removes
  whole multiclass classes; for the binary task those rows are ordinary
  `Attack` flows, so only the multiclass record measures unseen-class behaviour.
  This is stated in the gate report's `definitions`.
- **Multiclass calibration is an approximation** (per-class one-vs-rest, then
  renormalisation) and is labelled as such in the JSON `definitions`, so ECE
  gains are not over-read.

### Metrics

- **Not measured on real data.** `data/raw/` is still empty, so
  `reports/calibration_report.*`, `reports/gate_report.*` and the reliability /
  gate figures do **not** exist for CICIDS2017 or CICIoT2023. No ECE, Brier,
  log-loss, threshold or escalation rate for the real datasets has been
  measured.
- On the throw-away synthetic fixture (720 + 60 holdout rows, 4 features, 3
  classes, 2 seeds, all three families) the pipeline ran end to end and
  produced: calibration written for 12 runs, gate reports for 12 runs, and 12
  gate figures plus 6 reliability figures. Observed there, **not dataset
  results**: multiclass MLP seed 43 test ECE fell from 0.3980 to 0.1678, fit-split
  escalation rates landed at 0.056–0.139 (inside the 0.05–0.15 band), and the
  zero-day escalation rate ranged from 0.000 to 1.000 across runs depending on
  how far the fixture's synthetic holdout rows sit from the training
  distribution. Nothing from the fixture was copied into `reports/`.
- Real-data escalation rate, error-capture rate and the size of the Tier-2 load
  it implies: **not measured**.

### Open issues

- `reports/tier1_benchmark.json` does not exist yet, so `make calibrate-*` and
  `make gate-*` stop with "benchmark report not found ... Run
  `make tier1-benchmark DATASET=cicids2017` first." (exit 1) rather than
  producing anything.
- `gate.max_prob_threshold` / `gate.entropy_threshold` in `configs/config.yaml`
  are still Phase-0 placeholders; after a real run they must be replaced with
  the fitted values from `reports/gate_report.json`.
- `tier1.calibration` is a single global method; per-family calibration (for
  example isotonic for trees, temperature scaling for the MLP) is not supported.
- ECE depends on `eval.calibration_bins` (15); values from runs with different
  bin counts are not directly comparable.
- Sigmoid calibration emits a `DeprecationWarning` from scikit-learn 1.5.2's
  L-BFGS-B call under the installed SciPy 1.17. It is a third-party warning and
  was left unsuppressed; it may disappear with a newer pinned scikit-learn.
- The gate's zero-day measurement is only meaningful for the multiclass task
  (see Decisions) and requires `preprocess.zeroday_classes` to be set.

### Verification

- `python -m pytest` -> 145 passed (35 new), `python -m ruff check src tests` ->
  All checks passed; `make test` and `make lint` -> exit 0.
- Three-stage CLI smoke on the synthetic fixture (benchmark 12 runs ->
  calibrate 12 runs -> gate 12 runs) -> exit 0 at every stage; wrote
  `tier1_benchmark.json`, `calibration_report.json`, `gate_report.json` and
  their markdown tables plus 24 figures; every calibrator reloads in a fresh
  interpreter (regression test).
- Empty-data path: `make calibrate-cicids2017` and `make gate-cicids2017` each
  print the benchmark-missing message and exit non-zero, leaving `reports/` and
  `models/` without new artifacts.

---

## Phase 1–4 real-data verification attempt (2026-10-19) — BLOCKED

### Done

- Confirmed the checked-out branch is local `master` (4 commits ahead of
  `origin/main`); the working tree was clean before this entry.
- Ran `make test`: **145 passed**, 6 SciPy `disp`/`iprint` deprecation warnings,
  exit 0 (54.81 s).
- Ran `make lint`: Ruff reported all checks passed, exit 0.
- Reviewed `src/data/preprocess.py`, `src/tier1/train.py`,
  `src/tier1/calibrate.py` and `src/gate/gate.py` for leakage, selection and
  random seeds. Prioritised observations were reported in the session before
  this log edit: (1) the calibration default uses the dedicated calibration
  split while `AGENTS.md` says validation; (2) candidate selection uses p95
  latency measured on test rows (but does not use test labels); (3) duplicate
  removal includes the raw label despite being described as label-free. No
  unseeded stochastic operation was identified in the reviewed paths.
- Checked `data/raw/`: it contains only `.gitkeep`; neither
  `data/raw/cicids2017/` nor `data/raw/ciciot2023/` contains CSV files.

### Blocker and next action

- Real-data pipeline not run. `load`, EDA, preprocessing, Tier-1 benchmark,
  calibration, and gate were all stopped before execution because no real
  dataset is present. DEV_SAMPLE smoke and full-data runs are both **not
  measured**. No zero-day classes, model default, or fitted gate thresholds
  were selected or written; existing config values remain unchanged.
- To unblock CICIDS2017, download the **MachineLearningCVE** ZIP (the 8 daily
  CSV archive, approximately 450 MB) from the official UNB CIC page:
  <https://www.unb.ca/cic/datasets/ids-2017.html> (free registration), or use
  the mirror listed by `src/data/loader.py`:
  <http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCVE.zip>.
  Extract all eight `*.csv` files directly into `data/raw/cicids2017/`. The
  loader lists the expected capture-day filenames and records the dataset
  licence as CC BY 4.0; retain/cite the UNB CIC source.
- CICIoT2023 was also absent. It is optional for the CICIDS2017 run. If wanted,
  download the preprocessed CSV archive from the official UNB page
  <https://www.unb.ca/cic/datasets/iotdataset-2023.html>, or the all-parts
  Mendeley archive (DOI `10.17632/7zcswnmkfd`, approximately 8.6 GB) at
  <https://data.mendeley.com/datasets/7zcswnmkfd>, and place extracted CSVs in
  `data/raw/ciciot2023/`. Confirm source terms/licence when downloading; the
  loader documents CC BY 4.0 and citation of the UNB CIC IoT 2023 paper.

### Open issues

- Once CICIDS2017 CSVs are present, rerun the requested pipeline in order,
  first with `DEV_SAMPLE=<N>` for a smoke run, then on the full data. Derive
  zero-day choices from that dataset's measured EDA counts before preprocessing.
- Append measured Phase 1–4 verification results only after successful real
  runs. Then set `tier1.model` from the benchmark `chosen_default` and replace
  gate placeholders only with the fitted values in the gate report.

---

## Tier-1 explanations and ATT&CK lookup (2026-10-19) — PARTIAL

### Done

- Added `src/explain/shap_explainer.py` with cached SHAP `TreeExplainer` support
  for binary Random Forest / XGBoost models, top-k signed Attack contributions,
  absolute magnitudes, train-only percentile fitting, qualitative wording, and a
  configurable per-flow timing helper. Positive SHAP values are toward the
  model's `Attack` output; negative values are toward `Normal`. MLP and
  multiclass models are rejected rather than assigned an ambiguous direction.
- Added `configs/feature_dictionary.yaml`. Since no processed feature artifact
  exists, feature-specific meanings remain empty and the fallback is explicitly
  `TODO`; percentile words are derived from training rows when a feature list is
  available. No dataset-specific feature meanings or percentile cut points have
  been inferred.
- Added `src/explain/attack_lookup.py` and `configs/attack_lookup.yaml`. Draft
  candidates are recorded for DoS/DDoS (`T1498`, Network Denial of Service),
  PortScan/Recon (`T1046`, Network Service Discovery), and BruteForce (`T1110`,
  Brute Force). All taxonomy entries are marked
  `needs_human_verification: true`; candidate entries cite their MITRE ATT&CK
  technique pages. Empty candidate lists are intentional, not a claim that no
  technique applies.
- Added `retrieval_query` to combine protocol, port, explicitly supplied service
  name, and qualitative SHAP feature descriptions. Service names are not
  inferred from ports.
- Added `tests/test_explain.py` for deterministic Tree SHAP output shape,
  train-percentile wording, query construction, candidate verification flags,
  unsupported-model rejection, and the timing helper's output contract. These
  are unit tests with a small synthetic tree, not the project report's worked
  flows.
- Added SHAP and feature/lookup/timing settings to `configs/config.yaml` and its
  typed config model.

### Verification and limitations

- `make test` -> **151 passed**, 918 warnings, exit 0 (2026-10-19). Warnings
  include SHAP plotting-color deprecations and the existing scikit-learn/SciPy
  calibration deprecations.
- `make lint` -> all checks passed (2026-10-19).
- Environment repair: installed declared `shap==0.46.0` and restored declared
  `numpy==1.26.4`, `pandas==2.2.3`, and `pyarrow==17.0.0` in `.venv`. These are
  environment changes only; project dependency declarations were not changed.
- Actual Tier-1 SHAP per-flow overhead: **not measured**. There is no trained
  model artifact under `models/`, and the repository contains no project report
  with the specified normal HTTPS, SYN flood, and ambiguous SMB examples.
  Consequently, acceptance tests against those exact examples were not
  possible; no substitute flow values or real-model timing were fabricated.
- Feature-specific dictionary coverage is **not complete** until a real
  `models/<dataset>_features.json` is available and every listed feature has a
  reviewed meaning. The TODO fallback is deliberate.
- The ATT&CK mapping remains a draft and is not validated ground truth until a
  human reviews each label-to-technique association.

### Open issues

- Provide the report path or the three worked-example records and a fitted
  binary tree-model artifact to complete exact-example tests and measure
  per-flow overhead.
- After real preprocessing, populate and review `configs/feature_dictionary.yaml`
  for every model feature; fit its percentile cut points only on the training
  split.
- Human-review the ATT&CK candidates before using the lookup as Tier-2 mapping
  ground truth or publishing mapping accuracy.

---

## Local ATT&CK/CVE knowledge base and retrieval (2026-10-19)

### Done

- Downloaded `data/kb/enterprise-attack.json` from the official
  `mitre-attack/attack-stix-data` GitHub release (`v20.0`). Download date:
  2026-10-19. Verified file size **55,271,014 bytes** and SHA-256
  `81113ba98afa2928f3dcff62022539f7ce30c315c32e14422f11ff95ce534352` against
  the release asset metadata. Source record is in `data/kb/attack_source.json`.
- Added `src/rag/build_kb.py`: parses active ATT&CK `attack-pattern` objects,
  including sub-techniques; excludes revoked/deprecated objects; retains IDs,
  names, tactics, description, detection guidance and platforms; chunks source
  text with configured overlap and repeats identity/metadata in each chunk.
- Added the curator-owned `data/kb/cve_curated.csv` template with exactly three
  clearly marked example-placeholder rows. No CVE identifier or description was
  supplied from memory; example rows are excluded from the index.
- Added configurable Sentence Transformers embeddings and a normalized FAISS
  inner-product index under `models/kb_index/`, with passage metadata in
  `documents.jsonl`. Current encoder: `sentence-transformers/all-MiniLM-L6-v2`;
  switching is configurable through `rag.embedding_model`.
- Added `src/rag/retriever.py` with `retrieve(query, k=3)` dense retrieval and
  optional BM25+dense hybrid ranking. Results include text, technique/CVE ID,
  score and metadata.
- Added `data/kb/retrieval_eval.csv` with **60** SHAP-style query candidates.
  Expected technique IDs are validated against `configs/attack_lookup.yaml`;
  every row has `needs_review=true` because mappings and queries are unreviewed.
- Added `make kb-build` (builds index and runs evaluation) and `make kb-eval`.
  Evaluation JSON and Markdown are saved as `reports/retrieval_eval.json` and
  `.md`; reports include configuration, source version and Git provenance.
- `make kb-build` completed successfully: **706 unique technique IDs**, **1,133
  text chunks**, 384-dimensional vectors; no curated CVE documents were indexed.

### Measured retrieval evaluation (provisional; queries need human review)

Using the 60 currently unreviewed query rows and the real ATT&CK v20.0 index:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.0333 | 0.1500 | 0.3667 | 0.1225 |
| Hybrid | 0.0000 | 0.1000 | 0.1500 | 0.0544 |

- Dense Recall@3 is **below 0.70** and is flagged in the report with suggested
  fixes. Hybrid is worse on this provisional evaluation; it is reported as
  measured, not hidden. Do not treat these metrics as validated retrieval
  quality until a human reviews the candidate labels and queries.
- `make test` -> **157 passed**, 918 warnings, exit 0. Warnings are third-party
  SHAP/scikit-learn deprecations.
- `make lint` -> all checks passed, exit 0.

### Open issues

- Human-review all retrieval evaluation rows and ATT&CK label mappings. Then
  rerun `make kb-eval`; expected technique IDs are provisional, not ground truth.
- Curate CVE records from cited authoritative sources before indexing them. The
  current CVE document count is zero by design.
- Dense and hybrid retrieval quality is currently poor against the unreviewed
  candidate set. Use the report's suggested fixes and rerun measurement after
  reviewing query/label validity; do not tune against test data later used for a
  final report.

---

## Tier-2 strict LLM explainer (2026-10-19) — IMPLEMENTED, REAL-MODEL RUN BLOCKED

### Done

- Added `src/tier2/llm_backend.py`: abstract `LLMBackend`, local Ollama
  `/api/generate` client, and deterministic `MockBackend`. Ollama model tag,
  temperature, token limit and timeout come from `configs/config.yaml`. Calls
  log prompt tokens, output tokens and latency; failed HTTP calls log estimated
  prompt tokens, zero output tokens and measured request latency.
- Added `src/tier2/prompt.py` with the requested prompt as version `prompt_v1`.
- Added `src/tier2/validate.py`: strict Pydantic JSON schema, 3–4 sentence
  explanation check, canonical technique ID/name validation against the built
  local ATT&CK passages (or source STIX bundle), retrieved-passage grounding
  warning, and flags for numeric explanation values absent from flow/SHAP input.
  Failed generation or validation retries up to the configured maximum of two
  retries, then returns `Needs Human Review` while retaining `tier1_verdict`.
- Every parsed Tier-2 result includes `prompt_version`; `Tier2Run` carries raw
  response(s), parsed response, model, attempt count, per-call telemetry and
  cache status. Result cache keys are SHA-256 of canonical flow JSON, prompt
  version and model tag. Raw and parsed runs can be saved to
  `reports/tier2_samples/` through `save_tier2_sample`.
- Added mock tests for valid JSON, malformed JSON and retries, a source-backed
  technique ID absent from the supplied test KB, timeout fallback, retrieval and
  numeric-value warnings, prompt provenance and cache hits.
- The existing `OutputRecord` schema was not changed: Tier-2's requested
  categorical confidence and response contract are represented in a separate
  validated Tier-2 result. The original nine-field final pipeline record
  remains intact until an end-to-end adapter is specified.

### Verification and real-model blocker

- `make test` -> **164 passed**, 918 third-party warnings, exit 0.
- `make lint` -> all checks passed, exit 0.
- Local Ollama probe at `http://localhost:11434/api/tags` -> connection refused;
  no local model tag or response was available. Real model used: **not
  measured**. Real per-call prompt/output token counts and latency: **not
  measured**. No raw/parsed real-model samples were saved.
- The repository has no report containing the three worked example flows, so
  those examples were not reconstructed or replaced with fabricated inputs.
  Mock tests are code-path tests only, not real flow explanations.

### Open issues

- Start Ollama and pull/configure a local model tag (for example the user-listed
  model names), then rerun the three report examples and save raw/parsed outputs
  under `reports/tier2_samples/`.
- Provide the report location or example flow, SHAP and passage inputs. The
  example-level acceptance and real-model latency measurement remain blocked.
- The returned Tier-2 structure is separate from `OutputRecord`; wiring it into
  the full pipeline requires an explicit policy for converting categorical
  confidence to the final record's numeric confidence without guessing.

---

## Demo and report draft (2026-10-19) — UI/DOCS IMPLEMENTED, RUN BLOCKED

### Done

- Added `app/demo.py` and `make demo`. The Streamlit app accepts CSV uploads,
  lets users select rows from the uploaded file, displays Tier-1 confidence and
  gate routing, and for escalations presents the report card, ATT&CK link,
  recommended action, retrieved evidence and SHAP contribution chart when those
  artifacts are available. A live escalation counter updates during processing.
- Cost savings are shown only after the user enters prompt/output token prices.
  The displayed all-flows baseline assumes one call per flow at the mean observed
  per-call token count; this assumption is labeled in the UI. Without a priced
  uncached LLM call, cost is reported as not estimated. No token prices are
  hard-coded.
- Built-in sample flows were not added because the repository's report examples
  and real datasets are absent. Users can choose sample rows from their uploaded
  CSV instead. When fitted artifacts are missing, the app displays the blocker
  and does not generate placeholder predictions.
- Added `reports/draft_report.md` with the requested sections, TODO markers for
  unsupported claims, and citations to existing report files for numeric results.
  The only metric table uses the provisional retrieval figures in
  `reports/retrieval_eval.json`, explicitly marked unreviewed.
- Added `reports/expectations_audit.md`. The original report is not present in
  this repository, so the audit covers only expectations discoverable in
  `AGENTS.md`, config and current result reports; it does not claim to enumerate
  the missing report's every statement.
- Rewrote `README.md` with clean-clone setup, official dataset and ATT&CK source
  instructions, available pipeline commands, demo/cascade commands, repository
  layout and a clear note that E1–E7 experiment targets are not yet implemented.

### Verification and limitations

- Streamlit `AppTest` rendered the landing state with no flow upload and no
  trained artifacts; no predictions were made.
- `make test` -> **168 passed**, 918 third-party warnings, exit 0.
- `make lint` -> all checks passed. `ruff check app` -> all checks passed.
- The UI cannot complete a cascade run here: `data/raw/` and `data/processed/`
  contain only `.gitkeep`; Tier-1 benchmark/gate reports and model artifacts are
  absent; Ollama is unavailable. Full upload-to-record behavior is therefore not
  measured.
- A clean clone can install the pinned software and follow documented data
  setup, but it cannot reproduce E1–E7: those Make targets/runners do not exist
  yet, and the requisite data, artifacts, human-reviewed mapping and local model
  must be supplied first. No Pareto figure or experimental values were added.

### Open issues

- Supply the original report if a complete expectation-by-expectation audit is
  required, plus the report's example CSV flows if they are to be offered as
  curated demo samples.
- Run `make demo` after downloading real data, creating benchmark/calibration/
  gate/model artifacts, building the KB and starting Ollama. Verify all UI
  outputs against the real records and `OutputRecord` schema.
- Implement and validate the E1–E7 experiment targets before claiming clean-clone
  reproduction of the full experiment suite.

---

## Full cascade orchestration (2026-10-19) — IMPLEMENTED, DATA RUN DEFERRED

### Done

- Added `src/pipeline/cascade.py` with batched Tier-1 inference, optional fitted
  probability calibration, fitted gate-profile selection from the gate report's
  validation sweep, and ordered per-flow outputs. Only gate-escalated rows enter
  the concurrent Tier-2 executor.
- Tier-1-only outputs use the static ATT&CK lookup when a multiclass label has a
  candidate, and explicitly disclose that draft tags require human verification.
  Tier-2 routes through SHAP, retrieval query construction, local retrieval,
  LLM validation and fallback handling.
- Every output passes `OutputRecord` and retains exactly its nine AGENTS.md
  fields. Categorical Tier-2 confidence is not converted to a guessed numeric
  scale; the canonical record retains the calibrated Tier-1 numeric confidence.
- Added iterator/streaming and batch APIs, vectorized Tier-1 batches, a
  configurable Tier-2 concurrency limit, and JSONL writing. Run stats include
  total/escalated flows, LLM attempts, cache hits, failed attempts, fallbacks,
  per-stage latency aggregates and prompt/output token totals; the configured
  destination defaults to `reports/run_stats.json`.
- Added `python -m src.cli run --input ... --gate-profile p10 --out ...`, plus a
  `run-cascade` Make target. Artifact loading requires the benchmark-selected
  model, fitted preprocessor, gate sweep and configured KB index; it does not
  invent thresholds or substitute unprocessed input features.
- Added synthetic routing integration tests for the described Normal, SYN-flood,
  and ambiguous-SMB scenarios. The tests assert routing, schema, static tagging,
  vectorized batches and counters; they are explicitly not the report's exact
  flow records.
- Made Tier-2 cache writes merge under a lock so concurrent escalated rows do
  not overwrite one another's cache entries.

### Verification and deferred acceptance runs

- `make test` -> **167 passed**, 918 third-party warnings, exit 0.
- `make lint` -> all checks passed, exit 0.
- CLI help was verified for `python -m src.cli run`.
- The report with the three original flow records is not present; exact report
  reproduction was not possible. `data/raw/` still lacks real flow data and
  there are no Tier-1 benchmark/model/preprocessor/gate artifacts.
- Per the user's request to run the full project after phases are complete, the
  10,000-flow mock run and the 200-flow real-model run are **deferred, not
  measured**. No `reports/run_stats.json` from a real project run or
  `reports/tier2_samples/` real-model outputs were generated.
- Local Ollama was previously probed at `http://localhost:11434/api/tags` and
  refused the connection. A real-model run still requires Ollama, a configured
  model, real flows, and fitted artifacts.

### Open issues / runtime constraints

- Current Tree SHAP support is limited to binary Random Forest/XGBoost. The
  artifact loader leaves the explainer unavailable for multiclass or MLP models;
  escalations then become explicit human-review fallbacks. Extend SHAP support
  or choose a supported model/task before claiming end-to-end Tier-2 coverage.
- The static lookup is keyed by unified multiclass labels. A binary `Attack`
  prediction alone does not identify an attack class, so it receives no static
  technique tag rather than a guessed one.
- Run the deferred acceptance checks after real data/model artifacts and Ollama
  are available; verify all emitted JSONL records against `OutputRecord` and
  inspect the generated `reports/run_stats.json`.

---

## Report-claim experiment suite E1–E7 (2026-10-19) — BLOCKED

### Availability check

- Read the project rules and current progress before checking artifacts.
- `data/raw/` contains only `.gitkeep`; `data/processed/` also contains only
  `.gitkeep`. No CICIDS2017/CICIoT2023 CSVs, processed test split, or zero-day
  holdout are available.
- No benchmark, calibration or gate JSON report is present in `reports/`; there
  are no Tier-1 model/preprocessor/calibrator artifacts under `models/tier1/`.
  The ATT&CK FAISS index exists, but `data/kb/cve_curated.csv` contains only
  example placeholders.
- Local Ollama at `http://localhost:11434/api/tags` is unavailable. The existing
  ATT&CK mapping and its 60 retrieval queries are marked for human review and
  cannot serve as validated mapping ground truth.
- Cost assumptions (GPU-hour price or per-token prices) are not configured.

### Decision

- E1–E7 empirical results are **not measured**. `reports/results.md` records the
  blocked status for each experiment. No experiment Make targets, metrics JSON,
  confidence intervals, Pareto figure, random-control results, ablation results,
  zero-day results, mapping accuracy or rater cases were generated from absent
  data. The existing provisional retrieval evaluation is not substituted for
  these experiments.
- E3's equal-cost random-escalation control is essential to the central claim;
  without it, confidence gating is **not established to beat random escalation**.
- To unblock: download CICIDS2017 MachineLearningCVE CSVs into
  `data/raw/cicids2017/` (official UNB source and eight-file instructions are in
  the earlier Phase 1–4 blocked entry above), run the real load/preprocess/
  benchmark/calibrate/gate pipeline, start Ollama, review `attack_lookup.yaml`,
  and configure cost assumptions. Then generate and execute E1–E7 against the
  held-out data and obey `eval.sample_cap` for LLM calls.
- Per the user's instruction to run the project after the phases are complete,
  the 10,000-flow mock and 200-flow real-model runs also remain deferred.
