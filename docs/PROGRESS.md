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
