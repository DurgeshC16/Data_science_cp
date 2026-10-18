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
