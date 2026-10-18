# AGENTS.md — working agreement for Cascade-XID

This file is binding for any human or AI agent working in this repository.

## 1. Project summary

**Cascade-XID** is a research codebase for a confidence-gated, two-tier intrusion
detection system:

- **Tier 1** — a fast classical model (Random Forest / XGBoost) scores *every*
  network flow.
- **Confidence gate** — flows whose Tier-1 score is uncertain are escalated. The
  target escalation rate is roughly **5–15 %** of flows; the exact thresholds are
  calibrated, not guessed (see `configs/config.yaml` → `gate`).
- **Tier 2** — for escalated flows only: SHAP feature attributions + retrieval
  augmented generation (RAG) over MITRE ATT&CK / CVE passages + a **local, open
  weight LLM**. Tier 2 returns a classification, a plain-English explanation, an
  ATT&CK technique ID and a recommended action.

The system is a research prototype. It is **not** a production IDS.

## 2. Architecture

```
network flows (raw PCAP → parsed flow table, data/raw → data/interim → data/processed)
        │
        ▼
┌─────────────────────────────┐
│ Tier 1: RF / XGBoost        │  fast scoring of every flow
│ + probability calibration   │  (calibrator fitted on the validation split only)
└─────────────┬───────────────┘
              │ class probabilities
              ▼
┌─────────────────────────────┐
│ Confidence gate             │  escalate if max-prob < threshold
│ src/gate                    │  or predictive entropy > threshold
└──────┬──────────────────────┘
       │ ~5–15 % escalated
       ▼
┌───────────────────────────────────────────────────────────────┐
│ Tier 2 (src/tier2, src/explain, src/rag)                       │
│  1. SHAP attributions for the escalated flow                   │
│  2. FAISS retrieval over ATT&CK / CVE passages (data/kb)       │
│  3. Local open LLM prompt grounded in SHAP + retrieved passages│
│  4. Output validation: classification, explanation, ATT&CK id, │
│     recommended action; ids must appear in the retrieved text  │
└───────────────────────────────┬───────────────────────────────┘
                                ▼
                     OutputRecord (src/schema.py)
                                │
                                ▼
              reports/ JSONL results + JSON metrics + figures
```

| Directory | Responsibility |
| --- | --- |
| `src/data` | ingestion, cleaning, splitting (leakage-safe) |
| `src/tier1` | RF/XGBoost training, calibration, inference |
| `src/gate` | uncertainty scoring and escalation policy |
| `src/explain` | SHAP attribution for escalated flows |
| `src/rag` | KB ingestion, embeddings, FAISS index, retrieval |
| `src/tier2` | prompt construction, local LLM calls, output parsing |
| `src/pipeline` | end-to-end orchestration |
| `src/eval` | metrics, bootstrap intervals, figures |
| `configs/` | the single YAML configuration |
| `data/kb/` | ATT&CK / CVE passages and the FAISS index |
| `reports/` | metrics JSON, run logs, `figures/` |
| `tests/` | pytest suite |

## 3. Final output record

Every evaluated flow produces one record with **exactly** these fields
(validated by `src/schema.py::OutputRecord`):

| Field | Type | Meaning |
| --- | --- | --- |
| `flow_id` | `str` | stable identifier of the scored flow |
| `final_classification` | `str` | final label (benign or attack class) |
| `explanation` | `str` | plain-English rationale |
| `attack_technique` | `str \| null` | ATT&CK technique id (`T####` / `T####.###`), null if none |
| `confidence` | `float` in [0, 1] | final confidence |
| `recommended_action` | `str` | operational recommendation |
| `evidence_used` | `list[str]` | retrieved passage ids/snippets the answer relies on |
| `tier_reached` | `1 \| 2` | 1 = Tier 1 only, 2 = escalated |
| `llm_invoked` | `bool` | whether the LLM was actually called |

Do not add, rename or remove fields without updating this document, the schema
and the tests in the same change.

## 4. Rules

1. **Fixed seeds.** Every stochastic step (splitting, model fitting, Optuna,
   sampling, bootstrap) takes its seed from `configs/config.yaml` (`seed: 42`).
   No implicit randomness.
2. **No data leakage.** Fit every scaler, calibrator, encoder, imputer,
   feature selector and threshold **only** on its designated split:
   train → fit, validation → calibrate/threshold, test → report once.
   Never touch the test split during model selection.
3. **Everything is configurable and runnable.** No hard-coded paths, seeds,
   thresholds or model names in scripts. Each script reads
   `configs/config.yaml` (via `src.config.load_config`) and is reachable from a
   `Makefile` target.
4. **Every experiment writes metrics JSON.** Each run saves its metrics to
   `reports/<name>.json` (plus figures in `reports/figures/`), including the
   config values and git commit needed to reproduce it.
5. **Type hints and docstrings.** All public functions, classes and methods are
   fully type-annotated and have docstrings stating purpose, arguments and
   return value.
6. **NEVER fabricate.** Do not invent numbers, metrics, citations, ATT&CK
   technique IDs or CVE IDs — not in code, reports, docstrings, comments,
   notebooks or chat. Only use identifiers that appear verbatim in retrieved /
   downloaded sources, and cite the source. If a metric has not been measured,
   write "not measured".
7. **Stop instead of faking.** If something cannot be done in this environment
   (missing data, no LLM backend, no network, unsupported hardware), say so
   explicitly, record it in `docs/PROGRESS.md`, and stop that line of work. Do
   not substitute synthetic results for real ones or stub a result silently.
8. **Progress is append-only.** Every phase appends to `docs/PROGRESS.md`:
   what was done, decisions and their rationale, measured metrics, open issues.
   Never rewrite or delete earlier entries.
9. **Small, verifiable steps.** `make test` and `make lint` must pass before a
   phase is marked complete.
