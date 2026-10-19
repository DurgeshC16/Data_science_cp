# Experiment results — blocked pending real data and validated artifacts

**Status:** no requested E1–E7 experiment was run. All experiment metrics below
are **not measured**; this is a blocker/status report, not evidence for or against
the cascade claims. No synthetic results are substituted.

## E1 — Detection quality

| Task | Tier-1 only precision / recall / F1 | Full cascade precision / recall / F1 | Status |
| --- | --- | --- | --- |
| Binary | not measured | not measured | No labeled held-out flows or trained artifacts |
| Multiclass | not measured | not measured | No labeled held-out flows or trained artifacts |

The requested Tier-2-to-label conventions (including `Needs Human Review`
counted as flagged) have not been evaluated.

## E2 — Cost and coverage

| Gate profile | Escalation rate | Mean / p95 end-to-end latency | LLM tokens / 1,000 | Estimated cost / 1,000 |
| --- | --- | --- | --- | --- |
| p05 | not measured | not measured | not measured | not measured |
| p10 | not measured | not measured | not measured | not measured |
| p15 | not measured | not measured | not measured | not measured |

GPU-hour price and per-token prices are not configured, so no cost assumption
can be stated. The cost-vs-F1 Pareto figure was **not generated**.

## E3 — Baselines

| Baseline | Quality at comparable cost | Status |
| --- | --- | --- |
| Tier-1 only | not measured | No held-out flows/model artifacts |
| LLM on every flow (random subsample, extrapolated) | not measured | Local Ollama unavailable; no flows |
| Random escalation at gate-matched rate | not measured | No held-out flows or fitted gate profile |

The equal-cost random-escalation control has not been run; the central gating
claim is therefore **not established**.

## E4 — Tier-2 ablations

| Variant | Classification correctness | ATT&CK accuracy | Hallucinated-technique rate | Status |
| --- | --- | --- | --- | --- |
| Full RAG + SHAP | not measured | not measured | not measured | No real evaluated flows/model |
| No RAG | not measured | not measured | not measured | No real evaluated flows/model |
| No SHAP | not measured | not measured | not measured | No real evaluated flows/model |
| Neither | not measured | not measured | not measured | No real evaluated flows/model |

## E5 — Zero-day holdout

| Measure | Result |
| --- | --- |
| Unseen-class escalation rate | not measured |
| Tier-2 verdict quality on unseen classes | not measured |

There is no measured zero-day holdout or fitted model.

## E6 — ATT&CK mapping

| Measure | Result |
| --- | --- |
| Top-1 agreement, overall | not measured |
| Top-1 agreement, per attack class | not measured |

`configs/attack_lookup.yaml` still marks every label
`needs_human_verification: true`; it is not yet valid evaluation ground truth.

## E7 — Human evaluation

| Item | Status |
| --- | --- |
| 50 balanced escalated cases exported | not generated; no real escalated cases |
| Correctness / clarity / usefulness ratings | not measured |
| Inter-rater agreement | not measured |

## Required to proceed

- Download CICIDS2017 MachineLearningCVE CSVs into `data/raw/cicids2017/` as
  described in the earlier blocked entry in `docs/PROGRESS.md`.
- Run the real data pipeline to produce processed test/zero-day splits and
  benchmark, calibration, gate and Tier-1 model artifacts.
- Start Ollama with a configured model for Tier-2 experiments; human-review the
  ATT&CK lookup before using it as ground truth.
- Configure the applicable GPU-hour or token price assumptions before reporting
  estimated costs.
