# Cascade-XID report draft

**Draft status:** empirical claims are incomplete. The experiment matrix in
`reports/results.md` has not been measured on real data. Unsupported claims are
marked TODO rather than inferred from tests or synthetic fixtures.

## Implementation

Cascade-XID has code for leakage-safe data preprocessing, Tier-1 model
benchmarking, probability calibration, a confidence gate, Tier-1 Tree SHAP for
binary tree models, static ATT&CK candidate lookup, a local ATT&CK retrieval
index, strict Tier-2 output validation, and batch/streaming cascade orchestration.
See `src/data/`, `src/tier1/`, `src/gate/`, `src/explain/`, `src/rag/`,
`src/tier2/`, and `src/pipeline/cascade.py`.

The Tier-2 response keeps its categorical confidence separate; the final
`OutputRecord` retains the calibrated Tier-1 numeric confidence. This policy is
implemented in `src/pipeline/cascade.py`; TODO: decide whether this confidence
semantics is appropriate for the final report.

## Experimental Setup

The configured split is train/validation/calibration/test, and the implemented
training/calibration/gate stages use their configured disjoint splits
(`configs/config.yaml`). TODO: record the real dataset version, row counts,
selected model, fitted calibration method, gate profile, software versions,
hardware, and run commit after a real experiment run.

The local ATT&CK source metadata records ATT&CK v20.0 (`data/kb/attack_source.json`,
`reports/retrieval_eval.json`). The retrieval evaluation currently uses an
unreviewed candidate query set; see `reports/retrieval_eval.json`.

## Results

### Detection, cost, baselines, ablations and zero-day evaluation

E1–E5 results are **not measured**. This includes binary/multiclass detection
quality, gate-profile cost and latency, LLM-every-flow and random-escalation
baselines, Tier-2 ablations, and zero-day holdout quality. The experiment status
table and reasons are in `reports/results.md`.

### Retrieval

The existing provisional retrieval evaluation reports dense Recall@1/3/5 of
0.0333/0.1500/0.3667 and MRR of 0.1225; hybrid Recall@1/3/5 of
0.0000/0.1000/0.1500 and MRR of 0.0544 (`reports/retrieval_eval.json`). The
query rows and expected mappings all require human review, so these measurements
are not validated retrieval-quality claims. Dense Recall@3 is below the
configured acceptance threshold, and the report flags it with suggested fixes
(`reports/retrieval_eval.json`).

### ATT&CK mapping and human evaluation

E6 mapping accuracy and E7 human ratings/inter-rater agreement are **not
measured** (`reports/results.md`). The current label-to-technique table is
explicitly draft and requires human verification (`configs/attack_lookup.yaml`).

## Discussion

The available retrieval run is negative: hybrid retrieval scored below dense
retrieval on its current provisional query set (`reports/retrieval_eval.json`).
That set is unreviewed, so the result motivates query/label review and a rerun;
it does not establish which retrieval method is generally superior.

The central detection claim—that confidence gating improves on random
escalation at equal cost—is **not established** because E1 and E3 are not
measured (`reports/results.md`). TODO: discuss the measured result, including
negative findings, after the held-out experiments are run.

## Threats to Validity and Limitations

- No real raw dataset, processed test split, Tier-1 benchmark, or gate report is
  currently available (`reports/results.md`).
- The provisional ATT&CK mapping and retrieval queries require human review
  (`configs/attack_lookup.yaml`, `data/kb/retrieval_eval.csv`).
- No curated CVE records are in the current knowledge index; example rows in
  `data/kb/cve_curated.csv` are placeholders.
- Current SHAP support is binary Random Forest/XGBoost only; multiclass or MLP
  escalations fall back to human review (`docs/PROGRESS.md`).
- A local real-model run and human evaluation have not been performed
  (`docs/PROGRESS.md`, `reports/results.md`).
- TODO: add measured hardware/runtime, sampling uncertainty, label quality,
  data-shift and cost assumptions after conducting the experiments.

## Future Work

1. Download and document the real dataset, then produce leakage-safe splits and
   fitted Tier-1/calibration/gate artifacts.
2. Human-review ATT&CK mappings and retrieval evaluation rows; curate sourced
   CVE notes.
3. Implement and run E1–E7 with sample-cap enforcement and bootstrap intervals.
4. Include the equal-cost random-escalation control; if confidence gating does
   not beat it, report that negative result and revise the central claim.
5. Run the local model and export the human-rating kit only from actual
   escalated cases.

## References

No external literature references have been added to this draft. TODO: add only
verified, cited sources when the project authors provide/approve the report
bibliography.
