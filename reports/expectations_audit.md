# Expectations versus available evidence

The original project report was not found in the repository. This audit covers
expectations stated in `AGENTS.md`, `configs/config.yaml`, and
`reports/results.md`; it cannot claim to enumerate expectations from a missing
report.

| Documented expectation or claim | Available measurement | Replacement wording until measured |
| --- | --- | --- |
| Gate escalation should be roughly in the configured 5–15% band (`AGENTS.md`, `configs/config.yaml`) | Real-data escalation rate is not measured (`reports/results.md`) | “The gate is configured for a target band; achieved real-data escalation is not measured.” |
| Full cascade improves on Tier-1 alone (E1) | Tier-1 and cascade metrics not measured (`reports/results.md`) | “Relative detection quality is not measured.” |
| Confidence gate beats random escalation at equal cost (E3) | Random control and equal-cost comparison not measured (`reports/results.md`) | “The central gating claim is not established.” |
| RAG and SHAP improve Tier-2 quality (E4) | Ablations not measured (`reports/results.md`) | “The contribution of RAG and SHAP is not measured.” |
| Gate identifies unseen attacks and Tier 2 handles them (E5) | Zero-day holdout and verdict quality not measured (`reports/results.md`) | “Zero-day behavior is not measured.” |
| Static ATT&CK lookup is accurate ground truth (E6) | Accuracy not measured; mappings need human verification (`configs/attack_lookup.yaml`, `reports/results.md`) | “The lookup is a draft candidate table, not validated ground truth.” |
| Cascade reduces cost versus LLM on every flow (E2/E3) | Costs, token counts, and latencies not measured; prices are not configured (`reports/results.md`) | “Cost savings are not measured; any later estimate must state its price and token assumptions.” |
| Human analysts find outputs correct, clear and useful (E7) | No rater cases or ratings (`reports/results.md`) | “Human evaluation is not measured.” |
| Dense or hybrid retrieval is ready for operational claims | Current provisional scores: dense Recall@3 0.1500, hybrid 0.1000; all queries require review (`reports/retrieval_eval.json`) | “Retrieval performance is provisional and below the target; review queries/mappings and rerun.” |

The retrieval figures above are the only measured experiment values available
in reports, and they are explicitly provisional. They do not validate detection,
cost, zero-day, ATT&CK accuracy, ablation, or human-evaluation claims.
