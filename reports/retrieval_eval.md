# Retrieval evaluation

- ATT&CK version: `20.0`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Evaluation queries: 60 (all require human review)

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.0333 | 0.1500 | 0.3667 | 0.1225 |
| hybrid | 0.0000 | 0.1000 | 0.1500 | 0.0544 |

## Flag: dense Recall@3 below 0.70

- Review expected technique IDs and query wording with a human; all supplied rows are explicitly unverified.
- Add sourced ATT&CK detection guidance or curated CVE passages relevant to the query vocabulary.
- Tune chunk size/overlap and embedding model using a held-out, human-reviewed query set.
- Compare hybrid results and inspect false negatives before changing retrieval thresholds.
