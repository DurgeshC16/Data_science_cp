"""Dense and BM25+dense retrieval over the local knowledge-base index."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, Config, load_config
from src.explain.attack_lookup import load_attack_lookup

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DEFAULT_RETRIEVER: "KnowledgeRetriever | None" = None


def _resolve(path: Path) -> Path:
    """Resolve a relative path against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_commit() -> str:
    """Return the repository revision for experiment provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _git_dirty() -> bool:
    """Return whether tracked files differ from the recorded commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        return result.returncode != 0
    except (FileNotFoundError, OSError):
        return True


def _tokens(text: str) -> list[str]:
    """Normalize text into BM25 tokens."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class KnowledgeRetriever:
    """Load FAISS vectors and provide dense or hybrid search."""

    def __init__(self, config: Config) -> None:
        """Initialize model, FAISS index, and text metadata.

        Args:
            config: application config containing index and model paths.

        Raises:
            FileNotFoundError: if the index or metadata is not built.
        """
        self.config = config
        self.index_path = _resolve(config.rag.index_path)
        self.passages_path = _resolve(config.rag.passages_path)
        if not self.index_path.is_file() or not self.passages_path.is_file():
            raise FileNotFoundError(
                f"retrieval index is missing; run `make kb-build` first "
                f"(expected {self.index_path} and {self.passages_path})"
            )
        self.index = faiss.read_index(str(self.index_path))
        with self.passages_path.open("r", encoding="utf-8") as handle:
            self.documents = [json.loads(line) for line in handle if line.strip()]
        if self.index.ntotal != len(self.documents):
            raise ValueError(
                f"index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.documents)} documents"
            )
        self.model = SentenceTransformer(config.rag.embedding_model)
        self.vectors = np.zeros((self.index.ntotal, self.index.d), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, self.vectors)
        self.tokenized = [_tokens(document["text"]) for document in self.documents]
        self.doc_lengths = np.asarray([len(row) for row in self.tokenized], dtype=float)
        self.average_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.document_frequency: Counter[str] = Counter()
        for document in self.tokenized:
            self.document_frequency.update(set(document))
        self.idf = {
            term: math.log(1.0 + (len(self.tokenized) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.document_frequency.items()
        }

    def _dense_scores(self, query: str) -> np.ndarray:
        """Compute cosine similarity for a query against all indexed rows."""
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        return self.vectors @ query_vector[0]

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Compute Okapi BM25 scores over the local text corpus."""
        terms = _tokens(query)
        scores = np.zeros(len(self.documents), dtype=float)
        if not terms or self.average_length == 0.0:
            return scores
        for position, document in enumerate(self.tokenized):
            frequencies = Counter(document)
            length_norm = k1 * (1.0 - b + b * self.doc_lengths[position] / self.average_length)
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    scores[position] += self.idf.get(term, 0.0) * (
                        frequency * (k1 + 1.0) / (frequency + length_norm)
                    )
        return scores

    def retrieve(
        self,
        query: str,
        k: int = 3,
        mode: str = "dense",
        hybrid_alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve passages by cosine similarity or BM25+dense fusion.

        Args:
            query: free-text SHAP-derived query.
            k: number of results to return (default 3).
            mode: ``dense`` or ``hybrid``.
            hybrid_alpha: dense score weight in [0, 1]; config default if omitted.

        Returns:
            Ranked records with text, technique/CVE ID, and score.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")
        dense = self._dense_scores(query)
        if mode == "dense":
            scores = dense
        else:
            alpha = self.config.rag.hybrid_alpha if hybrid_alpha is None else hybrid_alpha
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("hybrid_alpha must be within [0, 1]")
            sparse = self._bm25_scores(query)
            dense_range = float(np.ptp(dense))
            sparse_range = float(np.ptp(sparse))
            dense_normalized = (dense - dense.min()) / dense_range if dense_range else dense
            sparse_normalized = (sparse - sparse.min()) / sparse_range if sparse_range else sparse
            scores = alpha * dense_normalized + (1.0 - alpha) * sparse_normalized
        positions = np.argsort(scores)[::-1][: min(k, len(self.documents))]
        output: list[dict[str, Any]] = []
        for position in positions:
            document = self.documents[int(position)]
            output.append(
                {
                    "text": document["text"],
                    "technique_id": document.get("technique_id"),
                    "cve_id": document.get("cve_id"),
                    "score": float(scores[position]),
                    "metadata": {
                        key: document.get(key)
                        for key in ("source_type", "name", "tactics", "platforms", "document_id")
                    },
                }
            )
        return output


def retrieve(
    query: str, k: int = 3, mode: str = "dense", config: Config | None = None
) -> list[dict[str, Any]]:
    """Convenience retrieval API backed by a lazily loaded singleton.

    Args:
        query: free-text retrieval query.
        k: result count (default 3).
        mode: ``dense`` or ``hybrid``.
        config: optional settings; the singleton is built from default config.

    Returns:
        Ranked records with text, ID fields and score.
    """
    global _DEFAULT_RETRIEVER
    if config is not None:
        return KnowledgeRetriever(config).retrieve(query, k, mode)
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = KnowledgeRetriever(load_config())
    return _DEFAULT_RETRIEVER.retrieve(query, k, mode)


def _read_eval_rows(path: Path, config: Config) -> list[dict[str, Any]]:
    """Load human-reviewable evaluation rows and validate expected IDs."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = load_attack_lookup(_resolve(config.shap.attack_lookup_path))["labels"]
    for row in rows:
        label = row["source_label"]
        expected = [value for value in row["expected_technique_ids"].split(";") if value]
        allowed = {candidate["id"] for candidate in lookup[label]["candidates"]}
        if not expected or not set(expected).issubset(allowed):
            raise ValueError(
                f"evaluation expected IDs for {label!r} must come from attack_lookup.yaml"
            )
        if row.get("needs_review", "").lower() != "true":
            raise ValueError("all retrieval evaluation rows must be marked needs_review=true")
    return rows


def _metric_summary(
    queries: Sequence[dict[str, Any]],
    ranked: Sequence[Sequence[dict[str, Any]]],
    cutoffs: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute Recall@k and MRR for the ranked query results."""
    recalls: dict[str, float] = {}
    for cutoff in cutoffs:
        per_query: list[float] = []
        for query, results in zip(queries, ranked):
            relevant = set(query["expected_ids"])
            hits = {
                result.get("technique_id") or result.get("cve_id")
                for result in results[:cutoff]
            }
            per_query.append(len(relevant & hits) / len(relevant))
        recalls[f"recall_at_{cutoff}"] = float(np.mean(per_query)) if per_query else 0.0
    reciprocal_ranks: list[float] = []
    for query, results in zip(queries, ranked):
        relevant = set(query["expected_ids"])
        reciprocal_ranks.append(
            next(
                (1.0 / rank for rank, result in enumerate(results, start=1)
                 if (result.get("technique_id") or result.get("cve_id")) in relevant),
                0.0,
            )
        )
    return {
        "n_queries": len(queries),
        **recalls,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def evaluate_retrieval(config: Config) -> dict[str, Any]:
    """Evaluate dense and hybrid search and save measured metrics JSON."""
    query_path = _resolve(config.rag.retrieval_eval_path)
    queries = [
        {
            "query": row["query"],
            "expected_ids": row["expected_technique_ids"].split(";"),
            "source_label": row["source_label"],
            "needs_review": row["needs_review"].lower() == "true",
        }
        for row in _read_eval_rows(query_path, config)
    ]
    retriever = KnowledgeRetriever(config)
    results = {
        mode: [retriever.retrieve(row["query"], k=5, mode=mode) for row in queries]
        for mode in ("dense", "hybrid")
    }
    metrics = {
        mode: _metric_summary(queries, ranked)
        for mode, ranked in results.items()
    }
    recall3 = metrics["dense"]["recall_at_3"]
    if recall3 < 0.7:
        warning = {
            "flagged": True,
            "reason": "dense Recall@3 is below 0.70",
            "suggested_fixes": [
                "Review expected technique IDs and query wording with a human; all supplied rows are explicitly unverified.",
                "Add sourced ATT&CK detection guidance or curated CVE passages relevant to the query vocabulary.",
                "Tune chunk size/overlap and embedding model using a held-out, human-reviewed query set.",
                "Compare hybrid results and inspect false negatives before changing retrieval thresholds.",
            ],
        }
    else:
        warning = {"flagged": False, "reason": None, "suggested_fixes": []}
    source_metadata_path = PROJECT_ROOT / "data/kb/attack_source.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "attack_version": source_metadata.get("attack_version", "not recorded"),
        "embedding_model": config.rag.embedding_model,
        "config": {
            "top_k": config.rag.top_k,
            "chunk_tokens": config.rag.chunk_tokens,
            "overlap_tokens": config.rag.overlap_tokens,
            "hybrid_alpha": config.rag.hybrid_alpha,
        },
        "evaluation_queries": str(query_path),
        "all_queries_need_human_review": True,
        "metrics": metrics,
        "dense_recall_at_3_warning": warning,
        "definitions": {
            "recall_at_k": "fraction of each row's expected IDs appearing in top-k results, averaged over rows",
            "mrr": "mean reciprocal rank of the first expected ID",
            "hybrid": "weighted sum of min-max normalized cosine and Okapi BM25 scores",
        },
    }
    report_path = _resolve(config.rag.retrieval_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Retrieval evaluation",
        "",
        f"- ATT&CK version: `{report['attack_version']}`",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- Evaluation queries: {len(queries)} (all require human review)",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in metrics.items():
        markdown.append(
            f"| {mode} | {values['recall_at_1']:.4f} | {values['recall_at_3']:.4f} "
            f"| {values['recall_at_5']:.4f} | {values['mrr']:.4f} |"
        )
    if warning["flagged"]:
        markdown.extend(["", "## Flag: dense Recall@3 below 0.70", ""])
        markdown.extend(f"- {item}" for item in warning["suggested_fixes"])
    else:
        markdown.extend(["", "All metric values are from the measured index/query run."])
    report_path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line retrieval evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", default=None, help="Retrieve one query instead of evaluating")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    try:
        if args.query:
            for result in KnowledgeRetriever(config).retrieve(args.query, args.k, args.mode):
                print(json.dumps(result, ensure_ascii=False))
        else:
            report = evaluate_retrieval(config)
            print(f"Saved retrieval evaluation to {report['report_path']}")
    except (FileNotFoundError, ValueError, ImportError) as error:
        logger.error("Retrieval failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
