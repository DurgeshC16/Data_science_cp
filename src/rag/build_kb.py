"""Build a sourced ATT&CK/CVE passage corpus and dense retrieval index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, Config, load_config

logger = logging.getLogger(__name__)

ATTACK_SOURCE_URL = (
    "https://github.com/mitre-attack/attack-stix-data/"
    "releases/download/v20.0/enterprise-attack.json"
)
SOURCE_METADATA_PATH = "data/kb/attack_source.json"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _resolve(path: Path) -> Path:
    """Resolve a configured relative path against the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_commit() -> str:
    """Return the current Git revision, or ``unknown`` outside a repository."""
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


def _tokenize(text: str) -> list[str]:
    """Split text into approximate lexical tokens for controlled chunking."""
    return TOKEN_PATTERN.findall(text)


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into overlapping approximate-token chunks.

    Args:
        text: full technique narrative.
        chunk_tokens: target lexical-token count per chunk.
        overlap_tokens: number of lexical tokens shared by adjacent chunks.

    Returns:
        Non-empty text chunks.

    Raises:
        ValueError: if chunk sizes are invalid.
    """
    if chunk_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("require chunk_tokens > overlap_tokens >= 0")
    tokens = _tokenize(text)
    if not tokens:
        return [""]
    stride = chunk_tokens - overlap_tokens
    return [
        " ".join(tokens[start : start + chunk_tokens])
        for start in range(0, len(tokens), stride)
    ]


def _attack_pattern_to_documents(
    item: dict[str, Any], chunk_tokens: int, overlap_tokens: int
) -> list[dict[str, Any]]:
    """Convert one active ATT&CK technique STIX object into chunk records."""
    technique_id = str(item.get("external_references", [{}])[0].get("external_id", ""))
    for reference in item.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            technique_id = str(reference.get("external_id", ""))
            break
    name = str(item.get("name", "")).strip()
    if not technique_id or not name:
        return []
    tactics = sorted(
        {
            str(phase.get("phase_name", "")).replace("-", " ").title()
            for phase in item.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
            and phase.get("phase_name")
        }
    )
    description = str(item.get("description", "")).strip()
    detection = str(item.get("x_mitre_detection", "")).strip()
    platforms = [str(value) for value in item.get("x_mitre_platforms", [])]
    header = "\n".join(
        [
            f"ATT&CK technique {technique_id}: {name}",
            f"Tactics: {', '.join(tactics) if tactics else 'not specified in source'}",
            f"Platforms: {', '.join(platforms) if platforms else 'not specified in source'}",
        ]
    )
    body_sections = [section for section in (description, f"Detection guidance: {detection}" if detection else "") if section]
    body = "\n\n".join(body_sections)
    chunks = chunk_text(body, chunk_tokens, overlap_tokens)
    return [
        {
            "document_id": f"attack:{technique_id}:{index}",
            "source_type": "attack_pattern",
            "technique_id": technique_id,
            "cve_id": None,
            "name": name,
            "tactics": tactics,
            "platforms": platforms,
            "revoked": False,
            "deprecated": False,
            "chunk_index": index,
            "text": f"{header}\n\n{chunk}".strip(),
        }
        for index, chunk in enumerate(chunks)
    ]


def parse_attack_bundle(
    bundle: dict[str, Any], chunk_tokens: int = 300, overlap_tokens: int = 50
) -> list[dict[str, Any]]:
    """Extract active Enterprise techniques and sub-techniques from STIX.

    Args:
        bundle: parsed official ATT&CK STIX bundle.
        chunk_tokens: approximate lexical tokens per chunk.
        overlap_tokens: approximate lexical tokens repeated between chunks.

    Returns:
        One or more document chunks for each non-revoked, non-deprecated
        ``attack-pattern`` technique, with its ID and name repeated in every
        chunk.
    """
    documents: list[dict[str, Any]] = []
    for item in bundle.get("objects", []):
        if item.get("type") != "attack-pattern":
            continue
        if item.get("revoked") is True or item.get("x_mitre_deprecated") is True:
            continue
        documents.extend(_attack_pattern_to_documents(item, chunk_tokens, overlap_tokens))
    return documents


def _cve_documents(path: Path) -> list[dict[str, Any]]:
    """Load curator-authored CVE rows, ignoring explicit template examples."""
    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"cve_id", "description", "affected_product", "related_attack_class"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"CVE CSV must include columns: {sorted(expected)}")
        for row in reader:
            cve_id = (row.get("cve_id") or "").strip()
            description = (row.get("description") or "").strip()
            if not cve_id or cve_id.startswith("EXAMPLE"):
                continue
            if not description or description.startswith("EXAMPLE"):
                continue
            output.append(
                {
                    "document_id": f"cve:{cve_id}",
                    "source_type": "cve",
                    "technique_id": None,
                    "cve_id": cve_id,
                    "name": cve_id,
                    "tactics": [],
                    "platforms": [],
                    "affected_product": (row.get("affected_product") or "").strip(),
                    "related_attack_class": (row.get("related_attack_class") or "").strip(),
                    "source_url": (row.get("source_url") or "").strip() or None,
                    "revoked": False,
                    "deprecated": False,
                    "chunk_index": 0,
                    "text": "\n".join(
                        [
                            f"CVE {cve_id}",
                            f"Affected product: {(row.get('affected_product') or '').strip()}",
                            f"Related attack class: {(row.get('related_attack_class') or '').strip()}",
                            f"Description: {description}",
                        ]
                    ),
                }
            )
    return output


def build_documents(config: Config, attack_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse source files into ATT&CK and curator-provided CVE documents."""
    source_path = attack_path or PROJECT_ROOT / "data/kb/enterprise-attack.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"ATT&CK bundle missing: {source_path}")
    raw_bytes = source_path.read_bytes()
    bundle = json.loads(raw_bytes)
    documents = parse_attack_bundle(
        bundle, config.rag.chunk_tokens, config.rag.overlap_tokens
    )
    cve_path = _resolve(config.rag.cve_path)
    documents.extend(_cve_documents(cve_path))
    metadata = {
        "source_url": ATTACK_SOURCE_URL,
        "attack_version": "20.0",
        "download_date": "2026-10-19",
        "bundle_path": str(source_path.relative_to(PROJECT_ROOT))
        if source_path.is_relative_to(PROJECT_ROOT)
        else str(source_path),
        "bundle_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    return documents, metadata


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL records in UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_index(config: Config) -> dict[str, Any]:
    """Build the configured FAISS index and metadata sidecar files."""
    documents, source_metadata = build_documents(config)
    if not documents:
        raise ValueError("no source documents available to embed")
    texts = [row["text"] for row in documents]
    logger.info("Loading embedding model %s", config.rag.embedding_model)
    model = SentenceTransformer(config.rag.embedding_model)
    model_limit = getattr(model[0].auto_model.config, "max_position_embeddings", None)
    if model_limit is not None and config.rag.chunk_tokens + 20 > model_limit:
        raise ValueError(
            f"rag.chunk_tokens={config.rag.chunk_tokens} plus document header exceeds "
            f"embedding model max_position_embeddings={model_limit}"
        )
    model.max_seq_length = config.rag.chunk_tokens + 20
    vectors = model.encode(
        texts,
        batch_size=config.rag.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path = _resolve(config.rag.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    passages_path = _resolve(config.rag.passages_path)
    _write_jsonl(passages_path, documents)
    source_metadata.update(
        {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "embedding_model": config.rag.embedding_model,
            "embedding_dimension": int(vectors.shape[1]),
            "document_chunks": len(documents),
            "unique_technique_ids": len(
                {row["technique_id"] for row in documents if row.get("technique_id")}
            ),
            "cve_documents": sum(row["source_type"] == "cve" for row in documents),
            "chunk_tokens": config.rag.chunk_tokens,
        "overlap_tokens": config.rag.overlap_tokens,
        "embedding_batch_size": config.rag.embedding_batch_size,
            "index_path": str(index_path),
            "passages_path": str(passages_path),
        }
    )
    source_path = PROJECT_ROOT / SOURCE_METADATA_PATH
    source_path.write_text(json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8")
    return source_metadata


def main(argv: list[str] | None = None) -> int:
    """Build the retrieval index and run the configured retrieval evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    try:
        metadata = build_index(config)
        logger.info(
            "Indexed %d chunks from ATT&CK v%s (SHA-256 %s)",
            metadata["document_chunks"],
            metadata["attack_version"],
            metadata["bundle_sha256"],
        )
        if not args.skip_evaluation:
            from src.rag.retriever import evaluate_retrieval

            report = evaluate_retrieval(config)
            logger.info("Wrote retrieval evaluation: %s", report["report_path"])
    except (FileNotFoundError, ValueError, ImportError) as error:
        logger.error("Knowledge-base build stopped: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
