"""Unit tests for STIX conversion and retrieval-evaluation inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import load_config
from src.rag.build_kb import _cve_documents, chunk_text, parse_attack_bundle
from src.rag.retriever import _metric_summary, _read_eval_rows


def _attack_pattern(
    technique_id: str,
    name: str,
    *,
    description: str = "A description.",
    revoked: bool = False,
    deprecated: bool = False,
) -> dict[str, object]:
    """Build a minimal STIX-shape fixture for parser control-flow tests."""
    return {
        "type": "attack-pattern",
        "name": name,
        "description": description,
        "revoked": revoked,
        "x_mitre_deprecated": deprecated,
        "x_mitre_platforms": ["Windows"],
        "x_mitre_detection": "Detection notes.",
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": "discovery"}
        ],
        "external_references": [
            {"source_name": "mitre-attack", "external_id": technique_id}
        ],
    }


def test_chunk_text_applies_overlap() -> None:
    chunks = chunk_text("a b c d e f g", chunk_tokens=4, overlap_tokens=1)

    assert chunks == ["a b c d", "d e f g", "g"]


def test_stix_conversion_keeps_technique_identity_in_each_chunk() -> None:
    long_description = "description " * 20
    bundle = {
        "objects": [
            _attack_pattern(
                "T1498",
                "Network Denial of Service",
                description=long_description,
            ),
            {**_attack_pattern("", "Revoked object", revoked=True), "external_references": []},
            {**_attack_pattern("", "Deprecated object", deprecated=True), "external_references": []},
            {"type": "relationship", "relationship_type": "uses"},
        ]
    }

    rows = parse_attack_bundle(bundle, chunk_tokens=8, overlap_tokens=2)

    assert len(rows) > 1
    assert all("T1498" in row["text"] for row in rows)
    assert all("Network Denial of Service" in row["text"] for row in rows)
    assert {row["tactics"][0] for row in rows} == {"Discovery"}
    assert all(row["platforms"] == ["Windows"] for row in rows)


def test_cve_template_examples_are_not_indexed(tmp_path: Path) -> None:
    path = tmp_path / "cve_curated.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cve_id", "description", "affected_product", "related_attack_class"])
        writer.writerow(["EXAMPLE-DO-NOT-USE", "EXAMPLE only", "EXAMPLE", "EXAMPLE"])

    assert _cve_documents(path) == []


def test_retrieval_eval_rows_are_over_40_and_use_lookup_candidates() -> None:
    config = load_config()
    rows = _read_eval_rows(
        Path(config.rag.retrieval_eval_path)
        if config.rag.retrieval_eval_path.is_absolute()
        else Path.cwd() / config.rag.retrieval_eval_path,
        config,
    )

    assert len(rows) >= 40
    assert all(row["needs_review"] == "true" for row in rows)


def test_recall_and_mrr_are_hand_computable() -> None:
    queries = [{"expected_ids": ["T1498"]}, {"expected_ids": ["T1046"]}]
    ranked = [
        [{"technique_id": "T1498"}, {"technique_id": "T1046"}],
        [{"technique_id": "T1110"}, {"technique_id": "T1046"}],
    ]

    metrics = _metric_summary(queries, ranked)

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.75


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_tokens > overlap_tokens"):
        chunk_text("example", chunk_tokens=2, overlap_tokens=2)
