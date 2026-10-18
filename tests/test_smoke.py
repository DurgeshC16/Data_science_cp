"""Smoke tests: the repository scaffold, config and output schema are usable.

These tests deliberately avoid importing the heavy ML stack (torch, shap, faiss)
so they run in a couple of seconds and can gate every commit.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, load_config
from src.schema import OutputRecord

REQUIRED_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "data/kb",
    "models",
    "reports/figures",
    "configs",
    "src/data",
    "src/tier1",
    "src/gate",
    "src/explain",
    "src/rag",
    "src/tier2",
    "src/pipeline",
    "src/eval",
    "tests",
    "notebooks",
    "docs",
]


@pytest.mark.parametrize("relative_path", REQUIRED_DIRS)
def test_required_directory_exists(relative_path: str) -> None:
    """Every directory from the project spec is present."""
    assert (PROJECT_ROOT / relative_path).is_dir()


def test_config_loads_and_is_pinned_to_seed_42() -> None:
    """Config parses, uses seed 42 and four splits that sum to one."""
    config = load_config()
    assert config.seed == 42
    total = (
        config.split.train
        + config.split.val
        + config.split.calibration
        + config.split.test
    )
    assert total == pytest.approx(1.0)
    # The 60/10/10/20 contract from the phase-2 specification.
    assert (config.split.train, config.split.val, config.split.calibration, config.split.test) == (
        pytest.approx(0.60),
        pytest.approx(0.10),
        pytest.approx(0.10),
        pytest.approx(0.20),
    )
    assert config.eval.sample_cap > 0
    assert (
        config.gate.target_escalation_rate_min
        <= config.gate.target_escalation_rate_max
    )


def test_output_record_round_trip() -> None:
    """A valid Tier-2 record serialises with exactly the specified fields."""
    record = OutputRecord(
        flow_id="flow-0001",
        final_classification="benign",
        explanation="No suspicious behaviour was observed in this flow.",
        attack_technique=None,
        confidence=0.62,
        recommended_action="Monitor only; no response required.",
        evidence_used=["kb-passage-placeholder"],
        tier_reached=2,
        llm_invoked=True,
    )
    payload = record.model_dump()
    assert set(payload) == {
        "flow_id",
        "final_classification",
        "explanation",
        "attack_technique",
        "confidence",
        "recommended_action",
        "evidence_used",
        "tier_reached",
        "llm_invoked",
    }
    assert OutputRecord.model_validate(payload) == record


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence", 1.5),
        ("attack_technique", "TA0001"),  # tactic id, not a technique id
        ("attack_technique", "not-an-id"),
        ("tier_reached", 3),
        ("flow_id", ""),
    ],
)
def test_output_record_rejects_invalid_values(field: str, value: object) -> None:
    """The schema rejects out-of-range confidence and malformed ATT&CK ids."""
    payload = {
        "flow_id": "flow-0002",
        "final_classification": "benign",
        "explanation": "Placeholder explanation.",
        "attack_technique": None,
        "confidence": 0.5,
        "recommended_action": "Monitor.",
        "evidence_used": [],
        "tier_reached": 1,
        "llm_invoked": False,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        OutputRecord(**payload)


def test_requirements_are_fully_pinned() -> None:
    """requirements.txt pins every direct dependency with ``==``."""
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "requirements.txt is empty"
    for line in lines:
        assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^\s;]+$", line), f"unpinned: {line}"


def test_makefile_exposes_required_targets() -> None:
    """The Makefile defines setup, test and lint targets."""
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("setup", "test", "lint"):
        assert re.search(rf"^{target}\s*:", makefile, flags=re.MULTILINE), target
