"""Mock-backend tests for strict Tier-2 generation and safety validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.tier2.llm_backend import MockBackend
from src.tier2.prompt import PROMPT_VERSION, build_prompt
from src.tier2.validate import Tier2Explainer, save_tier2_sample

FLOW = {"bytes": 1200, "destination_port": 443, "protocol": "TCP"}
SHAP = [{"feature": "bytes", "value": 1200, "direction": "toward Attack"}]
PASSAGES = [
    {
        "technique_id": "T1498",
        "text": "ATT&CK technique T1498: Network Denial of Service",
    }
]
KNOWN = {"T1498": "Network Denial of Service"}
TIER1 = "Attack"


def _valid_response() -> str:
    """Return a compliant answer grounded in this module's explicit test input."""
    return json.dumps(
        {
            "classification": "Likely Malicious",
            "explanation": (
                "The flow sent 1200 bytes to destination port 443. "
                "That traffic volume should be compared with the host's usual behavior. "
                "Review related flows before taking a blocking action."
            ),
            "attack_technique": {
                "id": "T1498",
                "name": "Network Denial of Service",
            },
            "confidence": "Medium",
            "recommended_action": "Review adjacent flows to the same destination before blocking.",
        }
    )


def _explainer(
    responses: list[str | BaseException], cache_path: Path
) -> tuple[Tier2Explainer, MockBackend]:
    config = load_config()
    backend = MockBackend(responses, model_name="mock-tier2")
    return Tier2Explainer(backend, config, cache_path), backend


def test_valid_json_is_parsed_and_contains_prompt_provenance(tmp_path: Path) -> None:
    explainer, backend = _explainer([_valid_response()], tmp_path / "cache.json")

    run = explainer.explain(
        FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN
    )

    assert run.parsed_output["classification"] == "Likely Malicious"
    assert run.parsed_output["prompt_version"] == "prompt_v1"
    assert run.parsed_output["tier1_verdict"] == TIER1
    assert run.parsed_output["attack_technique"]["id"] == "T1498"
    assert run.parsed_output["warnings"] == []
    assert backend.calls == 1
    assert run.call_metrics[0]["prompt_tokens"] > 0
    sample = save_tier2_sample(run, "normal_https", tmp_path / "samples")
    saved = json.loads(sample.read_text(encoding="utf-8"))
    assert saved["raw_outputs"] == [_valid_response()]
    assert saved["parsed_output"]["prompt_version"] == "prompt_v1"


def test_broken_json_retries_twice_then_falls_back_preserving_tier1(
    tmp_path: Path,
) -> None:
    explainer, backend = _explainer(
        ["{broken", "not json", "still not json"], tmp_path / "cache.json"
    )

    run = explainer.explain(FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN)

    assert backend.calls == 3
    assert run.attempts == 3
    assert run.parsed_output["classification"] == "Needs Human Review"
    assert run.parsed_output["tier1_verdict"] == TIER1
    assert run.parsed_output["fallback_used"] is True
    assert len(run.raw_outputs) == 3


def test_id_absent_from_supplied_kb_is_rejected_and_falls_back(tmp_path: Path) -> None:
    invalid = json.loads(_valid_response())
    # T1046 is a source-backed ATT&CK technique but is intentionally absent from
    # this test's restricted KB map; validation must not accept an unknown ID.
    invalid["attack_technique"] = {
        "id": "T1046",
        "name": "Network Service Discovery",
    }
    explainer, backend = _explainer(
        [json.dumps(invalid)] * 3, tmp_path / "cache.json"
    )

    run = explainer.explain(FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN)

    assert backend.calls == 3
    assert run.parsed_output["classification"] == "Needs Human Review"
    assert run.parsed_output["attack_technique"] is None
    assert "does not exist in the local KB" in run.parsed_output["warnings"][0]


def test_timeout_falls_back_and_preserves_tier1_verdict(tmp_path: Path) -> None:
    explainer, backend = _explainer(
        [TimeoutError("mock timeout")] * 3, tmp_path / "cache.json"
    )

    run = explainer.explain(FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN)

    assert backend.calls == 3
    assert run.parsed_output["classification"] == "Needs Human Review"
    assert run.parsed_output["tier1_verdict"] == TIER1
    assert run.parsed_output["attack_technique"] is None
    assert run.parsed_output["fallback_used"] is True


def test_nonretrieved_id_and_unseen_numeric_feature_value_are_flagged(
    tmp_path: Path,
) -> None:
    answer = json.loads(_valid_response())
    answer["explanation"] = (
        "The flow sent 1200 bytes to destination port 443. "
        "A rate of 900 packets per second would be unusual. "
        "Review adjacent flows before taking a blocking action."
    )
    explainer, _ = _explainer([json.dumps(answer)], tmp_path / "cache.json")

    run = explainer.explain(
        FLOW,
        SHAP,
        [{"technique_id": "T1046", "text": "Network Service Discovery"}],
        TIER1,
        known_techniques={**KNOWN, "T1046": "Network Service Discovery"},
    )

    assert "attack_technique_not_in_retrieved_passages" in run.parsed_output["warnings"]
    assert any(
        warning.startswith("explanation_cites_feature_values_not_in_input:900")
        for warning in run.parsed_output["warnings"]
    )


def test_cache_key_uses_flow_prompt_version_and_model(tmp_path: Path) -> None:
    explainer, backend = _explainer([_valid_response()], tmp_path / "cache.json")
    first = explainer.explain(FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN)
    second = explainer.explain(FLOW, SHAP, PASSAGES, TIER1, known_techniques=KNOWN)

    assert first.cache_key == second.cache_key
    assert second.cache_hit is True
    assert backend.calls == 1
    assert PROMPT_VERSION == "prompt_v1"


def test_prompt_uses_the_requested_components_and_version() -> None:
    prompt = build_prompt(FLOW, SHAP, PASSAGES)

    assert "SOC analyst assistant" in prompt
    assert '"destination_port": 443' in prompt
    assert "toward Attack" in prompt
    assert "T1498" in prompt
