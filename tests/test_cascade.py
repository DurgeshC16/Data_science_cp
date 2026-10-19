"""Routing integration tests with deterministic Tier-1 and mock Tier-2 parts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.pipeline.cascade import CascadePipeline
from src.pipeline.cascade import select_gate_thresholds
from src.schema import OutputRecord
from src.tier2.llm_backend import MockBackend
from src.tier2.validate import Tier2Explainer


class RoutingModel:
    """Synthetic adapter for testing routing only, not a trained detector."""

    family = "random_forest"
    task = "multiclass"
    labels = ["Normal", "DoS", "DDoS"]
    feature_columns = ["route_code"]

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return predeclared scenario probabilities for the unit fixture."""
        values = np.asarray(features["route_code"], dtype=float)
        self.batch_sizes.append(len(values))
        output = []
        for value in values:
            if value == 0.0:
                output.append([0.99, 0.005, 0.005])
            elif value == 1.0:
                output.append([0.01, 0.98, 0.01])
            else:
                output.append([0.40, 0.30, 0.30])
        return np.asarray(output, dtype=float)


class IdentityCalibrator:
    """Test calibrator that leaves the fake model probabilities unchanged."""

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Return the supplied matrix unchanged."""
        return probabilities


class FixtureShap:
    """Small fixture explainer; no SHAP claim or measured attribution is made."""

    def explain(self, flow: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one structural feature record for the mock prompt."""
        return [
            {
                "feature": "route_code",
                "value": flow["route_code"],
                "value_words": "high route_code",
                "direction": "toward Attack",
                "shap_value": 0.2,
                "magnitude": 0.2,
            }
        ]


class FixtureRetriever:
    """Return source-backed ATT&CK passage metadata for one mock query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Record the query and return a cited passage fixture."""
        self.queries.append(query)
        return [
            {
                "document_id": "attack:T1498:0",
                "technique_id": "T1498",
                "text": "ATT&CK technique T1498: Network Denial of Service",
                "score": 0.5,
            }
        ][:k]


def _tier2_json() -> str:
    """Produce a minimal strict response with no unsupported numeric claims."""
    return json.dumps(
        {
            "classification": "Needs Human Review",
            "explanation": (
                "The flow has an uncertain Tier-1 result. "
                "The retrieved reference provides context for analyst review. "
                "Compare neighboring flows before taking action."
            ),
            "attack_technique": {
                "id": "T1498",
                "name": "Network Denial of Service",
            },
            "confidence": "Low",
            "recommended_action": "Review this flow alongside nearby network activity.",
        }
    )


def test_synthetic_normal_syn_and_smb_scenarios_route_through_expected_tiers(
    tmp_path: Path,
) -> None:
    """Exercise scenario routing; these are not the absent report's exact flows."""
    config = load_config()
    model = RoutingModel()
    retriever = FixtureRetriever()
    backend = MockBackend([_tier2_json()], model_name="mock-tier2")
    tier2 = Tier2Explainer(backend, config, tmp_path / "cache.json")
    pipeline = CascadePipeline(
        model,  # type: ignore[arg-type]
        {"max_prob_threshold": 0.7, "entropy_threshold": 0.8},
        config=config,
        calibrator=IdentityCalibrator(),
        shap_explainer=FixtureShap(),
        retriever=retriever,
        tier2_explainer=tier2,
        attack_techniques={"T1498": "Network Denial of Service"},
        tier2_concurrency=2,
        run_stats_path=tmp_path / "run_stats.json",
    )
    rows = [
        {
            "flow_id": "example-a-normal",
            "route_code": 0.0,
            "protocol": "TCP",
            "destination_port": 443,
        },
        {
            "flow_id": "example-b-syn",
            "route_code": 1.0,
            "protocol": "TCP",
            "flag_syn": 1,
        },
        {
            "flow_id": "example-c-smb",
            "route_code": 2.0,
            "protocol": "TCP",
            "destination_port": 445,
            "service": "SMB",
        },
    ]

    records = pipeline.run_batch(iter(rows), batch_size=3)

    assert len(records) == 3
    assert all(isinstance(OutputRecord.model_validate(row.model_dump()), OutputRecord) for row in records)
    assert records[0].tier_reached == 1
    assert records[0].final_classification == "Normal"
    assert records[0].llm_invoked is False
    assert records[0].attack_technique is None
    assert records[1].tier_reached == 1
    assert records[1].final_classification == "DoS"
    assert records[1].attack_technique == "T1498"
    assert records[1].llm_invoked is False
    assert records[2].tier_reached == 2
    assert records[2].final_classification == "Needs Human Review"
    assert records[2].llm_invoked is True
    assert backend.calls == 1
    assert len(retriever.queries) == 1
    assert "port 445" in retriever.queries[0]
    stats = json.loads((tmp_path / "run_stats.json").read_text(encoding="utf-8"))
    assert stats["total_flows"] == 3
    assert stats["escalated"] == 1
    assert stats["llm_calls"] == 1


def test_batch_prediction_is_vectorized_and_outputs_exact_schema(tmp_path: Path) -> None:
    """The Tier-1 adapter sees batches and the record field set stays fixed."""
    config = load_config()
    model = RoutingModel()
    pipeline = CascadePipeline(
        model,  # type: ignore[arg-type]
        {"max_prob_threshold": 0.7, "entropy_threshold": 0.8},
        config=config,
        calibrator=IdentityCalibrator(),
        run_stats_path=tmp_path / "stats.json",
    )
    output = pipeline.run_batch(
        ({"route_code": 0.0} for _ in range(5)), batch_size=3
    )

    assert model.batch_sizes == [3, 2]
    expected_fields = set(OutputRecord.model_fields)
    assert all(set(record.model_dump()) == expected_fields for record in output)


def test_gate_profile_selects_nearest_fitted_sweep_rate() -> None:
    thresholds = select_gate_thresholds(
        {
            "thresholds": {
                "sweep": [
                    {
                        "escalation_rate": 0.05,
                        "max_prob_threshold": 0.8,
                        "entropy_threshold": 0.4,
                    },
                    {
                        "escalation_rate": 0.10,
                        "max_prob_threshold": 0.7,
                        "entropy_threshold": 0.6,
                    },
                ]
            }
        },
        "p10",
    )

    assert thresholds["achieved_fit_rate"] == 0.10
    assert thresholds["max_prob_threshold"] == 0.7
