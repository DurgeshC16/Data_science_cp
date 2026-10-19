"""Tests for Tree SHAP output, percentile wording, and ATT&CK lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explain.attack_lookup import load_attack_lookup, lookup_attack_candidates
from src.explain.shap_explainer import (
    Tier1TreeExplainer,
    benchmark_explainer,
    fit_training_percentiles,
    retrieval_query,
    verbalize_feature_value,
)
from src.tier1.train import TrainedModel


def _tree_model() -> tuple[TrainedModel, pd.DataFrame]:
    """Build a deterministic synthetic binary tree model for unit tests."""
    train = pd.DataFrame(
        {
            "feature_a": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "feature_b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    estimator = RandomForestClassifier(n_estimators=8, random_state=42).fit(
        train.to_numpy(), np.asarray([0, 0, 0, 1, 1, 1])
    )
    model = TrainedModel(
        family="random_forest",
        task="binary",
        seed=42,
        labels=["Normal", "Attack"],
        feature_columns=list(train.columns),
        estimator=estimator,
        params={},
        training_time_seconds=0.0,
    )
    return model, train


def test_explain_returns_top_three_with_attack_direction_and_magnitude() -> None:
    model, train = _tree_model()
    dictionary = {
        "features": {},
        "default_meaning": "TODO: reviewed meaning required",
        "percentile_rules": {"quantiles": [0.25, 0.5, 0.75]},
    }
    percentiles = fit_training_percentiles(train, model.feature_columns, dictionary)
    explainer = Tier1TreeExplainer(model, percentiles, dictionary)

    explanations = explainer.explain({"feature_a": 2.5, "feature_b": 1.0})

    assert len(explanations) == 2
    assert explanations[0]["magnitude"] >= explanations[1]["magnitude"]
    assert all(item["direction"] in {"toward Attack", "toward Normal", "neutral"} for item in explanations)
    assert all(item["value_words"].endswith("(meaning TODO)") for item in explanations)


def test_training_percentiles_and_words_are_derived_from_training_frame() -> None:
    dictionary = {
        "features": {"duration": {"meaning": "TODO: duration units unverified"}},
        "default_meaning": "TODO",
        "percentile_rules": {
            "quantiles": [0.25, 0.5, 0.75],
            "words": {"below_p25": "low", "p25_to_p75": "typical", "above_p75": "high"},
        },
    }
    cutoffs = fit_training_percentiles(pd.DataFrame({"duration": [1, 2, 3, 4]}), ["duration"], dictionary)

    assert cutoffs["duration"] == {"p25": 1.75, "p50": 2.5, "p75": 3.25}
    assert verbalize_feature_value("duration", 1.0, cutoffs, dictionary) == "low duration (meaning TODO)"
    assert verbalize_feature_value("duration", 4.0, cutoffs, dictionary) == "high duration (meaning TODO)"


def test_retrieval_query_uses_explicit_protocol_port_service_and_shap_words() -> None:
    query = retrieval_query(
        [{"value_words": "high bytes per packet"}, {"value_words": "short burst"}],
        destination_port=445,
        protocol="tcp",
        service="SMB",
    )

    assert query == "TCP flow, to port 445, SMB, high bytes per packet, short burst"


def test_attack_lookup_candidates_are_all_flagged_for_human_review() -> None:
    lookup = load_attack_lookup()
    assert all(entry["needs_human_verification"] is True for entry in lookup["labels"].values())
    assert lookup_attack_candidates("DoS")["candidates"][0]["id"] == "T1498"
    assert lookup_attack_candidates("PortScan/Recon")["candidates"][0]["id"] == "T1046"
    assert lookup_attack_candidates("BruteForce")["candidates"][0]["id"] == "T1110"


def test_explainer_rejects_non_tree_or_non_binary_models() -> None:
    model, train = _tree_model()
    model.family = "mlp"
    with pytest.raises(ValueError, match="TreeExplainer supports"):
        Tier1TreeExplainer(model, fit_training_percentiles(train, model.feature_columns))


def test_explainer_benchmark_reports_per_flow_latency() -> None:
    model, train = _tree_model()
    explainer = Tier1TreeExplainer(
        model, fit_training_percentiles(train, model.feature_columns)
    )

    timing = benchmark_explainer(explainer, train.iloc[:2], calls=2, warmup_calls=1)

    assert timing["calls"] == 2
    assert timing["warmup_calls"] == 1
    assert timing["median_ms"] > 0.0
    assert timing["p95_ms"] > 0.0
