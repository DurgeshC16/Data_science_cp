"""Tree SHAP explanations and train-percentile feature verbalisation."""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap
import yaml

from src.config import PROJECT_ROOT, load_config
from src.tier1.train import TrainedModel

logger = logging.getLogger(__name__)

# Keep the estimator alive alongside its explainer so Python cannot recycle its
# id and accidentally associate a later model with the wrong cached object.
_EXPLAINER_CACHE: dict[int, tuple[Any, shap.TreeExplainer]] = {}


def load_feature_dictionary(path: str | Path | None = None) -> dict[str, Any]:
    """Load the feature dictionary YAML.

    Args:
        path: optional YAML path; defaults to ``configs/feature_dictionary.yaml``.

    Returns:
        Parsed feature dictionary and percentile rendering rules.
    """
    configured_path = load_config().shap.feature_dictionary_path
    dictionary_path = (
        Path(path)
        if path
        else configured_path
        if configured_path.is_absolute()
        else PROJECT_ROOT / configured_path
    )
    with dictionary_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("features", {}), dict):
        raise ValueError(f"invalid feature dictionary: {dictionary_path}")
    return payload


def fit_training_percentiles(
    training_features: pd.DataFrame,
    feature_columns: Sequence[str],
    dictionary: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit percentile cut points from training rows only.

    Args:
        training_features: training split feature frame (never validation/test).
        feature_columns: ordered model features.
        dictionary: parsed dictionary; its quantiles default to quartiles.

    Returns:
        Per-feature quantile values keyed by percentile string.

    Raises:
        ValueError: If a requested feature is absent or quantiles are invalid.
    """
    missing = [name for name in feature_columns if name not in training_features]
    if missing:
        raise ValueError(f"training data missing model features: {missing}")
    rules = (dictionary or {}).get("percentile_rules", {})
    quantiles = [float(q) for q in rules.get("quantiles", [0.25, 0.5, 0.75])]
    if not quantiles or any(q < 0.0 or q > 1.0 for q in quantiles):
        raise ValueError("percentile quantiles must be non-empty and in [0, 1]")
    numeric = training_features[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    result: dict[str, dict[str, float]] = {}
    for feature in feature_columns:
        values = numeric[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"cannot fit feature percentiles: {feature} has no values")
        result[feature] = {
            f"p{int(q * 100):02d}": float(values.quantile(q)) for q in quantiles
        }
    return result


def verbalize_feature_value(
    feature: str,
    value: float,
    percentiles: dict[str, dict[str, float]],
    dictionary: dict[str, Any] | None = None,
) -> str:
    """Render a numeric feature relative to training-set percentile cut points.

    Args:
        feature: model feature name.
        value: flow's numeric feature value.
        percentiles: cut points returned by :func:`fit_training_percentiles`.
        dictionary: loaded feature dictionary and word rules.

    Returns:
        A concise qualitative phrase; no physical interpretation is inferred.
    """
    config = dictionary or {}
    cutoffs = percentiles.get(feature)
    if not cutoffs:
        return "value not characterized"
    low_cutoff = cutoffs.get("p25")
    high_cutoff = cutoffs.get("p75")
    words = config.get("percentile_rules", {}).get("words", {})
    if low_cutoff is not None and value < low_cutoff:
        band = words.get("below_p25", "low")
    elif high_cutoff is not None and value > high_cutoff:
        band = words.get("above_p75", "high")
    else:
        band = words.get("p25_to_p75", "typical")
    feature_entry = config.get("features", {}).get(feature, {})
    meaning = feature_entry.get("meaning", config.get("default_meaning", "TODO"))
    if meaning.startswith("TODO"):
        return f"{band} {feature} (meaning TODO)"
    override = config.get("percentile_rules", {}).get("overrides", {}).get(feature, {})
    return str(override.get(band, f"{band} {meaning}"))


def _cached_explainer(estimator: Any) -> shap.TreeExplainer:
    """Return a cached TreeExplainer for an estimator object."""
    key = id(estimator)
    cached = _EXPLAINER_CACHE.get(key)
    if cached is not None and cached[0] is estimator:
        return cached[1]
    explainer = shap.TreeExplainer(estimator)
    _EXPLAINER_CACHE[key] = (estimator, explainer)
    return explainer


def _class_shap_values(values: Any, class_index: int, n_features: int) -> np.ndarray:
    """Normalize SHAP's version-dependent output layout to one feature vector."""
    if isinstance(values, list):
        selected = np.asarray(values[class_index], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            # SHAP 0.46 commonly returns (rows, features, classes).
            if array.shape[1] == n_features:
                selected = array[0, :, class_index]
            # Some explainers return (classes, rows, features).
            elif array.shape[2] == n_features:
                selected = array[class_index, 0, :]
            else:
                raise ValueError(f"unrecognized SHAP output shape {array.shape}")
        elif array.ndim == 2:
            selected = array[0]
        else:
            raise ValueError(f"unrecognized SHAP output shape {array.shape}")
    selected = np.asarray(selected, dtype=float).reshape(-1)
    if len(selected) != n_features:
        raise ValueError(f"expected {n_features} SHAP values, got {len(selected)}")
    return selected


class Tier1TreeExplainer:
    """Cached Tree SHAP wrapper for a binary tree-based Tier-1 model.

    Direction is defined by the signed SHAP contribution to the model's
    ``Attack`` output: positive means toward Attack; negative means toward
    Normal. The current implementation deliberately rejects MLP and
    multiclass models rather than assigning an ambiguous direction.
    """

    def __init__(
        self,
        model: TrainedModel,
        percentiles: dict[str, dict[str, float]],
        dictionary: dict[str, Any] | None = None,
    ) -> None:
        """Create an explainer and validate the model contract.

        Args:
            model: fitted binary RF or XGBoost ``TrainedModel``.
            percentiles: training-only feature cut points.
            dictionary: optional loaded feature dictionary.

        Raises:
            ValueError: if the model is not a supported binary tree model.
        """
        if model.family not in {"random_forest", "xgboost"}:
            raise ValueError("TreeExplainer supports random_forest or xgboost, not this model")
        if set(model.labels) != {"Normal", "Attack"}:
            raise ValueError("directional explanations currently require binary Normal/Attack labels")
        if not model.feature_columns:
            raise ValueError("model feature_columns must not be empty")
        self.model = model
        self.percentiles = percentiles
        self.dictionary = dictionary or load_feature_dictionary()
        self._explainer = _cached_explainer(model.estimator)
        self.attack_index = model.labels.index("Attack")

    def explain(
        self,
        flow: pd.DataFrame | pd.Series | dict[str, float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Explain one flow with the top-k signed Attack contributions.

        Args:
            flow: one row keyed by model feature names.
            top_k: maximum number of feature records to return; defaults to the
                configured value (3).

        Returns:
            Records with feature, numeric value, qualitative value, direction,
            signed contribution and absolute magnitude.
        """
        count = top_k if top_k is not None else load_config().shap.top_k_features
        if count < 1:
            raise ValueError("top_k must be at least 1")
        if isinstance(flow, pd.DataFrame):
            if len(flow) != 1:
                raise ValueError("explain accepts exactly one flow")
            frame = flow
        elif isinstance(flow, pd.Series):
            frame = flow.to_frame().T
        else:
            frame = pd.DataFrame([flow])
        missing = [name for name in self.model.feature_columns if name not in frame]
        if missing:
            raise ValueError(f"flow is missing model features: {missing}")
        ordered = frame[self.model.feature_columns].apply(pd.to_numeric, errors="raise")
        values = self._explainer.shap_values(ordered.to_numpy(dtype=float))
        contributions = _class_shap_values(
            values, self.attack_index, len(self.model.feature_columns)
        )
        row = ordered.iloc[0]
        records = [
            {
                "feature": name,
                "value": float(row[name]),
                "value_words": verbalize_feature_value(
                    name, float(row[name]), self.percentiles, self.dictionary
                ),
                "direction": "toward Attack"
                if contributions[index] > 0
                else "toward Normal"
                if contributions[index] < 0
                else "neutral",
                "shap_value": float(contributions[index]),
                "magnitude": float(abs(contributions[index])),
            }
            for index, name in enumerate(self.model.feature_columns)
        ]
        return sorted(records, key=lambda record: record["magnitude"], reverse=True)[:count]


def benchmark_explainer(
    explainer: Tier1TreeExplainer,
    flows: pd.DataFrame,
    calls: int,
    warmup_calls: int,
) -> dict[str, float | int]:
    """Measure cached per-flow explanation latency on supplied rows.

    Args:
        explainer: initialized, cached Tier-1 Tree SHAP wrapper.
        flows: measured flow rows with all model features.
        calls: number of timed explanations.
        warmup_calls: untimed calls before measurement.

    Returns:
        Number of calls and median, p95, and mean milliseconds per flow.
    """
    if flows.empty:
        raise ValueError("cannot benchmark explainer on an empty frame")
    if calls < 1 or warmup_calls < 0:
        raise ValueError("calls must be positive and warmup_calls non-negative")
    for index in range(warmup_calls):
        explainer.explain(flows.iloc[[index % len(flows)]])
    durations: list[float] = []
    for index in range(calls):
        row = flows.iloc[[index % len(flows)]]
        started = time.perf_counter()
        explainer.explain(row)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "calls": calls,
        "warmup_calls": warmup_calls,
        "median_ms": float(statistics.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "mean_ms": float(statistics.mean(durations)),
    }


def retrieval_query(
    top_features: Sequence[dict[str, Any]],
    destination_port: int | str | None,
    protocol: str | None,
    service: str | None = None,
) -> str:
    """Build a concise retrieval query from SHAP features and flow metadata.

    Args:
        top_features: records returned by :meth:`Tier1TreeExplainer.explain`.
        destination_port: destination port, if available.
        protocol: protocol label/number, if available.
        service: service name only when independently known (never inferred).

    Returns:
        Short lower-case, comma-separated retrieval query.
    """
    parts: list[str] = []
    if protocol is not None:
        parts.append(f"{str(protocol).upper()} flow")
    elif destination_port is not None:
        parts.append("network flow")
    if destination_port is not None:
        parts.append(f"to port {destination_port}")
    if service:
        parts.append(str(service))
    for record in top_features:
        phrase = str(record.get("value_words", "")).strip()
        if phrase:
            parts.append(phrase)
    return ", ".join(parts) or "network flow"
