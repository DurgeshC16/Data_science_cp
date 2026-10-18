"""Tests for the confidence gate.

The fixtures are synthetic; none of the values are dataset results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import build_processed_dataset, run_tiny_benchmark, tiny_tier1_config
from typing import Any

from src.config import GateConfig, load_config
from src.tier1.calibrate import run_calibration
from src.gate.gate import (
    escalate_mask,
    escalation_rate,
    evaluate_gate,
    fit_thresholds,
    main,
    predictive_entropy,
    run_gate,
    top_probability,
    uncertainty_scores,
)

CLASSES = ["a", "b", "c"]


def _spread_probabilities(seed: int = 0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return a probability matrix with a wide confidence range and labels."""
    rng = np.random.default_rng(seed)
    broad = rng.dirichlet(np.full(3, 0.3), size=n)
    sharp = rng.dirichlet(np.full(3, 8.0), size=n)
    weight = rng.uniform(0.0, 1.0, size=(n, 1))
    matrix = weight * sharp + (1.0 - weight) * broad
    matrix /= matrix.sum(axis=1, keepdims=True)
    labels = np.asarray(
        [CLASSES[index] for index in rng.integers(0, 3, size=n)], dtype=object
    )
    return matrix, labels


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def test_top_probability_and_entropy_values() -> None:
    """Scores match their closed forms for uniform and one-hot rows."""
    uniform = np.full((1, 4), 0.25)
    one_hot = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert top_probability(uniform)[0] == pytest.approx(0.25)
    assert predictive_entropy(uniform, "natural")[0] == pytest.approx(math.log(4), abs=1e-9)
    assert predictive_entropy(uniform, "bits")[0] == pytest.approx(2.0, abs=1e-9)
    assert predictive_entropy(one_hot, "natural")[0] == pytest.approx(0.0, abs=1e-9)
    scores = uncertainty_scores(uniform, "bits")
    assert set(scores) == {"max_probability", "entropy"}


def test_entropy_rejects_unknown_base_and_bad_shape() -> None:
    """Unknown units and non-2-D input raise."""
    with pytest.raises(ValueError, match="unknown entropy base"):
        predictive_entropy(np.full((2, 2), 0.5), "shannons")
    with pytest.raises(ValueError, match="2-D"):
        top_probability(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="2-D"):
        predictive_entropy(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_escalate_mask_is_an_or_policy() -> None:
    """A flow escalates when either condition fires."""
    top = np.array([0.9, 0.9, 0.4, 0.4])
    entropy = np.array([0.1, 5.0, 0.1, 5.0])
    mask = escalate_mask(top, entropy, max_prob_threshold=0.5, entropy_threshold=1.0)
    assert mask.tolist() == [False, True, True, True]
    assert escalation_rate(mask) == pytest.approx(0.75)
    assert escalation_rate(np.array([], dtype=bool)) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        escalate_mask(np.array([0.5]), np.array([0.1, 0.2]), 0.5, 1.0)


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


def test_fit_thresholds_lands_in_the_target_band() -> None:
    """The fitted pair hits the configured escalation band."""
    matrix, _ = _spread_probabilities()
    scores = uncertainty_scores(matrix)
    fitted = fit_thresholds(
        scores["max_probability"], scores["entropy"], 0.05, 0.15, grid_steps=51
    )
    assert fitted["in_band"] is True
    assert 0.05 <= fitted["achieved_rate"] <= 0.15
    assert abs(fitted["achieved_rate"] - 0.10) <= 0.05
    assert len(fitted["sweep"]) == 51
    assert fitted["criterion"].startswith("escalation rate inside")
    rates = [entry["escalation_rate"] for entry in fitted["sweep"]]
    assert rates[0] <= rates[-1]


def test_fit_thresholds_raises_when_the_band_is_unreachable() -> None:
    """An unreachable band stops the run and reports the closest rate."""
    matrix, _ = _spread_probabilities()
    scores = uncertainty_scores(matrix)
    with pytest.raises(ValueError, match="closest achievable rate"):
        fit_thresholds(
            scores["max_probability"], scores["entropy"], 0.45, 0.55, grid_steps=2
        )


def test_fit_thresholds_validates_inputs() -> None:
    """Empty, mismatched and invalid-band inputs are rejected."""
    scores = np.array([0.2, 0.5, 0.9])
    with pytest.raises(ValueError, match="empty"):
        fit_thresholds(np.array([]), np.array([]), 0.05, 0.15)
    with pytest.raises(ValueError, match="same shape"):
        fit_thresholds(scores, np.array([0.1]), 0.05, 0.15)
    with pytest.raises(ValueError, match="invalid target band"):
        fit_thresholds(scores, scores, 0.5, 0.1)
    with pytest.raises(ValueError, match="grid_steps"):
        fit_thresholds(scores, scores, 0.05, 0.15, grid_steps=1)


def test_fit_thresholds_is_deterministic() -> None:
    """The same scores give the same thresholds."""
    matrix, _ = _spread_probabilities()
    scores = uncertainty_scores(matrix)
    first = fit_thresholds(scores["max_probability"], scores["entropy"], 0.05, 0.15)
    second = fit_thresholds(scores["max_probability"], scores["entropy"], 0.05, 0.15)
    assert first["max_prob_threshold"] == second["max_prob_threshold"]
    assert first["entropy_threshold"] == second["entropy_threshold"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_gate_counts_errors_and_classes() -> None:
    """Counts, accuracies and error capture match a hand-built example."""
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ]
    )
    labels = np.array(["a", "b", "b", "c"])
    mask = np.array([False, True, False, True])
    report = evaluate_gate(probabilities, labels, CLASSES, mask)
    assert report["n_flows"] == 4
    assert report["n_escalated"] == 2
    assert report["escalation_rate"] == pytest.approx(0.5)
    assert report["accuracy_all"] == pytest.approx(0.75)
    assert report["accuracy_tier1_only"] == pytest.approx(1.0)
    assert report["accuracy_escalated"] == pytest.approx(0.5)
    assert report["error_capture_rate"] == pytest.approx(1.0)
    assert report["per_class"]["a"] == {"n": 1, "escalated": 0, "escalation_rate": 0.0}
    assert report["per_class"]["b"]["escalation_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same length"):
        evaluate_gate(probabilities, labels, CLASSES, np.array([True]))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _prepare(tmp_path: Path, **gate_overrides: Any) -> tuple[Path, Path, Path, Any]:
    """Build the fixture, benchmark it and calibrate it."""
    processed, models = build_processed_dataset(
        tmp_path, n_per_class=45, zeroday_class="Infiltration", zeroday_rows=15
    )
    reports = tmp_path / "reports"
    config = tiny_tier1_config(gate_overrides=gate_overrides or None)
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)
    run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    return processed, models, reports, config


def test_run_gate_writes_artifacts_and_respects_the_target_band(tmp_path: Path) -> None:
    """The gate fits on the configured split and reports every split."""
    processed, models, reports, config = _prepare(tmp_path)
    report = run_gate(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    assert (reports / "gate_report.json").is_file()
    assert (reports / "gate_report.md").is_file()
    assert report["threshold_fit_split"] == "val"
    assert report["policy"].startswith("escalate when max_probability <")

    for record in report["runs"]:
        thresholds = record["thresholds"]
        assert thresholds["in_band"] is True
        assert (
            config.gate.target_escalation_rate_min
            <= thresholds["achieved_rate"]
            <= config.gate.target_escalation_rate_max
        )
        assert record["val"]["escalation_rate"] == pytest.approx(
            thresholds["achieved_rate"]
        )
        assert 0.0 <= record["test"]["escalation_rate"] <= 1.0
        assert 0.0 <= record["test"]["accuracy_tier1_only"] <= 1.0
        assert record["test"]["accuracy_all"] == pytest.approx(
            (
                record["test"]["accuracy_tier1_only"]
                * (record["test"]["n_flows"] - record["test"]["n_escalated"])
                + record["test"]["accuracy_escalated"]
                * record["test"]["n_escalated"]
            )
            / record["test"]["n_flows"]
        )
        assert record["calibrated_probabilities"] is True
        assert record["placeholder_thresholds"]["note"].startswith("config placeholders")
        assert "zeroday" in record
        assert record["zeroday"]["n_flows"] == 15
        # The holdout is defined on the multiclass taxonomy: for the binary task
        # those rows are ordinary "Attack" flows.
        assert record["zeroday"]["classes_seen"] == (
            ["Infiltration"] if record["task"] == "multiclass" else ["Attack"]
        )

    figures = sorted(path.name for path in (reports / "figures").glob("gate_*"))
    assert figures == [
        "gate_entropy_binary_random_forest.png",
        "gate_entropy_multiclass_random_forest.png",
        "gate_sweep_binary_random_forest.png",
        "gate_sweep_multiclass_random_forest.png",
    ]


def test_gate_requires_a_calibrator_when_calibration_is_enabled(tmp_path: Path) -> None:
    """Missing calibrators stop the run with the command that creates them."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    config = tiny_tier1_config()
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)
    with pytest.raises(FileNotFoundError, match="calibrate-toy"):
        run_gate(
            "toy",
            config,
            processed_dir=processed,
            models_dir=models,
            reports_dir=reports,
            figures_dir=reports / "figures",
        )


def test_gate_can_score_raw_probabilities(tmp_path: Path) -> None:
    """With calibration disabled the gate runs without calibrator artifacts."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    config = tiny_tier1_config(gate_overrides={"use_calibrated_probabilities": False})
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)
    report = run_gate(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    assert all(r["calibrated_probabilities"] is False for r in report["runs"])
    assert all(r["calibrator_path"] is None for r in report["runs"])
    assert report["zeroday_available"] is False


def test_thresholds_do_not_depend_on_the_test_split(tmp_path: Path) -> None:
    """Corrupting the test labels leaves the fitted thresholds identical."""
    processed, models, reports, config = _prepare(tmp_path)
    first = run_gate(
        "toy",
        config,
        tasks=["binary"],
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    test_path = processed / "toy_test.parquet"
    frame = pd.read_parquet(test_path)
    frame["multiclass_label"] = "Normal"
    frame["binary_label"] = "Normal"
    frame.to_parquet(test_path, index=False)
    second = run_gate(
        "toy",
        config,
        tasks=["binary"],
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    assert (
        first["runs"][0]["thresholds"]["max_prob_threshold"]
        == second["runs"][0]["thresholds"]["max_prob_threshold"]
    )
    assert (
        first["runs"][0]["thresholds"]["entropy_threshold"]
        == second["runs"][0]["thresholds"]["entropy_threshold"]
    )


def test_gate_cli_exits_one_without_benchmark(tmp_path: Path) -> None:
    """No benchmark report means no gate report and exit code 1."""
    build_processed_dataset(tmp_path, n_per_class=20)
    exit_code = main(
        [
            "--dataset",
            "toy",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--models-dir",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--figures-dir",
            str(tmp_path / "reports" / "figures"),
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "reports" / "gate_report.json").exists()


def test_gate_report_is_json_serialisable(tmp_path: Path) -> None:
    """The written report round-trips through JSON."""
    processed, models, reports, config = _prepare(tmp_path, grid_steps=21)
    run_gate(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    payload = json.loads((reports / "gate_report.json").read_text(encoding="utf-8"))
    assert payload["grid_steps"] == 21
    assert len(payload["runs"][0]["thresholds"]["sweep"]) == 21
    assert payload["definitions"]["test_usage"].startswith("thresholds are fitted")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_gate_config_defaults_and_validators() -> None:
    """The new gate fields parse and invalid values are rejected."""
    gate = load_config().gate
    assert gate.calibrator_fit_split == "calibration"
    assert gate.threshold_fit_split == "val"
    assert gate.use_calibrated_probabilities is True
    assert gate.entropy_base == "natural"
    assert gate.grid_steps >= 2
    payload = gate.model_dump()
    with pytest.raises(ValidationError):
        GateConfig.model_validate({**payload, "entropy_base": "hartleys"})
    with pytest.raises(ValidationError):
        GateConfig.model_validate({**payload, "calibrator_fit_split": "test"})
    with pytest.raises(ValidationError):
        GateConfig.model_validate({**payload, "grid_steps": 1})


def test_makefile_exposes_phase_four_targets() -> None:
    """Calibration and gate have one make target per dataset."""
    from src.config import PROJECT_ROOT
    import re

    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("calibrate-cicids2017", "calibrate-ciciot2023", "gate-cicids2017", "gate-ciciot2023"):
        assert re.search(rf"^{target}\s*:", makefile, flags=re.MULTILINE), target
    assert "src.tier1.calibrate" in makefile
    assert "src.gate.gate" in makefile
