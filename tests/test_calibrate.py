"""Tests for the Tier-1 probability-calibration module.

The fixtures are synthetic; none of the values are dataset results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import build_processed_dataset, run_tiny_benchmark, tiny_tier1_config
from src.config import load_config
from src.data.preprocess import SPLIT_NAMES
from src.tier1.calibrate import (
    CALIBRATION_METHODS,
    ProbabilityCalibrator,
    calibration_metrics,
    expected_calibration_error,
    fit_calibrator,
    load_calibrator,
    main,
    mean_log_loss,
    multiclass_brier_score,
    reliability_curve,
    run_calibration,
    save_calibrator,
)

CLASSES = ["a", "b", "c"]


def _overconfident_problem(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Return overconfident probabilities and labels from a weak classifier."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, size=n)
    # A model that mostly agrees with the truth but pushes its scores to 0/1.
    noisy = np.where(rng.random(n) < 0.25, rng.integers(0, 3, size=n), truth)
    scores = np.full((n, 3), 0.02)
    scores[np.arange(n), noisy] = 0.96
    scores /= scores.sum(axis=1, keepdims=True)
    labels = np.asarray([CLASSES[index] for index in truth], dtype=object)
    return scores, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_expected_calibration_error_hand_computed() -> None:
    """ECE matches a hand-computed value on a four-bin example."""
    probabilities = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    labels = np.array(["a", "a", "b", "a"])
    # bins=4 gives the edges [0, 0.25, 0.5, 0.75, 1.0]:
    #   bin [0.5, 0.75): both 0.6 rows, confidence 0.6, accuracy 1/2
    #   bin [0.75, 1.0]: both 0.9 rows, confidence 0.9, accuracy 2/2
    expected = 0.5 * abs(0.5 - 0.6) + 0.5 * abs(1.0 - 0.9)
    assert expected_calibration_error(
        probabilities, labels, ["a", "b"], bins=4
    ) == pytest.approx(expected, abs=1e-9)
    # The same matrix landed in a single bin with bins=2: accuracy equals mean
    # confidence there, so the ECE is zero.
    assert expected_calibration_error(
        probabilities, labels, ["a", "b"], bins=2
    ) == pytest.approx(0.0, abs=1e-9)


def test_brier_and_log_loss_hand_computed() -> None:
    """Brier and log loss match their definitions on a single row."""
    probabilities = np.array([[0.7, 0.2, 0.1]])
    labels = np.array(["a"])
    assert multiclass_brier_score(probabilities, labels, CLASSES) == pytest.approx(
        0.3**2 + 0.2**2 + 0.1**2
    )
    assert mean_log_loss(probabilities, labels, CLASSES) == pytest.approx(-np.log(0.7))


def test_calibration_metrics_report_accuracy_and_confidence() -> None:
    """The metric bundle carries accuracy, confidence and the calibration scores."""
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array(["a", "b"])
    metrics = calibration_metrics(probabilities, labels, ["a", "b"], bins=5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["mean_confidence"] == pytest.approx(0.6)
    assert metrics["n_samples"] == 2
    assert set(metrics) >= {"ece", "brier", "log_loss", "n_bins"}


def test_reliability_curve_bins_are_ordered() -> None:
    """Reliability curves return ordered confidence and accuracy series."""
    probabilities, labels = _overconfident_problem()
    curve = reliability_curve(probabilities, labels, CLASSES, bins=5)
    assert len(curve["confidence"]) == len(curve["accuracy"]) == len(curve["count"])
    assert curve["confidence"] == sorted(curve["confidence"])
    assert all(0.0 <= value <= 1.0 for value in curve["accuracy"])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_output_is_a_distribution(method: str) -> None:
    """Calibrated rows stay a valid probability distribution."""
    probabilities, labels = _overconfident_problem()
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, method, "toy", "multiclass", "random_forest", 42, "calibration"
    )
    calibrated = calibrator.calibrate(probabilities)
    assert calibrated.shape == probabilities.shape
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert calibrator.to_payload()["method"] == method
    assert calibrator.n_fit_rows == len(probabilities)


def test_isotonic_calibration_reduces_ece_on_overconfident_scores() -> None:
    """Isotonic calibration improves a badly overconfident probability set."""
    probabilities, labels = _overconfident_problem()
    before = calibration_metrics(probabilities, labels, CLASSES)
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, "isotonic", "toy", "multiclass", "random_forest", 42, "calibration"
    )
    after = calibration_metrics(calibrator.calibrate(probabilities), labels, CLASSES)
    assert after["ece"] < before["ece"]
    assert after["log_loss"] < before["log_loss"]


def test_none_method_is_the_identity() -> None:
    """``none`` returns the input matrix unchanged."""
    probabilities, labels = _overconfident_problem(n=50)
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, "none", "toy", "multiclass", "random_forest", 42, "calibration"
    )
    assert calibrator.estimators == {}
    assert np.allclose(calibrator.calibrate(probabilities), probabilities)


def test_unknown_method_and_bad_width_raise() -> None:
    """Unknown methods and mismatched column counts fail loudly."""
    probabilities, labels = _overconfident_problem(n=30)
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator(
            probabilities, labels, CLASSES, "platt-ish", "toy", "multiclass", "rf", 42, "calibration"
        )
    with pytest.raises(ValueError, match="probability matrix"):
        fit_calibrator(
            probabilities[:, :2], labels, CLASSES, "isotonic", "toy", "multiclass", "rf", 42, "calibration"
        )
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, "isotonic", "toy", "multiclass", "rf", 42, "calibration"
    )
    with pytest.raises(ValueError, match="probability matrix"):
        calibrator.calibrate(probabilities[:, :2])


def test_single_class_in_fit_split_keeps_the_identity() -> None:
    """A class with one label value is left uncalibrated and recorded."""
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    labels = np.array(["a", "a", "a", "a"])
    calibrator = fit_calibrator(
        probabilities, labels, ["a", "b"], "isotonic", "toy", "binary", "rf", 42, "calibration"
    )
    assert "b" in calibrator.identity_classes
    assert "a" not in calibrator.estimators


def test_calibrator_round_trip(tmp_path: Path) -> None:
    """A saved calibrator reloads and reproduces its outputs."""
    probabilities, labels = _overconfident_problem(n=60)
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, "sigmoid", "toy", "multiclass", "rf", 42, "calibration"
    )
    path = tmp_path / "calibrator.joblib"
    size = save_calibrator(calibrator, path)
    assert size == path.stat().st_size > 0
    loaded = load_calibrator(path)
    assert isinstance(loaded, ProbabilityCalibrator)
    assert np.allclose(loaded.calibrate(probabilities), calibrator.calibrate(probabilities))
    # The artifact must be a plain dict, not a pickled class defined in a module
    # that a CLI may run as __main__ (that breaks loading from another entry
    # point, e.g. `make calibrate-*` then `make gate-*`).
    import joblib

    raw = joblib.load(path)
    assert isinstance(raw, dict) and set(raw) == {"metadata", "estimators"}
    with pytest.raises(FileNotFoundError):
        load_calibrator(tmp_path / "missing.joblib")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_calibration_writes_artifacts_and_uses_the_configured_split(
    tmp_path: Path,
) -> None:
    """Calibration covers every benchmarked run and reports all splits."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=40)
    reports = tmp_path / "reports"
    config = tiny_tier1_config()
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)

    report = run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    assert (reports / "calibration_report.json").is_file()
    assert (reports / "calibration_report.md").is_file()
    assert report["fit_split"] == config.gate.calibrator_fit_split
    assert report["method"] == config.tier1.calibration

    for record in report["runs"]:
        assert set(record["metrics"]) == set(SPLIT_NAMES)
        assert record["n_calibrator_inputs"] == len(
            pd.read_parquet(processed / f"toy_{record['fit_split']}.parquet")
        )
        for split, metrics in record["metrics"].items():
            assert 0.0 <= metrics["calibrated"]["accuracy"] <= 1.0
            assert metrics["calibrated"]["n_samples"] > 0, split
        assert Path(record["calibrator_path"]).is_file()
        assert record["calibrator_size_bytes"] > 0
    figures = list((reports / "figures").glob("tier1_reliability_*.png"))
    assert len(figures) == len({(r["task"], r["family"]) for r in report["runs"]})


def test_calibration_fit_split_is_configurable(tmp_path: Path) -> None:
    """``gate.calibrator_fit_split = val`` fits on the validation split."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    config = tiny_tier1_config(gate_overrides={"calibrator_fit_split": "val"})
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)
    report = run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    val_rows = len(pd.read_parquet(processed / "toy_val.parquet"))
    assert report["fit_split"] == "val"
    assert all(r["n_calibrator_inputs"] == val_rows for r in report["runs"])


def test_calibrator_does_not_depend_on_the_test_split(tmp_path: Path) -> None:
    """Corrupting the test labels leaves every calibrator unchanged."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=40)
    reports = tmp_path / "reports"
    config = tiny_tier1_config()
    run_tiny_benchmark(processed, models, reports, families=("random_forest",), config=config)

    first = run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    def _probes(n_classes: int, seed: int = 5) -> np.ndarray:
        rng = np.random.default_rng(seed)
        matrix = rng.random((4, n_classes)) + 0.05
        return matrix / matrix.sum(axis=1, keepdims=True)

    before = {
        (r["task"], r["family"]): load_calibrator(Path(r["calibrator_path"])).calibrate(
            _probes(len(r["classes"]))
        )
        for r in first["runs"]
    }

    test_path = processed / "toy_test.parquet"
    frame = pd.read_parquet(test_path)
    frame["multiclass_label"] = "Normal"
    frame["binary_label"] = "Normal"
    frame.to_parquet(test_path, index=False)

    second = run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    for record in second["runs"]:
        key = (record["task"], record["family"])
        after = load_calibrator(Path(record["calibrator_path"])).calibrate(
            _probes(len(record["classes"]))
        )
        assert np.allclose(before[key], after), key


def test_calibration_supports_the_mlp_family(tmp_path: Path) -> None:
    """The MLP artifact reloads and calibrates like the sklearn families."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    config = tiny_tier1_config()
    run_tiny_benchmark(processed, models, reports, families=("mlp",), tasks=("binary",), config=config)
    report = run_calibration(
        "toy",
        config,
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    assert [r["family"] for r in report["runs"]] == ["mlp"]
    assert report["runs"][0]["metrics"]["test"]["calibrated"]["n_samples"] > 0


def test_calibrator_artifact_loads_from_another_process(tmp_path: Path) -> None:
    """A calibrator written by one CLI loads in a different interpreter."""
    probabilities, labels = _overconfident_problem(n=60)
    calibrator = fit_calibrator(
        probabilities, labels, CLASSES, "isotonic", "toy", "multiclass", "rf", 42, "calibration"
    )
    path = tmp_path / "calibrator.joblib"
    save_calibrator(calibrator, path)
    code = (
        "import sys; from pathlib import Path; "
        "from src.tier1.calibrate import load_calibrator; "
        "c = load_calibrator(Path(sys.argv[1])); "
        "sys.exit(0 if c.method == 'isotonic' else 2)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_calibration_cli_exits_one_without_benchmark(tmp_path: Path) -> None:
    """No benchmark report means no calibration, with a clear exit code."""
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
    assert not (tmp_path / "reports" / "calibration_report.json").exists()


def test_calibration_rejects_wrong_dataset_report(tmp_path: Path) -> None:
    """A benchmark report for another dataset is refused."""
    processed, models = build_processed_dataset(tmp_path, n_per_class=20)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "tier1_benchmark.json").write_text(
        json.dumps({"dataset": "other", "runs": [{"task": "binary"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not 'toy'"):
        run_calibration(
            "toy",
            tiny_tier1_config(),
            processed_dir=processed,
            models_dir=models,
            reports_dir=reports,
            figures_dir=reports / "figures",
        )


def test_calibration_methods_constant() -> None:
    """The supported method list matches the config literal."""
    assert CALIBRATION_METHODS == ("none", "isotonic", "sigmoid")
    assert load_config().tier1.calibration in CALIBRATION_METHODS
