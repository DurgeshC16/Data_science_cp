"""Probability calibration for the Tier-1 candidates.

Why
---
A Random Forest's ``predict_proba`` is an average of leaf votes, XGBoost's is a
gradient-boosted score and the MLP's is a softmax: none of them is a calibrated
probability. The confidence gate compares those probabilities against fixed
thresholds, so they must mean something first.

Protocol
--------
- The calibrator is fitted on ``gate.calibrator_fit_split`` (default
  ``calibration``, the dedicated Phase-2 split; ``val`` matches the wording in
  AGENTS.md). It is never fitted on train, and never on test.
- ``none`` leaves the probabilities untouched, ``isotonic`` fits a monotone
  piecewise-constant map per class and ``sigmoid`` (Platt) fits a logistic map on
  the class score. Multiclass calibration is one-vs-rest per class followed by a
  row renormalisation, which is an approximation and is reported as such.
- Calibration quality is reported on the fit split, on the validation split and
  (report-only, after every calibrator has been fitted) on the test split, using
  expected calibration error (ECE), the multiclass Brier score and log loss,
  before and after calibration. Test numbers never feed back into the fit.

Artifacts
---------
- ``models/tier1/<dataset>_<task>_<family>_seed<seed>_calibrator.joblib``
- ``reports/calibration_report.json``
- ``reports/calibration_report.md``
- ``reports/figures/tier1_reliability_<task>_<family>.png``

Usage
-----
::

    python -m src.tier1.calibrate --dataset cicids2017

Runs stop with an actionable message (exit code 1) when the benchmark or the
processed splits are missing instead of inventing probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import PROJECT_ROOT, Config, load_config  # noqa: E402
from src.data.preprocess import SPLIT_NAMES  # noqa: E402
from src.tier1.train import (  # noqa: E402
    TrainedModel,
    load_processed_splits,
    read_benchmark_runs,
    resolve_report_path,
    trained_model_from_artifact,
)

logger = logging.getLogger(__name__)

#: Supported calibration methods (mirrors ``tier1.calibration``).
CALIBRATION_METHODS: tuple[str, str, str] = ("none", "isotonic", "sigmoid")
#: Probability clipping used when taking logits or log probabilities.
_EPSILON = 1e-12


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Return the log-odds of ``probabilities``, clipped away from 0 and 1."""
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class ProbabilityCalibrator:
    """Per-class calibrator applied to a model's probability matrix.

    Attributes:
        dataset: dataset name.
        task: ``binary`` or ``multiclass``.
        family: Tier-1 model family.
        seed: seed of the model being calibrated.
        method: ``none``, ``isotonic`` or ``sigmoid``.
        classes: class order of the probability columns.
        fit_split: split the calibrator was fitted on.
        n_fit_rows: number of rows used for the fit.
        estimators: fitted sklearn estimator per class (empty for ``none``).
        identity_classes: classes left uncalibrated, with the reason.
    """

    dataset: str
    task: str
    family: str
    seed: int
    method: str
    classes: list[str]
    fit_split: str
    n_fit_rows: int
    estimators: dict[str, Any] = field(default_factory=dict)
    identity_classes: dict[str, str] = field(default_factory=dict)

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities.

        Args:
            probabilities: ``(n_rows, n_classes)`` raw probability matrix.

        Returns:
            Calibrated matrix with the same shape; every row sums to 1.

        Raises:
            ValueError: If the matrix width does not match ``classes``.
        """
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.classes):
            raise ValueError(
                f"expected a (n, {len(self.classes)}) probability matrix, got "
                f"{matrix.shape}"
            )
        if self.method == "none":
            return matrix

        calibrated = np.empty_like(matrix)
        for position, label in enumerate(self.classes):
            scores = matrix[:, position]
            estimator = self.estimators.get(label)
            if estimator is None:
                calibrated[:, position] = scores
            elif self.method == "isotonic":
                calibrated[:, position] = estimator.predict(scores)
            else:
                calibrated[:, position] = estimator.predict_proba(
                    _logit(scores).reshape(-1, 1)
                )[:, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        totals = calibrated.sum(axis=1, keepdims=True)
        # Guard against an all-zero row (possible with isotonic on a degenerate
        # calibration split): fall back to a uniform distribution.
        degenerate = totals[:, 0] <= 0.0
        if degenerate.any():
            calibrated[degenerate] = 1.0 / len(self.classes)
            totals = calibrated.sum(axis=1, keepdims=True)
        return calibrated / totals

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serialisable metadata (without the estimators)."""
        return {
            "dataset": self.dataset,
            "task": self.task,
            "family": self.family,
            "seed": self.seed,
            "method": self.method,
            "classes": list(self.classes),
            "fit_split": self.fit_split,
            "n_fit_rows": self.n_fit_rows,
            "identity_classes": self.identity_classes,
        }


def fit_calibrator(
    probabilities: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    method: str,
    dataset: str,
    task: str,
    family: str,
    seed: int,
    fit_split: str,
) -> ProbabilityCalibrator:
    """Fit one calibrator per class on the designated split.

    Args:
        probabilities: raw probabilities from the model on the fit split.
        labels: true string labels for the same rows.
        classes: class order of the probability columns.
        method: ``none``, ``isotonic`` or ``sigmoid``.
        dataset: dataset name.
        task: task name.
        family: model family.
        seed: model seed.
        fit_split: name of the split the rows come from.

    Returns:
        A fitted :class:`ProbabilityCalibrator`.

    Raises:
        ValueError: If ``method`` is unknown or ``probabilities`` has the wrong
            width.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"unknown calibration method {method!r}; expected one of "
            f"{list(CALIBRATION_METHODS)}"
        )
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError(
            f"expected a (n, {len(classes)}) probability matrix, got {matrix.shape}"
        )
    labels_array = np.asarray([str(label) for label in labels], dtype=object)

    calibrator = ProbabilityCalibrator(
        dataset=dataset,
        task=task,
        family=family,
        seed=int(seed),
        method=method,
        classes=[str(label) for label in classes],
        fit_split=fit_split,
        n_fit_rows=int(len(matrix)),
    )
    if method == "none":
        return calibrator

    for position, label in enumerate(calibrator.classes):
        target = (labels_array == label).astype(float)
        if target.min() == target.max():
            # The split contains a single class for this label; any calibrator
            # would be arbitrary, so the identity is kept and recorded.
            calibrator.identity_classes[label] = (
                "class has a single value in the calibration split"
            )
            continue
        scores = np.asarray(matrix[:, position], dtype=float)
        if method == "isotonic":
            estimator: Any = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(scores, target)
        else:
            estimator = LogisticRegression(solver="lbfgs", max_iter=1000).fit(
                _logit(scores).reshape(-1, 1), target
            )
        calibrator.estimators[label] = estimator
    return calibrator


def save_calibrator(calibrator: ProbabilityCalibrator, path: Path) -> int:
    """Persist a calibrator and return the artifact size in bytes.

    The payload is a plain dictionary (metadata plus fitted estimators), not the
    dataclass instance. Pickling an object whose class is defined in a module
    that a CLI runs as ``__main__`` records the class as
    ``__main__.ProbabilityCalibrator``, which then fails to unpickle from any
    other entry point (for example ``make calibrate-*`` followed by
    ``make gate-*``).

    Args:
        calibrator: fitted calibrator to persist.
        path: destination artifact path.

    Returns:
        Size of the written file in bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"metadata": calibrator.to_payload(), "estimators": calibrator.estimators},
        path,
    )
    return int(path.stat().st_size)


def load_calibrator(path: Path) -> ProbabilityCalibrator:
    """Load a calibrator artifact.

    Args:
        path: artifact path written by :func:`save_calibrator`.

    Returns:
        The stored :class:`ProbabilityCalibrator`.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the payload is not a calibrator artifact.
    """
    if not path.is_file():
        raise FileNotFoundError(f"calibrator not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"{path} does not contain a calibrator artifact")
    metadata = payload["metadata"]
    return ProbabilityCalibrator(
        dataset=str(metadata["dataset"]),
        task=str(metadata["task"]),
        family=str(metadata["family"]),
        seed=int(metadata["seed"]),
        method=str(metadata["method"]),
        classes=[str(label) for label in metadata["classes"]],
        fit_split=str(metadata["fit_split"]),
        n_fit_rows=int(metadata["n_fit_rows"]),
        estimators=dict(payload["estimators"]),
        identity_classes=dict(metadata.get("identity_classes", {})),
    )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: np.ndarray, labels: Sequence[str], classes: Sequence[str], bins: int = 15
) -> float:
    """Return the top-1 expected calibration error.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.
        labels: true string labels.
        classes: class order of the matrix columns.
        bins: number of equal-width confidence bins.

    Returns:
        ``sum_bins (n_bin / n) * |accuracy_bin - confidence_bin|``, in [0, 1].
    """
    matrix = np.asarray(probabilities, dtype=float)
    class_index = {str(label): position for position, label in enumerate(classes)}
    y_true = np.asarray([class_index.get(str(label), -1) for label in labels])
    confidence = matrix.max(axis=1)
    prediction = matrix.argmax(axis=1)
    correct = (prediction == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(matrix)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper >= 1.0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / total) * abs(
            correct[in_bin].mean() - confidence[in_bin].mean()
        )
    return float(error)


def multiclass_brier_score(
    probabilities: np.ndarray, labels: Sequence[str], classes: Sequence[str]
) -> float:
    """Return the mean squared error between probabilities and one-hot truth."""
    matrix = np.asarray(probabilities, dtype=float)
    class_index = {str(label): position for position, label in enumerate(classes)}
    one_hot = np.zeros_like(matrix)
    for row, label in enumerate(labels):
        one_hot[row, class_index[str(label)]] = 1.0
    return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


def mean_log_loss(
    probabilities: np.ndarray, labels: Sequence[str], classes: Sequence[str]
) -> float:
    """Return the mean negative log likelihood of the true class."""
    matrix = np.clip(np.asarray(probabilities, dtype=float), _EPSILON, 1.0)
    class_index = {str(label): position for position, label in enumerate(classes)}
    true_probabilities = np.asarray(
        [matrix[row, class_index[str(label)]] for row, label in enumerate(labels)]
    )
    return float(-np.mean(np.log(true_probabilities)))


def calibration_metrics(
    probabilities: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    bins: int = 15,
) -> dict[str, Any]:
    """Return accuracy plus the calibration metrics for one probability matrix.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.
        labels: true string labels.
        classes: class order of the matrix columns.
        bins: ECE / reliability bins.

    Returns:
        Dictionary with ``accuracy``, ``mean_confidence``, ``ece``, ``brier``,
        ``log_loss``, ``n_samples`` and ``n_bins``.
    """
    matrix = np.asarray(probabilities, dtype=float)
    class_index = {str(label): position for position, label in enumerate(classes)}
    y_true = np.asarray([class_index.get(str(label), -1) for label in labels])
    predictions = matrix.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predictions == y_true)),
        "mean_confidence": float(matrix.max(axis=1).mean()),
        "ece": expected_calibration_error(matrix, labels, classes, bins),
        "brier": multiclass_brier_score(matrix, labels, classes),
        "log_loss": mean_log_loss(matrix, labels, classes),
        "n_samples": int(len(matrix)),
        "n_bins": int(bins),
    }


def reliability_curve(
    probabilities: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    bins: int = 15,
) -> dict[str, list[float]]:
    """Return the top-1 reliability curve (confidence vs accuracy per bin)."""
    matrix = np.asarray(probabilities, dtype=float)
    class_index = {str(label): position for position, label in enumerate(classes)}
    y_true = np.asarray([class_index.get(str(label), -1) for label in labels])
    confidence = matrix.max(axis=1)
    correct = (matrix.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centres: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper >= 1.0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if not in_bin.any():
            continue
        centres.append(float(confidence[in_bin].mean()))
        observed.append(float(correct[in_bin].mean()))
        counts.append(float(in_bin.sum()))
    return {"confidence": centres, "accuracy": observed, "count": counts}


def plot_reliability(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
    task: str,
    family: str,
    method: str,
    path: Path,
) -> None:
    """Save a reliability diagram comparing raw and calibrated probabilities."""
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    axis.plot(
        before["confidence"],
        before["accuracy"],
        marker="o",
        label="raw",
    )
    axis.plot(
        after["confidence"],
        after["accuracy"],
        marker="s",
        label=f"calibrated ({method})",
    )
    axis.set_xlabel("mean predicted confidence (top-1)")
    axis.set_ylabel("observed accuracy")
    axis.set_title(f"Reliability — {task} / {family}")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

#: Name pattern of a calibrator artifact.
def calibrator_path(models_dir: Path, dataset: str, task: str, family: str, seed: int) -> Path:
    """Return the artifact path for one run's calibrator."""
    return models_dir / "tier1" / f"{dataset}_{task}_{family}_seed{seed}_calibrator.joblib"


def _predict(model: TrainedModel, split_data: Any, split: str) -> np.ndarray:
    """Return the model's raw probability matrix for one split."""
    return np.asarray(model.predict_proba(split_data.features(split)), dtype=float)


def run_calibration(
    dataset: str,
    config: Config,
    tasks: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit and evaluate a calibrator for every benchmarked Tier-1 run.

    Args:
        dataset: dataset name.
        config: project config (calibration method, splits, ECE bins).
        tasks: optional task filter.
        families: optional family filter.
        seeds: optional seed filter.
        processed_dir: override for ``data/processed`` (tests).
        models_dir: override for ``models`` (tests).
        reports_dir: override for ``reports`` (tests).
        figures_dir: override for ``reports/figures`` (tests).

    Returns:
        The calibration report that was written to
        ``reports/calibration_report.json``.

    Raises:
        FileNotFoundError: If the benchmark report, a model artifact or the
            processed splits are missing.
        ValueError: If the configured calibration method is unknown.
    """
    if config.tier1.calibration not in CALIBRATION_METHODS:
        raise ValueError(
            f"tier1.calibration={config.tier1.calibration!r} is not one of "
            f"{list(CALIBRATION_METHODS)}"
        )
    root = PROJECT_ROOT
    processed = processed_dir or (root / config.paths.processed)
    models = models_dir or (root / config.paths.models)
    reports = reports_dir or (root / config.paths.reports)
    figures = figures_dir or (root / config.paths.figures)
    for directory in (reports, figures, models / "tier1"):
        directory.mkdir(parents=True, exist_ok=True)

    runs = read_benchmark_runs(reports, dataset)
    if tasks:
        runs = [run for run in runs if run["task"] in set(tasks)]
    if families:
        runs = [run for run in runs if run["family"] in set(families)]
    if seeds:
        runs = [run for run in runs if int(run["seed"]) in {int(s) for s in seeds}]
    if not runs:
        raise ValueError(
            "no benchmark runs match the requested tasks/families/seeds"
        )

    split_data = load_processed_splits(
        dataset, config, processed_dir=processed, models_dir=models
    )
    fit_split = config.gate.calibrator_fit_split
    bins = config.eval.calibration_bins
    method = config.tier1.calibration

    records: list[dict[str, Any]] = []
    for run in runs:
        task = str(run["task"])
        family = str(run["family"])
        seed = int(run["seed"])
        model_path = resolve_report_path(run["model_path"], root)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"model artifact missing: {model_path}\n"
                f"Run `make tier1-benchmark DATASET={dataset}` first."
            )
        model = trained_model_from_artifact(model_path, task, family, seed)
        # The artifact's label order is the column order of predict_proba.
        labels = list(model.labels)

        raw_fit = _predict(model, split_data, fit_split)
        y_fit = split_data.target(fit_split, task)
        calibrator = fit_calibrator(
            raw_fit,
            y_fit,
            labels,
            method,
            dataset,
            task,
            family,
            seed,
            fit_split,
        )
        artifact = calibrator_path(models, dataset, task, family, seed)
        size = save_calibrator(calibrator, artifact)

        splits_report: dict[str, Any] = {}
        curves: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]] = {}
        # Test is only read here, after every fit for this run is complete.
        for split in SPLIT_NAMES:
            raw = _predict(model, split_data, split)
            calibrated = calibrator.calibrate(raw)
            y_true = split_data.target(split, task)
            splits_report[split] = {
                "raw": calibration_metrics(raw, y_true, labels, bins),
                "calibrated": calibration_metrics(calibrated, y_true, labels, bins),
            }
            curves[split] = (
                reliability_curve(raw, y_true, labels, bins),
                reliability_curve(calibrated, y_true, labels, bins),
            )

        figure_path = figures / f"tier1_reliability_{task}_{family}.png"
        if seed == int(runs[0]["seed"]):
            plot_reliability(
                curves["test"][0], curves["test"][1], task, family, method, figure_path
            )

        records.append(
            {
                "task": task,
                "family": family,
                "seed": seed,
                "model_path": run["model_path"],
                "calibrator_path": str(artifact.relative_to(root))
                if artifact.is_relative_to(root)
                else str(artifact),
                "calibrator_size_bytes": size,
                "method": method,
                "classes": labels,
                "n_calibrator_inputs": int(calibrator.n_fit_rows),
                "identity_classes": calibrator.identity_classes,
                "fit_split": fit_split,
                "metrics": splits_report,
                "reliability_curve_test": {
                    "raw": curves["test"][0],
                    "calibrated": curves["test"][1],
                },
                "figure": str(figure_path.relative_to(root))
                if figure_path.is_relative_to(root)
                else str(figure_path),
            }
        )
        logger.info(
            "Calibrated %s/%s seed %d: test ECE %.4f -> %.4f",
            task,
            family,
            seed,
            splits_report["test"]["raw"]["ece"],
            splits_report["test"]["calibrated"]["ece"],
        )

    report: dict[str, Any] = {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "fit_split": fit_split,
        "bins": bins,
        "classes_by_task": {
            task: sorted({label for record in records if record["task"] == task for label in record["classes"]})
            for task in sorted({record["task"] for record in records})
        },
        "runs": records,
        "definitions": {
            "ece": "top-1 expected calibration error, equal-width confidence bins",
            "brier": "mean squared error between the probability vector and the "
            "one-hot label (multiclass Brier score)",
            "log_loss": "mean negative log likelihood of the true class",
            "multiclass_note": "per-class one-vs-rest calibration, then row "
            "renormalisation (an approximation for >2 classes)",
            "test_usage": "test metrics are reported only; they never influence "
            "the calibrator",
        },
    }
    json_path = reports / "calibration_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, reports / "calibration_report.md")
    logger.info("Wrote %s and %s", json_path, reports / "calibration_report.md")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Write the calibration comparison table."""
    lines = [
        "# Probability calibration",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Method: `{report['method']}` (fitted on the `{report['fit_split']}` split)",
        f"- ECE / reliability bins: {report['bins']}",
        "- Test split: reported only, never used to fit the calibrator",
        "",
        "| Task | Family | Seed | Split | Rows | Accuracy | ECE raw | ECE calibrated "
        "| Brier raw | Brier calibrated | Log loss raw | Log loss calibrated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["runs"]:
        for split, metrics in record["metrics"].items():
            raw = metrics["raw"]
            calibrated = metrics["calibrated"]
            lines.append(
                f"| {record['task']} | {record['family']} | {record['seed']} | {split} "
                f"| {raw['n_samples']} | {calibrated['accuracy']:.4f} "
                f"| {raw['ece']:.4f} | {calibrated['ece']:.4f} "
                f"| {raw['brier']:.4f} | {calibrated['brier']:.4f} "
                f"| {raw['log_loss']:.4f} | {calibrated['log_loss']:.4f} |"
            )
    lines += [
        "",
        "> Accuracy is unchanged by a monotone per-class calibration only when the "
        "argmax is preserved; the multiclass path renormalises rows, so a small "
        "difference is expected. Uncalibrated classes are listed per run under "
        "`identity_classes` in the JSON report.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for probability calibration.

    Args:
        argv: argument list override (tests).

    Returns:
        Process exit code: 0 on success, 1 when required artifacts are missing.
    """
    parser = argparse.ArgumentParser(
        description="Fit probability calibrators for the Tier-1 candidates."
    )
    parser.add_argument("--dataset", required=True, help="cicids2017 or ciciot2023")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--tasks", default=None, help="comma-separated tasks")
    parser.add_argument("--families", default=None, help="comma-separated families")
    parser.add_argument("--seeds", default=None, help="comma-separated integer seeds")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)

    def _values(value: str | None) -> list[str] | None:
        return [part.strip() for part in value.split(",") if part.strip()] if value else None

    try:
        report = run_calibration(
            dataset=args.dataset,
            config=config,
            tasks=_values(args.tasks),
            families=_values(args.families),
            seeds=[int(v) for v in _values(args.seeds)] if args.seeds else None,
            processed_dir=Path(args.processed_dir) if args.processed_dir else None,
            models_dir=Path(args.models_dir) if args.models_dir else None,
            reports_dir=Path(args.reports_dir) if args.reports_dir else None,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    logger.info("Calibrated %d run(s).", len(report["runs"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
