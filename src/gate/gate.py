"""The confidence gate: uncertainty scoring and escalation policy.

The gate sits between Tier 1 and Tier 2. Tier-1 probabilities (calibrated when
``gate.use_calibrated_probabilities`` is true) are turned into two uncertainty
scores, and a flow is escalated to Tier 2 when **either**

- the top-1 probability is below ``max_prob_threshold`` (the model is unsure), or
- the predictive entropy exceeds ``entropy_threshold`` (the mass is spread out).

Thresholds are **fitted, never guessed**: a quantile grid is evaluated on
``gate.threshold_fit_split`` (``val``) and the pair whose escalation rate falls
inside ``[target_escalation_rate_min, target_escalation_rate_max]`` and is
closest to the middle of that band wins. If no candidate reaches the band the
run stops with the best achievable rate in the message rather than reporting a
threshold that misses its target.

Artifacts
---------
- ``reports/gate_report.json``
- ``reports/gate_report.md``
- ``reports/figures/gate_sweep_<task>_<family>.png``
- ``reports/figures/gate_entropy_<task>_<family>.png``

Usage
-----
::

    python -m src.gate.gate --dataset cicids2017

Requires ``make tier1-benchmark`` and, when calibrated probabilities are
enabled, ``make calibrate-<dataset>``. Without them the command exits 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, Config, load_config  # noqa: E402
from src.data.preprocess import SPLIT_NAMES  # noqa: E402
from src.tier1.calibrate import calibrator_path, load_calibrator  # noqa: E402
from src.tier1.train import (  # noqa: E402
    load_processed_splits,
    read_benchmark_runs,
    resolve_report_path,
    trained_model_from_artifact,
)

logger = logging.getLogger(__name__)

#: Entropy units and their log base.
ENTROPY_BASES: dict[str, float] = {"natural": float(np.e), "bits": 2.0}


def top_probability(probabilities: np.ndarray) -> np.ndarray:
    """Return the top-1 probability of every row.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.

    Returns:
        Array of length ``n_rows``.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {matrix.shape}")
    return matrix.max(axis=1)


def predictive_entropy(
    probabilities: np.ndarray, base: str = "natural"
) -> np.ndarray:
    """Return the predictive entropy of every row.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.
        base: ``natural`` (nats) or ``bits``.

    Returns:
        Array of length ``n_rows`` in the configured unit.

    Raises:
        ValueError: If ``base`` is unknown.
    """
    if base not in ENTROPY_BASES:
        raise ValueError(f"unknown entropy base {base!r}; expected {list(ENTROPY_BASES)}")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {matrix.shape}")
    clipped = np.clip(matrix, 1e-12, 1.0)
    if base == "natural":
        return -np.sum(matrix * np.log(clipped), axis=1)
    return -np.sum(matrix * np.log(clipped), axis=1) / np.log(ENTROPY_BASES[base])


def uncertainty_scores(
    probabilities: np.ndarray, base: str = "natural"
) -> dict[str, np.ndarray]:
    """Return both gate scores for a probability matrix.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.
        base: entropy unit.

    Returns:
        Dictionary with ``max_probability`` and ``entropy``.
    """
    return {
        "max_probability": top_probability(probabilities),
        "entropy": predictive_entropy(probabilities, base),
    }


def escalate_mask(
    max_probability: np.ndarray,
    entropy: np.ndarray,
    max_prob_threshold: float,
    entropy_threshold: float,
) -> np.ndarray:
    """Return the escalation mask for the OR policy.

    Args:
        max_probability: top-1 probabilities.
        entropy: predictive entropies.
        max_prob_threshold: escalate when the top probability is below this.
        entropy_threshold: escalate when the entropy exceeds this.

    Returns:
        Boolean array; ``True`` means the flow goes to Tier 2.
    """
    top = np.asarray(max_probability, dtype=float)
    uncertainty = np.asarray(entropy, dtype=float)
    if top.shape != uncertainty.shape:
        raise ValueError("max_probability and entropy must have the same shape")
    return (top < float(max_prob_threshold)) | (uncertainty > float(entropy_threshold))


def escalation_rate(mask: np.ndarray) -> float:
    """Return the fraction of escalated rows (0.0 for an empty mask)."""
    array = np.asarray(mask, dtype=bool)
    return float(array.mean()) if array.size else 0.0


def fit_thresholds(
    max_probability: np.ndarray,
    entropy: np.ndarray,
    target_min: float,
    target_max: float,
    grid_steps: int = 101,
) -> dict[str, Any]:
    """Fit gate thresholds on the designated split by quantile search.

    Each candidate level ``t`` sets ``max_prob_threshold`` to the ``t`` quantile
    of the top-1 probabilities (escalate the least confident ``t`` share) and
    ``entropy_threshold`` to the ``1 - t`` quantile of the entropies (escalate
    the most uncertain ``t`` share). Their union is evaluated, and the candidate
    inside ``[target_min, target_max]`` closest to the midpoint wins.

    Args:
        max_probability: top-1 probabilities on the threshold-fit split.
        entropy: predictive entropies on the same rows.
        target_min: lower bound of the acceptable escalation rate.
        target_max: upper bound of the acceptable escalation rate.
        grid_steps: number of quantile levels to evaluate.

    Returns:
        Dictionary with the fitted thresholds, the achieved rate, the band, the
        full sweep and the selection criterion.

    Raises:
        ValueError: If the inputs are empty, mismatched, or ``grid_steps`` is too
            small, or if no candidate reaches the target band (the message
            reports the best achievable rate).
    """
    top = np.asarray(max_probability, dtype=float)
    uncertainty = np.asarray(entropy, dtype=float)
    if top.shape != uncertainty.shape:
        raise ValueError("max_probability and entropy must have the same shape")
    if top.size == 0:
        raise ValueError("cannot fit gate thresholds on an empty split")
    if grid_steps < 2:
        raise ValueError("grid_steps must be at least 2")
    if not 0.0 <= target_min <= target_max <= 1.0:
        raise ValueError(
            f"invalid target band [{target_min}, {target_max}]; expected "
            "0 <= min <= max <= 1"
        )

    levels = np.linspace(0.0, 1.0, int(grid_steps))
    prob_thresholds = np.quantile(top, levels)
    entropy_thresholds = np.quantile(uncertainty, 1.0 - levels)
    midpoint = (target_min + target_max) / 2.0

    sweep: list[dict[str, float]] = []
    best: dict[str, Any] | None = None
    best_gap: float | None = None
    for level, prob_threshold, entropy_threshold in zip(
        levels, prob_thresholds, entropy_thresholds
    ):
        mask = escalate_mask(top, uncertainty, prob_threshold, entropy_threshold)
        rate = escalation_rate(mask)
        sweep.append(
            {
                "level": float(level),
                "max_prob_threshold": float(prob_threshold),
                "entropy_threshold": float(entropy_threshold),
                "escalation_rate": rate,
            }
        )
        if target_min <= rate <= target_max:
            gap = abs(rate - midpoint)
            if best is None or gap < best_gap or (
                gap == best_gap and rate < best["achieved_rate"]
            ):
                best = {
                    "max_prob_threshold": float(prob_threshold),
                    "entropy_threshold": float(entropy_threshold),
                    "achieved_rate": rate,
                    "level": float(level),
                }
                best_gap = gap

    if best is None:
        rates = [entry["escalation_rate"] for entry in sweep]
        closest = min(rates, key=lambda rate: min(abs(rate - target_min), abs(rate - target_max)))
        raise ValueError(
            f"no threshold pair reaches the target escalation band "
            f"[{target_min}, {target_max}]; closest achievable rate was "
            f"{closest:.4f} over {grid_steps} quantile levels. Widen the band, "
            "increase gate.grid_steps, or gather more validation rows."
        )

    return {
        **best,
        "target_min": float(target_min),
        "target_max": float(target_max),
        "target_midpoint": float(midpoint),
        "in_band": True,
        "grid_steps": int(grid_steps),
        "candidates_evaluated": int(len(sweep)),
        "criterion": "escalation rate inside the target band, closest to the "
        "band midpoint (ties broken by the lower rate)",
        "sweep": sweep,
    }


def evaluate_gate(
    probabilities: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Summarise what the gate would do on a split.

    Args:
        probabilities: ``(n_rows, n_classes)`` probability matrix.
        labels: true string labels for the rows.
        classes: class order of the probability columns.
        mask: escalation mask.

    Returns:
        Dictionary with the escalation rate, accuracy overall / kept /
        escalated, the share of model errors that were escalated, and per-class
        escalation counts.
    """
    matrix = np.asarray(probabilities, dtype=float)
    escalated = np.asarray(mask, dtype=bool)
    if len(matrix) != len(escalated):
        raise ValueError("probabilities and mask must have the same length")
    index = {str(label): position for position, label in enumerate(classes)}
    y_true = np.asarray([index.get(str(label), -1) for label in labels])
    predictions = matrix.argmax(axis=1)
    correct = predictions == y_true

    def _accuracy(selection: np.ndarray) -> float:
        return float(correct[selection].mean()) if selection.any() else 0.0

    errors = ~correct
    error_capture = (
        float(escalated[errors].mean()) if errors.any() else 0.0
    )
    per_class: dict[str, dict[str, Any]] = {}
    for label in classes:
        key = str(label)
        in_class = np.asarray([str(value) == key for value in labels])
        per_class[key] = {
            "n": int(in_class.sum()),
            "escalated": int(escalated[in_class].sum()),
            "escalation_rate": float(escalated[in_class].mean())
            if in_class.any()
            else 0.0,
        }
    return {
        "n_flows": int(len(matrix)),
        "n_escalated": int(escalated.sum()),
        "escalation_rate": escalation_rate(escalated),
        "accuracy_all": _accuracy(np.ones_like(escalated, dtype=bool)),
        "accuracy_tier1_only": _accuracy(~escalated),
        "accuracy_escalated": _accuracy(escalated),
        "error_capture_rate": error_capture,
        "per_class": per_class,
    }


def plot_sweep(
    sweep: list[dict[str, float]],
    target_min: float,
    target_max: float,
    chosen_level: float,
    task: str,
    family: str,
    path: Path,
) -> None:
    """Save the escalation-rate sweep with the target band marked."""
    levels = [entry["level"] for entry in sweep]
    rates = [entry["escalation_rate"] for entry in sweep]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axhspan(target_min, target_max, color="tab:green", alpha=0.15, label="target band")
    axis.plot(levels, rates, color="tab:blue", label="escalation rate")
    axis.axvline(chosen_level, color="tab:red", linestyle="--", label="chosen level")
    axis.set_xlabel("quantile level t (low-confidence / high-entropy share)")
    axis.set_ylabel("escalation rate")
    axis.set_title(f"Gate sweep — {task} / {family}")
    axis.set_ylim(0.0, 1.0)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_entropy(
    entropy: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    threshold: float,
    task: str,
    family: str,
    path: Path,
) -> None:
    """Save a per-class entropy histogram with the fitted threshold."""
    figure, axis = plt.subplots(figsize=(7, 4))
    label_array = np.asarray([str(label) for label in labels], dtype=object)
    bins = np.linspace(0.0, float(np.max(entropy)) * 1.05 + 1e-9, 30)
    for label in classes:
        selected = entropy[label_array == str(label)]
        if selected.size:
            axis.hist(selected, bins=bins, alpha=0.5, label=str(label))
    axis.axvline(threshold, color="tab:red", linestyle="--", label="entropy threshold")
    axis.set_xlabel("predictive entropy")
    axis.set_ylabel("flows")
    axis.set_title(f"Entropy by true class — {task} / {family}")
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _display(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible."""
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _zeroday_frame(dataset: str, processed: Path) -> pd.DataFrame | None:
    """Return the zero-day holdout frame when it exists and is non-empty."""
    path = processed / f"{dataset}_zeroday_test.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    return frame if len(frame) else None


def run_gate(
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
    """Fit the gate thresholds and report the escalation behaviour.

    Thresholds are fitted on ``gate.threshold_fit_split``; the validation and
    test splits are then scored with those frozen thresholds. The test split is
    reported only. The zero-day holdout, when present, is scored as well: it
    measures whether the gate escalates classes the Tier-1 model never saw.

    Args:
        dataset: dataset name.
        config: project config.
        tasks: optional task filter.
        families: optional family filter.
        seeds: optional seed filter.
        processed_dir: override for ``data/processed`` (tests).
        models_dir: override for ``models`` (tests).
        reports_dir: override for ``reports`` (tests).
        figures_dir: override for ``reports/figures`` (tests).

    Returns:
        The gate report that was written to ``reports/gate_report.json``.

    Raises:
        FileNotFoundError: If the benchmark, a model or a calibrator artifact is
            missing, or the processed splits are absent.
        ValueError: If no run matches the filters or the thresholds cannot reach
            the target band.
    """
    root = PROJECT_ROOT
    processed = processed_dir or (root / config.paths.processed)
    models = models_dir or (root / config.paths.models)
    reports = reports_dir or (root / config.paths.reports)
    figures = figures_dir or (root / config.paths.figures)
    for directory in (reports, figures):
        directory.mkdir(parents=True, exist_ok=True)

    runs = read_benchmark_runs(reports, dataset)
    if tasks:
        runs = [run for run in runs if run["task"] in set(tasks)]
    if families:
        runs = [run for run in runs if run["family"] in set(families)]
    if seeds:
        runs = [run for run in runs if int(run["seed"]) in {int(s) for s in seeds}]
    if not runs:
        raise ValueError("no benchmark runs match the requested tasks/families/seeds")

    split_data = load_processed_splits(
        dataset, config, processed_dir=processed, models_dir=models
    )
    fit_split = config.gate.threshold_fit_split
    base = config.gate.entropy_base
    use_calibrated = config.gate.use_calibrated_probabilities
    zeroday = _zeroday_frame(dataset, processed)

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
        calibrator = None
        if use_calibrated:
            artifact = calibrator_path(models, dataset, task, family, seed)
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"calibrator missing: {artifact}\n"
                    f"Run `make calibrate-{dataset}` first, or set "
                    "gate.use_calibrated_probabilities=false to score raw "
                    "probabilities."
                )
            calibrator = load_calibrator(artifact)

        def _probabilities(frame: pd.DataFrame) -> np.ndarray:
            raw = np.asarray(
                model.predict_proba(frame[model.feature_columns]), dtype=float
            )
            return calibrator.calibrate(raw) if calibrator is not None else raw

        val_probabilities = _probabilities(split_data.frames[fit_split])
        val_scores = uncertainty_scores(val_probabilities, base)
        thresholds = fit_thresholds(
            val_scores["max_probability"],
            val_scores["entropy"],
            config.gate.target_escalation_rate_min,
            config.gate.target_escalation_rate_max,
            config.gate.grid_steps,
        )

        def _gate_report(frame: pd.DataFrame, labels: Sequence[str]) -> dict[str, Any]:
            probabilities = _probabilities(frame)
            scores = uncertainty_scores(probabilities, base)
            mask = escalate_mask(
                scores["max_probability"],
                scores["entropy"],
                thresholds["max_prob_threshold"],
                thresholds["entropy_threshold"],
            )
            report = evaluate_gate(probabilities, labels, model.labels, mask)
            report["entropy_mean"] = float(scores["entropy"].mean())
            report["max_probability_mean"] = float(scores["max_probability"].mean())
            return report

        placeholder_mask = escalate_mask(
            val_scores["max_probability"],
            val_scores["entropy"],
            config.gate.max_prob_threshold,
            config.gate.entropy_threshold,
        )
        record: dict[str, Any] = {
            "task": task,
            "family": family,
            "seed": seed,
            "calibrated_probabilities": calibrator is not None,
            "calibrator_path": _display(
                calibrator_path(models, dataset, task, family, seed), root
            )
            if calibrator is not None
            else None,
            "entropy_base": base,
            "thresholds": thresholds,
            "placeholder_thresholds": {
                "max_prob_threshold": config.gate.max_prob_threshold,
                "entropy_threshold": config.gate.entropy_threshold,
                "escalation_rate_on_fit_split": escalation_rate(placeholder_mask),
                "note": "config placeholders, not FITTED thresholds",
            },
            fit_split: _gate_report(
                split_data.frames[fit_split], split_data.target(fit_split, task)
            ),
        }
        record["test"] = _gate_report(
            split_data.frames["test"], split_data.target("test", task)
        )
        if fit_split != "calibration" and "calibration" in SPLIT_NAMES:
            record["calibration"] = _gate_report(
                split_data.frames["calibration"],
                split_data.target("calibration", task),
            )
        if zeroday is not None:
            zeroday_labels = (
                zeroday["multiclass_label"].astype(str).to_numpy()
                if task == "multiclass"
                else zeroday["binary_label"].astype(str).to_numpy()
            )
            record["zeroday"] = _gate_report(zeroday, zeroday_labels)
            record["zeroday"]["classes_seen"] = sorted(set(zeroday_labels.tolist()))

        if seed == int(runs[0]["seed"]):
            plot_sweep(
                thresholds["sweep"],
                thresholds["target_min"],
                thresholds["target_max"],
                thresholds["level"],
                task,
                family,
                figures / f"gate_sweep_{task}_{family}.png",
            )
            test_probabilities = _probabilities(split_data.frames["test"])
            plot_entropy(
                uncertainty_scores(test_probabilities, base)["entropy"],
                split_data.target("test", task),
                model.labels,
                thresholds["entropy_threshold"],
                task,
                family,
                figures / f"gate_entropy_{task}_{family}.png",
            )
        records.append(record)
        logger.info(
            "Gate %s/%s seed %d: fit-split rate %.4f, test rate %.4f "
            "(thresholds max_prob<%.4f or entropy>%.4f)",
            task,
            family,
            seed,
            record[fit_split]["escalation_rate"],
            record["test"]["escalation_rate"],
            thresholds["max_prob_threshold"],
            thresholds["entropy_threshold"],
        )

    report: dict[str, Any] = {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold_fit_split": fit_split,
        "calibrator_fit_split": config.gate.calibrator_fit_split,
        "target_escalation_rate": {
            "min": config.gate.target_escalation_rate_min,
            "max": config.gate.target_escalation_rate_max,
        },
        "policy": "escalate when max_probability < max_prob_threshold OR "
        "entropy > entropy_threshold",
        "entropy_base": base,
        "grid_steps": config.gate.grid_steps,
        "runs": records,
        "zeroday_available": zeroday is not None,
        "definitions": {
            "escalation_rate": "escalated flows / all flows",
            "error_capture_rate": "share of Tier-1 misclassifications that the "
            "gate escalates",
            "accuracy_tier1_only": "accuracy on the flows Tier 1 answers alone",
            "zeroday": "holdout rows whose multiclass classes were removed from "
            "every modelling split; the gate should escalate them. The holdout "
            "is defined on the multiclass taxonomy, so for the binary task "
            "those rows are ordinary Attack flows and only the multiclass "
            "record measures unseen-class behaviour",
            "test_usage": "thresholds are fitted on the threshold-fit split; "
            "test rows are scored with the frozen thresholds and reported only",
        },
    }
    json_path = reports / "gate_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, reports / "gate_report.md")
    logger.info("Wrote %s and %s", json_path, reports / "gate_report.md")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    """Write the gate comparison table."""
    lines = [
        "# Confidence gate",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Policy: {report['policy']}",
        f"- Thresholds fitted on: `{report['threshold_fit_split']}`",
        f"- Calibrator fitted on: `{report['calibrator_fit_split']}`",
        f"- Target escalation rate: "
        f"{report['target_escalation_rate']['min']:.2f} - "
        f"{report['target_escalation_rate']['max']:.2f}",
        f"- Entropy base: `{report['entropy_base']}`, grid steps "
        f"{report['grid_steps']}",
        f"- Zero-day holdout available: {report['zeroday_available']}",
        "",
        "| Task | Family | Seed | Calibrated | max_prob < | entropy > | Fit rate "
        "| Test rate | Test Tier-1-only accuracy | Test error capture |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["runs"]:
        thresholds = record["thresholds"]
        fit_metrics = record[report["threshold_fit_split"]]
        test_metrics = record["test"]
        lines.append(
            f"| {record['task']} | {record['family']} | {record['seed']} "
            f"| {record['calibrated_probabilities']} "
            f"| {thresholds['max_prob_threshold']:.4f} "
            f"| {thresholds['entropy_threshold']:.4f} "
            f"| {fit_metrics['escalation_rate']:.4f} "
            f"| {test_metrics['escalation_rate']:.4f} "
            f"| {test_metrics['accuracy_tier1_only']:.4f} "
            f"| {test_metrics['error_capture_rate']:.4f} |"
        )
    lines += [
        "",
        "`Fit rate` is the escalation rate on the split the thresholds were "
        "fitted on; `Test rate` applies the frozen thresholds to the test split, "
        "which never influenced them.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the confidence gate.

    Args:
        argv: argument list override (tests).

    Returns:
        Process exit code: 0 on success, 1 when artifacts are missing or the
        target escalation band cannot be reached.
    """
    parser = argparse.ArgumentParser(
        description="Fit the confidence gate and report escalation behaviour."
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
        return (
            [part.strip() for part in value.split(",") if part.strip()]
            if value
            else None
        )

    try:
        report = run_gate(
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
    logger.info("Gate reported for %d run(s).", len(report["runs"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
