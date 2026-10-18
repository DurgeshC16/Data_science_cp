"""Shared helpers for the Tier-1, calibration and gate tests.

The helpers build a synthetic processed dataset in exactly the on-disk format
``src.data.preprocess`` produces, so the calibration and gate tests exercise the
real code paths. Numbers produced from these fixtures are test values, never
dataset results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import Config, Tier1Config, load_config
from src.data.preprocess import BINARY_COLUMN, MULTICLASS_COLUMN, SPLIT_NAMES

FEATURES = ["f0", "f1", "f2", "f3"]
CLASSES = ["DDoS", "DoS", "Normal"]


def build_processed_dataset(
    root: Path,
    dataset: str = "toy",
    classes: Sequence[str] = CLASSES,
    n_per_class: int = 45,
    seed: int = 0,
    features: Sequence[str] = FEATURES,
    zeroday_class: str | None = None,
    zeroday_rows: int = 0,
) -> tuple[Path, Path]:
    """Write synthetic processed splits and a feature list under ``root``.

    Args:
        root: temporary directory acting as the artefact root.
        dataset: dataset name used in the file names.
        classes: multiclass labels to generate.
        n_per_class: rows per class, split as in Phase 2 (60/10/10/20).
        seed: RNG seed.
        features: feature column names.
        zeroday_class: when set, write a zero-day holdout parquet containing
            this class only.
        zeroday_rows: number of zero-day rows to write.

    Returns:
        ``(processed_dir, models_dir)``.
    """
    rng = np.random.default_rng(seed)
    fractions = {"train": 0.6, "val": 0.1, "calibration": 0.1, "test": 0.2}
    rows: list[dict[str, Any]] = []
    row_id = 0
    for class_index, label in enumerate(classes):
        counts = {split: int(round(f * n_per_class)) for split, f in fractions.items()}
        counts["test"] = n_per_class - sum(
            value for split, value in counts.items() if split != "test"
        )
        for split in SPLIT_NAMES:
            for _ in range(counts[split]):
                offset = 3.0 * (class_index - 1)
                values = rng.normal(size=len(features))
                values[0] += offset
                rows.append(
                    {
                        "flow_id": f"{dataset}-{row_id:06d}",
                        "split": split,
                        **{
                            name: float(value)
                            for name, value in zip(features, values)
                        },
                        MULTICLASS_COLUMN: label,
                        BINARY_COLUMN: "Normal" if label == "Normal" else "Attack",
                    }
                )
                row_id += 1

    frame = pd.DataFrame(rows)
    processed = root / "processed"
    models = root / "models"
    processed.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        part = (
            frame[frame["split"] == split]
            .drop(columns=["split"])
            .reset_index(drop=True)
        )
        part.to_parquet(processed / f"{dataset}_{split}.parquet", index=False)

    if zeroday_class is not None and zeroday_rows:
        holdout_rows = []
        for index in range(zeroday_rows):
            values = rng.normal(size=len(features)) + 12.0
            holdout_rows.append(
                {
                    "flow_id": f"{dataset}-zeroday-{index:05d}",
                    **{
                        name: float(value)
                        for name, value in zip(features, values)
                    },
                    MULTICLASS_COLUMN: zeroday_class,
                    BINARY_COLUMN: "Attack",
                }
            )
        pd.DataFrame(holdout_rows).to_parquet(
            processed / f"{dataset}_zeroday_test.parquet", index=False
        )

    (models / f"{dataset}_features.json").write_text(
        json.dumps({"feature_columns": list(features)}), encoding="utf-8"
    )
    return processed, models


def tiny_tier1_config(
    benchmark_overrides: dict[str, Any] | None = None,
    mlp_overrides: dict[str, Any] | None = None,
    optuna_overrides: dict[str, Any] | None = None,
    gate_overrides: dict[str, Any] | None = None,
    calibration: str | None = None,
) -> Config:
    """Return the project config with fast test-sized Tier-1 settings.

    Args:
        benchmark_overrides: overrides merged into ``tier1.benchmark``.
        mlp_overrides: overrides merged into ``tier1.mlp``.
        optuna_overrides: overrides merged into ``tier1.optuna``.
        gate_overrides: overrides merged into ``gate``.
        calibration: override for ``tier1.calibration``.

    Returns:
        A validated :class:`~src.config.Config` copy.
    """
    base = load_config()
    payload = base.tier1.model_dump()
    payload["benchmark"] = {
        **base.tier1.benchmark.model_dump(),
        "latency_calls": 10,
        "warmup_calls": 1,
        "batch_size": 32,
        "throughput_max_rows": 32,
        **(benchmark_overrides or {}),
    }
    payload["mlp"] = {
        **base.tier1.mlp.model_dump(),
        "hidden_layer_options": [[8, 4]],
        "epochs": 3,
        "batch_size": 32,
        "patience": 2,
        **(mlp_overrides or {}),
    }
    payload["optuna"] = {
        **base.tier1.optuna.model_dump(),
        "n_trials": 1,
        "time_budget_seconds": 60,
        **(optuna_overrides or {}),
    }
    if calibration is not None:
        payload["calibration"] = calibration
    updates: dict[str, Any] = {"tier1": Tier1Config.model_validate(payload)}
    if gate_overrides:
        from src.config import GateConfig

        updates["gate"] = GateConfig.model_validate(
            {**base.gate.model_dump(), **gate_overrides}
        )
    return base.model_copy(update=updates)


def run_tiny_benchmark(
    processed: Path,
    models: Path,
    reports: Path,
    dataset: str = "toy",
    families: Iterable[str] = ("random_forest",),
    tasks: Iterable[str] = ("binary", "multiclass"),
    seeds: Sequence[int] = (42,),
    config: Config | None = None,
) -> dict[str, Any]:
    """Run the real Tier-1 benchmark on the synthetic fixture.

    Args:
        processed: processed-split directory.
        models: model-artifact directory.
        reports: report directory (the benchmark JSON is written here).
        dataset: dataset name.
        families: families to benchmark.
        tasks: tasks to benchmark.
        seeds: seeds to fit.
        config: config override; defaults to :func:`tiny_tier1_config`.

    Returns:
        The benchmark report dictionary.
    """
    from src.tier1.train import run_benchmark

    return run_benchmark(
        dataset,
        config or tiny_tier1_config(),
        tasks=list(tasks),
        families=list(families),
        seeds=list(seeds),
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
