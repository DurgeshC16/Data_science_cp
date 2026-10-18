"""Tests for the Tier-1 training and benchmark module.

The fixture is a small synthetic dataset written into a temporary directory in
the same on-disk format ``src.data.preprocess`` produces. It exercises the real
training code paths (Random Forest, XGBoost, PyTorch MLP, Optuna); none of the
numbers produced here are dataset results and none are written to ``reports/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, Config, Tier1Config, load_config
from src.data.preprocess import SPLIT_NAMES
from src.tier1.train import (
    FAMILY_NAMES,
    aggregate_runs,
    compute_metrics,
    fit_model,
    load_processed_splits,
    load_tier1_model,
    main,
    measure_batch_throughput,
    measure_latency,
    run_benchmark,
    save_model,
    select_default_model,
    tune_hyperparameters,
)

FEATURES = ["f0", "f1", "f2", "f3"]
CLASSES = ["DDoS", "DoS", "Normal"]


def _build_dataset(
    root: Path, dataset: str = "toy", n_per_class: int = 120, seed: int = 0
) -> tuple[Path, Path]:
    """Write synthetic processed splits and a feature list under ``root``.

    Args:
        root: temporary directory acting as the project root for artifacts.
        dataset: dataset name used in the file names.
        n_per_class: rows per synthetic class.
        seed: RNG seed for the features.

    Returns:
        ``(processed_dir, models_dir)``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    fractions = {"train": 0.6, "val": 0.1, "calibration": 0.1, "test": 0.2}
    row_id = 0
    for class_index, label in enumerate(CLASSES):
        counts = {
            split: int(round(fractions[split] * n_per_class)) for split in SPLIT_NAMES
        }
        counts["test"] = n_per_class - sum(
            counts[split] for split in SPLIT_NAMES if split != "test"
        )
        for split in SPLIT_NAMES:
            for _ in range(counts[split]):
                offset = 3.0 * (class_index - 1)
                features = rng.normal(size=len(FEATURES))
                features[0] += offset
                rows.append(
                    {
                        "flow_id": f"{dataset}-{row_id:06d}",
                        "split": split,
                        **{name: float(value) for name, value in zip(FEATURES, features)},
                        "multiclass_label": label,
                        "binary_label": "Normal" if label == "Normal" else "Attack",
                    }
                )
                row_id += 1

    frame = pd.DataFrame(rows)
    processed = root / "processed"
    models = root / "models"
    processed.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        part = frame[frame["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        part.to_parquet(processed / f"{dataset}_{split}.parquet", index=False)
    (models / f"{dataset}_features.json").write_text(
        json.dumps({"feature_columns": FEATURES}), encoding="utf-8"
    )
    return processed, models


def _tiny_config(
    benchmark_overrides: dict[str, object] | None = None,
    mlp_overrides: dict[str, object] | None = None,
    selection_overrides: dict[str, object] | None = None,
    optuna_overrides: dict[str, object] | None = None,
) -> Config:
    """Return a copy of the project config with tiny benchmark settings."""
    base = load_config()
    payload = base.tier1.model_dump()
    payload["benchmark"] = {
        **base.tier1.benchmark.model_dump(),
        "latency_calls": 25,
        "warmup_calls": 2,
        "batch_size": 32,
        "throughput_max_rows": 64,
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
    if selection_overrides:
        payload["selection"] = {
            **base.tier1.selection.model_dump(),
            **selection_overrides,
        }
    tier1 = Tier1Config.model_validate(payload)
    return base.model_copy(update={"tier1": tier1})


def _small_params(family: str) -> dict[str, object]:
    """Return a minimal valid parameter set for one family."""
    if family == "random_forest":
        return {
            "n_estimators": 20,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
    if family == "xgboost":
        return {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
        }
    return {
        "hidden_layers": [8, 4],
        "learning_rate": 0.01,
        "dropout": 0.0,
        "weight_decay": 0.0001,
        "batch_size": 32,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_processed_splits_requires_preprocessing(tmp_path: Path) -> None:
    """Missing artifacts raise with the command that creates them."""
    with pytest.raises(FileNotFoundError) as error:
        load_processed_splits(
            "toy",
            load_config(),
            processed_dir=tmp_path / "processed",
            models_dir=tmp_path / "models",
        )
    assert "preprocess" in str(error.value)


def test_load_processed_splits_is_disjoint_and_exposes_targets(
    tmp_path: Path,
) -> None:
    """All four splits load, share no flow_id and carry both label columns."""
    processed, models = _build_dataset(tmp_path)
    data = load_processed_splits("toy", load_config(), processed, models)
    assert set(data.frames) == set(SPLIT_NAMES)
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        ids = set(data.frames[split]["flow_id"].astype(str))
        assert not (ids & seen)
        seen |= ids
        assert set(data.target(split, "binary")) <= {"Normal", "Attack"}
        assert set(data.target(split, "multiclass")) <= set(CLASSES)
    assert data.features("train").shape == (len(data.frames["train"]), len(FEATURES))


def test_load_processed_splits_rejects_duplicate_flow_ids(tmp_path: Path) -> None:
    """A flow_id in two splits is a leakage bug and must raise."""
    processed, models = _build_dataset(tmp_path)
    train = pd.read_parquet(processed / "toy_train.parquet")
    val = pd.read_parquet(processed / "toy_val.parquet")
    val.loc[0, "flow_id"] = train.loc[0, "flow_id"]
    val.to_parquet(processed / "toy_val.parquet", index=False)
    with pytest.raises(ValueError, match="not disjoint"):
        load_processed_splits("toy", load_config(), processed, models)


# ---------------------------------------------------------------------------
# Metrics, aggregation, selection
# ---------------------------------------------------------------------------


def test_compute_metrics_matches_hand_computed_values() -> None:
    """Accuracy, macro scores, per-class report and matrix are consistent."""
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    metrics = compute_metrics(y_true, y_pred, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert metrics["per_class"]["a"]["support"] == 2
    assert metrics["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert metrics["n_samples"] == 4
    # Classes with no predictions get zero precision without raising.
    metrics_unseen = compute_metrics(np.array(["a"]), np.array(["a"]), ["a", "b"])
    assert metrics_unseen["per_class"]["b"]["support"] == 0
    assert metrics_unseen["per_class"]["b"]["f1"] == 0.0


def _fake_run(
    seed: int,
    family: str = "random_forest",
    task: str = "binary",
    macro_f1: float = 0.8,
    accuracy: float = 0.9,
    p95: float = 5.0,
) -> dict[str, object]:
    """Build a minimal run record shaped like a real benchmark run."""
    return {
        "task": task,
        "family": family,
        "seed": seed,
        "val": {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": macro_f1},
        "test": {
            "accuracy": accuracy,
            "macro_precision": macro_f1,
            "macro_recall": macro_f1,
            "macro_f1": macro_f1,
            "weighted_precision": macro_f1,
            "weighted_recall": macro_f1,
            "weighted_f1": macro_f1,
        },
        "training_time_seconds": 2.0 + seed % 2,
        "model_size_bytes": 1000,
        "latency": {"median_ms": p95 / 2, "p95_ms": p95},
        "batch_throughput": {"flows_per_sec": 10000.0},
        "accuracy_per_ms": accuracy / (p95 / 2),
    }


def test_aggregate_runs_reports_mean_and_std() -> None:
    """Seeds aggregate to mean and sample standard deviation."""
    runs = [_fake_run(42, accuracy=0.90), _fake_run(43, accuracy=0.80)]
    aggregate = aggregate_runs(runs)
    assert aggregate["n_seeds"] == 2
    assert aggregate["mean"]["test.accuracy"] == pytest.approx(0.85)
    assert aggregate["std"]["test.accuracy"] == pytest.approx(
        float(np.std([0.9, 0.8], ddof=1))
    )


def test_aggregate_runs_rejects_mixed_groups() -> None:
    """Aggregating across tasks or families is a programming error."""
    with pytest.raises(ValueError):
        aggregate_runs([_fake_run(42), _fake_run(43, family="mlp")])
    with pytest.raises(ValueError):
        aggregate_runs([])


def test_selection_rule_respects_latency_budget() -> None:
    """Selection maximises the validation metric within the latency budget."""
    config = _tiny_config(selection_overrides={"max_p95_latency_ms": 10.0})
    slow = aggregate_runs([_fake_run(42, family="random_forest", macro_f1=0.95, p95=40.0)])
    fast = aggregate_runs([_fake_run(42, family="xgboost", macro_f1=0.80, p95=2.0)])
    selection = select_default_model([slow, fast], config)
    assert selection["chosen"]["family"] == "xgboost"
    assert selection["fallback_used"] is False
    assert [entry["family"] for entry in selection["eligible"]] == ["xgboost"]


def test_selection_rule_falls_back_when_nothing_is_fast_enough() -> None:
    """No eligible candidate: pick the lowest latency and say so."""
    config = _tiny_config(selection_overrides={"max_p95_latency_ms": 1.0})
    a = aggregate_runs([_fake_run(42, family="random_forest", macro_f1=0.95, p95=40.0)])
    b = aggregate_runs([_fake_run(42, family="mlp", macro_f1=0.70, p95=20.0)])
    selection = select_default_model([a, b], config)
    assert selection["fallback_used"] is True
    assert selection["chosen"]["family"] == "mlp"
    assert "lowest p95" in selection["reason"]


# ---------------------------------------------------------------------------
# Fitting, timing, persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_fit_model_round_trip(family: str, tmp_path: Path) -> None:
    """Every family trains, predicts, saves and reloads."""
    processed, models = _build_dataset(tmp_path, n_per_class=30)
    config = _tiny_config()
    data = load_processed_splits("toy", config, processed, models)
    labels = sorted(set(data.target("train", "multiclass")))
    model = fit_model(
        family,
        data.features("train"),
        data.target("train", "multiclass"),
        data.features("val"),
        data.target("val", "multiclass"),
        labels,
        data.feature_columns,
        _small_params(family),
        config,
        seed=42,
        task="multiclass",
    )
    probabilities = model.predict_proba(data.features("test"))
    assert probabilities.shape == (len(data.frames["test"]), len(labels))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    predictions = model.predict(data.features("test"))
    assert set(predictions) <= set(labels)
    assert model.training_time_seconds > 0

    suffix = ".pt" if family == "mlp" else ".joblib"
    path = tmp_path / f"model{suffix}"
    size = save_model(model, path)
    assert path.is_file() and size == path.stat().st_size > 0
    payload = load_tier1_model(path)
    if family == "mlp":
        assert payload["classes"] == labels
        assert isinstance(payload["model"], type(model.estimator))
    else:
        assert payload["labels"] == labels
        assert payload["family"] == family


def test_latency_and_throughput_are_measured(tmp_path: Path) -> None:
    """Latency returns median/p95 and throughput a positive rate."""
    processed, models = _build_dataset(tmp_path, n_per_class=30)
    config = _tiny_config()
    data = load_processed_splits("toy", config, processed, models)
    labels = sorted(set(data.target("train", "binary")))
    model = fit_model(
        "random_forest",
        data.features("train"),
        data.target("train", "binary"),
        data.features("val"),
        data.target("val", "binary"),
        labels,
        data.feature_columns,
        _small_params("random_forest"),
        config,
        seed=42,
        task="binary",
    )
    x_test = data.features("test")
    latency = measure_latency(model, x_test, calls=25, warmup_calls=2)
    assert latency["calls"] == 25
    assert latency["median_ms"] > 0
    assert latency["p95_ms"] >= latency["median_ms"]
    throughput = measure_batch_throughput(model, x_test, batch_size=32)
    assert throughput["rows"] == len(x_test)
    assert throughput["flows_per_sec"] > 0
    with pytest.raises(ValueError):
        measure_latency(model, x_test[:0], calls=1, warmup_calls=0)


# ---------------------------------------------------------------------------
# Tuning uses validation only
# ---------------------------------------------------------------------------


def test_tuning_objective_is_validation_macro_f1(tmp_path: Path) -> None:
    """Tuning scores the validation split, never the test split."""
    processed, models = _build_dataset(tmp_path, n_per_class=60, seed=7)
    config = _tiny_config()

    # Corrupt the test labels so a model that peeked at test would be scored
    # very differently from the validation macro-F1.
    for split in ("test",):
        frame = pd.read_parquet(processed / f"toy_{split}.parquet")
        frame["multiclass_label"] = "Normal"
        frame.to_parquet(processed / f"toy_{split}.parquet", index=False)

    data = load_processed_splits("toy", config, processed, models)
    labels = sorted(set(data.target("train", "multiclass")))
    result = tune_hyperparameters(
        data, "multiclass", "random_forest", labels, config, n_trials=1
    )
    assert result["objective"] == "val_macro_f1"

    rebuilt = fit_model(
        "random_forest",
        data.features("train"),
        data.target("train", "multiclass"),
        data.features("val"),
        data.target("val", "multiclass"),
        labels,
        data.feature_columns,
        result["best_params_resolved"],
        config,
        seed=config.seed,
        task="multiclass",
    )
    val_f1 = compute_metrics(
        data.target("val", "multiclass"),
        rebuilt.predict(data.features("val")),
        labels,
    )["macro_f1"]
    test_f1 = compute_metrics(
        data.target("test", "multiclass"),
        rebuilt.predict(data.features("test")),
        labels,
    )["macro_f1"]
    assert result["best_value"] == pytest.approx(val_f1, abs=1e-9)
    assert test_f1 != pytest.approx(val_f1, abs=1e-6)


def test_tuning_is_bounded_by_trials_and_budget(tmp_path: Path) -> None:
    """n_trials and the wall-clock budget both bound the search."""
    processed, models = _build_dataset(tmp_path, n_per_class=20)
    config = _tiny_config()
    data = load_processed_splits("toy", config, processed, models)
    labels = sorted(set(data.target("train", "binary")))
    result = tune_hyperparameters(
        data, "binary", "xgboost", labels, config, n_trials=2, time_budget_seconds=120
    )
    assert result["n_trials_requested"] == 2
    assert result["n_trials_completed"] >= 1
    assert result["time_budget_seconds"] == 120
    assert result["best_params_resolved"]["n_estimators"] >= 50


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------


def test_run_benchmark_writes_all_artifacts(tmp_path: Path) -> None:
    """The full benchmark produces JSON, markdown, figures and every model."""
    processed, models = _build_dataset(tmp_path, n_per_class=45)
    reports = tmp_path / "reports"
    figures = reports / "figures"
    config = _tiny_config()
    report = run_benchmark(
        "toy",
        config,
        tasks=["binary", "multiclass"],
        families=list(FAMILY_NAMES),
        seeds=[42],
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=figures,
    )

    assert (reports / "tier1_benchmark.json").is_file()
    assert (reports / "tier1_benchmark.md").is_file()
    assert len(report["runs"]) == 2 * len(FAMILY_NAMES) * 1
    assert len(report["aggregates"]) == 2 * len(FAMILY_NAMES)
    assert report["chosen_default"]["family"] in FAMILY_NAMES
    assert report["selection"]["rule"]["selection_split"] == "val"

    for run in report["runs"]:
        assert run["test"]["n_samples"] == run["n_test_rows"]
        assert run["model_size_bytes"] > 0
        assert run["latency"]["p95_ms"] >= run["latency"]["median_ms"] > 0
        assert run["batch_throughput"]["flows_per_sec"] > 0
        assert run["accuracy_per_ms"] == pytest.approx(
            run["test"]["accuracy"] / run["latency"]["median_ms"]
        )
        assert Path(PROJECT_ROOT, run["model_path"]).is_file() or Path(
            run["model_path"]
        ).is_file()

    for task in ("binary", "multiclass"):
        for family in FAMILY_NAMES:
            figure = figures / f"tier1_confusion_{task}_{family}.png"
            assert figure.is_file() and figure.stat().st_size > 0
    assert len(report["figures"]) == 2 * len(FAMILY_NAMES)
    for figure in report["figures"]:
        assert Path(figure).is_file(), figure

    markdown = (reports / "tier1_benchmark.md").read_text(encoding="utf-8")
    assert "| Task | Family |" in markdown
    assert "Selected default Tier-1 model" in markdown

    payload = json.loads((reports / "tier1_benchmark.json").read_text(encoding="utf-8"))
    assert payload["features"]["names"] == FEATURES
    assert payload["definitions"]["selection_split"] == "val"
    assert payload["environment"]["torch"]
    assert payload["splits"]["rows"]["train"] == len(
        load_processed_splits("toy", config, processed, models).frames["train"]
    )


def test_inference_threads_are_pinned_for_sklearn_families(tmp_path: Path) -> None:
    """Saved sklearn models serve with the configured inference thread count."""
    processed, models = _build_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    config = _tiny_config()
    report = run_benchmark(
        "toy",
        config,
        tasks=["binary"],
        families=["random_forest", "xgboost"],
        seeds=[42],
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=reports / "figures",
    )
    for run in report["runs"]:
        payload = load_tier1_model(Path(run["model_path"]))
        assert payload["estimator"].n_jobs == config.tier1.inference_n_jobs
    settings = report["benchmark_settings"]
    assert settings["inference_n_jobs"] == config.tier1.inference_n_jobs
    assert settings["train_n_jobs"] == config.tier1.n_jobs
    assert settings["n_trials"] == 1


def test_figures_are_written_once_per_task_and_family(tmp_path: Path) -> None:
    """Multi-seed runs keep one confusion-matrix figure per task/family."""
    processed, models = _build_dataset(tmp_path, n_per_class=30)
    reports = tmp_path / "reports"
    figures = reports / "figures"
    config = _tiny_config()
    report = run_benchmark(
        "toy",
        config,
        tasks=["binary"],
        families=["random_forest"],
        seeds=[42, 43],
        processed_dir=processed,
        models_dir=models,
        reports_dir=reports,
        figures_dir=figures,
    )
    assert len(report["runs"]) == 2
    assert report["aggregates"][0]["n_seeds"] == 2
    assert report["aggregates"][0]["std"]["test.accuracy"] >= 0.0
    assert len(list(figures.glob("tier1_confusion_binary_random_forest.png"))) == 1


def test_cli_exits_one_without_processed_data(tmp_path: Path) -> None:
    """The CLI stops instead of fabricating a benchmark."""
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
            str(tmp_path / "figures"),
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "reports" / "tier1_benchmark.json").exists()


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    """Unknown families fail loudly instead of being ignored."""
    processed, models = _build_dataset(tmp_path, n_per_class=12)
    with pytest.raises(ValueError, match="unknown families"):
        run_benchmark(
            "toy",
            _tiny_config(),
            families=["knn"],
            seeds=[42],
            processed_dir=processed,
            models_dir=models,
            reports_dir=tmp_path / "reports",
            figures_dir=tmp_path / "figures",
        )


# ---------------------------------------------------------------------------
# Config and Makefile contract
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_benchmark_spec() -> None:
    """Config carries three seeds, both tasks, three families and a rule."""
    config = load_config()
    assert config.tier1.seeds == [42, 43, 44]
    assert config.tier1.families == list(FAMILY_NAMES)
    assert config.tier1.tasks == ["binary", "multiclass"]
    assert config.tier1.optuna.n_trials >= 1
    assert config.tier1.optuna.time_budget_seconds > 0
    assert config.tier1.selection.metric in {"macro_f1", "accuracy", "weighted_f1"}
    assert config.tier1.benchmark.latency_calls == 1000
    assert len(config.tier1.mlp.hidden_layer_options) >= 1


def test_config_rejects_invalid_tier1_settings() -> None:
    """Bad MLP layer counts, duplicate seeds and empty lists are rejected."""
    base = load_config().tier1.model_dump()
    with pytest.raises(ValidationError):
        Tier1Config.model_validate(
            {**base, "mlp": {**base["mlp"], "hidden_layer_options": [[8]]}}
        )
    with pytest.raises(ValidationError):
        Tier1Config.model_validate({**base, "seeds": [42, 42]})
    with pytest.raises(ValidationError):
        Tier1Config.model_validate({**base, "families": []})
    with pytest.raises(ValidationError):
        Tier1Config.model_validate(
            {**base, "xgboost": {**base["xgboost"], "learning_rate_min": 0.5}}
        )


def test_makefile_exposes_the_tier1_target() -> None:
    """One `make` target reproduces the benchmark."""
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^tier1-benchmark\s*:", makefile, flags=re.MULTILINE)
    assert "src.tier1.train" in makefile
    assert re.search(r"^DATASET\s*\?=", makefile, flags=re.MULTILINE)
