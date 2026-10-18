"""Train and benchmark the Tier-1 candidates.

Protocol
--------
1. Load the leakage-safe parquet splits written by
   :mod:`src.data.preprocess` (``data/processed/<dataset>_{train,val,calibration,test}.parquet``)
   and the feature list from ``models/<dataset>_features.json``.
2. For every configured task (``binary`` = Normal/Attack, ``multiclass`` =
   unified 10-class taxonomy) and model family (Random Forest, XGBoost, a small
   feed-forward network), run a **bounded** Optuna search that fits on the
   training split and scores macro-F1 on the **validation** split only.
3. Only after every task/family has been tuned, refit the best configuration
   for each configured seed and evaluate the **test** split. Test labels never
   influence tuning, early stopping or model selection.
4. Report accuracy, macro/weighted precision/recall/F1, per-class metrics,
   confusion matrices, model size, training time, single-flow latency
   (median/p95 over ``benchmark.latency_calls`` calls), batch throughput and
   accuracy per millisecond.
5. Aggregate the seeds as mean +/- std, write
   ``reports/tier1_benchmark.json`` and ``reports/tier1_benchmark.md``, save the
   confusion-matrix figures and pick the default Tier-1 model with the rule in
   ``tier1.selection``.

Model selection uses the **validation** macro-F1 (with the measured p95 latency
constraint), never the test score: choosing the champion on test would turn the
test split into a tuning signal.

Class imbalance is handled with class weights only; no row is resampled.

Artifacts
---------
- ``models/tier1/<dataset>_<task>_<family>_seed<seed>.joblib|.pt``
- ``reports/tier1_benchmark.json``
- ``reports/tier1_benchmark.md``
- ``reports/figures/tier1_confusion_<task>_<family>.png``

Usage
-----
::

    python -m src.tier1.train --dataset cicids2017
    python -m src.tier1.train --dataset cicids2017 --n-trials 5 --time-budget 120

The runs stop with an actionable message (and exit code 1) when the processed
splits do not exist, rather than inventing data.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import PROJECT_ROOT, Config, load_config  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    BINARY_COLUMN,
    MULTICLASS_COLUMN,
    SPLIT_NAMES,
)

logger = logging.getLogger(__name__)

#: Target column per task name.
TASK_COLUMNS: dict[str, str] = {
    "binary": BINARY_COLUMN,
    "multiclass": MULTICLASS_COLUMN,
}
#: Supported model families.
FAMILY_NAMES: tuple[str, str, str] = ("random_forest", "xgboost", "mlp")
#: Task names.
TASK_NAMES: tuple[str, str] = ("binary", "multiclass")
#: Suffix of the serialised model per family.
MODEL_SUFFIXES: dict[str, str] = {
    "random_forest": ".joblib",
    "xgboost": ".joblib",
    "mlp": ".pt",
}
#: Scalar metrics aggregated across seeds (dotted paths into a run record).
SCALAR_PATHS: tuple[str, ...] = (
    "val.accuracy",
    "val.macro_f1",
    "val.weighted_f1",
    "test.accuracy",
    "test.macro_precision",
    "test.macro_recall",
    "test.macro_f1",
    "test.weighted_precision",
    "test.weighted_recall",
    "test.weighted_f1",
    "training_time_seconds",
    "model_size_bytes",
    "latency.median_ms",
    "latency.p95_ms",
    "batch_throughput.flows_per_sec",
    "accuracy_per_ms",
)


def _git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"`` outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _display_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return ``path`` relative to the project root when it lives inside it.

    Keeps report entries portable without hard-coding directory names, which
    the config can move.
    """
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _get_path(record: dict[str, Any], dotted: str) -> float:
    """Read a nested numeric value from ``record`` using a dotted key."""
    value: Any = record
    for part in dotted.split("."):
        value = value[part]
    return float(value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    """Processed splits plus the ordered model feature list.

    Attributes:
        dataset: dataset name (``cicids2017`` or ``ciciot2023``).
        frames: mapping split name -> DataFrame with ``flow_id``, features and
            the label columns.
        feature_columns: ordered feature names from the fitted preprocessor.
    """

    dataset: str
    frames: dict[str, pd.DataFrame]
    feature_columns: list[str]

    def features(self, split: str) -> np.ndarray:
        """Return the feature matrix of ``split`` as float32 ndarray."""
        frame = self.frames[split]
        return np.ascontiguousarray(
            frame[self.feature_columns].to_numpy(dtype=np.float32)
        )

    def target(self, split: str, task: str) -> np.ndarray:
        """Return the string labels of ``split`` for ``task``."""
        column = TASK_COLUMNS[task]
        if column not in self.frames[split].columns:
            raise ValueError(
                f"task {task!r} needs column {column!r}, which is missing from "
                f"the {split} split; rerun preprocessing with the current "
                "label map"
            )
        return self.frames[split][column].astype(str).to_numpy()


def load_processed_splits(
    dataset: str,
    config: Config,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> SplitData:
    """Load the processed splits and fitted feature list for one dataset.

    Args:
        dataset: dataset name, used in the artifact file names.
        config: project config providing default directories.
        processed_dir: override for ``data/processed`` (tests).
        models_dir: override for ``models`` (tests).

    Returns:
        A :class:`SplitData` with all four splits loaded.

    Raises:
        FileNotFoundError: If the feature list or any split parquet is missing,
            with the command that produces it.
        ValueError: If splits share a ``flow_id`` or a feature is missing.
    """
    root = PROJECT_ROOT
    processed = processed_dir or (root / config.paths.processed)
    models = models_dir or (root / config.paths.models)

    features_path = models / f"{dataset}_features.json"
    if not features_path.is_file():
        raise FileNotFoundError(
            f"feature list not found: {features_path}\n"
            f"Run `make preprocess-{dataset}` (or "
            f"`python -m src.data.preprocess --dataset {dataset}`) first."
        )

    missing = [
        processed / f"{dataset}_{split}.parquet"
        for split in SPLIT_NAMES
        if not (processed / f"{dataset}_{split}.parquet").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "processed splits not found:\n  "
            + "\n  ".join(str(path) for path in missing)
            + f"\nRun `make preprocess-{dataset}` first."
        )

    payload = json.loads(features_path.read_text(encoding="utf-8"))
    feature_columns = list(payload["feature_columns"])
    frames = {
        split: pd.read_parquet(processed / f"{dataset}_{split}.parquet")
        for split in SPLIT_NAMES
    }

    for split, frame in frames.items():
        missing_features = [c for c in feature_columns if c not in frame.columns]
        if missing_features:
            raise ValueError(
                f"{split} split is missing feature columns: {missing_features}"
            )

    seen: dict[str, str] = {}
    for split, frame in frames.items():
        for flow_id in frame["flow_id"].astype(str):
            if flow_id in seen:
                raise ValueError(
                    f"flow_id {flow_id!r} appears in both {seen[flow_id]} and "
                    f"{split}; the splits are not disjoint"
                )
            seen[flow_id] = split

    logger.info(
        "Loaded %s splits: %s",
        dataset,
        {name: len(frame) for name, frame in frames.items()},
    )
    return SplitData(dataset=dataset, frames=frames, feature_columns=feature_columns)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TorchMLP(nn.Module):
    """Small feed-forward classifier: 2-3 ReLU hidden layers plus dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        n_classes: int,
        dropout: float,
    ) -> None:
        """Build the network.

        Args:
            input_dim: number of input features.
            hidden_layers: widths of the hidden layers (2 or 3 entries).
            n_classes: number of output classes.
            dropout: dropout probability after every hidden layer.
        """
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                [nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous = int(width)
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits for ``features``."""
        return self.net(features)


def _as_array(features: np.ndarray | pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Convert features to a contiguous float32 2-D array in column order."""
    if isinstance(features, pd.DataFrame):
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise ValueError(f"feature columns missing: {missing}")
        array = features[list(columns)].to_numpy(dtype=np.float32)
    else:
        array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {array.shape}")
    return np.ascontiguousarray(array)


@dataclass
class TrainedModel:
    """A fitted Tier-1 model with string-label predictions.

    Attributes:
        family: ``random_forest``, ``xgboost`` or ``mlp``.
        task: ``binary`` or ``multiclass``.
        seed: seed used for this fit.
        labels: ordered class labels; column ``i`` of ``predict_proba``
            corresponds to ``labels[i]``.
        feature_columns: ordered model features.
        estimator: sklearn estimator or :class:`TorchMLP`.
        params: hyper-parameters used for the fit.
        training_time_seconds: wall-clock fit time.
        extra: family-specific training diagnostics (e.g. MLP early stopping).
    """

    family: str
    task: str
    seed: int
    labels: list[str]
    feature_columns: list[str]
    estimator: Any
    params: dict[str, Any]
    training_time_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return class probabilities in ``labels`` order."""
        array = _as_array(features, self.feature_columns)
        if self.family == "mlp":
            self.estimator.eval()
            with torch.no_grad():
                logits = self.estimator(torch.from_numpy(array))
                return torch.softmax(logits, dim=1).numpy()
        return np.asarray(self.estimator.predict_proba(array), dtype=float)

    def predict(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predicted string labels."""
        probabilities = self.predict_proba(features)
        index = np.argmax(probabilities, axis=1)
        return np.asarray(self.labels, dtype=object)[index]


def _class_weights(
    y_index: np.ndarray, n_classes: int, enabled: bool
) -> np.ndarray | None:
    """Return balanced class weights computed from the given labels only."""
    if not enabled:
        return None
    counts = np.bincount(y_index, minlength=n_classes).astype(float)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=float)
    weights[present] = len(y_index) / (present.sum() * counts[present])
    return weights


def _encode(
    labels: list[str], y: np.ndarray, allow_unseen: bool = False
) -> np.ndarray:
    """Map string labels to their index in the sorted ``labels`` list.

    Args:
        labels: ordered training labels.
        y: string labels to encode.
        allow_unseen: when True, labels absent from ``labels`` are mapped to
            ``-1`` instead of raising. Used for validation labels during MLP
            early stopping; the final metrics always use the string labels so
            unseen rows are counted as errors.

    Returns:
        Integer label indices.

    Raises:
        ValueError: If a label is unknown and ``allow_unseen`` is False.
    """
    index = {label: position for position, label in enumerate(labels)}
    encoded: list[int] = []
    for value in y:
        position = index.get(str(value), -1)
        if position < 0 and not allow_unseen:
            raise ValueError(
                f"label {value!r} is not in the training labels {labels}"
            )
        encoded.append(position)
    return np.asarray(encoded, dtype=np.int64)


def _fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    params: dict[str, Any],
    config: Config,
    seed: int,
    class_weighting: bool,
) -> TrainedModel:
    """Fit a Random Forest on the training split."""
    weights = _class_weights(y_train, n_classes, class_weighting)
    class_weight = (
        {index: float(weight) for index, weight in enumerate(weights)}
        if weights is not None
        else None
    )
    started = time.perf_counter()
    estimator = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=class_weight,
        random_state=seed,
        n_jobs=config.tier1.n_jobs,
    )
    estimator.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return TrainedModel(
        family="random_forest",
        task="",
        seed=seed,
        labels=[],
        feature_columns=[],
        estimator=estimator,
        params=params,
        training_time_seconds=elapsed,
    )


def _fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    params: dict[str, Any],
    config: Config,
    seed: int,
    class_weighting: bool,
) -> TrainedModel:
    """Fit an XGBoost classifier on the training split."""
    weights = _class_weights(y_train, n_classes, class_weighting)
    sample_weight = weights[y_train] if weights is not None else None
    started = time.perf_counter()
    estimator = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        gamma=float(params["gamma"]),
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        tree_method="hist",
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        random_state=seed,
        n_jobs=config.tier1.n_jobs,
        verbosity=0,
    )
    estimator.fit(x_train, y_train, sample_weight=sample_weight)
    elapsed = time.perf_counter() - started
    return TrainedModel(
        family="xgboost",
        task="",
        seed=seed,
        labels=[],
        feature_columns=[],
        estimator=estimator,
        params=params,
        training_time_seconds=elapsed,
    )


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    params: dict[str, Any],
    config: Config,
    seed: int,
    class_weighting: bool,
) -> TrainedModel:
    """Fit the feed-forward network with early stopping on validation macro-F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_layers = [int(width) for width in params["hidden_layers"]]
    model = TorchMLP(
        input_dim=x_train.shape[1],
        hidden_layers=hidden_layers,
        n_classes=n_classes,
        dropout=float(params["dropout"]),
    )
    weights = _class_weights(y_train, n_classes, class_weighting)
    loss_weight = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    batch_size = int(params["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    x_val_tensor = torch.from_numpy(x_val)
    best_state: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0
    started = time.perf_counter()
    for epoch in range(1, config.tier1.mlp.epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(x_val_tensor).argmax(dim=1).numpy()
        macro_f1 = float(
            f1_score(
                y_val,
                predictions,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        )
        epochs_run = epoch
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.tier1.mlp.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return TrainedModel(
        family="mlp",
        task="",
        seed=seed,
        labels=[],
        feature_columns=[],
        estimator=model,
        params=params,
        training_time_seconds=elapsed,
        extra={
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
        },
    )


def fit_model(
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: list[str],
    feature_columns: list[str],
    params: dict[str, Any],
    config: Config,
    seed: int,
    task: str = "",
    class_weighting: bool = True,
) -> TrainedModel:
    """Fit one model family with the given hyper-parameters.

    Args:
        family: ``random_forest``, ``xgboost`` or ``mlp``.
        x_train: training features (already scaled by the fitted preprocessor).
        y_train: training labels (strings).
        x_val: validation features; used only for MLP early stopping.
        y_val: validation labels (strings).
        labels: ordered class labels taken from the training split.
        feature_columns: ordered model feature names.
        params: hyper-parameters for the family.
        config: project config (n_jobs, MLP epochs/patience).
        seed: seed for this fit.
        task: task name recorded on the model.
        class_weighting: whether balanced class weights are applied.

    Returns:
        A :class:`TrainedModel` with the fitted estimator and fit time.

    Raises:
        ValueError: If ``family`` is unknown.
    """
    y_train_index = _encode(labels, y_train)
    y_val_index = _encode(labels, y_val, allow_unseen=True)
    if family == "random_forest":
        model = _fit_random_forest(
            x_train, y_train_index, len(labels), params, config, seed, class_weighting
        )
    elif family == "xgboost":
        model = _fit_xgboost(
            x_train, y_train_index, len(labels), params, config, seed, class_weighting
        )
    elif family == "mlp":
        model = _fit_mlp(
            x_train,
            y_train_index,
            x_val,
            y_val_index,
            len(labels),
            params,
            config,
            seed,
            class_weighting,
        )
    else:  # pragma: no cover - validated by the config
        raise ValueError(f"unknown model family: {family}")

    model.task = task
    model.labels = labels
    model.feature_columns = list(feature_columns)
    return model


# ---------------------------------------------------------------------------
# Hyper-parameter search (validation only)
# ---------------------------------------------------------------------------


def suggest_params(
    trial: optuna.Trial, family: str, config: Config
) -> dict[str, Any]:
    """Sample one hyper-parameter configuration for ``family``.

    Args:
        trial: active Optuna trial.
        family: model family being tuned.
        config: project config holding the search bounds.

    Returns:
        A JSON-serialisable parameter dictionary.
    """
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, config.tier1.random_forest.n_estimators_max, step=50
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, config.tier1.random_forest.max_depth_max
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                config.tier1.random_forest.min_samples_leaf_max,
                log=True,
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
        }
    if family == "xgboost":
        xgb = config.tier1.xgboost
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, xgb.n_estimators_max, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 2, xgb.max_depth_max),
            "learning_rate": trial.suggest_float(
                "learning_rate", xgb.learning_rate_min, xgb.learning_rate_max, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 10.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
    if family == "mlp":
        options = config.tier1.mlp.hidden_layer_options
        keys = ["x".join(str(width) for width in option) for option in options]
        key = trial.suggest_categorical("hidden_layers", keys)
        return {
            "hidden_layers": [int(part) for part in key.split("x")],
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512]
            ),
        }
    raise ValueError(f"unknown model family: {family}")


def tune_hyperparameters(
    split_data: SplitData,
    task: str,
    family: str,
    labels: list[str],
    config: Config,
    n_trials: int | None = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the bounded Optuna search using train (fit) and val (objective).

    The test split is never passed to this function.

    Args:
        split_data: loaded splits.
        task: ``binary`` or ``multiclass``.
        family: model family.
        labels: ordered training labels.
        config: project config.
        n_trials: override for ``tier1.optuna.n_trials``.
        time_budget_seconds: override for the wall-clock budget.

    Returns:
        Dictionary with best parameters, best validation macro-F1, completed
        trial count and the budget used.
    """
    x_train = split_data.features("train")
    y_train = split_data.target("train", task)
    x_val = split_data.features("val")
    y_val = split_data.target("val", task)
    trials = n_trials if n_trials is not None else config.tier1.optuna.n_trials
    budget = (
        time_budget_seconds
        if time_budget_seconds is not None
        else config.tier1.optuna.time_budget_seconds
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, family, config)
        model = fit_model(
            family,
            x_train,
            y_train,
            x_val,
            y_val,
            labels,
            split_data.feature_columns,
            params,
            config,
            seed=config.seed,
            task=task,
        )
        predictions = model.predict(x_val)
        return float(
            f1_score(
                y_val,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name=f"{split_data.dataset}-{task}-{family}",
    )
    study.optimize(objective, n_trials=trials, timeout=budget, gc_after_trial=True)
    return {
        "task": task,
        "family": family,
        "objective": "val_macro_f1",
        "n_trials_requested": trials,
        "n_trials_completed": len(study.trials),
        "time_budget_seconds": budget,
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "best_params_resolved": suggest_resolved(study.best_params, family),
    }


def suggest_resolved(params: dict[str, Any], family: str) -> dict[str, Any]:
    """Convert Optuna's categorical encoding back to model parameters.

    Args:
        params: ``study.best_params`` value.
        family: model family.

    Returns:
        Parameter dictionary accepted by :func:`fit_model`.
    """
    resolved = dict(params)
    if family == "mlp" and "hidden_layers" in resolved:
        resolved["hidden_layers"] = [
            int(part) for part in str(resolved["hidden_layers"]).split("x")
        ]
    return resolved


# ---------------------------------------------------------------------------
# Metrics and timing
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted scores, per-class report and matrix.

    Args:
        y_true: true string labels.
        y_pred: predicted string labels.
        labels: label order used for the per-class report and matrix.

    Returns:
        JSON-serialisable metric dictionary.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class: dict[str, dict[str, float]] = {}
    for position, label in enumerate(labels):
        true_positive = float(matrix[position, position])
        support = float(matrix[position, :].sum())
        predicted = float(matrix[:, position].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "matrix": matrix.tolist(),
        },
        "n_samples": int(len(y_true)),
    }


def measure_latency(
    model: TrainedModel,
    x_rows: np.ndarray,
    calls: int,
    warmup_calls: int,
) -> dict[str, Any]:
    """Measure single-flow inference latency in milliseconds.

    Args:
        model: fitted model.
        x_rows: rows to cycle through for the calls.
        calls: number of timed single-row predictions.
        warmup_calls: untimed predictions before measuring.

    Returns:
        Dictionary with ``calls``, ``median_ms``, ``p95_ms`` and ``mean_ms``.

    Raises:
        ValueError: If ``x_rows`` is empty.
    """
    if len(x_rows) == 0:
        raise ValueError("cannot measure latency on an empty feature matrix")
    for index in range(warmup_calls):
        row = x_rows[index % len(x_rows) : index % len(x_rows) + 1]
        model.predict_proba(row)
    timings = np.empty(calls, dtype=float)
    for index in range(calls):
        position = index % len(x_rows)
        row = x_rows[position : position + 1]
        started = time.perf_counter()
        model.predict_proba(row)
        timings[index] = (time.perf_counter() - started) * 1000.0
    return {
        "calls": calls,
        "warmup_calls": warmup_calls,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(timings.mean()),
    }


def measure_batch_throughput(
    model: TrainedModel, x_rows: np.ndarray, batch_size: int
) -> dict[str, Any]:
    """Measure batch inference throughput in flows per second.

    Args:
        model: fitted model.
        x_rows: feature rows to score (already capped by the caller).
        batch_size: chunk size.

    Returns:
        Dictionary with ``rows``, ``batch_size``, ``seconds`` and
        ``flows_per_sec``.
    """
    started = time.perf_counter()
    for start in range(0, len(x_rows), batch_size):
        model.predict_proba(x_rows[start : start + batch_size])
    elapsed = time.perf_counter() - started
    return {
        "rows": int(len(x_rows)),
        "batch_size": int(batch_size),
        "seconds": float(elapsed),
        "flows_per_sec": float(len(x_rows) / elapsed) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: TrainedModel, path: Path) -> int:
    """Serialise ``model`` to ``path`` and return the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.family == "mlp":
        torch.save(
            {
                "state_dict": model.estimator.state_dict(),
                "input_dim": int(model.estimator.net[0].in_features),
                "hidden_layers": [
                    int(layer.out_features)
                    for layer in model.estimator.net
                    if isinstance(layer, nn.Linear)
                ][:-1],
                "dropout": float(model.params["dropout"]),
                "classes": model.labels,
                "feature_columns": model.feature_columns,
                "task": model.task,
                "seed": model.seed,
                "params": model.params,
            },
            path,
        )
    else:
        joblib.dump(
            {
                "estimator": model.estimator,
                "family": model.family,
                "task": model.task,
                "labels": model.labels,
                "feature_columns": model.feature_columns,
                "seed": model.seed,
                "params": model.params,
            },
            path,
        )
    return int(path.stat().st_size)


def pin_inference_threads(model: TrainedModel, n_jobs: int) -> None:
    """Pin the sklearn-family estimators to a fixed inference thread count.

    ``RandomForestClassifier`` and the XGBoost wrapper parallelise *every*
    ``predict_proba`` call, so scoring a single flow with ``n_jobs=-1`` spends
    almost all of its time in worker start-up. Training keeps
    ``tier1.n_jobs``; serving (and therefore the reported latency and
    throughput) uses ``tier1.inference_n_jobs``.

    Args:
        model: fitted model, modified in place.
        n_jobs: thread count to use for inference.
    """
    if model.family == "mlp":
        return
    if hasattr(model.estimator, "n_jobs"):
        model.estimator.n_jobs = int(n_jobs)


def load_tier1_model(path: Path) -> dict[str, Any]:
    """Load a saved Tier-1 artifact.

    Args:
        path: ``.joblib`` (sklearn) or ``.pt`` (MLP) artifact path.

    Returns:
        For ``.joblib`` the saved payload; for ``.pt`` a dictionary with a
        rebuilt :class:`TorchMLP` under ``"model"`` plus its metadata.

    Raises:
        ValueError: If the artifact suffix is not supported.
    """
    if path.suffix == ".joblib":
        return joblib.load(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        n_classes = len(payload["classes"])
        network = TorchMLP(
            input_dim=int(payload["input_dim"]),
            hidden_layers=payload["hidden_layers"],
            n_classes=n_classes,
            dropout=float(payload["dropout"]),
        )
        network.load_state_dict(payload["state_dict"])
        network.eval()
        return {**payload, "model": network}
    raise ValueError(f"unsupported model artifact: {path}")


def resolve_report_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path stored in a report against the project root.

    Args:
        value: absolute path, or a path relative to the project root.
        root: project root used for relative values.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    path = Path(value)
    return path if path.is_absolute() else root / path


def trained_model_from_artifact(
    path: Path, task: str, family: str, seed: int
) -> TrainedModel:
    """Rebuild a :class:`TrainedModel` from a saved Tier-1 artifact.

    Args:
        path: ``.joblib`` (sklearn) or ``.pt`` (MLP) artifact written by
            :func:`save_model`.
        task: task name recorded on the model.
        family: model family.
        seed: seed recorded for the artifact.

    Returns:
        A :class:`TrainedModel` ready for prediction.

    Raises:
        ValueError: If the artifact does not contain the expected fields.
    """
    payload = load_tier1_model(path)
    if path.suffix == ".pt":
        estimator = payload["model"]
        labels = payload["classes"]
    else:
        estimator = payload["estimator"]
        labels = payload["labels"]
    feature_columns = payload.get("feature_columns")
    if not feature_columns or not labels:
        raise ValueError(f"artifact {path} is missing feature columns or labels")
    return TrainedModel(
        family=family,
        task=task,
        seed=int(seed),
        labels=list(labels),
        feature_columns=list(feature_columns),
        estimator=estimator,
        params=dict(payload.get("params", {})),
        training_time_seconds=0.0,
    )


def read_benchmark_runs(reports_dir: Path, dataset: str) -> list[dict[str, Any]]:
    """Read the run records written by :func:`run_benchmark`.

    Args:
        reports_dir: directory holding ``tier1_benchmark.json``.
        dataset: dataset that the report must belong to.

    Returns:
        The ``runs`` list from the benchmark report.

    Raises:
        FileNotFoundError: If the benchmark report does not exist.
        ValueError: If the report was produced for a different dataset.
    """
    path = reports_dir / "tier1_benchmark.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark report not found: {path}\n"
            f"Run `make tier1-benchmark DATASET={dataset}` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"benchmark report {path} is for dataset "
            f"{payload.get('dataset')!r}, not {dataset!r}"
        )
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"benchmark report {path} contains no runs")
    return list(runs)


# ---------------------------------------------------------------------------
# Aggregation and selection
# ---------------------------------------------------------------------------


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate seed-level runs into mean and standard deviation.

    Args:
        runs: run records sharing one ``(task, family)``.

    Returns:
        Dictionary with ``n_seeds``, per-metric ``mean`` and ``std`` maps.

    Raises:
        ValueError: If ``runs`` is empty or mixes tasks/families.
    """
    if not runs:
        raise ValueError("cannot aggregate an empty list of runs")
    tasks = {run["task"] for run in runs}
    families = {run["family"] for run in runs}
    if len(tasks) != 1 or len(families) != 1:
        raise ValueError("runs must share one task and one family")
    values = {
        path: np.asarray([_get_path(run, path) for run in runs], dtype=float)
        for path in SCALAR_PATHS
    }
    return {
        "task": next(iter(tasks)),
        "family": next(iter(families)),
        "n_seeds": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "mean": {path: float(array.mean()) for path, array in values.items()},
        "std": {
            path: float(array.std(ddof=1)) if len(array) > 1 else 0.0
            for path, array in values.items()
        },
    }


def select_default_model(
    aggregates: list[dict[str, Any]], config: Config
) -> dict[str, Any]:
    """Pick the champion with the configured latency-constrained rule.

    Selection uses the **validation** score, never the test score. Latency is
    a hardware property measured on the test rows but carries no label
    information.

    Args:
        aggregates: per ``(task, family)`` aggregates.
        config: project config with ``tier1.selection``.

    Returns:
        Dictionary describing the rule, eligible candidates and the choice.

    Raises:
        ValueError: If ``aggregates`` is empty.
    """
    if not aggregates:
        raise ValueError("no benchmark aggregates to select from")
    rule = config.tier1.selection
    metric_paths = {
        "accuracy": "val.accuracy",
        "macro_f1": "val.macro_f1",
        "weighted_f1": "val.weighted_f1",
    }
    metric_path = metric_paths[rule.metric]

    def score(record: dict[str, Any]) -> float:
        return float(record["mean"][metric_path])

    def latency(record: dict[str, Any]) -> float:
        return float(record["mean"]["latency.p95_ms"])

    eligible = [record for record in aggregates if latency(record) <= rule.max_p95_latency_ms]
    fallback = False
    if not eligible:
        fallback = True
        winner = min(aggregates, key=latency)
        reason = (
            "no candidate met the p95 latency budget; chose the lowest p95 "
            "latency instead"
        )
    else:
        def sort_key(record: dict[str, Any]) -> tuple[float, float]:
            if rule.tie_breaker == "p95_latency_ms":
                tie = -latency(record)
            else:
                tie = float(record["mean"][f"val.{rule.tie_breaker}"])
            return (score(record), tie)

        winner = max(eligible, key=sort_key)
        reason = (
            f"highest mean {metric_path} subject to mean p95 latency <= "
            f"{rule.max_p95_latency_ms} ms"
        )
    return {
        "rule": {
            "metric": metric_path,
            "max_p95_latency_ms": rule.max_p95_latency_ms,
            "tie_breaker": rule.tie_breaker,
            "selection_split": "val",
        },
        "fallback_used": fallback,
        "eligible": [
            {
                "task": record["task"],
                "family": record["family"],
                "val_metric": score(record),
                "p95_latency_ms": latency(record),
            }
            for record in eligible
        ],
        "chosen": {
            "task": winner["task"],
            "family": winner["family"],
            "val_metric": score(winner),
            "p95_latency_ms": latency(winner),
            "test_macro_f1_mean": float(winner["mean"]["test.macro_f1"]),
        },
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    metrics: dict[str, Any],
    task: str,
    family: str,
    seed: int,
    path: Path,
) -> None:
    """Save a confusion-matrix heatmap for one run."""
    payload = metrics["confusion_matrix"]
    labels = payload["labels"]
    matrix = np.asarray(payload["matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.7), max(5, len(labels) * 0.6)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{task} — {family} (seed {seed})")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format_mean_std(aggregate: dict[str, Any], path: str, digits: int = 4) -> str:
    """Format ``mean ± std`` for a metric path."""
    mean = aggregate["mean"][path]
    std = aggregate["std"][path]
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    """Write the human-readable benchmark comparison table.

    Args:
        report: benchmark report dictionary.
        path: destination markdown file.
    """
    settings = report["benchmark_settings"]
    lines: list[str] = [
        "# Tier-1 benchmark",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Seeds: {len(settings['seeds'])} "
        f"({', '.join(str(s) for s in settings['seeds'])})",
        f"- Tasks: {', '.join(settings['tasks'])}",
        f"- Families: {', '.join(settings['families'])}",
        f"- Features: {report['features']['n_features']}",
        f"- Tuning: Optuna, objective `val_macro_f1`, "
        f"{settings['n_trials']} trials and "
        f"{settings['time_budget_seconds']} s budget "
        "per (task, family), train fit / validation score",
        f"- Inference threads: {settings['inference_n_jobs']} "
        f"(training used n_jobs={settings['train_n_jobs']})",
        f"- Latency protocol: {settings['latency_calls']} single-flow calls after "
        f"{settings['warmup_calls']} warm-up calls; batch size "
        f"{settings['batch_size']} over at most "
        f"{settings['throughput_max_rows']} test rows",
        "- Test split: evaluated once per seed after every task/family was "
        "tuned; never used for tuning or selection",
        "- `accuracy_per_ms` = test accuracy divided by median single-flow "
        "latency (ms)",
        "",
        "Values are mean ± std across seeds.",
        "",
        "| Task | Family | Test accuracy | Macro P | Macro R | Macro F1 | "
        "Weighted F1 | Val macro F1 | Size (bytes) | Train (s) | Latency med (ms) | "
        "p95 (ms) | Throughput (flows/s) | Acc/ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ]
    for aggregate in report["aggregates"]:
        lines.append(
            "| {task} | {family} | {acc} | {mp} | {mr} | {mf1} | {wf1} | {vmf1} "
            "| {size} | {train} | {lat} | {p95} | {thr} | {apms} |".format(
                task=aggregate["task"],
                family=aggregate["family"],
                acc=_format_mean_std(aggregate, "test.accuracy"),
                mp=_format_mean_std(aggregate, "test.macro_precision"),
                mr=_format_mean_std(aggregate, "test.macro_recall"),
                mf1=_format_mean_std(aggregate, "test.macro_f1"),
                wf1=_format_mean_std(aggregate, "test.weighted_f1"),
                vmf1=_format_mean_std(aggregate, "val.macro_f1"),
                size=_format_mean_std(aggregate, "model_size_bytes", 0),
                train=_format_mean_std(aggregate, "training_time_seconds", 2),
                lat=_format_mean_std(aggregate, "latency.median_ms", 3),
                p95=_format_mean_std(aggregate, "latency.p95_ms", 3),
                thr=_format_mean_std(aggregate, "batch_throughput.flows_per_sec", 0),
                apms=_format_mean_std(aggregate, "accuracy_per_ms", 4),
            )
        )
    choice = report["selection"]["chosen"]
    lines += [
        "",
        "## Selected default Tier-1 model",
        "",
        f"- Rule: {report['selection']['reason']}",
        f"- Selected: **{choice['family']} / {choice['task']}**",
        f"- Validation metric: {choice['val_metric']:.4f}",
        f"- Mean p95 latency: {choice['p95_latency_ms']:.3f} ms",
        f"- Mean test macro-F1 (reported, not used for selection): "
        f"{choice['test_macro_f1_mean']:.4f}",
        f"- Latency-budget fallback used: {report['selection']['fallback_used']}",
        "",
        "Confusion-matrix figures: `reports/figures/tier1_confusion_<task>_<family>.png`.",
        "",
        "> Numbers in this file are written only by an actual benchmark run. If "
        "the dataset is absent, the command exits without writing this file.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _environment() -> dict[str, str]:
    """Return versions needed to reproduce measured timings."""
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
        "torch": torch.__version__,
        "optuna": optuna.__version__,
    }


def run_benchmark(
    dataset: str,
    config: Config,
    tasks: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_trials: int | None = None,
    time_budget_seconds: float | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Tune, fit, benchmark and report every configured Tier-1 candidate.

    Args:
        dataset: dataset name, used for input and output file names.
        config: project config.
        tasks: task override (defaults to ``tier1.tasks``).
        families: family override (defaults to ``tier1.families``).
        seeds: seed override (defaults to ``tier1.seeds``).
        n_trials: Optuna trial override.
        time_budget_seconds: Optuna wall-clock override.
        processed_dir: override for ``data/processed`` (tests).
        models_dir: override for ``models`` (tests).
        reports_dir: override for ``reports`` (tests).
        figures_dir: override for ``reports/figures`` (tests).

    Returns:
        The benchmark report that was written to
        ``reports/tier1_benchmark.json``.

    Raises:
        FileNotFoundError: If processed splits or the feature list are missing.
        ValueError: If an unknown task or family is requested.
    """
    task_list = list(tasks) if tasks is not None else list(config.tier1.tasks)
    family_list = list(families) if families is not None else list(config.tier1.families)
    seed_list = [int(seed) for seed in (seeds if seeds is not None else config.tier1.seeds)]
    if not task_list or not family_list:
        raise ValueError("at least one task and one family are required")
    unknown_tasks = [task for task in task_list if task not in TASK_COLUMNS]
    unknown_families = [family for family in family_list if family not in FAMILY_NAMES]
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")
    if unknown_families:
        raise ValueError(f"unknown families: {unknown_families}")
    if not seed_list:
        raise ValueError("at least one seed is required")

    root = PROJECT_ROOT
    processed = processed_dir or (root / config.paths.processed)
    models = models_dir or (root / config.paths.models)
    reports = reports_dir or (root / config.paths.reports)
    figures = figures_dir or (root / config.paths.figures)
    model_dir = models / "tier1"
    for directory in (reports, figures, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_data = load_processed_splits(
        dataset, config, processed_dir=processed, models_dir=models
    )
    torch.set_num_threads(config.tier1.benchmark.torch_threads)

    labels_by_task: dict[str, list[str]] = {}
    unseen_by_task: dict[str, dict[str, int]] = {}
    for task in task_list:
        train_labels = sorted(set(split_data.target("train", task).tolist()))
        labels_by_task[task] = train_labels
        unseen_by_task[task] = {
            split: int(
                np.sum(
                    ~np.isin(
                        split_data.target(split, task),
                        np.asarray(train_labels, dtype=object),
                    )
                )
            )
            for split in SPLIT_NAMES
        }

    # Phase A: tuning. Only train/val are touched here.
    trials_used = n_trials if n_trials is not None else config.tier1.optuna.n_trials
    budget_used = (
        time_budget_seconds
        if time_budget_seconds is not None
        else config.tier1.optuna.time_budget_seconds
    )

    tuning: list[dict[str, Any]] = []
    for task in task_list:
        for family in family_list:
            logger.info("Tuning %s / %s ...", task, family)
            result = tune_hyperparameters(
                split_data,
                task,
                family,
                labels_by_task[task],
                config,
                n_trials=n_trials,
                time_budget_seconds=time_budget_seconds,
            )
            tuning.append(result)
            logger.info(
                "  best val macro-F1 %.4f after %d trials",
                result["best_value"],
                result["n_trials_completed"],
            )

    # Phase B: final fits and the single test evaluation per seed.
    runs: list[dict[str, Any]] = []
    tuned = {(entry["task"], entry["family"]): entry for entry in tuning}
    for task in task_list:
        labels = labels_by_task[task]
        x_train = split_data.features("train")
        y_train = split_data.target("train", task)
        x_val = split_data.features("val")
        y_val = split_data.target("val", task)
        x_test = split_data.features("test")
        y_test = split_data.target("test", task)
        for family in family_list:
            params = tuned[(task, family)]["best_params_resolved"]
            for seed in seed_list:
                logger.info("Final fit %s / %s / seed %d", task, family, seed)
                model = fit_model(
                    family,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    labels,
                    split_data.feature_columns,
                    params,
                    config,
                    seed=seed,
                    task=task,
                )
                pin_inference_threads(model, config.tier1.inference_n_jobs)
                val_metrics = compute_metrics(y_val, model.predict(x_val), labels)
                test_predictions = model.predict(x_test)
                test_metrics = compute_metrics(y_test, test_predictions, labels)
                latency = measure_latency(
                    model,
                    x_test,
                    calls=config.tier1.benchmark.latency_calls,
                    warmup_calls=config.tier1.benchmark.warmup_calls,
                )
                throughput_rows = x_test[
                    : config.tier1.benchmark.throughput_max_rows
                ]
                throughput = measure_batch_throughput(
                    model, throughput_rows, config.tier1.benchmark.batch_size
                )
                model_path = model_dir / (
                    f"{dataset}_{task}_{family}_seed{seed}{MODEL_SUFFIXES[family]}"
                )
                model_size = save_model(model, model_path)
                figure_path = figures / f"tier1_confusion_{task}_{family}.png"
                if seed == seed_list[0]:
                    plot_confusion_matrix(test_metrics, task, family, seed, figure_path)
                record = {
                    "task": task,
                    "family": family,
                    "seed": seed,
                    "params": params,
                    "n_train_rows": int(len(y_train)),
                    "n_val_rows": int(len(y_val)),
                    "n_test_rows": int(len(y_test)),
                    "unseen_labels_in_split": unseen_by_task[task],
                    "val": val_metrics,
                    "test": test_metrics,
                    "training_time_seconds": float(model.training_time_seconds),
                    "model_path": _display_path(model_path, root),
                    "model_size_bytes": model_size,
                    "latency": latency,
                    "batch_throughput": throughput,
                    "accuracy_per_ms": float(
                        test_metrics["accuracy"] / latency["median_ms"]
                    )
                    if latency["median_ms"] > 0
                    else 0.0,
                    "tuning_best_val_macro_f1": float(
                        tuned[(task, family)]["best_value"]
                    ),
                    "extra": model.extra,
                }
                runs.append(record)

    aggregates = [
        aggregate_runs([run for run in runs if run["task"] == task and run["family"] == family])
        for task in task_list
        for family in family_list
    ]
    selection = select_default_model(aggregates, config)

    class_counts = {
        split: {
            str(label): int(count)
            for label, count in sorted(
                frame[MULTICLASS_COLUMN].astype(str).value_counts().items()
            )
        }
        for split, frame in split_data.frames.items()
    }
    report: dict[str, Any] = {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": config.model_dump(mode="json"),
        "environment": _environment(),
        "features": {
            "n_features": len(split_data.feature_columns),
            "names": list(split_data.feature_columns),
        },
        "splits": {
            "rows": {
                split: int(len(frame)) for split, frame in split_data.frames.items()
            },
            "class_counts": class_counts,
        },
        "benchmark_settings": {
            "n_trials": int(trials_used),
            "time_budget_seconds": float(budget_used),
            "seeds": seed_list,
            "tasks": task_list,
            "families": family_list,
            "latency_calls": config.tier1.benchmark.latency_calls,
            "warmup_calls": config.tier1.benchmark.warmup_calls,
            "batch_size": config.tier1.benchmark.batch_size,
            "throughput_max_rows": config.tier1.benchmark.throughput_max_rows,
            "train_n_jobs": config.tier1.n_jobs,
            "inference_n_jobs": config.tier1.inference_n_jobs,
        },
        "tuning": tuning,
        "runs": runs,
        "aggregates": aggregates,
        "selection": selection,
        "chosen_default": {
            "task": selection["chosen"]["task"],
            "family": selection["chosen"]["family"],
        },
        "definitions": {
            "accuracy_per_ms": "test accuracy / median single-flow latency (ms)",
            "latency": "one predict_proba call per flow, isolated single row, "
            f"inference_n_jobs={config.tier1.inference_n_jobs}",
            "batch_throughput": "flows/second on at most "
            f"{config.tier1.benchmark.throughput_max_rows} test rows",
            "selection_split": "val",
        },
        "figures": [
            _display_path(figures / f"tier1_confusion_{task}_{family}.png", root)
            for task in task_list
            for family in family_list
        ],
        "notes": [
            "Test metrics are computed only after all tuning finished.",
            "Model selection uses validation macro-F1 and measured p95 latency, "
            "never test accuracy.",
            "Class weights are computed from the training split; no row is "
            "resampled.",
        ],
    }

    json_path = reports / "tier1_benchmark.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    write_markdown(report, reports / "tier1_benchmark.md")
    logger.info("Wrote %s and %s", json_path, reports / "tier1_benchmark.md")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_values(value: str) -> list[str]:
    """Split a comma-separated CLI value."""
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the Tier-1 benchmark.

    Args:
        argv: argument list override (tests).

    Returns:
        Process exit code: 0 on success, 1 when the processed data is missing.
    """
    parser = argparse.ArgumentParser(
        description="Train and benchmark Tier-1 candidates (RF, XGBoost, MLP)."
    )
    parser.add_argument("--dataset", required=True, help="cicids2017 or ciciot2023")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--tasks", default=None, help="comma-separated tasks: binary,multiclass"
    )
    parser.add_argument(
        "--families",
        default=None,
        help="comma-separated families: random_forest,xgboost,mlp",
    )
    parser.add_argument("--seeds", default=None, help="comma-separated integer seeds")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials")
    parser.add_argument(
        "--time-budget", type=float, default=None, help="Optuna seconds per study"
    )
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)
    try:
        report = run_benchmark(
            dataset=args.dataset,
            config=config,
            tasks=_parse_values(args.tasks) if args.tasks else None,
            families=_parse_values(args.families) if args.families else None,
            seeds=[int(value) for value in _parse_values(args.seeds)]
            if args.seeds
            else None,
            n_trials=args.n_trials,
            time_budget_seconds=args.time_budget,
            processed_dir=Path(args.processed_dir) if args.processed_dir else None,
            models_dir=Path(args.models_dir) if args.models_dir else None,
            reports_dir=Path(args.reports_dir) if args.reports_dir else None,
            figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        )
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    choice = report["chosen_default"]
    logger.info(
        "Benchmark complete. Selected default: %s / %s",
        choice["family"],
        choice["task"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
