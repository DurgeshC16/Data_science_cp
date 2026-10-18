"""Leak-free preprocessing and stratified splitting for Cascade-XID.

Pipeline
--------
1. **Structural cleaning** (label-free, applied to the whole pool before any
   split): ``+-inf`` -> ``NaN``, exact duplicate rows removed, identifier
   columns dropped, constant columns dropped.
2. **Label unification** via ``configs/label_map.yaml``: every raw label is
   mapped to ``multiclass_label`` (10-class taxonomy) and ``binary_label``
   (Normal/Attack). Unmapped labels raise instead of being guessed.
3. **Zero-day holdout**: configured rare attack classes are removed from the
   pool *before* splitting and written to
   ``data/processed/zeroday_test.parquet``.
4. **Stratified 60/10/10/20 split** on ``multiclass_label``: train,
   val, calibration, test. The calibration split is used only to fit the
   probability calibrator and the gate thresholds; it never trains a model.
5. **Train-only preprocessing**: missing-value statistics, column dropping,
   imputation values and the scaler are fitted on the *training split only*
   and applied unchanged to val/calibration/test/holdout.
6. **Class imbalance** is handled through model class weights (config option).
   Rows are never resampled, and nothing is resampled before splitting.

Artifacts
---------
- ``data/processed/<dataset>_{train,val,calibration,test}.parquet``
- ``data/processed/zeroday_test.parquet`` (spec path; most recent dataset) and
  ``data/processed/<dataset>_zeroday_test.parquet`` (per-dataset copy)
- ``models/<dataset>_preprocessor.joblib`` (fitted imputer + scaler)
- ``models/<dataset>_features.json`` (feature list, dtypes, dropped columns)
- ``models/<dataset>_class_weights.json`` (train-only class weights)
- ``reports/split_summary.md`` and ``reports/split_summary.json``

Usage
-----
::

    python -m src.data.preprocess --dataset cicids2017
    python -m src.data.preprocess --dataset ciciot2023 --propose-zeroday

Retained shortcut risks
-----------------------
Identifier columns (IPs, MACs, host addresses, flow ids, timestamps) are
removed because they let a model memorise hosts or capture order. Some retained
features can still act as shortcuts and are documented rather than dropped:

- ``destination_port`` -- attack traffic in these captures targets a small,
  fixed set of ports, so a model can learn the port instead of the behaviour.
  It is kept because port choice is genuinely predictive, but any generalisation
  claim must be accompanied by an ablation with this feature removed.
- ``flow_bytes_s`` / ``flow_packets_s`` -- derived rate features are infinite
  for zero-duration flows (converted to ``NaN`` and imputed). Their missingness
  itself correlates with the capture tooling, not only with the attack.

These are recorded in the per-dataset section of ``reports/split_summary.md``.

Rules (AGENTS.md): no data is invented, every stochastic step takes
``configs/config.yaml`` ``seed``, and no statistic is fitted on val,
calibration or test.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import PROJECT_ROOT, Config, load_config

logger = logging.getLogger(__name__)

#: Unified multiclass taxonomy (see ``configs/label_map.yaml``).
UNIFIED_CLASSES: tuple[str, ...] = (
    "Normal",
    "DoS",
    "DDoS",
    "PortScan/Recon",
    "BruteForce",
    "WebAttack",
    "Botnet/Mirai",
    "Spoofing",
    "Infiltration",
    "Other",
)

#: Binary view of the taxonomy.
BINARY_CLASSES: tuple[str, ...] = ("Normal", "Attack")

#: Column holding the unified multiclass target.
MULTICLASS_COLUMN = "multiclass_label"
#: Column holding the unified binary target.
BINARY_COLUMN = "binary_label"
#: Columns that are targets/bookkeeping and therefore never model features.
LABEL_COLUMNS: tuple[str, ...] = (MULTICLASS_COLUMN, BINARY_COLUMN)
#: Stable per-row identifier used to prove splits are disjoint. Bookkeeping
#: only -- it is never a model feature.
FLOW_ID_COLUMN = "flow_id"

#: Split names and the on-disk suffix used for each.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "calibration", "test")

#: A class needs at least this many rows to survive a 4-way stratified split.
MIN_ROWS_PER_CLASS = 4
#: Below this count a class is reported as "rare" in the split summary.
RARE_CLASS_WARNING_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def load_label_map(config: Config) -> dict[str, dict[str, dict[str, Any]]]:
    """Load ``configs/label_map.yaml``.

    Args:
        config: project config providing ``data.label_map``.

    Returns:
        Mapping ``{dataset: {raw_label: {multiclass_label, binary_label, ...}}}``.

    Raises:
        FileNotFoundError: If the label map file is missing.
        ValueError: If the file is not a mapping of datasets.
    """
    path = PROJECT_ROOT / config.data.label_map
    if not path.is_file():
        raise FileNotFoundError(f"label map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"label map must be a mapping, got {type(raw).__name__}")
    return raw


def _normalise_label_key(value: str) -> str:
    """Collapse whitespace and case so label lookups are robust.

    Args:
        value: raw label string.

    Returns:
        Upper-cased, whitespace-collapsed label used only for *matching*; the
        original label string is preserved in the data.
    """
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def apply_label_map(
    df: pd.DataFrame,
    dataset: str,
    label_map: dict[str, dict[str, dict[str, Any]]],
    label_col: str = "label",
) -> pd.DataFrame:
    """Attach unified ``multiclass_label`` and ``binary_label`` columns.

    Labels are matched case-insensitively with whitespace collapsed, so loader
    casing differences do not silently drop rows. Any label that is not in the
    map raises -- the project never guesses a mapping.

    Args:
        df: DataFrame with a raw ``label`` column.
        dataset: dataset key in the label map (``"cicids2017"``/``"ciciot2023"``).
        label_map: mapping as returned by :func:`load_label_map`.
        label_col: name of the raw label column.

    Returns:
        The same DataFrame with ``multiclass_label`` and ``binary_label`` added.

    Raises:
        KeyError: If ``dataset`` is absent from the label map.
        ValueError: If ``label_col`` is missing or any label is unmapped.
    """
    if dataset not in label_map:
        raise KeyError(f"dataset {dataset!r} not present in the label map")
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found in DataFrame")

    entries = label_map[dataset]
    lookup: dict[str, str] = {}
    for raw_label, mapping in entries.items():
        if not isinstance(mapping, dict):
            continue
        key = _normalise_label_key(raw_label)
        multiclass = mapping.get("multiclass_label")
        if multiclass not in UNIFIED_CLASSES:
            raise ValueError(
                f"label map entry {dataset}/{raw_label!r} has invalid "
                f"multiclass_label {multiclass!r}"
            )
        lookup[key] = multiclass

    keys = df[label_col].map(_normalise_label_key)
    unmapped = sorted(set(keys) - set(lookup))
    if unmapped:
        sample = ", ".join(repr(u) for u in unmapped[:10])
        more = "" if len(unmapped) <= 10 else f" (+{len(unmapped) - 10} more)"
        raise ValueError(
            f"unmapped {dataset} labels: {sample}{more}. Add them to "
            f"{PROJECT_ROOT / 'configs' / 'label_map.yaml'} before preprocessing; "
            "guessing a mapping is not allowed."
        )

    df[MULTICLASS_COLUMN] = keys.map(lookup)
    df[BINARY_COLUMN] = np.where(df[MULTICLASS_COLUMN] == "Normal", "Normal", "Attack")
    return df


# ---------------------------------------------------------------------------
# Structural cleaning
# ---------------------------------------------------------------------------


def find_identifier_columns(
    df: pd.DataFrame,
    patterns: list[str],
    extra_columns: list[str] | None = None,
) -> list[str]:
    """Return columns that look like identifiers (IPs, ids, timestamps).

    Identifier columns let a model memorise hosts or capture order instead of
    learning flow behaviour, so they are removed before splitting.

    Args:
        df: DataFrame to inspect.
        patterns: regexes matched case-insensitively against column names.
        extra_columns: explicit column names to treat as identifiers.

    Returns:
        Sorted list of identifier column names present in ``df``.
    """
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    found: set[str] = set()
    for column in df.columns:
        if any(p.search(column) for p in compiled):
            found.add(column)
    if extra_columns:
        found.update(c for c in extra_columns if c in df.columns)
    return sorted(found)


def clean_structural(
    df: pd.DataFrame,
    config: Config,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply label-free cleaning that must happen before any split.

    Steps, in order (each decision is recorded in the returned stats):

    1. ``+-inf`` -> ``NaN``. CICIDS2017 rate features (``flow_bytes_s``,
       ``flow_packets_s``) are infinite for zero-duration flows; treating them
       as missing is safer than feeding an unbounded value to a scaler.
    2. **Exact duplicate rows are removed before splitting.** The CICIDS2017
       capture days contain repeated flows; leaving them in place would put
       byte-identical rows in train and test (leakage).
    3. Identifier columns are dropped (see :func:`find_identifier_columns`).
    4. Constant columns (``nunique(dropna=False) <= 1``) are dropped: they
       carry no information and break correlation/scaling. This is a structural,
       label-free decision taken over the whole pool.

    Args:
        df: DataFrame with normalised column names and a raw ``label`` column.
        config: project config (``preprocess`` and ``data.drop_columns``).
        dataset: dataset name used only for logging.

    Returns:
        Tuple of the cleaned DataFrame and a stats dictionary describing every
        action taken.
    """
    stats: dict[str, Any] = {
        "rows_in": int(len(df)),
        "columns_in": int(df.shape[1]),
        "inf_values_replaced": 0,
        "duplicate_rows_removed": 0,
        "identifier_columns_dropped": [],
        "constant_columns_dropped": [],
    }

    work = df.copy()

    # 1. inf -> NaN (the loader already does this; repeated defensively).
    numeric_cols = work.select_dtypes(include="number").columns
    if len(numeric_cols):
        n_inf = int(np.isinf(work[numeric_cols].to_numpy()).sum())
        if n_inf:
            work[numeric_cols] = work[numeric_cols].replace([np.inf, -np.inf], np.nan)
            stats["inf_values_replaced"] = n_inf

    # 2. Exact duplicates BEFORE splitting.
    if config.preprocess.drop_duplicate_rows:
        before = len(work)
        work = work.drop_duplicates(keep="first").reset_index(drop=True)
        stats["duplicate_rows_removed"] = int(before - len(work))

    # 3. Identifier columns.
    explicit = list(config.data.drop_columns)
    if config.data.flow_id_column:
        explicit.append(config.data.flow_id_column)
    identifiers = find_identifier_columns(
        work, config.preprocess.identifier_patterns, explicit
    )
    identifiers = [c for c in identifiers if c != "label"]
    if identifiers:
        work = work.drop(columns=identifiers)
        stats["identifier_columns_dropped"] = identifiers

    # 4. Constant columns (label/identifier-free structural check).
    if config.preprocess.drop_constant_columns:
        protected = set(LABEL_COLUMNS) | {"label"}
        constant = [
            c
            for c in work.columns
            if c not in protected and work[c].nunique(dropna=False) <= 1
        ]
        if constant:
            work = work.drop(columns=constant)
            stats["constant_columns_dropped"] = constant

    stats["rows_out"] = int(len(work))
    stats["columns_out"] = int(work.shape[1])
    logger.info(
        "%s: cleaned %d -> %d rows, %d -> %d columns (dupes=%d, ids=%d, constants=%d)",
        dataset,
        stats["rows_in"],
        stats["rows_out"],
        stats["columns_in"],
        stats["columns_out"],
        stats["duplicate_rows_removed"],
        len(stats["identifier_columns_dropped"]),
        len(stats["constant_columns_dropped"]),
    )
    return work, stats


def assign_flow_ids(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add a deterministic ``flow_id`` bookkeeping column.

    The id is derived from the dataset name and the row's position *after*
    cleaning, so it never encodes a label, host or timestamp. It is excluded
    from ``feature_columns`` and exists only for traceability and for proving
    that splits are disjoint.

    Args:
        df: cleaned DataFrame.
        dataset: dataset name used as the id prefix.

    Returns:
        The same DataFrame with a ``flow_id`` column inserted first.
    """
    df.insert(0, FLOW_ID_COLUMN, [f"{dataset}-{i:08d}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------------------
# Zero-day holdout
# ---------------------------------------------------------------------------


def propose_zeroday_candidates(
    df: pd.DataFrame,
    max_classes: int = 2,
    label_col: str = MULTICLASS_COLUMN,
) -> list[tuple[str, int]]:
    """Return the rarest attack classes as zero-day candidates.

    Only classes present in the data are proposed; the counts come from the
    loaded data, never from assumptions.

    Args:
        df: DataFrame with a unified label column.
        max_classes: maximum number of candidates to return.
        label_col: unified label column name.

    Returns:
        List of ``(class, count)`` pairs, rarest first, excluding ``Normal``.
    """
    counts = df[label_col].value_counts()
    attack = counts[counts.index != "Normal"].sort_values()
    return [(str(label), int(count)) for label, count in attack.head(max_classes).items()]


def resolve_zeroday_classes(
    classes: list[str],
    dataset: str,
    label_map: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """Resolve configured zero-day names to unified taxonomy labels.

    A configured name may be either a unified class (``"Other"``) or a raw
    dataset label as shown in the EDA report (``"HEARTBLEED"``), which is then
    mapped through ``configs/label_map.yaml``. Because a raw label can map onto
    a shared unified class, holding out a raw label also holds out every other
    raw label that maps to the same unified class; the resolved list is logged.

    Args:
        classes: configured names (unified or raw).
        dataset: dataset key in the label map.
        label_map: mapping as returned by :func:`load_label_map`.

    Returns:
        Unique unified class names, order preserved.

    Raises:
        KeyError: If ``dataset`` is absent from the label map.
        ValueError: If a configured name matches neither a unified class nor a
            raw label of this dataset.
    """
    if dataset not in label_map:
        raise KeyError(f"dataset {dataset!r} not present in the label map")
    raw_lookup = {
        _normalise_label_key(raw_label): mapping["multiclass_label"]
        for raw_label, mapping in label_map[dataset].items()
        if isinstance(mapping, dict) and mapping.get("multiclass_label")
    }

    resolved: list[str] = []
    for name in classes:
        if name in UNIFIED_CLASSES:
            target = name
        else:
            target = raw_lookup.get(_normalise_label_key(name), "")
        if not target:
            raise ValueError(
                f"zero-day class {name!r} is neither a unified class nor a "
                f"{dataset} raw label. Unified: {list(UNIFIED_CLASSES)}; "
                f"raw labels: {sorted(label_map[dataset])}"
            )
        if target not in resolved:
            resolved.append(target)
    logger.info("Zero-day classes resolved to unified labels: %s", resolved)
    return resolved


def take_zeroday_holdout(
    df: pd.DataFrame,
    classes: list[str],
    label_col: str = MULTICLASS_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split configured zero-day classes off the modelling pool.

    Args:
        df: cleaned, label-unified DataFrame.
        classes: unified labels to hold out entirely (resolve raw names first
            with :func:`resolve_zeroday_classes`).
        label_col: unified label column name.

    Returns:
        ``(pool_without_holdout, holdout)``.

    Raises:
        ValueError: If a configured class is absent from the data, which would
            silently defeat the zero-day simulation.
    """
    if not classes:
        empty = df.iloc[0:0].copy()
        return df, empty

    present = set(df[label_col].unique())
    missing = [c for c in classes if c not in present]
    if missing:
        raise ValueError(
            f"zero-day classes not found in the data: {missing}. "
            f"Available labels: {sorted(present)}"
        )

    mask = df[label_col].isin(classes)
    holdout = df.loc[mask].reset_index(drop=True)
    pool = df.loc[~mask].reset_index(drop=True)
    logger.info(
        "Zero-day holdout: %d rows across %s removed from the modelling pool.",
        len(holdout),
        classes,
    )
    return pool, holdout


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
    config: Config,
    label_col: str = MULTICLASS_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Create stratified train / val / calibration / test splits.

    Two successive stratified ``train_test_split`` calls are used so that every
    split keeps the class proportions of the pool. The calibration split is a
    separate, disjoint frame and is never passed to a model-fitting step.

    Args:
        df: DataFrame to split (zero-day rows already removed).
        config: project config providing fractions, seed and stratification.
        label_col: column to stratify on.

    Returns:
        Mapping split name -> DataFrame, with ``train``/``val``/``calibration``/
        ``test`` keys.

    Raises:
        ValueError: If ``label_col`` is missing, if the fractions are invalid,
            or if a class is too small to be stratified across four splits.
    """
    if label_col not in df.columns:
        raise ValueError(f"stratification column {label_col!r} not found")
    fractions = {
        "train": config.split.train,
        "val": config.split.val,
        "calibration": config.split.calibration,
        "test": config.split.test,
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")

    counts = df[label_col].value_counts()
    too_small = {str(k): int(v) for k, v in counts.items() if v < MIN_ROWS_PER_CLASS}
    if too_small:
        raise ValueError(
            f"classes with fewer than {MIN_ROWS_PER_CLASS} rows cannot be "
            f"stratified across four splits: {too_small}. Move them to "
            "preprocess.zeroday_classes (holdout) or remove them explicitly."
        )
    rare = {str(k): int(v) for k, v in counts.items() if v < RARE_CLASS_WARNING_THRESHOLD}
    if rare:
        logger.warning("Rare classes (<%d rows): %s", RARE_CLASS_WARNING_THRESHOLD, rare)

    stratify = df[label_col] if config.split.stratify else None
    seed = config.seed

    # First cut: hold out the test+val+calibration remainder together.
    held_out_fraction = fractions["val"] + fractions["calibration"] + fractions["test"]
    train_df, remainder = train_test_split(
        df,
        test_size=held_out_fraction,
        stratify=stratify,
        random_state=seed,
    )

    # Second cut: test vs (val+calibration).
    remainder_fraction = fractions["val"] + fractions["calibration"]
    val_cal_fraction = remainder_fraction / held_out_fraction
    test_df, val_cal = train_test_split(
        remainder,
        test_size=1.0 - val_cal_fraction,
        stratify=remainder[label_col] if stratify is not None else None,
        random_state=seed,
    )

    # Third cut: calibration vs val.
    calibration_fraction = fractions["calibration"] / remainder_fraction
    val_df, calibration_df = train_test_split(
        val_cal,
        test_size=calibration_fraction,
        stratify=val_cal[label_col] if stratify is not None else None,
        random_state=seed,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "calibration": calibration_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    logger.info(
        "Split sizes: %s",
        {name: len(frame) for name, frame in splits.items()},
    )
    return splits


# ---------------------------------------------------------------------------
# Train-only preprocessing
# ---------------------------------------------------------------------------


@dataclass
class FittedPreprocessor:
    """Imputer + scaler fitted on the training split only.

    Attributes:
        dataset: dataset name.
        feature_columns: ordered model feature names.
        feature_dtypes: original pandas dtype string per feature.
        impute_strategy: ``median``, ``mean`` or ``drop_rows``.
        impute_values: per-column fill value computed on the training split.
        scaling: ``standard``, ``robust`` or ``none``.
        scaler: fitted scikit-learn scaler, or ``None`` when scaling is off.
        dropped_columns: columns removed because train missingness exceeded the
            configured threshold or the dtype was not numeric.
        max_column_missing_fraction: configured missingness cutoff.
    """

    dataset: str
    feature_columns: list[str]
    feature_dtypes: dict[str, str]
    impute_strategy: str
    impute_values: dict[str, float]
    scaling: str
    scaler: StandardScaler | RobustScaler | None
    dropped_columns: list[str] = field(default_factory=list)
    max_column_missing_fraction: float = 0.30

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted imputation and scaling to a DataFrame.

        Args:
            df: DataFrame containing the fitted feature columns, plus optional
                ``flow_id`` and label columns which are passed through.

        Returns:
            DataFrame with ``flow_id``, the scaled feature columns and any label
            columns present in ``df``. Rows with missing values are dropped only
            when ``impute_strategy == "drop_rows"``.
        """
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing at transform time: {missing}")

        features = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)

        if self.impute_strategy == "drop_rows":
            keep = features.notna().all(axis=1)
            features = features.loc[keep]
            passthrough = df.loc[keep]
        else:
            features = features.fillna(self.impute_values)
            passthrough = df

        if self.scaler is not None:
            scaled = self.scaler.transform(features.to_numpy(dtype=float))
            features = pd.DataFrame(
                scaled, columns=self.feature_columns, index=features.index
            )

        out = pd.DataFrame({FLOW_ID_COLUMN: passthrough[FLOW_ID_COLUMN].to_numpy()})
        for column in self.feature_columns:
            out[column] = features[column].to_numpy()
        for column in LABEL_COLUMNS:
            if column in passthrough.columns:
                out[column] = passthrough[column].to_numpy()
        return out.reset_index(drop=True)


def fit_preprocessor(
    train_df: pd.DataFrame,
    config: Config,
    dataset: str,
) -> FittedPreprocessor:
    """Fit imputation values and the scaler on the training split only.

    Columns are dropped when their *train* missing fraction exceeds
    ``max_column_missing_fraction`` or when they are not numeric. The same
    decision is then applied unchanged to the other splits, so no statistic is
    ever computed on val/calibration/test.

    Args:
        train_df: training split (only this frame may influence the fit).
        config: project config providing ``preprocess`` settings.
        dataset: dataset name.

    Returns:
        A fitted :class:`FittedPreprocessor`.
    """
    # `label` is the raw string target kept for audits; it is never a feature.
    protected = set(LABEL_COLUMNS) | {FLOW_ID_COLUMN, config.data.label_column}
    candidates = [
        c
        for c in train_df.columns
        if c not in protected and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    non_numeric = [
        c
        for c in train_df.columns
        if c not in protected and not pd.api.types.is_numeric_dtype(train_df[c])
    ]

    numeric_train = train_df[candidates].replace([np.inf, -np.inf], np.nan)
    missing_fraction = numeric_train.isna().mean()
    too_sparse = [c for c in candidates if missing_fraction[c] > config.preprocess.max_column_missing_fraction]
    feature_columns = [c for c in candidates if c not in too_sparse]

    dropped = sorted(set(non_numeric) | set(too_sparse))
    if non_numeric:
        logger.warning("Dropping non-numeric columns (not scaler-compatible): %s", non_numeric)
    if too_sparse:
        logger.warning(
            "Dropping columns missing > %.0f%% of TRAIN rows: %s",
            100 * config.preprocess.max_column_missing_fraction,
            too_sparse,
        )

    if not feature_columns:
        raise ValueError("no numeric feature columns left after cleaning")

    # Imputation statistics: median is the default because flow-feature
    # distributions are heavy-tailed (a few very large flows) and dropping rows
    # would delete whole attack classes. The values come from train only.
    if config.preprocess.impute_strategy == "drop_rows":
        impute_values: dict[str, float] = {}
    elif config.preprocess.impute_strategy == "mean":
        impute_values = {
            c: float(numeric_train[c].mean()) for c in feature_columns
        }
    else:
        impute_values = {
            c: float(numeric_train[c].median()) for c in feature_columns
        }

    if config.preprocess.impute_strategy != "drop_rows":
        imputed = numeric_train[feature_columns].fillna(impute_values)
        # A fully-missing column would still be NaN; guard against it loudly.
        if imputed.isna().any().any():
            bad = list(imputed.columns[imputed.isna().any()])
            raise ValueError(
                f"columns with no usable training values after imputation: {bad}"
            )
    else:
        imputed = numeric_train[feature_columns].dropna()

    scaler: StandardScaler | RobustScaler | None
    if config.preprocess.scaling == "standard":
        scaler = StandardScaler()
    elif config.preprocess.scaling == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        if len(imputed) == 0:
            raise ValueError("training split has no complete rows to fit the scaler")
        scaler.fit(imputed.to_numpy(dtype=float))

    return FittedPreprocessor(
        dataset=dataset,
        feature_columns=feature_columns,
        feature_dtypes={c: str(train_df[c].dtype) for c in feature_columns},
        impute_strategy=config.preprocess.impute_strategy,
        impute_values=impute_values,
        scaling=config.preprocess.scaling,
        scaler=scaler,
        dropped_columns=dropped,
        max_column_missing_fraction=config.preprocess.max_column_missing_fraction,
    )


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def compute_split_class_weights(
    train_df: pd.DataFrame,
    config: Config,
    label_col: str = MULTICLASS_COLUMN,
) -> dict[str, float] | None:
    """Compute balanced class weights from the training split only.

    Weights are passed to the Tier-1 model; no row is duplicated or removed, so
    the imbalance handling never leaks across splits.

    Args:
        train_df: training split.
        config: project config (``preprocess.class_weighting``).
        label_col: target column.

    Returns:
        Mapping ``{class: weight}``, or ``None`` when weighting is disabled.
    """
    if config.preprocess.class_weighting == "none":
        return None
    classes = np.array(sorted(train_df[label_col].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df[label_col].to_numpy(),
    )
    return {str(cls): float(weight) for cls, weight in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _class_table(frames: dict[str, pd.DataFrame], label_col: str) -> list[str]:
    """Build a Markdown table of per-class counts across splits.

    Args:
        frames: mapping split name -> DataFrame.
        label_col: unified label column.

    Returns:
        Markdown lines.
    """
    all_labels = sorted(
        {label for frame in frames.values() for label in frame[label_col].unique()},
        key=lambda v: (v != "Normal", v),
    )
    lines = [
        "| Class | " + " | ".join(f"{name}" for name in frames) + " | Total |",
        "| --- | " + " | ".join("---:" for _ in frames) + " | ---: |",
    ]
    for label in all_labels:
        counts = [int((frame[label_col] == label).sum()) for frame in frames.values()]
        lines.append(
            f"| `{label}` | " + " | ".join(str(c) for c in counts) + f" | {sum(counts)} |"
        )
    totals = [len(frame) for frame in frames.values()]
    lines.append(
        "| **Total** | " + " | ".join(f"**{t}**" for t in totals) + f" | {sum(totals)} |"
    )
    return lines


def _split_percentages(frames: dict[str, pd.DataFrame], label_col: str) -> list[str]:
    """Build a Markdown table of per-class percentages within each split.

    Args:
        frames: mapping split name -> DataFrame.
        label_col: unified label column.

    Returns:
        Markdown lines.
    """
    labels = sorted(
        {label for frame in frames.values() for label in frame[label_col].unique()},
        key=lambda v: (v != "Normal", v),
    )
    lines = [
        "| Class | " + " | ".join(f"{name} %" for name in frames) + " |",
        "| --- | " + " | ".join("---:" for _ in frames) + " |",
    ]
    for label in labels:
        cells = []
        for frame in frames.values():
            share = 100.0 * float((frame[label_col] == label).sum()) / max(len(frame), 1)
            cells.append(f"{share:.2f}")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return lines


def _git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"``.

    Returns:
        Short commit hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _render_dataset_section(
    dataset: str,
    splits: dict[str, pd.DataFrame],
    holdout: pd.DataFrame,
    preprocessor: FittedPreprocessor,
    cleaning: dict[str, Any],
    config: Config,
) -> str:
    """Render the Markdown section for one dataset.

    Args:
        dataset: dataset name.
        splits: the four modelling splits.
        holdout: zero-day holdout frame.
        preprocessor: fitted preprocessor.
        cleaning: stats from :func:`clean_structural`.
        config: project config.

    Returns:
        Markdown text for this dataset.
    """
    frames = {name: splits[name] for name in SPLIT_NAMES}
    if len(holdout):
        frames["zeroday"] = holdout
    lines: list[str] = [f"## Dataset: `{dataset}`", ""]
    lines.append(
        f"Seed `{config.seed}`; fractions train/val/calibration/test = "
        f"{config.split.train}/{config.split.val}/{config.split.calibration}/"
        f"{config.split.test}; git commit `{_git_commit()}`."
    )
    lines.append("")
    lines.append("### Class counts per split")
    lines.append("")
    lines.extend(_class_table(frames, MULTICLASS_COLUMN))
    lines.append("")
    lines.append("### Class percentages within each split")
    lines.append("")
    lines.extend(_split_percentages(frames, MULTICLASS_COLUMN))
    lines.append("")
    lines.append("### Binary view")
    lines.append("")
    lines.extend(_class_table(frames, BINARY_COLUMN))
    lines.append("")
    lines.append("### Preprocessing")
    lines.append("")
    lines.append(f"- Rows before cleaning: {cleaning['rows_in']:,}")
    lines.append(f"- Exact duplicate rows removed before splitting: {cleaning['duplicate_rows_removed']:,}")
    lines.append(f"- inf values replaced with NaN: {cleaning['inf_values_replaced']:,}")
    lines.append(
        "- Identifier columns dropped: "
        + (
            ", ".join(f"`{c}`" for c in cleaning["identifier_columns_dropped"])
            or "none"
        )
    )
    lines.append(
        "- Constant columns dropped: "
        + (", ".join(f"`{c}`" for c in cleaning["constant_columns_dropped"]) or "none")
    )
    lines.append(
        f"- Imputation: `{preprocessor.impute_strategy}` fitted on train; "
        f"columns dropped for train missingness > "
        f"{preprocessor.max_column_missing_fraction:.0%}: "
        + (", ".join(f"`{c}`" for c in preprocessor.dropped_columns) or "none")
    )
    lines.append(
        f"- Scaling: `{preprocessor.scaling}` fitted on train only "
        f"({len(preprocessor.feature_columns)} features)"
    )
    lines.append(
        f"- Class weighting: `{config.preprocess.class_weighting}` "
        "(weights only; no resampling)"
    )
    lines.append("")
    lines.append("### Retained shortcut risks")
    lines.append("")
    lines.append(
        "- `destination_port`: retained as a feature because port choice is "
        "genuinely predictive, but it is also a dataset artefact -- attack "
        "traffic in these captures targets a handful of fixed ports. A model "
        "can learn the port instead of the behaviour. Monitoring/ablation of "
        "this feature is required before any claim of generalisation."
    )
    lines.append(
        "- `flow_id`: synthetic bookkeeping id, excluded from features."
    )
    lines.append(
        "- `protocol`, `source_port`-like columns are dropped or documented in "
        "the cleaning stats above; check the identifier drop list per dataset."
    )
    lines.append("")
    return "\n".join(lines)


def write_split_summary(
    report_path: Path,
    dataset: str,
    section: str,
) -> None:
    """Insert or replace one dataset section in ``reports/split_summary.md``.

    Other dataset sections already in the file are preserved, so processing a
    second dataset does not erase the first one's summary.

    Args:
        report_path: Markdown file to update.
        dataset: dataset name (used as the section key).
        section: rendered section text.
    """
    header = "# Split summary\n\n"
    existing = report_path.read_text(encoding="utf-8") if report_path.is_file() else header
    if not existing.strip():
        existing = header
    if not existing.startswith("# Split summary"):
        existing = header + existing

    parts = re.split(r"(?m)^## Dataset: `", existing)
    kept: list[str] = []
    for part in parts[1:]:
        name = part.split("`", 1)[0]
        if name.strip() == dataset:
            continue
        kept.append("## Dataset: `" + part.rstrip() + "\n")

    body = existing.split("## Dataset: `", 1)[0].rstrip() + "\n"
    pieces = [body] + kept + [section.rstrip() + "\n"]
    report_path.write_text("\n".join(pieces), encoding="utf-8")
    logger.info("Split summary updated: %s", report_path)


def write_split_summary_json(
    path: Path,
    dataset: str,
    payload: dict[str, Any],
) -> None:
    """Merge one dataset's split metrics into ``reports/split_summary.json``.

    Args:
        path: JSON file to update.
        dataset: dataset name.
        payload: measured values for this dataset.
    """
    existing: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle) or {}
    existing[dataset] = payload
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_preprocessing(
    dataset: str,
    df: pd.DataFrame,
    config: Config | None = None,
    config_path: str | Path | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full cleaning / holdout / split / fit pipeline for one dataset.

    Args:
        dataset: ``cicids2017`` or ``ciciot2023`` (key in the label map).
        df: cleaned DataFrame from :mod:`src.data.loader` (before splitting).
        config: optional in-memory config; wins over ``config_path``.
        config_path: optional YAML path used when ``config`` is ``None``.
        processed_dir: override for the processed-data directory (tests).
        models_dir: override for the model-artifact directory (tests).
        reports_dir: override for the reports directory (tests).

    Returns:
        Dictionary with the split frames, fitted preprocessor, class weights and
        cleaning stats.
    """
    cfg = config if config is not None else load_config(config_path)
    root = PROJECT_ROOT
    processed = processed_dir if processed_dir is not None else root / cfg.paths.processed
    models = models_dir if models_dir is not None else root / cfg.paths.models
    reports = reports_dir if reports_dir is not None else root / cfg.paths.reports
    for directory in (processed, models, reports):
        directory.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(cfg)

    # 1-2. Structural cleaning, then label unification.
    cleaned, cleaning_stats = clean_structural(df, cfg, dataset)
    cleaned = apply_label_map(cleaned, dataset, label_map, cfg.data.label_column)
    cleaned = assign_flow_ids(cleaned, dataset)

    # 3. Zero-day holdout taken before the split so it can never be trained on.
    zeroday_unified = resolve_zeroday_classes(
        list(cfg.preprocess.zeroday_classes), dataset, label_map
    )
    pool, holdout = take_zeroday_holdout(cleaned, zeroday_unified, MULTICLASS_COLUMN)

    # 4. Stratified split on the unified multiclass label.
    splits = split_dataset(pool, cfg, MULTICLASS_COLUMN)

    # 5. Fit preprocessing on train only, then apply to every split.
    preprocessor = fit_preprocessor(splits["train"], cfg, dataset)
    transformed = {
        name: preprocessor.transform(frame) for name, frame in splits.items()
    }
    if len(holdout):
        holdout_features = preprocessor.transform(holdout)
    else:
        holdout_features = holdout

    # 6. Class weights from train only (no resampling anywhere).
    class_weights = compute_split_class_weights(splits["train"], cfg)

    # Persist splits.
    for name, frame in transformed.items():
        out = processed / f"{dataset}_{name}.parquet"
        frame.to_parquet(out, index=False)
        logger.info("Saved %s split to %s", name, out)
    holdout_spec = processed / "zeroday_test.parquet"
    holdout_dataset = processed / f"{dataset}_zeroday_test.parquet"
    holdout_features.to_parquet(holdout_dataset, index=False)
    if len(holdout_features):
        holdout_features.to_parquet(holdout_spec, index=False)
    else:
        logger.warning(
            "No zero-day classes configured for %s; %s not written. "
            "Set preprocess.zeroday_classes and rerun.",
            dataset,
            holdout_spec,
        )

    # Persist preprocessor, feature list and class weights.
    joblib.dump(preprocessor, models / f"{dataset}_preprocessor.joblib")
    features_payload = {
        "dataset": dataset,
        "flow_id_column": FLOW_ID_COLUMN,
        "label_columns": list(LABEL_COLUMNS),
        "feature_columns": preprocessor.feature_columns,
        "feature_dtypes": preprocessor.feature_dtypes,
        "dropped_columns": preprocessor.dropped_columns,
        "scaling": preprocessor.scaling,
        "impute_strategy": preprocessor.impute_strategy,
    }
    (models / f"{dataset}_features.json").write_text(
        json.dumps(features_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if class_weights is not None:
        (models / f"{dataset}_class_weights.json").write_text(
            json.dumps(class_weights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Reports.
    section = _render_dataset_section(
        dataset, splits, holdout, preprocessor, cleaning_stats, cfg
    )
    write_split_summary(reports / "split_summary.md", dataset, section)
    write_split_summary_json(
        reports / "split_summary.json",
        dataset,
        {
            "git_commit": _git_commit(),
            "seed": cfg.seed,
            "split_fractions": {
                "train": cfg.split.train,
                "val": cfg.split.val,
                "calibration": cfg.split.calibration,
                "test": cfg.split.test,
            },
            "rows": {name: int(len(frame)) for name, frame in splits.items()},
            "zeroday_rows": int(len(holdout)),
            "zeroday_classes": list(cfg.preprocess.zeroday_classes),
            "counts": {
                name: {str(k): int(v) for k, v in frame[MULTICLASS_COLUMN].value_counts().items()}
                for name, frame in splits.items()
            },
            "cleaning": cleaning_stats,
            "n_features": len(preprocessor.feature_columns),
            "dropped_columns": preprocessor.dropped_columns,
            "class_weighting": cfg.preprocess.class_weighting,
            "class_weights": class_weights,
        },
    )

    return {
        "splits": transformed,
        "raw_splits": splits,
        "holdout": holdout_features,
        "preprocessor": preprocessor,
        "class_weights": class_weights,
        "cleaning": cleaning_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when the dataset is not available.
    """
    import argparse

    from src.data.loader import DataNotFoundError, load_cicids2017, load_ciciot2023

    parser = argparse.ArgumentParser(
        description="Clean, hold out zero-day classes and split a dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "ciciot2023"],
        required=True,
        help="Which dataset to preprocess.",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument(
        "--dev-sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N rows before preprocessing (fast development).",
    )
    parser.add_argument(
        "--propose-zeroday",
        action="store_true",
        help="Print the rarest attack classes as zero-day candidates and exit.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        loader = load_cicids2017 if args.dataset == "cicids2017" else load_ciciot2023
        df = loader(args.config, dev_sample=args.dev_sample)
    except DataNotFoundError:
        return 1

    cfg = load_config(args.config)

    if args.propose_zeroday:
        cleaned, _ = clean_structural(df, cfg, args.dataset)
        cleaned = apply_label_map(
            cleaned, args.dataset, load_label_map(cfg), cfg.data.label_column
        )
        for label, count in propose_zeroday_candidates(
            cleaned, cfg.preprocess.zeroday_max_classes
        ):
            print(f"{label}: {count:,} rows")
        return 0

    result = run_preprocessing(args.dataset, df, config=cfg)
    print(
        f"Preprocessed {args.dataset}: "
        + ", ".join(f"{k}={len(v):,}" for k, v in result["splits"].items())
        + f", zeroday={len(result['holdout']):,}"
    )
    print(f"Features: {len(result['preprocessor'].feature_columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
