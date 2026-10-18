"""Tests for src/data/preprocess.py.

The tests use a small **synthetic fixture** (built in ``_make_fixture``) purely
to exercise the splitting/preprocessing logic; no result in this file is a
measured dataset result. The fixture mimics the quirks of CICIDS2017: identifier
columns, a constant column, a mostly-missing column, infinite rate values,
duplicate rows and a rare attack class.

Covered guarantees:
- exact duplicates are removed before splitting; no row is in two splits
- class proportions are preserved within tolerance
- the calibration split is disjoint from train (and never fitted on)
- zero-day holdout classes are absent from train/val/calibration/test
- scaler and imputation statistics are computed from the train split only
- identifier and constant columns are dropped from the feature list
- class weighting is weights-only (row counts unchanged, no oversampling)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.data.preprocess import (
    FLOW_ID_COLUMN,
    MULTICLASS_COLUMN,
    SPLIT_NAMES,
    FittedPreprocessor,
    apply_label_map,
    clean_structural,
    load_label_map,
    run_preprocessing,
    split_dataset,
)

DATASET = "cicids2017"

#: Raw label -> row count in the fixture (rare class last).
FIXTURE_CLASSES: dict[str, int] = {
    "BENIGN": 1200,
    "DDOS": 500,
    "PORTSCAN": 200,
    "WEB ATTACK - BRUTE FORCE": 80,
    "HEARTBLEED": 20,
}


def _make_fixture(duplicates: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a CICIDS2017-shaped fixture DataFrame.

    Args:
        duplicates: number of exact duplicate rows appended at the end.
        seed: RNG seed for the synthetic values.

    Returns:
        DataFrame with raw ``label`` plus identifier, constant and missing
        columns, and ``inf`` values in ``flow_bytes_s``.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for class_index, (label, count) in enumerate(FIXTURE_CLASSES.items()):
        # Give HEARTBLEED an extreme feature scale so that including the
        # holdout in any statistic would be detectable.
        scale = 1e9 if label == "HEARTBLEED" else 1.0
        frame = pd.DataFrame(
            {
                "label": [label] * count,
                "source_ip": [f"10.0.{class_index}.{i % 250}" for i in range(count)],
                "dst_ip": [f"192.168.{class_index}.1"] * count,
                "timestamp": np.arange(count, dtype=np.int64) + class_index * 10_000,
                "destination_port": rng.choice([80, 443, 22, 8080], size=count),
                "flow_duration": rng.normal(1000 * scale, 50 * scale, count),
                "total_fwd_packets": rng.integers(1, 50, count) * scale,
                "flow_bytes_s": rng.normal(500 * scale, 10 * scale, count),
                "constant_feature": 7,
                "mostly_missing": np.where(rng.random(count) < 0.8, np.nan, 1.0),
            }
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    # Infinite rate values, as produced by zero-duration flows.
    df.loc[df.index[:5], "flow_bytes_s"] = np.inf
    if duplicates:
        df = pd.concat([df, df.iloc[:duplicates]], ignore_index=True)
    return df


def _config(zeroday: list[str] | None = None, **preprocess_overrides: object) -> Config:
    """Load the real config and override preprocessing settings for tests.

    Args:
        zeroday: zero-day class names to hold out.
        **preprocess_overrides: extra ``preprocess`` field overrides.

    Returns:
        A deep-copied :class:`~src.config.Config` with test overrides applied.
    """
    config = load_config().model_copy(deep=True)
    updates = {
        "zeroday_classes": zeroday if zeroday is not None else ["HEARTBLEED"],
        **preprocess_overrides,
    }
    config.preprocess = config.preprocess.model_copy(update=updates)
    return config


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    """Return the synthetic fixture DataFrame."""
    return _make_fixture()


@pytest.fixture()
def processed(tmp_path: Path):
    """Run preprocessing on the fixture into a temporary directory.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Tuple of (result, processed_dir, models_dir, reports_dir).
    """
    df = _make_fixture()
    config = _config()
    result = run_preprocessing(
        DATASET,
        df,
        config=config,
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    return (
        result,
        tmp_path / "processed",
        tmp_path / "models",
        tmp_path / "reports",
    )


# ---------------------------------------------------------------------------
# Splits: disjointness and proportions
# ---------------------------------------------------------------------------


def test_no_row_appears_in_two_splits(processed) -> None:
    """Every flow id belongs to exactly one split or the holdout."""
    result, _, _, _ = processed
    splits = result["splits"]
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        ids = set(splits[name][FLOW_ID_COLUMN])
        assert not (ids & seen), f"{name} shares rows with an earlier split"
        seen |= ids
    holdout_ids = set(result["holdout"][FLOW_ID_COLUMN])
    assert not (holdout_ids & seen), "holdout shares rows with a modelling split"


def test_split_sizes_cover_the_cleaned_pool(processed) -> None:
    """Split sizes plus holdout equal the deduplicated pool size."""
    result, _, _, _ = processed
    total = sum(len(result["splits"][name]) for name in SPLIT_NAMES)
    total += len(result["holdout"])
    assert total == result["cleaning"]["rows_out"]


def test_class_proportions_preserved_within_tolerance(processed) -> None:
    """Each split's class share matches the configured fraction."""
    result, _, _, _ = processed
    splits = result["splits"]
    config = _config()
    targets = {
        "train": config.split.train,
        "val": config.split.val,
        "calibration": config.split.calibration,
        "test": config.split.test,
    }
    pool_size = sum(len(splits[s]) for s in SPLIT_NAMES)
    for name, target in targets.items():
        frame = splits[name]
        assert len(frame) / pool_size == pytest.approx(target, abs=0.02)
        for label in frame[MULTICLASS_COLUMN].unique():
            # Compare the split's share of this class against the pool's share.
            pool_total = sum(
                int((splits[s][MULTICLASS_COLUMN] == label).sum()) for s in SPLIT_NAMES
            )
            pool_share_overall = pool_total / pool_size
            class_rows = int((frame[MULTICLASS_COLUMN] == label).sum())
            observed = class_rows / len(frame)
            assert observed == pytest.approx(pool_share_overall, abs=0.02), (
                f"{name}/{label}: observed {observed:.3f} vs pool "
                f"{pool_share_overall:.3f} (target ratio {target:.2f})"
            )


def test_calibration_split_is_disjoint_from_train(processed) -> None:
    """The calibration split shares no rows with the training split."""
    result, _, _, _ = processed
    train_ids = set(result["splits"]["train"][FLOW_ID_COLUMN])
    calibration_ids = set(result["splits"]["calibration"][FLOW_ID_COLUMN])
    assert not (train_ids & calibration_ids)
    assert len(calibration_ids) > 0


# ---------------------------------------------------------------------------
# Zero-day holdout
# ---------------------------------------------------------------------------


def test_holdout_classes_absent_from_every_modelling_split(processed) -> None:
    """Held-out classes appear only in the zero-day frame."""
    result, _, _, _ = processed
    for name in SPLIT_NAMES:
        assert "Other" not in set(result["splits"][name][MULTICLASS_COLUMN])
    assert "Other" in set(result["holdout"][MULTICLASS_COLUMN])


def test_holdout_saved_to_spec_path(processed) -> None:
    """The holdout is written to data/processed/zeroday_test.parquet."""
    result, processed_dir, _, _ = processed
    spec_path = processed_dir / "zeroday_test.parquet"
    assert spec_path.is_file()
    reloaded = pd.read_parquet(spec_path)
    assert len(reloaded) == len(result["holdout"])


def test_propose_zeroday_ranks_rarest_attack_classes(fixture_df: pd.DataFrame) -> None:
    """Candidates are the rarest attack classes, Normal excluded."""
    from src.data.preprocess import propose_zeroday_candidates

    config = _config()
    labelled = apply_label_map(
        clean_structural(fixture_df, config, DATASET)[0],
        DATASET,
        load_label_map(config),
    )
    candidates = propose_zeroday_candidates(labelled, max_classes=2)
    assert [name for name, _ in candidates] == ["Other", "WebAttack"]
    assert all(name != "Normal" for name, _ in candidates)
    counts = [count for _, count in candidates]
    assert counts == sorted(counts)


def test_missing_zeroday_class_raises(fixture_df: pd.DataFrame) -> None:
    """A configured class that is not in the data fails loudly."""
    from src.data.preprocess import take_zeroday_holdout

    labelled = apply_label_map(
        clean_structural(fixture_df, _config(), DATASET)[0],
        DATASET,
        load_label_map(_config()),
    )
    with pytest.raises(ValueError, match="zero-day classes not found"):
        take_zeroday_holdout(labelled, ["Infiltration"])


def test_too_small_class_raises(fixture_df: pd.DataFrame) -> None:
    """A class too small to stratify raises with actionable guidance."""
    config = _config(zeroday=[])
    cleaned = apply_label_map(
        clean_structural(fixture_df, config, DATASET)[0],
        DATASET,
        load_label_map(config),
    )
    tiny = cleaned.index[:2]
    cleaned.loc[tiny, MULTICLASS_COLUMN] = "Infiltration"
    with pytest.raises(ValueError, match="fewer than"):
        split_dataset(cleaned, config)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_exact_duplicates_removed_before_split(processed) -> None:
    """Duplicate rows are dropped, never spread across splits."""
    result, _, _, _ = processed
    assert result["cleaning"]["duplicate_rows_removed"] > 0
    frames = [result["splits"][name] for name in SPLIT_NAMES]
    combined = pd.concat(frames, ignore_index=True)
    feature_cols = [
        c
        for c in combined.columns
        if c not in (FLOW_ID_COLUMN, *["multiclass_label", "binary_label"])
    ]
    assert not combined.duplicated(subset=feature_cols).any()


def test_identifier_columns_are_dropped(processed) -> None:
    """IP address and timestamp columns never reach the feature list."""
    result, _, _, _ = processed
    features = set(result["preprocessor"].feature_columns)
    for column in ("source_ip", "dst_ip", "timestamp"):
        assert column not in features
    dropped = set(result["cleaning"]["identifier_columns_dropped"])
    assert {"source_ip", "dst_ip", "timestamp"} <= dropped
    # destination_port is deliberately retained (documented shortcut risk).
    assert "destination_port" in features


def test_constant_columns_are_dropped(processed) -> None:
    """Zero-variance columns are removed."""
    result, _, _, _ = processed
    assert "constant_feature" in result["cleaning"]["constant_columns_dropped"]
    assert "constant_feature" not in result["preprocessor"].feature_columns


def test_inf_is_replaced_and_all_features_are_finite(processed) -> None:
    """Infinite values are imputed, leaving no NaN/inf in the saved splits."""
    result, processed_dir, _, _ = processed
    assert result["cleaning"]["inf_values_replaced"] >= 5
    frame = pd.read_parquet(processed_dir / f"{DATASET}_train.parquet")
    features = [c for c in result["preprocessor"].feature_columns]
    values = frame[features].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_mostly_missing_column_dropped(processed) -> None:
    """Columns missing more than the threshold in train are dropped."""
    result, _, _, _ = processed
    assert "mostly_missing" in result["preprocessor"].dropped_columns


# ---------------------------------------------------------------------------
# Train-only fitting
# ---------------------------------------------------------------------------


def test_scaler_statistics_come_from_train_only(processed) -> None:
    """Scaler mean/scale equal statistics of the raw training split."""
    result, _, _, _ = processed
    preprocessor: FittedPreprocessor = result["preprocessor"]
    raw_train = result["raw_splits"]["train"]
    config = _config()

    # Refitting on the train split reproduces the persisted scaler exactly.
    from src.data.preprocess import fit_preprocessor

    refit = fit_preprocessor(raw_train, config, DATASET)
    assert refit.feature_columns == preprocessor.feature_columns
    np.testing.assert_allclose(
        preprocessor.scaler.mean_, refit.scaler.mean_, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        preprocessor.scaler.scale_, refit.scaler.scale_, rtol=0, atol=1e-12
    )

    # And it matches an independent computation over the train split.
    index = preprocessor.feature_columns.index("flow_duration")
    expected_mean = float(raw_train["flow_duration"].mean())
    assert preprocessor.scaler.mean_[index] == pytest.approx(expected_mean, rel=1e-12)
    expected_std = float(raw_train["flow_duration"].std(ddof=0))
    assert preprocessor.scaler.scale_[index] == pytest.approx(expected_std, rel=1e-12)


def test_holdout_rows_do_not_influence_imputation(tmp_path: Path) -> None:
    """Imputation means exclude the zero-day holdout (extreme by construction)."""
    df = _make_fixture()
    config = _config(zeroday=["HEARTBLEED"], impute_strategy="mean")
    result = run_preprocessing(
        DATASET,
        df,
        config=config,
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    preprocessor: FittedPreprocessor = result["preprocessor"]
    raw_train = result["raw_splits"]["train"]
    assert preprocessor.impute_values["flow_duration"] == pytest.approx(
        float(raw_train["flow_duration"].mean()), rel=1e-12
    )
    # The holdout rows sit around 1e9; including them would shift the mean far.
    assert not np.isclose(preprocessor.impute_values["flow_duration"], 1e8)


def test_transform_applies_train_statistics_to_other_splits(processed) -> None:
    """Scaled val rows use the train mean, not their own mean."""
    result, _, _, _ = processed
    val_scaled = result["splits"]["val"]
    feature = "flow_duration"
    assert abs(float(val_scaled[feature].mean())) < 0.5  # centred on train mean


# ---------------------------------------------------------------------------
# Artifacts: preprocessor, features, class weights, report
# ---------------------------------------------------------------------------


def test_artifacts_are_saved_and_reloadable(processed) -> None:
    """The preprocessor, feature list and class weights are persisted."""
    result, _, models_dir, _ = processed
    preprocessor_path = models_dir / f"{DATASET}_preprocessor.joblib"
    assert preprocessor_path.is_file()
    reloaded: FittedPreprocessor = joblib.load(preprocessor_path)
    assert reloaded.feature_columns == result["preprocessor"].feature_columns

    import json

    features = json.loads((models_dir / f"{DATASET}_features.json").read_text())
    assert features["feature_columns"] == result["preprocessor"].feature_columns
    assert set(features["feature_dtypes"]) == set(features["feature_columns"])
    assert features["label_columns"] == ["multiclass_label", "binary_label"]

    weights_path = models_dir / f"{DATASET}_class_weights.json"
    assert weights_path.is_file()
    weights = json.loads(weights_path.read_text())
    assert set(weights) == set(result["splits"]["train"][MULTICLASS_COLUMN].unique())


def test_class_weights_are_balanced_and_do_not_oversample(processed) -> None:
    """Balanced weights come from train counts; row counts are untouched."""
    result, _, _, _ = processed
    train = result["raw_splits"]["train"]
    weights = result["class_weights"]
    counts = train[MULTICLASS_COLUMN].value_counts()
    n_classes = len(counts)
    n_train = len(train)
    for label, weight in weights.items():
        expected = n_train / (n_classes * counts[label])
        assert weight == pytest.approx(expected, rel=1e-9)
    # No oversampling: rarer classes get larger weights, and the modelling rows
    # still sum to the cleaned pool minus the zero-day holdout.
    ordered = counts.sort_values(ascending=False)
    assert weights[ordered.index[-1]] > weights[ordered.index[0]]
    assert sum(len(result["splits"][name]) for name in SPLIT_NAMES) == (
        result["cleaning"]["rows_out"] - len(result["holdout"])
    )


def test_split_summary_lists_class_counts_per_split(processed) -> None:
    """reports/split_summary.md contains each split and class counts."""
    result, _, _, reports_dir = processed
    text = (reports_dir / "split_summary.md").read_text(encoding="utf-8")
    assert f"## Dataset: `{DATASET}`" in text
    for name in SPLIT_NAMES:
        assert name in text
    for label in ("Normal", "DDoS", "PortScan/Recon"):
        assert f"`{label}`" in text
    assert "destination_port" in text  # documented retained shortcut risk

    import json

    summary = json.loads((reports_dir / "split_summary.json").read_text())
    assert summary[DATASET]["rows"]["train"] == len(result["splits"]["train"])
    assert summary[DATASET]["zeroday_classes"] == ["HEARTBLEED"]


def test_unmapped_label_raises(fixture_df: pd.DataFrame) -> None:
    """A label missing from the map raises instead of being guessed."""
    config = _config()
    labelled = clean_structural(fixture_df, config, DATASET)[0]
    labelled.loc[labelled.index[:3], "label"] = "SOME UNSEEN ATTACK"
    with pytest.raises(ValueError, match="unmapped"):
        apply_label_map(labelled, DATASET, load_label_map(config))
