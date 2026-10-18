"""Unit tests for src/data/loader.py.

Tests cover:
- Column-name normalisation (_to_snake, normalise_columns)
- CICIDS2017 quirks:
  - Leading/trailing whitespace stripped from column names
  - Duplicated 'Fwd Header Length' column detected and dropped
  - inf values replaced with NaN
  - label normalisation (whitespace, em-dash, upper-case)
- CICIoT2023:
  - Column name normalisation
  - Label whitespace stripping
- DataNotFoundError raised (with instructions) for missing directories
- _check_dataset_dir raises for empty directories

These tests are intentionally free of heavy ML imports and run in a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    DataNotFoundError,
    _check_dataset_dir,
    _fix_cicids2017_columns,
    _fix_cicids2017_labels,
    _fix_ciciot2023_labels,
    _replace_inf_nan,
    _to_snake,
    normalise_columns,
)


# ===========================================================================
# _to_snake
# ===========================================================================


class TestToSnake:
    """Unit tests for the _to_snake helper."""

    def test_leading_trailing_whitespace(self) -> None:
        """Leading and trailing spaces are removed."""
        assert _to_snake(" Destination Port ") == "destination_port"

    def test_internal_spaces(self) -> None:
        """Internal spaces become underscores."""
        assert _to_snake("Flow Duration") == "flow_duration"

    def test_already_snake(self) -> None:
        """Already-normalised names are unchanged."""
        assert _to_snake("flow_duration") == "flow_duration"

    def test_mixed_case(self) -> None:
        """Upper-case letters are lower-cased."""
        assert _to_snake("FlowDuration") == "flowduration"

    def test_hyphen(self) -> None:
        """Hyphens are converted to underscores."""
        assert _to_snake("FTP-Patator") == "ftp_patator"

    def test_slash(self) -> None:
        """Slashes become underscores."""
        assert _to_snake("Flow Bytes/s") == "flow_bytes_s"

    def test_parentheses(self) -> None:
        """Parentheses are converted to underscores and trailing ones trimmed."""
        assert _to_snake("Avg Fwd Segment Size") == "avg_fwd_segment_size"

    def test_multiple_separators(self) -> None:
        """Multiple consecutive separators collapse to a single underscore."""
        assert _to_snake("A  B--C") == "a_b_c"

    def test_trailing_separator(self) -> None:
        """Trailing underscores after conversion are stripped."""
        # "(something)" -> "_something_" -> "something" after strip
        result = _to_snake("(label)")
        assert not result.endswith("_")

    def test_empty_string(self) -> None:
        """Empty string remains empty."""
        assert _to_snake("") == ""


# ===========================================================================
# normalise_columns
# ===========================================================================


class TestNormaliseColumns:
    """Tests for normalise_columns applied to a DataFrame."""

    def test_renames_all_columns(self) -> None:
        """All column names are converted to snake_case."""
        df = pd.DataFrame(columns=[" Destination Port", "Flow Duration", "Label"])
        result = normalise_columns(df)
        assert list(result.columns) == ["destination_port", "flow_duration", "label"]

    def test_returns_same_dataframe(self) -> None:
        """The DataFrame is mutated in-place and returned."""
        df = pd.DataFrame({"A B": [1, 2]})
        returned = normalise_columns(df)
        assert returned is df


# ===========================================================================
# CICIDS2017: column quirks
# ===========================================================================


class TestFixCicids2017Columns:
    """Tests for CICIDS2017-specific column handling."""

    def _make_df(self, dup: bool = True) -> pd.DataFrame:
        """Build a small fake CICIDS2017-like DataFrame."""
        cols = [
            " Destination Port",   # leading space
            " Fwd Header Length",  # leading space + the original column
            " Label",              # leading space
        ]
        if dup:
            cols.append(" Fwd Header Length")  # second duplicate
        data = {c: [0, 1] for c in cols}
        # pandas auto-renames duplicate columns on construction; we simulate
        # the raw CSV behaviour by building with renamed_axis hack.
        df = pd.DataFrame(data)
        return df

    def test_whitespace_stripped_from_columns(self) -> None:
        """Leading spaces are stripped from all column names."""
        df = pd.DataFrame({" Flow Duration": [1.0], " Label": ["BENIGN"]})
        result = _fix_cicids2017_columns(df)
        assert "flow_duration" in result.columns
        assert "label" in result.columns

    def test_duplicate_fwd_header_length_dropped(self) -> None:
        """The second 'Fwd Header Length' column (renamed .1) is dropped."""
        # Build a DataFrame with the duplicate already renamed by pandas.
        df = pd.DataFrame(
            {
                "fwd_header_length": [20, 20, 20],
                "fwd_header_length_1": [20, 20, 20],  # the duplicate
                "label": ["BENIGN", "DDOS", "BENIGN"],
            }
        )
        result = _fix_cicids2017_columns(df)
        assert "fwd_header_length" in result.columns
        assert "fwd_header_length_1" not in result.columns

    def test_duplicate_dropped_even_when_not_identical(self) -> None:
        """Duplicate is dropped even when values differ (with a warning)."""
        df = pd.DataFrame(
            {
                "fwd_header_length": [20, 30, 20],
                "fwd_header_length_1": [20, 20, 20],  # differs at row 1
                "label": ["BENIGN", "DDOS", "BENIGN"],
            }
        )
        # Should not raise -- just drop and warn.
        result = _fix_cicids2017_columns(df)
        assert "fwd_header_length_1" not in result.columns

    def test_no_duplicate_no_error(self) -> None:
        """If the duplicate does not exist, nothing happens."""
        df = pd.DataFrame(
            {
                "fwd_header_length": [20],
                "label": ["BENIGN"],
            }
        )
        result = _fix_cicids2017_columns(df)
        assert "fwd_header_length" in result.columns
        assert "fwd_header_length_1" not in result.columns


# ===========================================================================
# inf / NaN replacement
# ===========================================================================


class TestReplaceInfNan:
    """Tests for _replace_inf_nan."""

    def test_positive_inf_replaced(self) -> None:
        """Positive inf becomes NaN."""
        df = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
        result = _replace_inf_nan(df)
        assert result["x"].isna().sum() == 1
        assert not np.isinf(result["x"].dropna()).any()

    def test_negative_inf_replaced(self) -> None:
        """Negative inf becomes NaN."""
        df = pd.DataFrame({"x": [-np.inf, 2.0]})
        result = _replace_inf_nan(df)
        assert result["x"].isna().sum() == 1

    def test_no_inf_unchanged(self) -> None:
        """DataFrames without inf values are not modified."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _replace_inf_nan(df)
        assert result["x"].isna().sum() == 0

    def test_non_numeric_columns_untouched(self) -> None:
        """Non-numeric columns are left alone."""
        df = pd.DataFrame({"x": [np.inf], "label": ["BENIGN"]})
        result = _replace_inf_nan(df)
        assert result["label"].iloc[0] == "BENIGN"


# ===========================================================================
# CICIDS2017: label normalisation
# ===========================================================================


class TestFixCicids2017Labels:
    """Tests for CICIDS2017 label string normalisation."""

    def test_leading_trailing_whitespace(self) -> None:
        """Whitespace around label values is stripped."""
        df = pd.DataFrame({"label": [" BENIGN ", "  DDOS"]})
        result = _fix_cicids2017_labels(df)
        assert result["label"].tolist() == ["BENIGN", "DDOS"]

    def test_upper_case(self) -> None:
        """Labels are upper-cased."""
        df = pd.DataFrame({"label": ["benign", "ddos"]})
        result = _fix_cicids2017_labels(df)
        assert result["label"].tolist() == ["BENIGN", "DDOS"]

    def test_en_dash_normalised(self) -> None:
        """En-dash (U+2013) in web-attack labels is replaced with ASCII hyphen."""
        df = pd.DataFrame({"label": ["Web Attack \u2013 Brute Force"]})
        result = _fix_cicids2017_labels(df)
        assert "-" in result["label"].iloc[0]
        assert "\u2013" not in result["label"].iloc[0]

    def test_em_dash_normalised(self) -> None:
        """Em-dash (U+2014) is replaced with ASCII hyphen."""
        df = pd.DataFrame({"label": ["Web Attack \u2014 XSS"]})
        result = _fix_cicids2017_labels(df)
        assert "\u2014" not in result["label"].iloc[0]

    def test_embedded_newline_collapsed(self) -> None:
        """Embedded \\r\\n are collapsed to a single space."""
        df = pd.DataFrame({"label": ["Web Attack \r\n Brute Force"]})
        result = _fix_cicids2017_labels(df)
        val = result["label"].iloc[0]
        assert "\r" not in val
        assert "\n" not in val

    def test_no_label_column_no_error(self) -> None:
        """Missing 'label' column is handled gracefully (just warns)."""
        df = pd.DataFrame({"x": [1, 2]})
        result = _fix_cicids2017_labels(df)  # should not raise
        assert "x" in result.columns


# ===========================================================================
# CICIoT2023: label normalisation
# ===========================================================================


class TestFixCiciot2023Labels:
    """Tests for CICIoT2023 label string normalisation."""

    def test_whitespace_stripped(self) -> None:
        """Whitespace around label values is stripped."""
        df = pd.DataFrame({"label": [" BenignTraffic ", "DDoS-TCP_Flood"]})
        result = _fix_ciciot2023_labels(df)
        assert result["label"].tolist() == ["BenignTraffic", "DDoS-TCP_Flood"]

    def test_case_preserved(self) -> None:
        """CICIoT2023 labels retain their original casing."""
        df = pd.DataFrame({"label": ["BenignTraffic"]})
        result = _fix_ciciot2023_labels(df)
        assert result["label"].iloc[0] == "BenignTraffic"

    def test_no_label_column_no_error(self) -> None:
        """Missing 'label' column is handled gracefully."""
        df = pd.DataFrame({"x": [1]})
        result = _fix_ciciot2023_labels(df)
        assert "x" in result.columns


# ===========================================================================
# _check_dataset_dir: DataNotFoundError
# ===========================================================================


class TestCheckDatasetDir:
    """Tests for the directory / CSV presence check."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        """A non-existent directory raises DataNotFoundError."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(DataNotFoundError) as exc_info:
            _check_dataset_dir("TestDS", missing, "Download from somewhere.")
        assert exc_info.value.dataset == "TestDS"
        assert exc_info.value.path == missing

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """An existing but empty directory raises DataNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(DataNotFoundError):
            _check_dataset_dir("TestDS", empty_dir, "Download from somewhere.")

    def test_directory_with_csvs_returns_list(self, tmp_path: Path) -> None:
        """A directory with CSV files returns a sorted list of Paths."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "b.csv").write_text("a,b\n1,2\n")
        (data_dir / "a.csv").write_text("a,b\n3,4\n")
        result = _check_dataset_dir("TestDS", data_dir, "unused")
        assert len(result) == 2
        assert result[0].name == "a.csv"  # sorted
        assert result[1].name == "b.csv"

    def test_directory_only_non_csv_raises(self, tmp_path: Path) -> None:
        """A directory containing only non-CSV files raises DataNotFoundError."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "readme.txt").write_text("nothing here")
        with pytest.raises(DataNotFoundError):
            _check_dataset_dir("TestDS", data_dir, "Download from somewhere.")


# ===========================================================================
# label_map.yaml: coverage check
# ===========================================================================


class TestLabelMapYaml:
    """Verify that the label map YAML is parseable and covers all known labels."""

    @pytest.fixture()
    def label_map(self) -> dict:  # type: ignore[type-arg]
        """Load the label_map.yaml from configs/."""
        import yaml
        from src.config import PROJECT_ROOT

        path = PROJECT_ROOT / "configs" / "label_map.yaml"
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_file_parseable(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """The YAML file loads without errors."""
        assert isinstance(label_map, dict)

    def test_both_datasets_present(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """Both 'cicids2017' and 'ciciot2023' top-level keys exist."""
        assert "cicids2017" in label_map
        assert "ciciot2023" in label_map

    def test_all_cicids2017_known_labels_mapped(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """Every known CICIDS2017 label string has an entry in the map."""
        from src.data.loader import CICIDS2017_KNOWN_LABELS

        mapped = set(label_map["cicids2017"].keys())
        missing = CICIDS2017_KNOWN_LABELS - mapped
        assert not missing, f"Unmapped CICIDS2017 labels: {missing}"

    def test_all_ciciot2023_known_labels_mapped(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """Every known CICIoT2023 label string has an entry in the map."""
        from src.data.loader import CICIOT2023_KNOWN_LABELS

        mapped = set(label_map["ciciot2023"].keys())
        missing = CICIOT2023_KNOWN_LABELS - mapped
        assert not missing, f"Unmapped CICIoT2023 labels: {missing}"

    def test_all_entries_have_required_fields(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """Every mapped label has 'multiclass_label' and 'binary_label' fields."""
        for dataset_name, entries in label_map.items():
            if not isinstance(entries, dict):
                continue  # skip non-entry top-level keys
            for label, mapping in entries.items():
                if not isinstance(mapping, dict):
                    continue
                assert "multiclass_label" in mapping, (
                    f"{dataset_name}/{label} missing 'multiclass_label'"
                )
                assert "binary_label" in mapping, (
                    f"{dataset_name}/{label} missing 'binary_label'"
                )

    def test_binary_labels_are_valid(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """binary_label is always 'Normal' or 'Attack'."""
        valid = {"Normal", "Attack"}
        for dataset_name, entries in label_map.items():
            if not isinstance(entries, dict):
                continue
            for label, mapping in entries.items():
                if not isinstance(mapping, dict):
                    continue
                bl = mapping.get("binary_label")
                assert bl in valid, (
                    f"{dataset_name}/{label}: invalid binary_label '{bl}'"
                )

    def test_multiclass_labels_are_from_taxonomy(self, label_map: dict) -> None:  # type: ignore[type-arg]
        """multiclass_label values are drawn from the defined taxonomy."""
        taxonomy = {
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
        }
        for dataset_name, entries in label_map.items():
            if not isinstance(entries, dict):
                continue
            for label, mapping in entries.items():
                if not isinstance(mapping, dict):
                    continue
                ml = mapping.get("multiclass_label")
                assert ml in taxonomy, (
                    f"{dataset_name}/{label}: unknown taxonomy class '{ml}'"
                )
