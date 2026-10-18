"""Dataset loaders for CICIDS2017 (MachineLearningCVE) and CICIoT2023.

Usage
-----
Run directly to check data availability and print download instructions::

    python -m src.data.loader --dataset cicids2017
    python -m src.data.loader --dataset ciciot2023

Or import programmatically::

    from src.data.loader import load_cicids2017, load_ciciot2023

Both loaders:
- normalise column names (strip whitespace, convert to snake_case)
- handle dataset-specific quirks (see individual function docstrings)
- optionally sample N rows for fast development (``dev_sample`` config option)
- save cleaned data to ``data/interim/<dataset>.parquet``

Rules (AGENTS.md):
- No data is invented or synthesised.
- If a dataset folder is missing or empty, download instructions are printed
  and a ``DataNotFoundError`` is raised.
- Seeds come from ``configs/config.yaml`` (``seed: 42``).
"""

from __future__ import annotations

import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataNotFoundError(FileNotFoundError):
    """Raised when a required dataset directory is absent or empty.

    Args:
        dataset: human-readable dataset name (e.g. ``"CICIDS2017"``).
        path: the expected local directory path.
        instructions: download instructions to print.
    """

    def __init__(self, dataset: str, path: Path, instructions: str) -> None:
        self.dataset = dataset
        self.path = path
        self.instructions = instructions
        super().__init__(str(path))


# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------

_SNAKE_MULTI = re.compile(r"[\s\-/\\()\[\]]+")
_SNAKE_TRAIL = re.compile(r"_+$")


def _to_snake(name: str) -> str:
    """Convert a raw column name to snake_case.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Replace runs of whitespace, hyphens, slashes, and brackets with ``_``.
    3. Collapse multiple underscores and strip trailing underscores.
    4. Lower-case the result.

    Args:
        name: raw column name string.

    Returns:
        Normalised snake_case column name.
    """
    name = name.strip()
    name = _SNAKE_MULTI.sub("_", name)
    name = _SNAKE_TRAIL.sub("", name)
    return name.lower()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to snake_case in-place and return the DataFrame.

    Args:
        df: DataFrame whose column names may contain raw whitespace or mixed
            casing.

    Returns:
        The same DataFrame with renamed columns (mutated).
    """
    df.columns = [_to_snake(c) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# CICIDS2017 download instructions
# ---------------------------------------------------------------------------

_CICIDS2017_INSTRUCTIONS = textwrap.dedent(
    """\
    CICIDS2017 -- Dataset not found
    ================================
    Expected location:  data/raw/cicids2017/  (one or more *.csv files)

    Download instructions:

    1. Go to the official UNB CIC page:
       https://www.unb.ca/cic/datasets/ids-2017.html

    2. Register (free) and download the
       "MachineLearningCVE" ZIP (~450 MB, 8 daily CSV files).
       Direct mirror (as of 2024):
       http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCVE.zip

    3. Extract all *.csv files into:
       data/raw/cicids2017/

    Expected files (one per capture day):
       Monday-WorkingHours.pcap_ISCX.csv
       Tuesday-WorkingHours.pcap_ISCX.csv
       Wednesday-workingHours.pcap_ISCX.csv
       Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
       Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
       Friday-WorkingHours-Morning.pcap_ISCX.csv
       Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
       Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

    Licence: CC BY 4.0 -- cite the UNB CIC paper.
    """
)

# ---------------------------------------------------------------------------
# CICIoT2023 download instructions
# ---------------------------------------------------------------------------

_CICIOT2023_INSTRUCTIONS = textwrap.dedent(
    """\
    CICIoT2023 -- Dataset not found
    =================================
    Expected location:  data/raw/ciciot2023/  (one or more *.csv files)

    Download instructions:

    Option A -- Official UNB CIC page:
       https://www.unb.ca/cic/datasets/iotdataset-2023.html
       Register (free) and download the preprocessed CSV archive.

    Option B -- Mendeley Data (DOI: 10.17632/7zcswnmkfd):
       https://data.mendeley.com/datasets/7zcswnmkfd
       Click "Download all" to obtain all part-CSVs (approx. 8.6 GB total).

    Option C -- Kaggle community mirror (smaller pre-cleaned subset):
       https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset-2023
       kaggle datasets download -d madhavmalhotra/unb-cic-iot-dataset-2023

    3. Place all extracted *.csv files in:
       data/raw/ciciot2023/

    Licence: CC BY 4.0 -- cite the UNB CIC IoT 2023 paper.
    """
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_dataset_dir(dataset_name: str, dir_path: Path, instructions: str) -> list[Path]:
    """Return a sorted list of CSV files inside ``dir_path``.

    Args:
        dataset_name: human-readable name used in error messages.
        dir_path: absolute path expected to contain ``*.csv`` files.
        instructions: multiline download-instructions string to print.

    Returns:
        List of ``Path`` objects for every ``*.csv`` file found.

    Raises:
        DataNotFoundError: if the directory does not exist or contains no CSVs.
    """
    if not dir_path.is_dir():
        print(instructions, file=sys.stderr)
        raise DataNotFoundError(dataset_name, dir_path, instructions)

    csvs = sorted(dir_path.glob("*.csv"))
    if not csvs:
        print(instructions, file=sys.stderr)
        raise DataNotFoundError(dataset_name, dir_path, instructions)

    return csvs


def _read_csvs(
    paths: list[Path],
    *,
    encoding: str = "utf-8",
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a list of CSV files into a single concatenated DataFrame.

    Args:
        paths: CSV file paths to load (in order).
        encoding: character encoding to try first.
        low_memory: passed directly to ``pd.read_csv``.

    Returns:
        Concatenated DataFrame with a reset integer index.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        logger.info("Reading %s ...", p.name)
        try:
            chunk = pd.read_csv(p, encoding=encoding, low_memory=low_memory)
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed for %s; retrying with latin-1", p.name)
            chunk = pd.read_csv(p, encoding="latin-1", low_memory=low_memory)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CICIDS2017 loader
# ---------------------------------------------------------------------------

#: Known raw label strings from the MachineLearningCVE files.
#: These are the values seen *after* stripping whitespace and upper-casing.
#: They are documented here for reference; actual normalisation is applied
#: to the raw column before comparison.
CICIDS2017_KNOWN_LABELS: frozenset[str] = frozenset(
    {
        "BENIGN",
        "FTP-PATATOR",
        "SSH-PATATOR",
        "DOS HULK",
        "DOS GOLDENEYE",
        "DOS SLOWLORIS",
        "DOS SLOWHTTPTEST",
        "HEARTBLEED",
        "WEB ATTACK - BRUTE FORCE",
        "WEB ATTACK - XSS",
        "WEB ATTACK - SQL INJECTION",
        "INFILTRATION",
        "BOT",
        "PORTSCAN",
        "DDOS",
    }
)


def _fix_cicids2017_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle CICIDS2017-specific column quirks.

    Quirks addressed (in order):
    1. Leading/trailing whitespace in column names (CICFlowMeter artefact).
    2. Duplicated ``'Fwd Header Length'`` column -- pandas renames the second
       copy to ``'Fwd Header Length.1'`` (or the snake_case equivalent
       ``fwd_header_length_1``); the first and second are verified identical
       and the duplicate is dropped.

    Args:
        df: DataFrame loaded directly from the raw CSVs.

    Returns:
        DataFrame with cleaned column names and without the duplicate column.
    """
    # Step 1: strip whitespace on raw names *before* converting to snake_case,
    #         because the snake converter also strips, but we log the originals.
    raw_names_stripped = [c.strip() for c in df.columns]
    if list(df.columns) != raw_names_stripped:
        n_dirty = sum(c != s for c, s in zip(df.columns, raw_names_stripped))
        logger.info(
            "CICIDS2017: stripped leading/trailing whitespace from %d column names.", n_dirty
        )

    # Apply snake_case normalisation.
    df = normalise_columns(df)

    # Step 2: handle the Fwd Header Length duplicate.
    dup_col = "fwd_header_length_1"
    orig_col = "fwd_header_length"
    if dup_col in df.columns:
        if orig_col in df.columns:
            # Verify they are identical (ignoring NaN positions).
            are_equal = df[orig_col].equals(df[dup_col])
            if not are_equal:
                logger.warning(
                    "CICIDS2017: 'fwd_header_length' and 'fwd_header_length_1' differ! "
                    "Keeping 'fwd_header_length' and dropping the duplicate anyway."
                )
            else:
                logger.info(
                    "CICIDS2017: confirmed 'fwd_header_length_1' == 'fwd_header_length';"
                    " dropping duplicate."
                )
        df = df.drop(columns=[dup_col])

    return df


def _fix_cicids2017_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``label`` column in CICIDS2017.

    Quirks addressed:
    - Leading/trailing whitespace in label values (e.g. ``" BENIGN "``)
    - Unicode en-dash / em-dash (vs ASCII hyphen) in web-attack labels
    - Inconsistent casing (all upper-cased after normalisation)
    - Embedded newlines in some files (e.g. ``"Web Attack \\r\\n Brute Force"``)

    Args:
        df: DataFrame with a normalised ``label`` column.

    Returns:
        DataFrame with a cleaned ``label`` column.
    """
    if "label" not in df.columns:
        logger.warning("CICIDS2017: no 'label' column found after normalisation.")
        return df

    series = df["label"].astype(str)
    # Strip whitespace and newlines.
    series = series.str.strip().str.replace(r"\s+", " ", regex=True)
    # Normalise em-dash / en-dash to ASCII hyphen " - ".
    series = series.str.replace("\u2013", "-", regex=False)  # en-dash
    series = series.str.replace("\u2014", "-", regex=False)  # em-dash
    # Upper-case for uniform comparison.
    series = series.str.upper()
    # Collapse multiple spaces that may have appeared after dash replacement.
    series = series.str.replace(r"  +", " ", regex=True)
    df["label"] = series
    return df


def _replace_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``+-inf`` with ``NaN`` across all numeric columns.

    CICIDS2017 rate-based features (``flow_bytes_s``, ``flow_packets_s``)
    frequently contain ``numpy.inf`` from zero-duration flows.  Converting
    them to ``NaN`` makes imputation straightforward in later stages.

    Args:
        df: DataFrame potentially containing infinite values.

    Returns:
        DataFrame with ``+-inf`` replaced by ``NaN`` in every numeric column.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    inf_mask = np.isinf(df[numeric_cols].values)
    n_inf = int(inf_mask.sum())
    if n_inf > 0:
        logger.info("Replaced %d inf values with NaN.", n_inf)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def load_cicids2017(
    config_path: str | Path | None = None,
    *,
    dev_sample: int | None = None,
) -> pd.DataFrame:
    """Load all CICIDS2017 MachineLearningCVE CSV files into one DataFrame.

    The following quirks are handled automatically:
    1. Leading/trailing whitespace in raw column names.
    2. Duplicated ``Fwd Header Length`` column (verified identical; duplicate
       dropped).
    3. ``+-inf`` values replaced with ``NaN``.
    4. Inconsistent label strings normalised (whitespace, em-dash, upper-case).
    5. Mixed character encodings (UTF-8 with fallback to latin-1).

    After cleaning, the DataFrame is saved to ``data/interim/cicids2017.parquet``.

    Args:
        config_path: path to the YAML config file. If ``None``, the default
            ``configs/config.yaml`` is used.
        dev_sample: if not ``None``, randomly sample this many rows
            (reproducible via ``seed`` from config). Overrides the config
            ``data.dev_sample`` key if present.

    Returns:
        Cleaned DataFrame with snake_case column names.

    Raises:
        DataNotFoundError: if ``data/raw/cicids2017/`` is absent or empty.
    """
    cfg = load_config(config_path)
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "cicids2017"
    interim_dir = root / cfg.paths.interim
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Check data presence -- prints instructions and raises if missing.
    csvs = _check_dataset_dir("CICIDS2017", raw_dir, _CICIDS2017_INSTRUCTIONS)
    logger.info("CICIDS2017: found %d CSV file(s) in %s", len(csvs), raw_dir)

    # Load all files.
    df = _read_csvs(csvs)
    logger.info("CICIDS2017: raw shape = %s", df.shape)

    # Fix column names and dataset quirks.
    df = _fix_cicids2017_columns(df)
    df = _replace_inf_nan(df)
    df = _fix_cicids2017_labels(df)

    # Optional dev sample.
    sample_n = dev_sample or getattr(cfg.data, "dev_sample", None)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=cfg.seed).reset_index(drop=True)
        logger.info("CICIDS2017: sampled %d rows (seed=%d).", sample_n, cfg.seed)

    logger.info("CICIDS2017: final shape = %s", df.shape)

    # Persist to interim parquet.
    out_path = interim_dir / "cicids2017.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("CICIDS2017: saved to %s", out_path)

    return df


# ---------------------------------------------------------------------------
# CICIoT2023 loader
# ---------------------------------------------------------------------------

#: Known raw label strings from CICIoT2023 (based on the published dataset
#: paper and community analysis). All values are case-sensitive as found in
#: the CSVs; they are normalised to snake_case ``label`` after loading.
CICIOT2023_KNOWN_LABELS: frozenset[str] = frozenset(
    {
        "BenignTraffic",
        "DDoS-ACK_Fragmentation",
        "DDoS-UDP_Fragmentation",
        "DDoS-ICMP_Fragmentation",
        "DDoS-SlowLoris",
        "DDoS-HTTP_Flood",
        "DDoS-TCP_Flood",
        "DDoS-UDP_Flood",
        "DDoS-ICMP_Flood",
        "DDoS-PSHACK_Flood",
        "DDoS-SYN_Flood",
        "DDoS-RSTFINFlood",
        "DDoS-SynonymousIP_Flood",
        "DoS-UDP_Flood",
        "DoS-TCP_Flood",
        "DoS-SYN_Flood",
        "DoS-HTTP_Flood",
        "Recon-HostDiscovery",
        "Recon-OSScan",
        "Recon-PortScan",
        "Recon-PingSweep",
        "VulnerabilityScan",
        "Mirai-greip_flood",
        "Mirai-greeth_flood",
        "Mirai-udpplain",
        "MITM-ArpSpoofing",
        "DNS_Spoofing",
        "Backdoor_Malware",
        "CommandInjection",
        "SqlInjection",
        "XSS",
        "Uploading_Attack",
        "BruteForce-SSH",
        "BruteForce-Web",
        "DictionaryBruteForce",
    }
)


def _fix_ciciot2023_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise CICIoT2023 column names to snake_case.

    CICIoT2023 column names use a mix of camelCase, PascalCase and mixed
    styles (e.g. ``'Header_Length'``, ``'Protocol Type'``, ``'IAT'``).
    This function normalises them uniformly.

    Args:
        df: raw DataFrame from the CSV files.

    Returns:
        DataFrame with snake_case column names.
    """
    return normalise_columns(df)


def _fix_ciciot2023_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``label`` column in CICIoT2023.

    Values are stripped of whitespace but otherwise preserved verbatim since
    they are already well-structured (PascalCase/mixed attack names).

    Args:
        df: DataFrame with column names already normalised.

    Returns:
        DataFrame with cleaned ``label`` values.
    """
    if "label" not in df.columns:
        logger.warning("CICIoT2023: no 'label' column found after normalisation.")
        return df

    df["label"] = df["label"].astype(str).str.strip()
    return df


def load_ciciot2023(
    config_path: str | Path | None = None,
    *,
    dev_sample: int | None = None,
) -> pd.DataFrame:
    """Load all CICIoT2023 CSV files into one DataFrame.

    Column names are normalised to snake_case.  The ``label`` column is
    stripped of whitespace.  ``+-inf`` values are replaced with ``NaN``.
    Data is saved to ``data/interim/ciciot2023.parquet``.

    Args:
        config_path: path to the YAML config file. If ``None``, the default
            ``configs/config.yaml`` is used.
        dev_sample: if not ``None``, randomly sample this many rows
            (reproducible via ``seed`` from config).

    Returns:
        Cleaned DataFrame with snake_case column names.

    Raises:
        DataNotFoundError: if ``data/raw/ciciot2023/`` is absent or empty.
    """
    cfg = load_config(config_path)
    root = PROJECT_ROOT
    raw_dir = root / "data" / "raw" / "ciciot2023"
    interim_dir = root / cfg.paths.interim
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Check data presence -- prints instructions and raises if missing.
    csvs = _check_dataset_dir("CICIoT2023", raw_dir, _CICIOT2023_INSTRUCTIONS)
    logger.info("CICIoT2023: found %d CSV file(s) in %s", len(csvs), raw_dir)

    # Load all files.
    df = _read_csvs(csvs)
    logger.info("CICIoT2023: raw shape = %s", df.shape)

    # Fix column names and quirks.
    df = _fix_ciciot2023_columns(df)
    df = _replace_inf_nan(df)
    df = _fix_ciciot2023_labels(df)

    # Optional dev sample.
    sample_n = dev_sample or getattr(cfg.data, "dev_sample", None)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=cfg.seed).reset_index(drop=True)
        logger.info("CICIoT2023: sampled %d rows (seed=%d).", sample_n, cfg.seed)

    logger.info("CICIoT2023: final shape = %s", df.shape)

    # Persist to interim parquet.
    out_path = interim_dir / "ciciot2023.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("CICIoT2023: saved to %s", out_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Load a dataset and save to data/interim/.")
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "ciciot2023"],
        required=True,
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--dev-sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N rows for fast development (optional).",
    )
    args = parser.parse_args()

    try:
        if args.dataset == "cicids2017":
            df = load_cicids2017(args.config, dev_sample=args.dev_sample)
        else:
            df = load_ciciot2023(args.config, dev_sample=args.dev_sample)
        print(f"Loaded {len(df):,} rows x {df.shape[1]} columns. Label counts:")
        if "label" in df.columns:
            print(df["label"].value_counts().to_string())
    except DataNotFoundError:
        # Instructions already printed by _check_dataset_dir.
        sys.exit(1)
