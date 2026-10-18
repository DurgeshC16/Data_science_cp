"""Exploratory Data Analysis for CICIDS2017 and CICIoT2023.

Produces:
- ``reports/eda_<dataset>.md``: shape, dtypes, class counts, missing/inf counts,
  constant columns, duplicate rows, highly correlated pairs (threshold 0.95),
  and 10 key feature distributions per class.
- ``reports/figures/eda_<dataset>_class_dist.png``: bar chart of class counts
  (log scale).
- ``reports/figures/eda_<dataset>_feat_<feature>.png``: per-class KDE/histogram
  for each of the 10 selected key features.

Usage (CLI)::

    python -m src.data.eda --dataset cicids2017
    python -m src.data.eda --dataset ciciot2023

Rules (AGENTS.md):
- No numbers are invented. All statistics come from the actual data.
- If parquet is missing (loader not yet run) or data folder is empty, the
  script reports that and exits without fabricating results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for server/CI use

from src.config import PROJECT_ROOT, load_config
from src.data.loader import (
    DataNotFoundError,
    load_cicids2017,
    load_ciciot2023,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key features to plot per dataset (selected from published feature importance
# studies; the actual columns present are verified at runtime)
# ---------------------------------------------------------------------------

# For CICIDS2017 (snake_case names after normalisation)
_CICIDS2017_KEY_FEATURES: list[str] = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_bytes_s",
    "flow_packets_s",
    "fwd_packet_length_max",
    "bwd_packet_length_max",
    "flow_iat_mean",
    "fwd_iat_mean",
    "packet_length_mean",
]

# For CICIoT2023 (snake_case names after normalisation)
_CICIOT2023_KEY_FEATURES: list[str] = [
    "flow_duration",
    "header_length",
    "rate",
    "srate",
    "drate",
    "iat",
    "magnitude",
    "radius",
    "variance",
    "weight",
]

# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------


def _highly_correlated_pairs(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """Find pairs of numeric features with |Pearson correlation| > threshold.

    Only the upper triangle is returned (each pair appears once).  Large
    datasets are sub-sampled to 50 000 rows to keep this tractable.

    Args:
        df: DataFrame (label column excluded before call).
        threshold: minimum absolute correlation to report.

    Returns:
        List of (col_a, col_b, correlation) tuples, sorted by descending |r|.
    """
    numeric = df.select_dtypes(include="number")
    if len(numeric) > 50_000:
        numeric = numeric.sample(50_000, random_state=42)
    # Drop constant columns to avoid NaN correlation.
    numeric = numeric.loc[:, numeric.std() > 0]
    corr_matrix = numeric.corr()
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr_matrix.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            r = corr_matrix.loc[col_a, col_b]
            if abs(r) > threshold:
                pairs.append((col_a, col_b, float(r)))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _plot_class_dist(
    df: pd.DataFrame,
    dataset_name: str,
    figures_dir: Path,
) -> Path:
    """Bar chart of class counts on a log scale.

    Args:
        df: DataFrame containing a ``label`` column.
        dataset_name: used in the figure title and filename.
        figures_dir: directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(counts) // 2), 5))
    bars = ax.bar(range(len(counts)), counts.values, color="#4C72B0", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{dataset_name}: class distribution (n={len(df):,})")
    ax.bar_label(bars, fmt="%d", fontsize=6, padding=2)
    fig.tight_layout()
    out = figures_dir / f"eda_{dataset_name}_class_dist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved class distribution figure: %s", out)
    return out


def _plot_feature_distributions(
    df: pd.DataFrame,
    dataset_name: str,
    feature_cols: list[str],
    figures_dir: Path,
    max_classes: int = 10,
    sample_per_class: int = 5_000,
) -> list[Path]:
    """Violin/KDE plot of each key feature, coloured by class.

    For readability, if there are more than ``max_classes`` unique labels, the
    ``max_classes - 1`` largest classes plus "Other" are shown.

    Args:
        df: cleaned DataFrame.
        dataset_name: used in titles/filenames.
        feature_cols: list of feature column names to plot (only those present
            in ``df`` are plotted).
        figures_dir: output directory.
        max_classes: maximum number of classes to show per plot.
        sample_per_class: max rows per class for the plot (performance).

    Returns:
        List of paths to saved figures.
    """
    present = [c for c in feature_cols if c in df.columns and c != "label"]
    if not present:
        logger.warning("None of the requested key features found in DataFrame.")
        return []

    # Build a colour map for classes.
    all_classes = df["label"].value_counts().index.tolist()
    if len(all_classes) > max_classes:
        top_classes = all_classes[: max_classes - 1]
        plot_df = df.copy()
        plot_df["label"] = plot_df["label"].apply(
            lambda x: x if x in top_classes else "Other"
        )
    else:
        plot_df = df.copy()
        top_classes = all_classes

    unique_classes = plot_df["label"].unique().tolist()
    colours = plt.cm.tab20.colors  # type: ignore[attr-defined]
    colour_map = {cls: colours[i % len(colours)] for i, cls in enumerate(unique_classes)}

    saved: list[Path] = []
    for feat in present:
        fig, ax = plt.subplots(figsize=(10, 4))
        for cls in unique_classes:
            subset = plot_df.loc[plot_df["label"] == cls, feat].dropna()
            if len(subset) > sample_per_class:
                subset = subset.sample(sample_per_class, random_state=42)
            if len(subset) < 2:
                continue
            # Clip to the 1st–99th percentile to avoid extreme outliers
            # collapsing the KDE.
            lo, hi = subset.quantile(0.01), subset.quantile(0.99)
            subset = subset.clip(lower=lo, upper=hi)
            try:
                subset.plot.kde(ax=ax, label=cls, color=colour_map[cls], alpha=0.6)
            except Exception:
                ax.hist(
                    subset,
                    bins=40,
                    density=True,
                    label=cls,
                    color=colour_map[cls],
                    alpha=0.4,
                )
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset_name}: {feat}")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        out = figures_dir / f"eda_{dataset_name}_feat_{feat}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        saved.append(out)
    return saved


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def _build_report(
    df: pd.DataFrame,
    dataset_name: str,
    figures_dir: Path,
    corr_threshold: float = 0.95,
) -> str:
    """Generate a Markdown EDA report string.

    Args:
        df: cleaned DataFrame.
        dataset_name: human-readable dataset identifier.
        figures_dir: used for figure references in the Markdown.
        corr_threshold: absolute Pearson correlation cutoff for pairs section.

    Returns:
        Multi-line Markdown string.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# EDA Report: {dataset_name}\n")
    add("_Generated by `src/data/eda.py`.  All statistics are measured from the "
        "actual data; no values are invented._\n")

    # ------------------------------------------------------------------
    # 1. Shape and dtypes
    # ------------------------------------------------------------------
    add("## 1. Shape\n")
    add("| Rows | Columns |")
    add("|------|---------|")
    add(f"| {len(df):,} | {df.shape[1]} |\n")

    add("## 2. Column dtypes\n")
    dtype_counts = df.dtypes.value_counts()
    add("| Dtype | Count |")
    add("|-------|-------|")
    for dtype, cnt in dtype_counts.items():
        add(f"| `{dtype}` | {cnt} |")
    add("")

    # ------------------------------------------------------------------
    # 2. Class counts
    # ------------------------------------------------------------------
    add("## 3. Class distribution\n")
    if "label" in df.columns:
        counts = df["label"].value_counts()
        total = len(df)
        add("| Label | Count | % |")
        add("|-------|-------|---|")
        for lbl, cnt in counts.items():
            add(f"| `{lbl}` | {cnt:,} | {100.0 * cnt / total:.2f} |")
        add("")
        imbalance_ratio = counts.iloc[0] / counts.iloc[-1] if len(counts) > 1 else float("nan")
        add(f"**Imbalance ratio (largest/smallest class):** {imbalance_ratio:.1f}\n")
    else:
        add("_No `label` column found._\n")

    # ------------------------------------------------------------------
    # 3. Missing / inf counts
    # ------------------------------------------------------------------
    add("## 4. Missing and inf values per column\n")
    numeric = df.select_dtypes(include="number")
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]

    if len(missing_nonzero) == 0:
        add("No missing values detected.\n")
    else:
        add(f"**{len(missing_nonzero)} column(s) have missing values:**\n")
        add("| Column | Missing | % |")
        add("|--------|---------|---|")
        for col, cnt in missing_nonzero.sort_values(ascending=False).items():
            add(f"| `{col}` | {cnt:,} | {100.0 * cnt / len(df):.2f} |")
        add("")

    # Inf counts (numeric only)
    inf_counts: dict[str, int] = {}
    for col in numeric.columns:
        n_inf = int(np.isinf(df[col]).sum())
        if n_inf > 0:
            inf_counts[col] = n_inf

    if inf_counts:
        add(f"**{len(inf_counts)} column(s) still contain inf values (after loader):**\n")
        add("| Column | Inf count |")
        add("|--------|-----------|")
        for col, cnt in sorted(inf_counts.items(), key=lambda t: -t[1]):
            add(f"| `{col}` | {cnt:,} |")
        add("")
    else:
        add("No inf values detected after loader cleaning.\n")

    # ------------------------------------------------------------------
    # 4. Constant columns
    # ------------------------------------------------------------------
    add("## 5. Constant columns\n")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        add(f"**{len(constant_cols)} constant column(s) detected (zero variance):**\n")
        for c in constant_cols:
            add(f"- `{c}`")
        add("")
    else:
        add("No constant columns detected.\n")

    # ------------------------------------------------------------------
    # 5. Duplicate rows
    # ------------------------------------------------------------------
    add("## 6. Duplicate rows\n")
    n_dupes = int(df.duplicated().sum())
    pct_dupes = 100.0 * n_dupes / len(df)
    add(f"**{n_dupes:,} duplicate rows** ({pct_dupes:.2f}% of total).\n")
    if pct_dupes > 5:
        add("> [!WARNING]\n> High duplicate-row rate may indicate data leakage "
            "between splits. Deduplicate before splitting.\n")

    # ------------------------------------------------------------------
    # 6. Highly correlated pairs
    # ------------------------------------------------------------------
    add(f"## 7. Highly correlated feature pairs (|r| > {corr_threshold})\n")
    label_col = "label" if "label" in df.columns else None
    df_no_label = df.drop(columns=[label_col]) if label_col else df
    pairs = _highly_correlated_pairs(df_no_label, threshold=corr_threshold)
    if pairs:
        add(f"**{len(pairs)} pair(s) found:**\n")
        add("| Feature A | Feature B | Pearson r |")
        add("|-----------|-----------|-----------|")
        for col_a, col_b, r in pairs[:50]:  # cap at 50 rows for readability
            add(f"| `{col_a}` | `{col_b}` | {r:.4f} |")
        if len(pairs) > 50:
            add(f"\n_... and {len(pairs) - 50} more pairs not shown._")
        add("")
    else:
        add(f"No pairs with |r| > {corr_threshold} found.\n")

    # ------------------------------------------------------------------
    # 7. Figure references
    # ------------------------------------------------------------------
    add("## 8. Figures\n")
    add(f"![Class distribution](figures/eda_{dataset_name}_class_dist.png)\n")
    add("### Key feature distributions per class\n")

    key_feats = (
        _CICIDS2017_KEY_FEATURES
        if "cicids2017" in dataset_name.lower()
        else _CICIOT2023_KEY_FEATURES
    )
    for feat in key_feats:
        fig_name = f"eda_{dataset_name}_feat_{feat}.png"
        add(f"![{feat}](figures/{fig_name})\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------


def run_eda(
    dataset_name: str,
    df: pd.DataFrame,
    config_path: str | Path | None = None,
    corr_threshold: float = 0.95,
) -> Path:
    """Run EDA on a loaded DataFrame and write report + figures.

    Args:
        dataset_name: ``"cicids2017"`` or ``"ciciot2023"``.
        df: cleaned DataFrame from the loader.
        config_path: path to the YAML config (default: ``configs/config.yaml``).
        corr_threshold: absolute Pearson correlation cutoff.

    Returns:
        Path to the generated Markdown report.
    """
    cfg = load_config(config_path)
    root = PROJECT_ROOT
    reports_dir = root / cfg.paths.reports
    figures_dir = root / cfg.paths.figures
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    key_feats = (
        _CICIDS2017_KEY_FEATURES
        if "cicids2017" in dataset_name.lower()
        else _CICIOT2023_KEY_FEATURES
    )

    # Generate figures.
    if "label" in df.columns:
        _plot_class_dist(df, dataset_name, figures_dir)
        _plot_feature_distributions(df, dataset_name, key_feats, figures_dir)

    # Generate markdown report.
    report_md = _build_report(df, dataset_name, figures_dir, corr_threshold=corr_threshold)
    report_path = reports_dir / f"eda_{dataset_name}.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("EDA report saved to %s", report_path)

    return report_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run EDA on CICIDS2017 or CICIoT2023 and save report/figures."
    )
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "ciciot2023"],
        required=True,
        help="Which dataset to analyse.",
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
        help="Sample N rows before EDA (fast development).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson r above which pairs are reported (default 0.95).",
    )
    args = parser.parse_args()

    try:
        if args.dataset == "cicids2017":
            df = load_cicids2017(args.config, dev_sample=args.dev_sample)
        else:
            df = load_ciciot2023(args.config, dev_sample=args.dev_sample)
    except DataNotFoundError:
        sys.exit(1)

    report_path = run_eda(
        args.dataset, df, config_path=args.config, corr_threshold=args.corr_threshold
    )
    print(f"Report written to: {report_path}")
