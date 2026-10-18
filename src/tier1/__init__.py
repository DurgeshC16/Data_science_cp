"""Cascade-XID Tier-1 subpackage: training, benchmarking and inference.

The public entry points live in :mod:`src.tier1.train` and are exposed lazily.
Importing them eagerly here would make ``python -m src.tier1.train`` (the
``make tier1-benchmark`` path) load the module twice, which runpy warns about
and which can execute module-level code twice.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = frozenset(
    {
        "TrainedModel",
        "aggregate_runs",
        "compute_metrics",
        "fit_model",
        "load_processed_splits",
        "load_tier1_model",
        "run_benchmark",
        "select_default_model",
        "tune_hyperparameters",
    }
)


def __getattr__(name: str) -> Any:
    """Import and return a Tier-1 entry point on first access.

    Args:
        name: attribute requested on the package.

    Returns:
        The attribute from :mod:`src.tier1.train`.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    if name in _EXPORTS:
        return getattr(import_module("src.tier1.train"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
