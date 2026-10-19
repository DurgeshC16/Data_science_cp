"""Draft static mapping from unified labels to MITRE ATT&CK candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT, load_config


def load_attack_lookup(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the human-reviewable ATT&CK draft table.

    Args:
        path: optional YAML path; defaults to ``configs/attack_lookup.yaml``.

    Returns:
        Parsed lookup payload.

    Raises:
        ValueError: if an entry is missing candidates or review status.
    """
    configured_path = load_config().shap.attack_lookup_path
    lookup_path = (
        Path(path)
        if path
        else configured_path
        if configured_path.is_absolute()
        else PROJECT_ROOT / configured_path
    )
    with lookup_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    labels = payload.get("labels", {})
    for label, entry in labels.items():
        if entry.get("needs_human_verification") is not True:
            raise ValueError(f"{label}: needs_human_verification must be true")
        for candidate in entry.get("candidates", []):
            if not candidate.get("id") or not candidate.get("name") or not candidate.get("source"):
                raise ValueError(f"{label}: candidate requires id, name, and source")
    return payload


def lookup_attack_candidates(
    unified_label: str, path: str | Path | None = None
) -> dict[str, Any]:
    """Return draft ATT&CK candidate(s) and the required review flag.

    Args:
        unified_label: one label from the project's unified taxonomy.
        path: optional lookup YAML path.

    Returns:
        Label, candidates, and ``needs_human_verification`` fields.

    Raises:
        KeyError: if the unified label is not in the lookup table.
    """
    entry = load_attack_lookup(path)["labels"].get(unified_label)
    if entry is None:
        raise KeyError(f"no ATT&CK lookup entry for unified label {unified_label!r}")
    return {
        "label": unified_label,
        "candidates": list(entry["candidates"]),
        "needs_human_verification": True,
    }
