"""Versioned prompt construction for the SOC analyst assistant."""

from __future__ import annotations

import json
from typing import Any, Sequence

PROMPT_VERSION = "prompt_v1"

_PROMPT_TEMPLATE = """You are a SOC analyst assistant. A network flow was flagged as uncertain by an ML detector. Flow details: {flow}. The detector's top contributing features were: {shap_features}. Reference context (retrieved from MITRE ATT&CK / CVE): {passages}. Using only the flow details and the reference context, respond in JSON with: classification (Likely Benign / Likely Malicious / Needs Human Review); explanation (3-4 plain-English sentences a junior analyst can follow); attack_technique (the single best-matching ID and name, or null); confidence (Low / Medium / High); recommended_action (one concrete next step). Return only a JSON object, without Markdown fences."""


def _json_payload(value: Any) -> str:
    """Render prompt components as stable readable JSON."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_prompt(
    flow: dict[str, Any],
    shap_features: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Build the versioned prompt from flow, SHAP, and retrieved passages.

    Args:
        flow: the flow fields supplied to Tier 2.
        shap_features: top contributing features from Tier 1 explanation.
        passages: retrieved KB passage records.
        prompt_version: prompt template version to use.

    Returns:
        A complete prompt string.

    Raises:
        ValueError: if an unsupported prompt version is requested.
    """
    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"unsupported prompt version: {prompt_version!r}")
    return _PROMPT_TEMPLATE.format(
        flow=_json_payload(flow),
        shap_features=_json_payload(list(shap_features)),
        passages=_json_payload(list(passages)),
    )
