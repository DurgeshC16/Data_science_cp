"""Strict Tier-2 JSON parsing, safety checks, retry, and result caching."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import PROJECT_ROOT, Config, load_config
from src.tier2.llm_backend import (
    BackendCallError,
    BackendResponse,
    LLMBackend,
    OllamaBackend,
    approximate_token_count,
)
from src.tier2.prompt import PROMPT_VERSION, build_prompt

logger = logging.getLogger(__name__)

_ATTACK_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?(?!\d)")
_ATTACK_IN_TEXT_RE = re.compile(r"(?<![A-Za-z0-9])T\d{4}(?:\.\d{3})?(?!\d)")


class AttackTechnique(BaseModel):
    """One sourced ATT&CK technique ID and its canonical name."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=5)
    name: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _valid_id_shape(cls, value: str) -> str:
        """Reject values that do not have a MITRE ATT&CK technique-ID shape."""
        if not _ATTACK_ID_RE.fullmatch(value):
            raise ValueError("attack_technique.id must be T#### or T####.###")
        return value


class Tier2Answer(BaseModel):
    """Strictly parsed LLM response before provenance/safety metadata is added."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    classification: Literal[
        "Likely Benign", "Likely Malicious", "Needs Human Review"
    ]
    explanation: str = Field(min_length=1)
    attack_technique: AttackTechnique | None
    confidence: Literal["Low", "Medium", "High"]
    recommended_action: str = Field(min_length=1)

    @field_validator("explanation")
    @classmethod
    def _three_or_four_sentences(cls, value: str) -> str:
        """Require the requested three or four plain-English sentences."""
        sentence_parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", value.strip())
            if part.strip()
        ]
        if not 3 <= len(sentence_parts) <= 4:
            raise ValueError("explanation must contain 3-4 sentences")
        return value.strip()


class ValidatedTier2Output(BaseModel):
    """Validated answer plus Tier-1 provenance and anti-hallucination flags."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal[
        "Likely Benign", "Likely Malicious", "Needs Human Review"
    ]
    explanation: str
    attack_technique: AttackTechnique | None
    confidence: Literal["Low", "Medium", "High"]
    recommended_action: str
    tier1_verdict: str
    prompt_version: str
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False


@dataclass(frozen=True)
class Tier2Run:
    """Raw and parsed output plus provenance and call telemetry."""

    raw_outputs: list[str]
    parsed_output: dict[str, Any]
    model: str
    attempts: int
    call_metrics: list[dict[str, Any]]
    cache_key: str
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the run."""
        return {
            "raw_outputs": list(self.raw_outputs),
            "parsed_output": self.parsed_output,
            "model": self.model,
            "attempts": self.attempts,
            "call_metrics": list(self.call_metrics),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
        }


def _resolve(path: Path) -> Path:
    """Resolve configured relative paths against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_json(value: Any) -> str:
    """Serialize cache-key inputs deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def make_cache_key(flow: dict[str, Any], prompt_version: str, model: str) -> str:
    """Hash exactly the requested flow, prompt version, and model tuple."""
    source = _canonical_json(
        {"flow": flow, "prompt_version": prompt_version, "model": model}
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_attack_technique_names(config: Config | None = None) -> dict[str, str]:
    """Load active technique IDs and names from built KB metadata or source STIX.

    Args:
        config: project configuration, used for the KB metadata path.

    Returns:
        Mapping from source-backed ATT&CK technique IDs to names.
    """
    cfg = config or load_config()
    passages_path = _resolve(cfg.rag.passages_path)
    if passages_path.is_file():
        names: dict[str, str] = {}
        with passages_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                passage = json.loads(line)
                technique_id = passage.get("technique_id")
                name = passage.get("name")
                if technique_id and name:
                    names[str(technique_id)] = str(name)
        return names

    source_path = PROJECT_ROOT / "data/kb/enterprise-attack.json"
    if not source_path.is_file():
        return {}
    bundle = json.loads(source_path.read_text(encoding="utf-8"))
    names = {}
    for item in bundle.get("objects", []):
        if (
            item.get("type") != "attack-pattern"
            or item.get("revoked") is True
            or item.get("x_mitre_deprecated") is True
        ):
            continue
        for reference in item.get("external_references", []):
            if reference.get("source_name") == "mitre-attack":
                identifier = reference.get("external_id")
                if identifier:
                    names[str(identifier)] = str(item.get("name", ""))
                break
    return names


def _input_numeric_values(
    flow: dict[str, Any], shap_features: Sequence[dict[str, Any]]
) -> set[str]:
    """Collect normalized numeric values explicitly present in prompt inputs."""
    values: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, (int, float)):
            values.add(_normalize_number(str(item)))
        elif isinstance(item, str):
            for found in _NUMBER_RE.findall(item):
                values.add(_normalize_number(found))

    visit(flow)
    visit(shap_features)
    return values


def _normalize_number(value: str) -> str:
    """Normalize numeric text so ``1.0`` and ``1`` compare equally."""
    stripped = value.strip().rstrip("%")
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def _cited_attack_ids(passages: Sequence[dict[str, Any]]) -> set[str]:
    """Collect technique IDs explicitly present in retrieved passages."""
    ids: set[str] = set()
    for passage in passages:
        value = passage.get("technique_id")
        if value:
            ids.add(str(value))
        text = str(passage.get("text", ""))
        ids.update(_ATTACK_IN_TEXT_RE.findall(text))
    return ids


def validate_answer(
    raw_text: str,
    *,
    flow: dict[str, Any],
    shap_features: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    tier1_verdict: str,
    known_techniques: dict[str, str],
    prompt_version: str = PROMPT_VERSION,
) -> ValidatedTier2Output:
    """Parse strict JSON and apply KB, retrieval-grounding, and value checks.

    Args:
        raw_text: untrusted raw LLM output.
        flow: flow fields supplied to the prompt.
        shap_features: feature values supplied to the prompt.
        passages: retrieved text and metadata supplied to the prompt.
        tier1_verdict: original Tier-1 class/verdict, retained in every output.
        known_techniques: technique IDs and names present in the ATT&CK KB.
        prompt_version: prompt version used to generate the answer.

    Returns:
        Validated output and warning flags.

    Raises:
        ValueError: malformed output, unknown technique ID/name, or numeric
            feature values not present in the supplied input.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed JSON: {error.msg}") from error
    answer = Tier2Answer.model_validate(payload)
    warnings: list[str] = []
    technique = answer.attack_technique
    if technique is not None:
        canonical_name = known_techniques.get(technique.id)
        if canonical_name is None:
            raise ValueError(f"ATT&CK ID {technique.id} does not exist in the local KB")
        if canonical_name.casefold() != technique.name.casefold():
            raise ValueError(
                f"ATT&CK name for {technique.id} must match KB name {canonical_name!r}"
            )
        if technique.id not in _cited_attack_ids(passages):
            warnings.append("attack_technique_not_in_retrieved_passages")

    supplied_numbers = _input_numeric_values(flow, shap_features)
    hallucinated_numbers = {
        _normalize_number(found)
        for found in _NUMBER_RE.findall(_ATTACK_IN_TEXT_RE.sub("", answer.explanation))
    } - supplied_numbers
    if hallucinated_numbers:
        warnings.append(
            "explanation_cites_feature_values_not_in_input:"
            + ",".join(sorted(hallucinated_numbers))
        )

    return ValidatedTier2Output(
        **answer.model_dump(),
        tier1_verdict=tier1_verdict,
        prompt_version=prompt_version,
        warnings=warnings,
    )


def _fallback_output(tier1_verdict: str, prompt_version: str, reason: str) -> dict[str, Any]:
    """Build safe Needs Human Review output retaining the Tier-1 verdict."""
    result = ValidatedTier2Output(
        classification="Needs Human Review",
        explanation=(
            "The Tier-2 response could not be validated. "
            f"Tier 1 classified this flow as {tier1_verdict}. "
            "A human analyst should review the flow and retrieved evidence before taking action."
        ),
        attack_technique=None,
        confidence="Low",
        recommended_action="Review the raw flow and retrieved evidence before blocking or allowlisting it.",
        tier1_verdict=tier1_verdict,
        prompt_version=prompt_version,
        warnings=[reason],
        fallback_used=True,
    )
    return result.model_dump(mode="json")


class Tier2Explainer:
    """Run Tier-2 generation, strict validation, bounded retry, and caching."""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        config: Config | None = None,
        cache_path: Path | None = None,
    ) -> None:
        """Initialize a backend and persistent result cache.

        Args:
            backend: chosen LLM backend; defaults to configured Ollama.
            config: project config; defaults to the repository config.
            cache_path: optional cache path override for tests.
        """
        self.config = config or load_config()
        if backend is not None:
            self.backend = backend
        elif self.config.llm.backend == "ollama":
            self.backend = OllamaBackend(self.config)
        else:
            raise ValueError(
                f"configured LLM backend {self.config.llm.backend!r} is not implemented"
            )
        self.cache_path = cache_path or _resolve(self.config.llm.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.RLock()

    def _read_cache(self) -> dict[str, Any]:
        """Read the JSON result cache, treating missing files as empty."""
        with self._cache_lock:
            if not self.cache_path.is_file():
                return {}
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
            except (json.JSONDecodeError, OSError):
                logger.warning("Ignoring unreadable Tier-2 cache at %s", self.cache_path)
                return {}

    def _write_cache_entry(self, key: str, entry: dict[str, Any]) -> None:
        """Merge and atomically persist one entry without losing concurrent writes."""
        with self._cache_lock:
            cache = self._read_cache()
            cache[key] = entry
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.cache_path.parent, delete=False
                ) as handle:
                    json.dump(cache, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    temporary = Path(handle.name)
                temporary.replace(self.cache_path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    def explain(
        self,
        flow: dict[str, Any],
        shap_features: Sequence[dict[str, Any]],
        passages: Sequence[dict[str, Any]],
        tier1_verdict: str,
        *,
        known_techniques: dict[str, str] | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> Tier2Run:
        """Generate and validate one explanation with at most two retries.

        Args:
            flow: flow details sent to the LLM.
            shap_features: SHAP features sent to the LLM.
            passages: RAG passages sent to the LLM.
            tier1_verdict: Tier-1 result preserved if fallback is needed.
            known_techniques: optional local KB map override for tests.
            prompt_version: versioned template identifier.

        Returns:
            Raw response(s), strict parsed output, model provenance and telemetry.
        """
        cache_key = make_cache_key(flow, prompt_version, self.backend.model_name)
        cache = self._read_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return Tier2Run(
                raw_outputs=list(cached["raw_outputs"]),
                parsed_output=dict(cached["parsed_output"]),
                model=self.backend.model_name,
                attempts=int(cached["attempts"]),
                call_metrics=list(cached["call_metrics"]),
                cache_key=cache_key,
                cache_hit=True,
            )

        techniques = (
            known_techniques
            if known_techniques is not None
            else load_attack_technique_names(self.config)
        )
        prompt = build_prompt(flow, shap_features, passages, prompt_version)
        raw_outputs: list[str] = []
        call_metrics: list[dict[str, Any]] = []
        parsed_output: dict[str, Any] | None = None
        last_error = "no valid response"
        retry_limit = min(self.config.llm.max_retries, 2)

        for attempt in range(retry_limit + 1):
            response_received = False
            try:
                response: BackendResponse = self.backend.generate(prompt)
                response_received = True
                raw_outputs.append(response.text)
                call_metrics.append(
                    {
                        "prompt_tokens": response.prompt_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_ms": response.latency_ms,
                    }
                )
                parsed_output = validate_answer(
                    response.text,
                    flow=flow,
                    shap_features=shap_features,
                    passages=passages,
                    tier1_verdict=tier1_verdict,
                    known_techniques=techniques,
                    prompt_version=prompt_version,
                ).model_dump(mode="json")
                break
            except Exception as error:  # backend/JSON/schema checks are untrusted boundaries
                last_error = f"{type(error).__name__}: {error}"
                if isinstance(error, BackendCallError):
                    call_metrics.append(
                        {
                            "prompt_tokens": error.prompt_tokens,
                            "output_tokens": error.output_tokens,
                            "latency_ms": error.latency_ms,
                            "error": str(error),
                        }
                    )
                elif not response_received:
                    call_metrics.append(
                        {
                            "prompt_tokens": approximate_token_count(prompt),
                            "output_tokens": 0,
                            "latency_ms": None,
                            "error": last_error,
                        }
                    )
                logger.warning("Tier-2 attempt %d/%d failed: %s", attempt + 1, retry_limit + 1, last_error)
                if attempt < retry_limit:
                    prompt += (
                        "\n\nYour previous output failed validation. Correct the JSON/schema errors "
                        "and return only the required JSON object. Do not add facts or values."
                    )

        if parsed_output is None:
            parsed_output = _fallback_output(tier1_verdict, prompt_version, last_error)
        run = Tier2Run(
            raw_outputs=raw_outputs,
            parsed_output=parsed_output,
            model=self.backend.model_name,
            attempts=len(call_metrics),
            call_metrics=call_metrics,
            cache_key=cache_key,
            cache_hit=False,
        )
        self._write_cache_entry(
            cache_key,
            {
                "raw_outputs": run.raw_outputs,
                "parsed_output": run.parsed_output,
                "attempts": run.attempts,
                "call_metrics": run.call_metrics,
            },
        )
        return run


def save_tier2_sample(
    run: Tier2Run,
    sample_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Save a run's raw model responses and parsed output for audit/review.

    Args:
        run: result returned by :meth:`Tier2Explainer.explain`.
        sample_name: readable sample identifier; path separators are removed.
        output_dir: optional output directory; defaults to
            ``reports/tier2_samples``.

    Returns:
        Path to the JSON file written.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_name).strip("._")
    if not safe_name:
        raise ValueError("sample_name must contain at least one safe filename character")
    directory = output_dir or (PROJECT_ROOT / "reports/tier2_samples")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{safe_name}.json"
    destination.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def explain_flow(
    flow: dict[str, Any],
    shap_features: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    tier1_verdict: str,
    *,
    backend: LLMBackend | None = None,
    config: Config | None = None,
    cache_path: Path | None = None,
) -> Tier2Run:
    """Convenience function to perform one Tier-2 explanation.

    Args:
        flow: original flow details.
        shap_features: top SHAP attributions.
        passages: retrieved ATT&CK/CVE evidence.
        tier1_verdict: Tier-1 classification retained in every parsed output.
        backend: optional backend override.
        config: optional application config.
        cache_path: optional result-cache override.

    Returns:
        A :class:`Tier2Run` containing raw/parsed output and telemetry.
    """
    return Tier2Explainer(backend, config, cache_path).explain(
        flow, shap_features, passages, tier1_verdict
    )
