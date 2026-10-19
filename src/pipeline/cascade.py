"""End-to-end batched Tier-1/gate/Tier-2 cascade orchestration."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Config, load_config
from src.data.loader import normalise_columns
from src.data.preprocess import FLOW_ID_COLUMN
from src.explain.attack_lookup import load_attack_lookup
from src.explain.shap_explainer import (
    Tier1TreeExplainer,
    fit_training_percentiles,
    load_feature_dictionary,
    retrieval_query,
)
from src.gate.gate import escalate_mask, uncertainty_scores
from src.rag.retriever import KnowledgeRetriever
from src.schema import OutputRecord
from src.tier1.calibrate import calibrator_path as probability_calibrator_path
from src.tier1.calibrate import load_calibrator
from src.tier1.train import (
    TrainedModel,
    resolve_report_path,
    trained_model_from_artifact,
)
from src.tier2.llm_backend import LLMBackend
from src.tier2.validate import Tier2Explainer, Tier2Run, load_attack_technique_names

logger = logging.getLogger(__name__)


def _resolve(path: Path) -> Path:
    """Resolve a relative project path against the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_commit() -> str:
    """Return the current Git revision or ``unknown`` outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _git_dirty() -> bool:
    """Return whether tracked working files differ from the current revision."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        return result.returncode != 0
    except (FileNotFoundError, OSError):
        return True


def select_gate_thresholds(
    gate_run: dict[str, Any], gate_profile: str
) -> dict[str, float]:
    """Select the fitted validation threshold pair nearest a named escalation profile.

    Args:
        gate_run: one matching run from ``reports/gate_report.json``.
        gate_profile: target rate written as ``p10`` for ten percent.

    Returns:
        Fitted max-probability and entropy thresholds, plus achieved validation rate.

    Raises:
        ValueError: for an invalid profile or a report without fit-sweep results.
    """
    match = re.fullmatch(r"p(\d{1,2})", gate_profile.lower())
    if match is None or int(match.group(1)) > 100:
        raise ValueError("gate_profile must be pNN, where NN is from 00 to 100")
    target = int(match.group(1)) / 100.0
    fitted = gate_run.get("thresholds", {})
    sweep = fitted.get("sweep", [])
    if not sweep:
        raise ValueError("gate report has no threshold sweep; thresholds cannot be guessed")
    selected = min(sweep, key=lambda item: abs(float(item["escalation_rate"]) - target))
    return {
        "max_prob_threshold": float(selected["max_prob_threshold"]),
        "entropy_threshold": float(selected["entropy_threshold"]),
        "target_rate": target,
        "achieved_fit_rate": float(selected["escalation_rate"]),
    }


def _static_technique_id(label: str, lookup: dict[str, Any]) -> str | None:
    """Return the first draft technique candidate for a label, if one exists."""
    candidates = lookup.get("labels", {}).get(label, {}).get("candidates", [])
    return str(candidates[0]["id"]) if candidates else None


def _passage_evidence_id(passage: dict[str, Any]) -> str | None:
    """Return a stable retrieved-document ID or source technique/CVE ID."""
    metadata = passage.get("metadata", {})
    value = (
        passage.get("document_id")
        or metadata.get("document_id")
        or passage.get("technique_id")
        or passage.get("cve_id")
    )
    return str(value) if value else None


class CascadePipeline:
    """Score flows in vectorized Tier-1 batches and escalate only uncertain rows."""

    def __init__(
        self,
        model: TrainedModel,
        gate_thresholds: dict[str, float],
        *,
        config: Config | None = None,
        calibrator: Any | None = None,
        preprocessor: Any | None = None,
        shap_explainer: Any | None = None,
        retriever: Any | None = None,
        tier2_explainer: Any | None = None,
        attack_lookup: dict[str, Any] | None = None,
        attack_techniques: dict[str, str] | None = None,
        tier2_concurrency: int | None = None,
        run_stats_path: Path | None = None,
    ) -> None:
        """Configure an inference cascade from already fitted components.

        Args:
            model: fitted Tier-1 model.
            gate_thresholds: calibrated max-probability and entropy thresholds.
            config: project configuration.
            calibrator: fitted probability calibrator, or ``None`` for raw scores.
            preprocessor: fitted train-only feature preprocessor, or ``None`` if
                input rows are already transformed.
            shap_explainer: injected SHAP explainer; created lazily when omitted.
            retriever: retriever with ``retrieve(query, k=...)``.
            tier2_explainer: Tier-2 runner with ``explain``.
            attack_lookup: static ATT&CK mapping.
            attack_techniques: source-backed canonical technique names for LLM
                validation.
            tier2_concurrency: worker limit for escalated rows.
            run_stats_path: optional run-stats JSON destination.
        """
        self.config = config or load_config()
        self.model = model
        self.calibrator = calibrator
        self.preprocessor = preprocessor
        self.shap_explainer = shap_explainer
        self.retriever = retriever
        self.tier2_explainer = tier2_explainer
        self.attack_lookup = attack_lookup or load_attack_lookup(
            _resolve(self.config.shap.attack_lookup_path)
        )
        self.attack_techniques = attack_techniques or {}
        self.gate_thresholds = gate_thresholds
        self.tier2_concurrency = (
            tier2_concurrency
            if tier2_concurrency is not None
            else self.config.pipeline.tier2_concurrency
        )
        if self.tier2_concurrency < 1:
            raise ValueError("tier2_concurrency must be at least one")
        self.run_stats_path = run_stats_path or _resolve(
            self.config.pipeline.run_stats_path
        )
        self._stats: dict[str, Any] = {}
        self._flow_offset = 0
        self.flow_details: dict[str, dict[str, Any]] = {}

    def _new_stats(self) -> dict[str, Any]:
        """Create empty per-run counters and latency accumulators."""
        return {
            "total_flows": 0,
            "escalated": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "failures": 0,
            "fallbacks": 0,
            "tokens": {"prompt": 0, "output": 0},
            "latency_ms": {
                name: {"total": 0.0, "count": 0, "mean": 0.0}
                for name in (
                    "tier1",
                    "calibration",
                    "gate",
                    "shap",
                    "retrieval",
                    "tier2",
                    "llm",
                    "total_per_flow",
                )
            },
        }

    def _record_latency(self, name: str, elapsed_ms: float, count: int = 1) -> None:
        """Add a latency measurement and refresh its per-unit mean."""
        accumulator = self._stats["latency_ms"][name]
        accumulator["total"] += float(elapsed_ms)
        accumulator["count"] += int(count)
        accumulator["mean"] = accumulator["total"] / max(accumulator["count"], 1)

    def _prepare_features(self, raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Transform one raw batch with the fitted train-only preprocessor."""
        frame = pd.DataFrame(raw_rows)
        missing = [name for name in self.model.feature_columns if name not in frame]
        if missing:
            raise ValueError(f"input batch is missing model features: {missing}")
        if self.preprocessor is not None:
            return self.preprocessor.transform(frame)
        return frame[[FLOW_ID_COLUMN, *self.model.feature_columns]]

    def _tier1_record(
        self,
        flow_id: str,
        flow: dict[str, Any],
        label: str,
        confidence: float,
    ) -> OutputRecord:
        """Build a Tier-1-only record and attach a static candidate when available."""
        technique_id = _static_technique_id(label, self.attack_lookup)
        protocol = flow.get("protocol")
        destination_port = flow.get("destination_port")
        context = []
        if protocol is not None:
            context.append(f"Observed protocol: {protocol}.")
        if destination_port is not None:
            context.append(f"Observed destination port: {destination_port}.")
        confidence_source = "calibrated" if self.calibrator is not None else "model"
        explanation = (
            f"Tier 1 classified this flow as {label} with {confidence:.0%} "
            f"{confidence_source} confidence. The confidence gate accepted the flow, "
            "so Tier 2 was not invoked."
        )
        if context:
            explanation += " " + " ".join(context)
        if technique_id is not None:
            explanation += " The static ATT&CK tag is a draft candidate and requires human verification."
        action = (
            "No immediate action; retain the flow for routine monitoring."
            if label == "Normal"
            else "Review the Tier-1 alert and verify it against trusted network telemetry."
        )
        return OutputRecord(
            flow_id=flow_id,
            final_classification=label,
            explanation=explanation,
            attack_technique=technique_id,
            confidence=confidence,
            recommended_action=action,
            evidence_used=[],
            tier_reached=1,
            llm_invoked=False,
        )

    def _prompt_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        """Remove targets and stable identifiers from LLM flow context."""
        excluded = {
            FLOW_ID_COLUMN,
            "label",
            "binary_label",
            "multiclass_label",
            "ground_truth",
            "target",
            "class",
        }
        identifier_patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in self.config.preprocess.identifier_patterns
        ]
        return {
            key: value
            for key, value in flow.items()
            if key not in excluded
            and not any(pattern.search(key) for pattern in identifier_patterns)
        }

    def _escalated_record(
        self,
        item: dict[str, Any],
    ) -> tuple[OutputRecord, dict[str, Any]]:
        """Run SHAP, retrieval and Tier 2 for one escalated flow."""
        started_total = time.perf_counter()
        local: dict[str, Any] = {
            "shap_ms": 0.0,
            "retrieval_ms": 0.0,
            "tier2_ms": 0.0,
            "shap_features": [],
            "retrieval_query": None,
            "passages": [],
            "tier2_output": None,
            "tier1_verdict": item["tier1_label"],
            "tier1_confidence": item["confidence"],
            "run": None,
            "failure": False,
            "fallback": False,
        }
        passages: list[dict[str, Any]] = []
        run: Tier2Run | None = None
        try:
            if self.shap_explainer is None:
                raise RuntimeError("SHAP explainer is not configured for this Tier-1 model")
            started = time.perf_counter()
            shap_features = self.shap_explainer.explain(item["model_flow"])
            local["shap_features"] = shap_features
            local["shap_ms"] = (time.perf_counter() - started) * 1000.0

            raw_flow = self._prompt_flow(item["raw_flow"])
            query = retrieval_query(
                shap_features,
                destination_port=raw_flow.get("destination_port"),
                protocol=str(raw_flow["protocol"]) if raw_flow.get("protocol") is not None else None,
                service=str(raw_flow["service"]) if raw_flow.get("service") is not None else None,
            )
            local["retrieval_query"] = query
            if self.retriever is None:
                raise RuntimeError("knowledge retriever is not configured")
            started = time.perf_counter()
            passages = self.retriever.retrieve(query, k=self.config.rag.top_k)
            local["passages"] = passages
            local["retrieval_ms"] = (time.perf_counter() - started) * 1000.0

            if self.tier2_explainer is None:
                raise RuntimeError("Tier-2 LLM explainer is not configured")
            started = time.perf_counter()
            run = self.tier2_explainer.explain(
                raw_flow,
                shap_features,
                passages,
                item["tier1_label"],
                known_techniques=self.attack_techniques,
            )
            local["tier2_ms"] = (time.perf_counter() - started) * 1000.0
            local["run"] = run
            parsed = run.parsed_output
            local["tier2_output"] = parsed
            candidate = parsed.get("attack_technique")
            technique_id = candidate.get("id") if isinstance(candidate, dict) else None
            evidence = [
                evidence_id
                for passage in passages
                if (evidence_id := _passage_evidence_id(passage)) is not None
            ]
            llm_invoked = bool(not run.cache_hit and run.attempts > 0)
            record = OutputRecord(
                flow_id=item["flow_id"],
                final_classification=str(parsed["classification"]),
                explanation=str(parsed["explanation"]),
                attack_technique=technique_id,
                # OutputRecord's established confidence contract is numeric;
                # retain calibrated Tier-1 confidence rather than inventing a
                # numeric conversion for the LLM's Low/Medium/High value.
                confidence=float(item["confidence"]),
                recommended_action=str(parsed["recommended_action"]),
                evidence_used=evidence,
                tier_reached=2,
                llm_invoked=llm_invoked,
            )
            local["fallback"] = bool(parsed.get("fallback_used", False))
            local["failures"] = sum(
                1 for metric in run.call_metrics if metric.get("error")
            )
        except Exception as error:
            logger.exception("Tier-2 cascade failed for flow %s", item["flow_id"])
            local["failure"] = True
            local["fallback"] = True
            local["error"] = f"{type(error).__name__}: {error}"
            static_id = _static_technique_id(item["tier1_label"], self.attack_lookup)
            evidence = [
                evidence_id
                for passage in passages
                if (evidence_id := _passage_evidence_id(passage)) is not None
            ]
            record = OutputRecord(
                flow_id=item["flow_id"],
                final_classification="Needs Human Review",
                explanation=(
                    f"Tier 2 could not complete validation for this flow. "
                    f"Tier 1 classified it as {item['tier1_label']}. "
                    "A human analyst should review the flow and available evidence before action."
                ),
                attack_technique=static_id,
                confidence=float(item["confidence"]),
                recommended_action="Review the flow, Tier-1 verdict, and retrieved evidence manually.",
                evidence_used=evidence,
                tier_reached=2,
                llm_invoked=bool(run is not None and not run.cache_hit and run.attempts > 0),
            )
        local["total_ms"] = (time.perf_counter() - started_total) * 1000.0
        local["llm_invoked"] = bool(run is not None and not run.cache_hit and run.attempts > 0)
        local["cache_hit"] = bool(run is not None and run.cache_hit)
        local["run"] = run
        return record, local

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        executor: ThreadPoolExecutor,
    ) -> list[OutputRecord]:
        """Run vectorized Tier 1 and dispatch only escalated rows to Tier 2."""
        if not rows:
            return []
        start_index = self._flow_offset
        raw_rows: list[dict[str, Any]] = []
        flow_ids: list[str] = []
        for offset, input_row in enumerate(rows):
            row = dict(input_row)
            flow_id = str(row.get(FLOW_ID_COLUMN) or f"stream-{start_index + offset:08d}")
            row[FLOW_ID_COLUMN] = flow_id
            raw_rows.append(row)
            flow_ids.append(flow_id)
        self._flow_offset += len(rows)

        features = self._prepare_features(raw_rows)
        feature_frame = features[self.model.feature_columns]
        started = time.perf_counter()
        raw_probabilities = self.model.predict_proba(feature_frame)
        tier1_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        probabilities = (
            self.calibrator.calibrate(raw_probabilities)
            if self.calibrator is not None
            else np.asarray(raw_probabilities, dtype=float)
        )
        calibration_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        scores = uncertainty_scores(probabilities, self.config.gate.entropy_base)
        mask = escalate_mask(
            scores["max_probability"],
            scores["entropy"],
            self.gate_thresholds["max_prob_threshold"],
            self.gate_thresholds["entropy_threshold"],
        )
        gate_ms = (time.perf_counter() - started) * 1000.0

        class_indices = np.argmax(probabilities, axis=1)
        labels = [self.model.labels[int(index)] for index in class_indices]
        confidences = scores["max_probability"]
        records: list[OutputRecord | None] = [None] * len(rows)
        escalated_items: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(raw_rows):
            if not bool(mask[index]):
                records[index] = self._tier1_record(
                    flow_ids[index], row, labels[index], float(confidences[index])
                )
            else:
                model_flow = feature_frame.iloc[[index]].iloc[0].to_dict()
                escalated_items.append(
                    (
                        index,
                        {
                            "flow_id": flow_ids[index],
                            "raw_flow": row,
                            "model_flow": model_flow,
                            "tier1_label": labels[index],
                            "confidence": float(confidences[index]),
                        },
                    )
                )

        self._stats["total_flows"] += len(rows)
        self._stats["escalated"] += len(escalated_items)
        self._record_latency("tier1", tier1_ms, len(rows))
        self._record_latency("calibration", calibration_ms, len(rows))
        self._record_latency("gate", gate_ms, len(rows))
        self._record_latency(
            "total_per_flow",
            tier1_ms + calibration_ms + gate_ms,
            len(rows),
        )

        if escalated_items:
            tier2_results = list(
                executor.map(self._escalated_record, [item for _, item in escalated_items])
            )
            for (index, _), (record, details) in zip(escalated_items, tier2_results):
                records[index] = record
                self.flow_details[record.flow_id] = {
                    "tier1_verdict": details["tier1_verdict"],
                    "tier1_confidence": details["tier1_confidence"],
                    "tier_reached": record.tier_reached,
                    "shap_features": details["shap_features"],
                    "retrieval_query": details["retrieval_query"],
                    "passages": details["passages"],
                    "tier2_output": details["tier2_output"],
                }
                self._record_latency("shap", details["shap_ms"])
                self._record_latency("retrieval", details["retrieval_ms"])
                self._record_latency("tier2", details["total_ms"])
                if details.get("failure"):
                    self._stats["failures"] += 1
                self._stats["failures"] += int(details.get("failures", 0))
                self._stats["fallbacks"] += int(details.get("fallback", False))
                self._stats["cache_hits"] += int(details.get("cache_hit", False))
                run: Tier2Run | None = details.get("run")
                if run is not None and not run.cache_hit:
                    self._stats["llm_calls"] += run.attempts
                    for metric in run.call_metrics:
                        self._stats["tokens"]["prompt"] += int(metric.get("prompt_tokens", 0))
                        self._stats["tokens"]["output"] += int(metric.get("output_tokens", 0))
                        latency_ms = metric.get("latency_ms")
                        if latency_ms is not None:
                            self._record_latency("llm", float(latency_ms))
                self._record_latency("total_per_flow", details["total_ms"], count=0)

        return [record for record in records if record is not None]

    def details_for_flow(self, flow_id: str) -> dict[str, Any] | None:
        """Return details retained for an escalated flow in the latest run."""
        return self.flow_details.get(flow_id)

    def run_stream(
        self,
        rows: Iterable[dict[str, Any]],
        batch_size: int | None = None,
    ) -> Iterator[OutputRecord]:
        """Yield schema-validated records from an input iterator in batches.

        Args:
            rows: iterator of raw or already transformed flow mappings.
            batch_size: optional batch size; defaults to pipeline config.

        Yields:
            One :class:`OutputRecord` per input flow, preserving input order.
        """
        size = batch_size or self.config.pipeline.batch_size
        if size < 1:
            raise ValueError("batch_size must be at least one")
        self._stats = self._new_stats()
        self._flow_offset = 0
        self.flow_details = {}
        executor = ThreadPoolExecutor(
            max_workers=self.tier2_concurrency, thread_name_prefix="cascade-tier2"
        )
        try:
            batch: list[dict[str, Any]] = []
            for row in rows:
                batch.append(dict(row))
                if len(batch) >= size:
                    yield from self._process_batch(batch, executor)
                    batch = []
            if batch:
                yield from self._process_batch(batch, executor)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            self._write_run_stats()

    def run_batch(
        self,
        rows: Iterable[dict[str, Any]],
        batch_size: int | None = None,
    ) -> list[OutputRecord]:
        """Return all outputs from the iterator/batch execution path."""
        return list(self.run_stream(rows, batch_size))

    def write_jsonl(
        self,
        rows: Iterable[dict[str, Any]],
        output_path: Path,
        batch_size: int | None = None,
    ) -> int:
        """Stream validated outputs as JSONL and return the number written."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.run_stream(rows, batch_size):
                handle.write(record.model_dump_json() + "\n")
                count += 1
        return count

    def _write_run_stats(self) -> None:
        """Persist counters, latencies, token totals, and run provenance."""
        self.run_stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_dirty(),
            "config": self.config.model_dump(mode="json"),
            "dataset": self.config.data.dataset,
            "model_family": self.model.family,
            "model_task": self.model.task,
            "gate_thresholds": self.gate_thresholds,
            "tier2_concurrency": self.tier2_concurrency,
            **self._stats,
        }
        self.run_stats_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_cascade_from_artifacts(
    dataset: str,
    gate_profile: str,
    *,
    config: Config | None = None,
    backend: LLMBackend | None = None,
    tier2_concurrency: int | None = None,
) -> CascadePipeline:
    """Load the benchmark-selected model and fitted supporting artifacts.

    Args:
        dataset: dataset key used in model/report artifact names.
        gate_profile: target validation escalation profile such as ``p10``.
        config: explicit project config, or default config.
        backend: optional Tier-2 backend override (tests).
        tier2_concurrency: optional concurrent Tier-2 flow limit.

    Returns:
        A configured cascade ready for streaming inference.

    Raises:
        FileNotFoundError: if required trained artifacts/reports are absent.
        ValueError: if benchmark/report records do not match.
    """
    cfg = config or load_config()
    reports_dir = _resolve(cfg.paths.reports)
    models_dir = _resolve(cfg.paths.models)
    processed_dir = _resolve(cfg.paths.processed)
    benchmark_path = reports_dir / "tier1_benchmark.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Tier-1 benchmark not found: {benchmark_path}; train the model before running the cascade"
        )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selected = benchmark.get("selection", {}).get("chosen", {})
    task = str(selected.get("task", ""))
    family = str(selected.get("family", ""))
    if not task or not family:
        raise ValueError("benchmark report does not contain selection.chosen task/family")
    runs = [
        run
        for run in benchmark.get("runs", [])
        if run.get("task") == task and run.get("family") == family
    ]
    preferred_seed = cfg.tier1.seeds[0]
    matching = [run for run in runs if int(run["seed"]) == preferred_seed]
    if not matching:
        matching = runs
    if not matching:
        raise ValueError(f"benchmark contains no run for selected {task}/{family}")
    model_run = matching[0]
    seed = int(model_run["seed"])
    model_path = resolve_report_path(model_run["model_path"])
    model = trained_model_from_artifact(model_path, task, family, seed)

    calibrator = None
    if cfg.gate.use_calibrated_probabilities:
        path = probability_calibrator_path(models_dir, dataset, task, family, seed)
        calibrator = load_calibrator(path)

    gate_path = reports_dir / "gate_report.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"gate report not found: {gate_path}; run the gate fitting step first")
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_run = next(
        (
            record
            for record in gate_report.get("runs", [])
            if record.get("task") == task
            and record.get("family") == family
            and int(record.get("seed", -1)) == seed
        ),
        None,
    )
    if gate_run is None:
        raise ValueError(f"gate report has no run for {task}/{family}/seed {seed}")
    thresholds = select_gate_thresholds(gate_run, gate_profile)

    preprocessor_path = models_dir / f"{dataset}_preprocessor.joblib"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"fitted preprocessor missing: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    train_path = processed_dir / f"{dataset}_train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"training split missing; expected {train_path}")
    train_frame = pd.read_parquet(train_path)
    dictionary = load_feature_dictionary(_resolve(cfg.shap.feature_dictionary_path))
    percentiles = fit_training_percentiles(
        train_frame, model.feature_columns, dictionary
    )
    shap_explainer = None
    if model.family in {"random_forest", "xgboost"} and set(model.labels) == {
        "Normal",
        "Attack",
    }:
        shap_explainer = Tier1TreeExplainer(model, percentiles, dictionary)
    retriever = KnowledgeRetriever(cfg)
    technique_names = load_attack_technique_names(cfg)
    tier2 = Tier2Explainer(backend=backend, config=cfg)
    return CascadePipeline(
        model,
        thresholds,
        config=cfg,
        calibrator=calibrator,
        preprocessor=preprocessor,
        shap_explainer=shap_explainer,
        retriever=retriever,
        tier2_explainer=tier2,
        attack_lookup=load_attack_lookup(_resolve(cfg.shap.attack_lookup_path)),
        attack_techniques=technique_names,
        tier2_concurrency=tier2_concurrency,
    )


def read_csv_rows(path: Path, chunk_size: int) -> Iterator[dict[str, Any]]:
    """Read and normalize a CSV as a streaming iterator of records.

    Args:
        path: input flow CSV.
        chunk_size: CSV chunk size used to bound memory.

    Yields:
        One normalized row mapping at a time.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    for frame in pd.read_csv(path, chunksize=chunk_size):
        yield from normalise_columns(frame).to_dict(orient="records")
