"""Typed access to ``configs/config.yaml``.

Every script in this project must read its parameters from the config file rather
than hard-coding them, so that a single YAML edit (or ``make`` target) reproduces
a run. Paths in the YAML are interpreted relative to the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class PathsConfig(BaseModel):
    """Directory layout used by the pipeline (relative to the project root)."""

    model_config = ConfigDict(extra="forbid")

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    kb: Path = Path("data/kb")
    models: Path = Path("models")
    reports: Path = Path("reports")
    figures: Path = Path("reports/figures")


class DataConfig(BaseModel):
    """Input-data settings. ``dataset`` stays null until real data is downloaded."""

    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    raw_glob: str = "*.csv"
    label_column: str = "label"
    flow_id_column: str | None = None
    drop_columns: list[str] = Field(default_factory=list)
    dev_sample: int | None = Field(default=None, gt=0)
    label_map: Path = Path("configs/label_map.yaml")


class SplitConfig(BaseModel):
    """Train/validation/calibration/test fractions. Must sum to 1.

    The calibration split is reserved for fitting the probability calibrator and
    the gate thresholds. It must never be used for model training.
    """

    model_config = ConfigDict(extra="forbid")

    train: float = Field(gt=0.0, lt=1.0)
    val: float = Field(gt=0.0, lt=1.0)
    calibration: float = Field(default=0.10, gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    stratify: bool = True

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitConfig":
        total = self.train + self.val + self.calibration + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")
        return self


class PreprocessConfig(BaseModel):
    """Cleaning, splitting and train-only preprocessing settings.

    ``class_weighting`` only controls class weights handed to the Tier-1 model;
    this project never resamples rows (no oversampling) before splitting.
    """

    model_config = ConfigDict(extra="forbid")

    impute_strategy: Literal["median", "mean", "drop_rows"] = "median"
    max_column_missing_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    drop_constant_columns: bool = True
    drop_duplicate_rows: bool = True
    scaling: Literal["standard", "robust", "none"] = "standard"
    class_weighting: Literal["none", "balanced"] = "balanced"
    identifier_patterns: list[str] = Field(default_factory=list)
    extra_identifier_columns: list[str] = Field(default_factory=list)
    zeroday_classes: list[str] = Field(default_factory=list)
    zeroday_max_classes: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_zeroday(self) -> "PreprocessConfig":
        if len(self.zeroday_classes) > self.zeroday_max_classes:
            raise ValueError(
                "zeroday_classes has more entries than zeroday_max_classes "
                f"({len(self.zeroday_classes)} > {self.zeroday_max_classes})"
            )
        return self


class OptunaConfig(BaseModel):
    """Bounded hyper-parameter search on the validation split only."""

    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(default=20, ge=1)
    time_budget_seconds: float = Field(default=600.0, gt=0.0)


class SelectionConfig(BaseModel):
    """Rule that picks the default Tier-1 model from the benchmark table.

    A candidate is eligible when its mean p95 single-flow latency is at most
    ``max_p95_latency_ms``; the winner maximises ``metric``, with ``tie_breaker``
    used when the metric values are equal.
    """

    model_config = ConfigDict(extra="forbid")

    metric: Literal["macro_f1", "accuracy", "weighted_f1"] = "macro_f1"
    max_p95_latency_ms: float = Field(default=25.0, gt=0.0)
    tie_breaker: Literal["accuracy", "macro_f1", "weighted_f1", "p95_latency_ms"] = (
        "accuracy"
    )


class BenchmarkConfig(BaseModel):
    """Timing protocol for the Tier-1 benchmark."""

    model_config = ConfigDict(extra="forbid")

    latency_calls: int = Field(default=1000, ge=1)
    warmup_calls: int = Field(default=50, ge=0)
    batch_size: int = Field(default=1024, ge=1)
    throughput_max_rows: int = Field(default=50000, ge=1)
    torch_threads: int = Field(default=1, ge=1)


class RandomForestConfig(BaseModel):
    """Search bounds for the Random Forest family."""

    model_config = ConfigDict(extra="forbid")

    n_estimators_max: int = Field(default=500, gt=0)
    max_depth_max: int = Field(default=40, gt=0)
    min_samples_leaf_max: int = Field(default=20, gt=0)


class XGBoostConfig(BaseModel):
    """Search bounds for the XGBoost family."""

    model_config = ConfigDict(extra="forbid")

    n_estimators_max: int = Field(default=500, gt=0)
    max_depth_max: int = Field(default=10, gt=0)
    learning_rate_min: float = Field(default=0.01, gt=0.0)
    learning_rate_max: float = Field(default=0.3, gt=0.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> "XGBoostConfig":
        if self.learning_rate_min >= self.learning_rate_max:
            raise ValueError("learning_rate_min must be below learning_rate_max")
        return self


class MlpConfig(BaseModel):
    """Small feed-forward network settings (2-3 hidden layers).

    ``hidden_layer_options`` is the categorical search space; every option must
    contain 2 or 3 positive layer widths.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_layer_options: list[list[int]] = Field(
        default_factory=lambda: [[64, 32], [128, 64], [128, 64, 32]]
    )
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=256, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    patience: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check_layers(self) -> "MlpConfig":
        if not self.hidden_layer_options:
            raise ValueError("hidden_layer_options must not be empty")
        for option in self.hidden_layer_options:
            if len(option) not in (2, 3):
                raise ValueError(
                    f"each MLP option needs 2 or 3 hidden layers, got {option}"
                )
            if any(width <= 0 for width in option):
                raise ValueError(f"MLP widths must be positive, got {option}")
        return self


class Tier1Config(BaseModel):
    """Tier-1 benchmark settings: families, tasks, seeds, tuning and timing."""

    model_config = ConfigDict(extra="forbid")

    #: Default family used by downstream phases. The benchmark writes the family
    #: chosen by ``selection`` into reports/tier1_benchmark.json; keep this value
    #: in sync with that choice (nothing silently overrides it at runtime).
    model: Literal["random_forest", "xgboost", "mlp"] = "random_forest"
    calibration: Literal["none", "isotonic", "sigmoid"] = "isotonic"
    families: list[Literal["random_forest", "xgboost", "mlp"]] = Field(
        default_factory=lambda: ["random_forest", "xgboost", "mlp"]
    )
    tasks: list[Literal["binary", "multiclass"]] = Field(
        default_factory=lambda: ["binary", "multiclass"]
    )
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    n_jobs: int = -1
    #: Threads used when scoring. RandomForest/XGBoost parallelise *every*
    #: predict call, so serving one flow with ``n_jobs=-1`` is dominated by
    #: worker start-up; a fixed count keeps the reported latency meaningful and
    #: comparable across families. Training still uses ``n_jobs``.
    inference_n_jobs: int = Field(default=1, ge=1)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    mlp: MlpConfig = Field(default_factory=MlpConfig)

    @model_validator(mode="after")
    def _check_lists(self) -> "Tier1Config":
        if not self.families:
            raise ValueError("tier1.families must not be empty")
        if not self.tasks:
            raise ValueError("tier1.tasks must not be empty")
        if not self.seeds:
            raise ValueError("tier1.seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("tier1.seeds must be unique")
        return self


class GateConfig(BaseModel):
    """Confidence-gate settings.

    ``max_prob_threshold`` / ``entropy_threshold`` are PLACEHOLDERS: the fitted
    values are computed on ``threshold_fit_split`` and written to
    ``reports/gate_report.json``. A flow is escalated when its top probability
    is below ``max_prob_threshold`` **or** its predictive entropy exceeds
    ``entropy_threshold``.
    """

    model_config = ConfigDict(extra="forbid")

    max_prob_threshold: float = Field(ge=0.0, le=1.0)
    entropy_threshold: float = Field(ge=0.0)
    target_escalation_rate_min: float = Field(ge=0.0, le=1.0)
    target_escalation_rate_max: float = Field(ge=0.0, le=1.0)
    #: Score the gate with calibrated probabilities. When true, a fitted
    #: calibrator artifact must exist for every scored run.
    use_calibrated_probabilities: bool = True
    #: Split used to fit the probability calibrator. ``calibration`` is the
    #: dedicated Phase-2 split; ``val`` matches the wording in AGENTS.md. Either
    #: way it is disjoint from train and test.
    calibrator_fit_split: Literal["calibration", "val"] = "calibration"
    #: Split used to fit the gate thresholds.
    threshold_fit_split: Literal["val"] = "val"
    #: Threshold grid resolution (quantile steps per score).
    grid_steps: int = Field(default=101, ge=2)
    #: Entropy units: natural log (nats) or base-2 (bits).
    entropy_base: Literal["natural", "bits"] = "natural"

    @model_validator(mode="after")
    def _check_ranges(self) -> "GateConfig":
        if self.target_escalation_rate_min > self.target_escalation_rate_max:
            raise ValueError(
                "target_escalation_rate_min must not exceed target_escalation_rate_max"
            )
        return self


class ShapConfig(BaseModel):
    """SHAP explainer settings for the escalated flows."""

    model_config = ConfigDict(extra="forbid")

    background_samples: int = Field(default=200, gt=0)
    max_display_features: int = Field(default=10, gt=0)
    top_k_features: int = Field(default=3, gt=0)
    benchmark_calls: int = Field(default=100, gt=0)
    benchmark_warmup_calls: int = Field(default=10, ge=0)
    feature_dictionary_path: Path = Path("configs/feature_dictionary.yaml")
    attack_lookup_path: Path = Path("configs/attack_lookup.yaml")


class RagConfig(BaseModel):
    """Retrieval settings over MITRE ATT&CK / CVE passages."""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: Path = Path("models/kb_index/faiss.index")
    passages_path: Path = Path("models/kb_index/documents.jsonl")
    top_k: int = Field(default=5, gt=0)
    min_score: float = 0.0
    chunk_tokens: int = Field(default=300, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    cve_path: Path = Path("data/kb/cve_curated.csv")
    retrieval_eval_path: Path = Path("data/kb/retrieval_eval.csv")
    retrieval_report_path: Path = Path("reports/retrieval_eval.json")

    @model_validator(mode="after")
    def _valid_chunks(self) -> "RagConfig":
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("rag.overlap_tokens must be less than rag.chunk_tokens")
        return self


class LlmConfig(BaseModel):
    """Local open-LLM backend settings for Tier 2."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["ollama", "llamacpp"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)
    require_grounded_ids: bool = True
    max_retries: int = Field(default=2, ge=0, le=2)
    cache_path: Path = Path("models/tier2_cache.json")


class EvalConfig(BaseModel):
    """Evaluation settings, including the cap on scored flows."""

    model_config = ConfigDict(extra="forbid")

    sample_cap: int = Field(default=2000, gt=0)
    #: Bins used for the expected-calibration-error and reliability diagrams.
    calibration_bins: int = Field(default=15, ge=2)
    bootstrap_resamples: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    metrics: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Batch and concurrency settings for end-to-end cascade inference."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=256, gt=0)
    tier2_concurrency: int = Field(default=4, gt=0)
    run_stats_path: Path = Path("reports/run_stats.json")


class Config(BaseModel):
    """Root config object mirroring configs/config.yaml."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "cascade-xid"
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tier1: Tier1Config = Field(default_factory=Tier1Config)
    gate: GateConfig
    shap: ShapConfig = Field(default_factory=ShapConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def resolved_paths(self, root: Path = PROJECT_ROOT) -> dict[str, Path]:
        """Return the configured directories as absolute paths."""
        return {
            name: (root / value).resolve()
            for name, value in self.paths.model_dump().items()
        }


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config.

    Args:
        path: Optional path to a YAML file. Defaults to ``configs/config.yaml``
            at the project root.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML violates the schema.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.model_validate(raw)


def ensure_directories(config: Config, root: Path = PROJECT_ROOT) -> None:
    """Create every configured directory if it is missing."""
    for directory in config.resolved_paths(root).values():
        directory.mkdir(parents=True, exist_ok=True)
