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


class Tier1Config(BaseModel):
    """Fast classical model settings (PLACEHOLDER values until Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["random_forest", "xgboost"] = "random_forest"
    calibration: Literal["none", "isotonic", "sigmoid"] = "isotonic"
    n_estimators: int = Field(default=300, gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    n_trials: int = Field(default=25, ge=0)


class GateConfig(BaseModel):
    """Confidence-gate thresholds (PLACEHOLDER values until Phase 3)."""

    model_config = ConfigDict(extra="forbid")

    max_prob_threshold: float = Field(ge=0.0, le=1.0)
    entropy_threshold: float = Field(ge=0.0)
    target_escalation_rate_min: float = Field(ge=0.0, le=1.0)
    target_escalation_rate_max: float = Field(ge=0.0, le=1.0)

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


class RagConfig(BaseModel):
    """Retrieval settings over MITRE ATT&CK / CVE passages."""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: Path = Path("data/kb/faiss.index")
    passages_path: Path = Path("data/kb/passages.jsonl")
    top_k: int = Field(default=5, gt=0)
    min_score: float = 0.0


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
    max_retries: int = Field(default=1, ge=0)


class EvalConfig(BaseModel):
    """Evaluation settings, including the cap on scored flows."""

    model_config = ConfigDict(extra="forbid")

    sample_cap: int = Field(default=2000, gt=0)
    bootstrap_resamples: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    metrics: list[str] = Field(default_factory=list)


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
