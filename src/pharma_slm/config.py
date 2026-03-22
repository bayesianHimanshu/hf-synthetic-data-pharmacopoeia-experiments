from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = Field(default_factory=list)

class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True

class ProjectConfig(BaseModel):
    name: str = "pharma-slm"
    version: str = "0.1.0"
    hf_username: str
    hf_token_env: str = "HUGGINGFACE_API_KEY"


class FigureExtractionConfig(BaseModel):
    enabled: bool = False
    vision_model: str = "HuggingFaceTB/SmolVLM-Instruct"
    dpi: int = 150
    min_drawing_area: float = 5000.0
    max_new_tokens: int = 256
    batch_size: int = 4

class DataConfig(BaseModel):
    pdf_url: str
    pdf_filename: str
    raw_chunks_path: str
    chunk_size: int = 3000
    min_chunk_len: int = 100
    raw_hub_repo: str
    figure_extraction: FigureExtractionConfig = Field(default_factory=FigureExtractionConfig)

class SynthesisConfig(BaseModel):
    model_name: str
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    temperature: float = 0.0
    max_tokens: int = 2048
    stop_tokens: list[str] = Field(default_factory=list)
    output_path: str
    synthetic_hub_repo: str
    min_output_len: int = 10

class TrainingConfig(BaseModel):
    base_model_id: str
    dataset_id: str
    output_dir: str
    adapter_subdir: str = "final_adapter"
    max_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    eval_split: float = 0.1
    eval_steps: int = 50
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 50
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

class MergeConfig(BaseModel):
    base_model_id: str
    adapter_dir: str
    merged_hub_repo: str
    torch_dtype: str = "bfloat16"

class InferenceConfig(BaseModel):
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = True

class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "pharma-slm"
    run_name: str | None = None   # if None, auto-generated; recommend setting to model+version
    tags: list[str] = Field(default_factory=list)


class BayesianConfig(BaseModel):
    entropy_threshold: float = 1.0
    skip_tokens: list[str] = Field(default_factory=lambda: ["", "|", "-", " ", "\n"])

class OtelExporterConfig(BaseModel):
    type: Literal["console", "file"]
    path: str | None = None   # for file

class TelemetryConfig(BaseModel):
    service_name: str = "pharma-slm"
    exporters: list[OtelExporterConfig] = Field(
        default_factory=lambda: [OtelExporterConfig(type="console")]
    )

class OutputsConfig(BaseModel):
    plots_dir: str = "outputs/plots"
    csv_path: str = "outputs/plots/metrics.csv"

class PharmaConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    synthesis: SynthesisConfig
    training: TrainingConfig
    merge: MergeConfig
    inference: InferenceConfig
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    outputs: OutputsConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on scalar conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "default.yaml"

def load_config(
    default_path: str | Path = _DEFAULT_CONFIG,
    override_path: str | Path | None = None,
) -> PharmaConfig:
    """Load and validate config.
    """
    with open(default_path) as f:
        base: dict = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path) as f:
            override: dict = yaml.safe_load(f) or {}
        base = _deep_merge(base, override)

    return PharmaConfig.model_validate(base)
