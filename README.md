# Pharma-SLM: Technical Documentation

A pipeline to build a pharmaceutical domain-specific small language model (SLM) from Pharmacopoeia PDFs using synthetic data generation, QLoRA fine-tuning, and uncertainty-aware inference.

**Public Data Used:** Indian Pharmacopoeia 2010 Edition: https://qps.nhsrcindia.org/sites/default/files/2022-01/INDIAN%20PHARMACOPOEIA%202010%20Volume%201.pdf

**HuggingFace RAW Dataset:** https://huggingface.co/datasets/EulerianKnight/pharmacopeia-raw-chunks

**HuggingFace Synthetic Dataset:** https://huggingface.co/datasets/EulerianKnight/pharmacopeia-synthetic-instruct

**HuggingFace Fine-Tuned Model:** https://huggingface.co/EulerianKnight/gemma-2b-pharmacopeia-slm

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Configuration System](#configuration-system)
3. [Pipeline Steps](#pipeline-steps)
4. [Key Components](#key-components)
5. [Observability](#observability)
6. [GPU Environments](#gpu-environments)
7. [Running the Pipeline](#running-the-pipeline)

---


## Project Structure

```
.
├── configs/
│   ├── default.yaml            # All settings with defaults
│   ├── l40s.yaml               # L40S 48GB override (bfloat16 LoRA)
│   └── lightning_studio.yaml   # L4 24GB override (QLoRA 4-bit)
├── scripts/
│   ├── 01_extract_data.py
│   ├── 02_generate_synthetic.py
│   ├── 03_push_raw_data.py
│   ├── 04_push_synthetic_data.py
│   ├── 05_train.py
│   ├── 06_merge_and_push.py
│   ├── 07_inference.py
│   └── 08_bayesian_inference.py
├── src/pharma_slm/
│   ├── config.py               # Pydantic v2 config models + YAML loader
│   ├── telemetry.py            # OpenTelemetry setup
│   ├── data/
│   │   ├── extraction.py       # PDF download, text extraction, chunking
│   │   ├── figure_extraction.py# Two-pass vision LLM figure detection
│   │   └── inspection.py       # Synthetic data QC sampling
│   ├── synthesis/
│   │   └── generator.py        # vLLM batch inference + prompt
│   ├── training/
│   │   ├── trainer.py          # SFTTrainer orchestration
│   │   ├── callbacks.py        # OTel metrics + matplotlib plots
│   │   └── merge.py            # LoRA merge + Hub push
│   ├── inference/
│   │   ├── runner.py           # Single-chunk inference
│   │   └── bayesian.py         # Entropy-based uncertainty scoring
│   └── hub/
│       └── upload.py           # Versioned Hub dataset uploads
├── outputs/
│   ├── data/                   # Extracted & synthetic JSONL files
│   ├── checkpoints/            # Training checkpoints
│   └── plots/                  # loss_curve.png, lr_schedule.png, metrics.csv
├── requirements.txt
└── pyproject.toml
```

---

## Configuration System

Configuration uses a **two-layer YAML merge**: `configs/default.yaml` defines all settings; a sparse override file (e.g. `configs/l40s.yaml`) overrides only what differs.

```bash
python scripts/05_train.py --config configs/l40s.yaml
```

All config is validated by **Pydantic v2** models defined in `src/pharma_slm/config.py`.

### Key Config Sections

| Section | Purpose |
|---|---|
| `project` | Name, version, HF username |
| `data` | PDF URL, chunk size, figure extraction settings |
| `synthesis` | Teacher model, vLLM parameters |
| `training` | Base model, LoRA, quantization, optimizer |
| `merge` | Adapter merge and Hub push target |
| `inference` | Fine-tuned model ID, generation params |
| `bayesian` | Entropy threshold, tokens to skip |
| `wandb` | W&B project, run name, tags |
| `outputs` | Paths for plots, CSV, checkpoints |
| `telemetry` | OTel service name, exporters |

---

## Pipeline Steps

### 01 — Extract Data

**Script:** `scripts/01_extract_data.py`
**Module:** `pharma_slm.data.extraction`

Downloads the Pharmacopoeia PDF (cached on disk) and extracts text using `pymupdf4llm.to_markdown()`. Text is split into fixed-size character chunks and saved as JSONL.

**Optional figure extraction** (set `data.figure_extraction.enabled: true`):
- **Pass 1** — PyMuPDF detects pages with embedded images or large vector drawings
- **Pass 2** — SmolVLM renders each candidate page and generates a scientific description
- Descriptions are appended to the document text before chunking

**Output:** `outputs/data/pharmacopeia_raw_chunks.jsonl`

---

### 02 — Generate Synthetic Data

**Script:** `scripts/02_generate_synthetic.py`
**Module:** `pharma_slm.synthesis.generator`

Loads raw chunks and runs a teacher model via **vLLM** to generate structured pharmaceutical content (tables, figure descriptions, Q&A pairs) using `TABLE_FIGURE_PROMPT`. Outputs JSONL with `original_chunk` and `synthetic_table_figure` fields.

**Multi-GPU:** Set `synthesis.tensor_parallel_size` to the number of GPUs.

**Output:** `outputs/data/synthetic_pharmacopeia_tables.jsonl`

---

### 03 & 04 — Push to Hugging Face Hub

**Modules:** `pharma_slm.hub.upload`

- **03** pushes raw chunks as a plain-text dataset
- **04** formats synthetic pairs as `{"messages": [user, assistant]}` chat records and pushes as an instruct dataset

Both use **versioned Hub branches**: the `main` branch always holds the latest version, and a `v{version}` branch (e.g. `v0.1.0`) is created for reproducibility. Dataset names never change — only the version branch.

---

### 05 — Train

**Script:** `scripts/05_train.py`
**Module:** `pharma_slm.training.trainer`

Fine-tunes a causal LM using **QLoRA** (PEFT + bitsandbytes) via HuggingFace TRL's `SFTTrainer`.

| Component | Detail |
|---|---|
| Base model | `google/gemma-2b-it` (configurable) |
| Quantization | 4-bit NF4 (L4) or bfloat16 full (L40S) |
| LoRA targets | q/k/v/o/gate/up/down projections |
| Optimizer | `paged_adamw_8bit` |
| Scheduler | Cosine with warmup |
| Gradient checkpointing | Enabled by default |
| Eval | 10% hold-out split |

**Multi-GPU:** Detected automatically via `LOCAL_RANK` env var (set by `torchrun`). When multi-GPU, `device_map` is omitted so Accelerate handles placement.

**Callbacks:**
- `OtelMetricsCallback` — emits training loss and learning rate as OTel gauges
- `PlottingCallback` — saves `loss_curve.png`, `lr_schedule.png`, `metrics.csv` after training

**Output:** LoRA adapter saved to `outputs/checkpoints/<model>/final_adapter/`

---

### 06 — Merge & Push

**Module:** `pharma_slm.training.merge`

Merges the LoRA adapter into the base model weights using `PeftModel.merge_and_unload()` and pushes the resulting standalone model to the Hub (same versioned-branch strategy as datasets).

---

### 07 — Inference

**Module:** `pharma_slm.inference.runner`

Loads the merged model from the Hub and runs structured extraction on a pharmacopoeia text chunk using the same `TABLE_FIGURE_PROMPT` used during training. Input can be `--text`, `--file`, or a built-in sample.

---

### 08 — Bayesian Inference

**Module:** `pharma_slm.inference.bayesian`

Runs inference with `output_scores=True` to capture per-token logit distributions. Computes **Shannon entropy** for each token:

```
H(t) = -Σ p(x) log₂ p(x)
```

Tokens above `bayesian.entropy_threshold` are flagged as uncertain. Numeric tokens (drug limits, concentrations, specifications) are highlighted as high-stakes. The output is routed to either:
- **Database commit** — all tokens below threshold
- **SME review queue** — one or more tokens above threshold

---

## Key Components

### Prompt (Single Source of Truth)

`TABLE_FIGURE_PROMPT` is defined once in `pharma_slm.synthesis.generator` and imported by both the synthesis and inference modules, ensuring training and inference use identical prompting.

### Versioned Hub Uploads

```
main branch  ->  always the latest dataset/model
v0.1.0       ->  pinnable snapshot of this experiment run
v0.2.0       ->  next experiment, same repo name
```

### LoRA Configuration (default)

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

## Observability

OpenTelemetry is configured in `pharma_slm.telemetry`. Call `setup_telemetry(cfg.telemetry)` once at startup. Two exporters are supported:

| Exporter | Config | Output |
|---|---|---|
| `console` | `type: console` | Stdout (pretty-printed spans) |
| `file` | `type: file, path: outputs/otel_traces.jsonl` | JSONL trace file |

Every major operation (PDF extraction, figure detection, synthesis, training, inference) is wrapped in an OTel span with relevant attributes.

---

## GPU Environments

| Config | GPU | VRAM | Quantization | Batch Size |
|---|---|---|---|---|
| `lightning_studio.yaml` | L4 | 24 GB | 4-bit NF4 (QLoRA) | 2 |
| `l40s.yaml` | L40S | 48 GB | bfloat16 (full LoRA) | 4 |

The L40S config skips 4-bit quantization (`load_in_4bit: false`) to get higher training throughput and model quality at the cost of memory. Gradient checkpointing is enabled by default to manage activation memory.

---

## Running the Pipeline

```bash
# Install dependencies
pip install -e ".[train]"

# Step 1: Extract PDF (figure extraction optional)
python scripts/01_extract_data.py --config configs/l40s.yaml

# Step 2: Generate synthetic data (requires vLLM)
pip install vllm
python scripts/02_generate_synthetic.py --config configs/l40s.yaml

# Steps 3 & 4: Push datasets to Hub
python scripts/03_push_raw_data.py --config configs/l40s.yaml
python scripts/04_push_synthetic_data.py --config configs/l40s.yaml

# Step 5: Train (single GPU)
PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/05_train.py --config configs/l40s.yaml

# Step 6: Merge adapter and push model
python scripts/06_merge_and_push.py --config configs/l40s.yaml

# Step 7: Run inference
python scripts/07_inference.py --config configs/l40s.yaml --text "Assay. Not less than 99.0%..."

# Step 8: Bayesian uncertainty check
python scripts/08_bayesian_inference.py --config configs/l40s.yaml --text "Assay. Not less than 99.0%..."
```

> All scripts accept `--config` to specify an environment override. Omit it to use `configs/default.yaml` alone.
