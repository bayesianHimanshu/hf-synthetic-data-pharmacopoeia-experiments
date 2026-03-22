#!/usr/bin/env python3
"""Step 2 - Generate synthetic training data from raw chunks using vLLM.

Loads raw chunks from the local JSONL file produced by step 01, then runs
batch inference with the teacher model to produce structured outputs.

Usage:
    python scripts/02_generate_synthetic.py
    python scripts/02_generate_synthetic.py --config configs/lightning_studio.yaml

Multi-GPU (vLLM tensor parallelism):
    Set synthesis.tensor_parallel_size in your config yaml, then run normally -
    vLLM handles multi-GPU internally without torchrun.
"""
import argparse
import json
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.synthesis import generate_synthetic_data

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an override YAML merged on top of default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(DEFAULT_CONFIG, args.config)
    setup_telemetry(cfg.telemetry)

    raw_chunks_path = Path(cfg.data.raw_chunks_path)
    if not raw_chunks_path.exists():
        raise FileNotFoundError(
            f"Raw chunks not found at {raw_chunks_path}. Run 01_extract_data.py first."
        )

    print(f"Loading raw chunks from {raw_chunks_path}.")
    with open(raw_chunks_path) as f:
        raw_chunks = [json.loads(line)["text"] for line in f]
    print(f"Loaded {len(raw_chunks)} chunks.")

    generate_synthetic_data(cfg.synthesis, raw_chunks)


if __name__ == "__main__":
    main()
