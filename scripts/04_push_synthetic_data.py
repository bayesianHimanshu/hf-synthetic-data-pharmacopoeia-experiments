#!/usr/bin/env python3
"""Step 4 - Format synthetic data as conversations and push to Hugging Face Hub.

Reads the synthetic JSONL from step 02, wraps each record in a user/assistant
chat format, filters short outputs, and uploads to Hub as an instruct dataset.

Usage:
    python scripts/04_push_synthetic_data.py
    python scripts/04_push_synthetic_data.py --config configs/my_experiment.yaml
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.hub import push_synthetic_instruct

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

    push_synthetic_instruct(cfg.synthesis, cfg.project)


if __name__ == "__main__":
    main()
