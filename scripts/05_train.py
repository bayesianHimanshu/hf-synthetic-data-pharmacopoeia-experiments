#!/usr/bin/env python3
"""Step 5 - Fine-tune Gemma 2B with QLoRA on the synthetic instruct dataset.

Usage:
    Single GPU:
        python scripts/05_train.py
        python scripts/05_train.py --config configs/lightning_studio.yaml

    Multi-GPU (Lightning Studio / any torchrun environment):
        torchrun --nproc_per_node=4 scripts/05_train.py --config configs/lightning_studio.yaml

    The multi-GPU path omits device_map="auto" automatically (detected via
    the LOCAL_RANK env var set by torchrun).
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.training import run_training

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

    run_training(cfg.training, cfg.outputs, cfg.wandb)


if __name__ == "__main__":
    main()
