#!/usr/bin/env python3
"""Step 6 — Merge LoRA adapters into the base model and push to Hub.

The merged model is a standalone model (no adapter files needed at inference
time) pushed to the repo specified in config.merge.merged_hub_repo.

Usage:
    python scripts/06_merge_and_push.py
    python scripts/06_merge_and_push.py --config configs/my_experiment.yaml
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.training import merge_adapter_and_push

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

    merge_adapter_and_push(cfg.merge, cfg.project)


if __name__ == "__main__":
    main()
