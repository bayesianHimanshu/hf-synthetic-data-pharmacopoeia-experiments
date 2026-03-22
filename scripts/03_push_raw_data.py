#!/usr/bin/env python3
"""Step 3 — Push raw pharmacopoeia chunks to Hugging Face Hub.

Usage:
    python scripts/03_push_raw_data.py
    python scripts/03_push_raw_data.py --config configs/my_experiment.yaml
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.hub import push_raw_chunks

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

    push_raw_chunks(cfg.data, cfg.project)


if __name__ == "__main__":
    main()
