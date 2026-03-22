#!/usr/bin/env python3
"""Step 1 - Download the Pharmacopoeia PDF and chunk it into a JSONL file.

Usage:
    python scripts/01_extract_data.py
    python scripts/01_extract_data.py --config configs/my_experiment.yaml
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.data import download_pdf, extract_and_chunk_pdf, save_chunks_jsonl

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

    download_pdf(cfg.data)
    chunks = extract_and_chunk_pdf(cfg.data)
    save_chunks_jsonl(chunks, cfg.data)


if __name__ == "__main__":
    main()
