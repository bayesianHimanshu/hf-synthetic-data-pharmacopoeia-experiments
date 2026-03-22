#!/usr/bin/env python3
"""Step 7 - Run extraction inference on a pharmacopoeia text chunk.

Provide input text via --text (inline) or --file (path to a .txt file).
If neither is given, a built-in sample chunk is used.

Usage:
    python scripts/07_inference.py
    python scripts/07_inference.py --text "Ammonium Acetate contains 150 g..."
    python scripts/07_inference.py --file my_chunk.txt
    python scripts/07_inference.py --config configs/my_experiment.yaml --text "..."
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.inference import run_inference

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"

SAMPLE_TEXT = """\
Ammonia-Ammonium Chloride Solution, Strong: A white or off-white crystals; mp, about 186°.
Contains Ammonium Acetate, 0.1 M equivalent to 7.71 g.
Ammonium Acetate Solution contains 150 g.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an override YAML merged on top of default.yaml",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Pharmacopoeia text chunk to process (inline string)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a .txt file containing the pharmacopoeia chunk",
    )
    args = parser.parse_args()

    cfg = load_config(DEFAULT_CONFIG, args.config)
    setup_telemetry(cfg.telemetry)

    if args.file:
        text = args.file.read_text()
    elif args.text:
        text = args.text
    else:
        print("No input provided — using built-in sample text.\n")
        text = SAMPLE_TEXT

    result = run_inference(cfg.inference, text.strip())

    print("\n[SLM EXTRACTION OUTPUT]")
    print(result)


if __name__ == "__main__":
    main()
