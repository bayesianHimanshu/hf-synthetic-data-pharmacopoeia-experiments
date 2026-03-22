#!/usr/bin/env python3
"""Step 8 - Run Bayesian (token-level uncertainty) inference.

Generates a structured extraction from the fine-tuned model, then computes the
Shannon entropy of each token's logit distribution. Tokens above the configured
entropy threshold are flagged - numeric tokens (drug limits, specifications) are
highlighted more prominently as they carry higher stakes.

Outputs a final routing decision:
  Zero uncertainty -> Data Integrity Confirmed, route to database
  High uncertainty detected -> route to SME for manual review

Usage:
    python scripts/08_bayesian_inference.py
    python scripts/08_bayesian_inference.py --text "Ammonium Acetate contains 150 g..."
    python scripts/08_bayesian_inference.py --file my_chunk.txt
    python scripts/08_bayesian_inference.py --config configs/my_experiment.yaml

To tune sensitivity, adjust bayesian.entropy_threshold in your config YAML:
  - Lower value (e.g. 0.5) -> more tokens flagged (stricter)
  - Higher value (e.g. 2.0) -> fewer tokens flagged (more lenient)
"""
import argparse
from pathlib import Path

from pharma_slm import load_config, setup_telemetry
from pharma_slm.inference import run_bayesian_inference

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

    run_bayesian_inference(cfg.inference, cfg.bayesian, text.strip())


if __name__ == "__main__":
    main()
