from __future__ import annotations

import json
import random
from pathlib import Path

from pharma_slm.config import SynthesisConfig


def inspect_synthetic_data(cfg: SynthesisConfig, num_samples: int = 3) -> None:
    path = Path(cfg.output_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path) as f:
        data = [json.loads(line) for line in f]

    print(f"Total synthetic records: {len(data)}\n")
    samples = random.sample(data, min(num_samples, len(data)))

    for i, sample in enumerate(samples):
        print(f"SAMPLE {i + 1}")
        print("\n[ORIGINAL PHARMACOPEIA CHUNK]")
        print(sample.get("original_chunk", "")[:500] + "...")
        print("\n[SYNTHETIC OUTPUT (Table / Figure / Q&A)]")
        print(sample.get("synthetic_table_figure", "No output found."))
        print()
