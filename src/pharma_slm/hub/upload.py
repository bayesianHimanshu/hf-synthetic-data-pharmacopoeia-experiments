from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from pharma_slm.config import DataConfig, ProjectConfig, SynthesisConfig
from pharma_slm.synthesis.generator import TABLE_FIGURE_PROMPT
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)


def _push_dataset_versioned(
    ds: Dataset,
    repo_id: str,
    version: str,
    token: str | None,
) -> None:
    """Push dataset to Hub on main (latest) and also create a versioned branch.
    """
    # Push to main (latest)
    print(f"Pushing to {repo_id} (main) ...")
    ds.push_to_hub(repo_id, token=token)

    # Create a versioned branch pointing to the same commit
    branch = f"v{version}"
    api = HfApi(token=token)
    api.create_branch(repo_id=repo_id, repo_type="dataset", branch=branch, exist_ok=True)
    print(f"  Branch '{branch}' created/updated → pin with: revision='{branch}'")


def push_raw_chunks(data_cfg: DataConfig, project_cfg: ProjectConfig) -> None:
    """Push the local raw-chunks JSONL to the Hugging Face Hub."""
    token = os.environ.get(project_cfg.hf_token_env)

    with tracer.start_as_current_span("hub.push_raw_chunks") as span:
        span.set_attribute("hub.repo", data_cfg.raw_hub_repo)
        span.set_attribute("hub.version", project_cfg.version)

        print(f"Loading {data_cfg.raw_chunks_path} ...")
        ds = load_dataset("json", data_files=data_cfg.raw_chunks_path, split="train")
        print(f"Pushing {len(ds)} records ...")

        _push_dataset_versioned(ds, data_cfg.raw_hub_repo, project_cfg.version, token)

        span.set_attribute("hub.records_pushed", len(ds))
        print("Done.")


def push_synthetic_instruct(synth_cfg: SynthesisConfig, project_cfg: ProjectConfig) -> None:
    """Format synthetic outputs as user/assistant conversations and push to Hub."""
    token = os.environ.get(project_cfg.hf_token_env)

    with tracer.start_as_current_span("hub.push_synthetic_instruct") as span:
        span.set_attribute("hub.repo", synth_cfg.synthetic_hub_repo)
        span.set_attribute("hub.version", project_cfg.version)

        formatted_data: list[dict] = []
        skipped = 0

        with open(synth_cfg.output_path) as f:
            for line in f:
                record = json.loads(line)
                synthetic_out = record.get("synthetic_table_figure", "")

                if len(synthetic_out.strip()) < synth_cfg.min_output_len:
                    skipped += 1
                    continue

                formatted_data.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": TABLE_FIGURE_PROMPT.format(
                                    text=record["original_chunk"]
                                ),
                            },
                            {"role": "assistant", "content": synthetic_out},
                        ]
                    }
                )

        print(
            f"Prepared {len(formatted_data)} conversational records "
            f"(skipped {skipped} short outputs)."
        )

        ds = Dataset.from_list(formatted_data)
        _push_dataset_versioned(ds, synth_cfg.synthetic_hub_repo, project_cfg.version, token)

        span.set_attribute("hub.records_pushed", len(formatted_data))
        span.set_attribute("hub.records_skipped", skipped)
        print("Done.")
