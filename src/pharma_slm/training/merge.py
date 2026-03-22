from __future__ import annotations

import os

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pharma_slm.config import MergeConfig, ProjectConfig
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)


def merge_adapter_and_push(cfg: MergeConfig, project_cfg: ProjectConfig) -> None:
    """Merge LoRA adapter into the base model and push the standalone model to Hub.
    """
    token = os.environ.get(project_cfg.hf_token_env)

    with tracer.start_as_current_span("merge.run") as span:
        span.set_attribute("merge.base_model", cfg.base_model_id)
        span.set_attribute("merge.adapter_dir", cfg.adapter_dir)
        span.set_attribute("merge.target_repo", cfg.merged_hub_repo)
        span.set_attribute("merge.version", project_cfg.version)

        dtype = getattr(torch, cfg.torch_dtype)

        print(f"Loading base model {cfg.base_model_id} in {cfg.torch_dtype} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            torch_dtype=dtype,
            device_map="auto",
        )

        print(f"Loading tokenizer from {cfg.adapter_dir} ...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.adapter_dir)

        print("Applying LoRA adapter ...")
        model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)

        print("Merging and unloading adapter ...")
        model = model.merge_and_unload()

        # Push to main (latest)
        print(f"Pushing merged model to {cfg.merged_hub_repo} (main) ...")
        model.push_to_hub(cfg.merged_hub_repo, token=token)
        tokenizer.push_to_hub(cfg.merged_hub_repo, token=token)

        # Create versioned branch pointing to this commit
        branch = f"v{project_cfg.version}"
        api = HfApi(token=token)
        api.create_branch(
            repo_id=cfg.merged_hub_repo,
            repo_type="model",
            branch=branch,
            exist_ok=True,
        )
        print(f"Branch '{branch}' created/updated; pin with: revision='{branch}'")
        print("Completed.")
