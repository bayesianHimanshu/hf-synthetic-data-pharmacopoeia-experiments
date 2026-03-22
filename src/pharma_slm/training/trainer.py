from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from pharma_slm.config import OutputsConfig, TrainingConfig, WandbConfig
from pharma_slm.telemetry import get_tracer
from pharma_slm.training.callbacks import OtelMetricsCallback, PlottingCallback

tracer = get_tracer(__name__)


def _is_multi_gpu() -> bool:
    """True when launched via torchrun (LOCAL_RANK env var is set by torch.distributed)."""
    return "LOCAL_RANK" in os.environ


def build_model_and_tokenizer(cfg: TrainingConfig):
    """Load quantised base model + tokenizer.

    Single-GPU / CPU:  device_map="auto" is passed (HF handles placement).
    Multi-GPU torchrun: device_map must NOT be set; accelerate handles sharding.
    """
    print(f"Loading tokenizer for {cfg.base_model_id}.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.quantization.compute_dtype),
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
    )

    load_kwargs: dict = {"quantization_config": bnb_config}
    if not _is_multi_gpu():
        load_kwargs["device_map"] = "auto"

    print(
        f"Loading base model in 4-bit {'(multi-GPU torchrun mode)' if _is_multi_gpu() else '(single-GPU mode)'}."
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, **load_kwargs)
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def _setup_wandb(wandb_cfg: WandbConfig, train_cfg: TrainingConfig) -> None:
    """Initialise a W&B run before the HF Trainer starts.

    Sets WANDB_PROJECT so the Trainer's built-in W&B integration picks up the
    right project automatically. Logs all training hyperparameters as config.
    """
    import wandb  # noqa: PLC0415

    run_name = wandb_cfg.run_name or f"{train_cfg.base_model_id.split('/')[-1]}-v{train_cfg.lora.r}r"
    wandb.init(
        project=wandb_cfg.project,
        name=run_name,
        tags=wandb_cfg.tags,
        config={
            "base_model": train_cfg.base_model_id,
            "dataset": train_cfg.dataset_id,
            "lora_r": train_cfg.lora.r,
            "lora_alpha": train_cfg.lora.alpha,
            "lora_dropout": train_cfg.lora.dropout,
            "learning_rate": train_cfg.learning_rate,
            "epochs": train_cfg.num_train_epochs,
            "batch_size": train_cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
            "lr_scheduler": train_cfg.lr_scheduler_type,
            "warmup_ratio": train_cfg.warmup_ratio,
            "max_length": train_cfg.max_length,
            "bf16": train_cfg.bf16,
            "quant_type": train_cfg.quantization.quant_type,
        },
    )
    print(f"W&B run '{run_name}' started in project '{wandb_cfg.project}'.")


def run_training(train_cfg: TrainingConfig, outputs_cfg: OutputsConfig, wandb_cfg: WandbConfig | None = None) -> None:
    with tracer.start_as_current_span("training.run") as span:
        span.set_attribute("training.base_model", train_cfg.base_model_id)
        span.set_attribute("training.dataset", train_cfg.dataset_id)
        span.set_attribute("training.epochs", train_cfg.num_train_epochs)
        span.set_attribute("training.learning_rate", train_cfg.learning_rate)
        span.set_attribute("training.lora_r", train_cfg.lora.r)

        if wandb_cfg and wandb_cfg.enabled:
            _setup_wandb(wandb_cfg, train_cfg)

        model, tokenizer = build_model_and_tokenizer(train_cfg)

        print(f"Loading dataset {train_cfg.dataset_id} ...")
        dataset = load_dataset(train_cfg.dataset_id, split="train")

        lora_config = PeftLoraConfig(
            r=train_cfg.lora.r,
            lora_alpha=train_cfg.lora.alpha,
            target_modules=train_cfg.lora.target_modules,
            lora_dropout=train_cfg.lora.dropout,
            bias=train_cfg.lora.bias,
            task_type="CAUSAL_LM",
        )

        sft_args = SFTConfig(
            output_dir=train_cfg.output_dir,
            dataset_text_field="messages",
            max_length=train_cfg.max_length,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            optim=train_cfg.optim,
            learning_rate=train_cfg.learning_rate,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_ratio=train_cfg.warmup_ratio,
            num_train_epochs=train_cfg.num_train_epochs,
            logging_steps=train_cfg.logging_steps,
            save_strategy=train_cfg.save_strategy,
            save_steps=train_cfg.save_steps,
            eval_strategy=train_cfg.save_strategy,  # must match save_strategy for load_best_model_at_end
            eval_steps=train_cfg.eval_steps,
            load_best_model_at_end=train_cfg.load_best_model_at_end,
            metric_for_best_model=train_cfg.metric_for_best_model,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            fp16=train_cfg.fp16,
            bf16=train_cfg.bf16,
            report_to="wandb" if (wandb_cfg and wandb_cfg.enabled) else "none",
        )

        callbacks = [
            OtelMetricsCallback(),
            PlottingCallback(
                plots_dir=outputs_cfg.plots_dir,
                csv_path=outputs_cfg.csv_path,
            ),
        ]
        split = dataset.train_test_split(test_size=train_cfg.eval_split, seed=42)
        print("Initialising SFTTrainer ...")
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        print("Starting training.")
        trainer.train()

        span.set_attribute("training.global_steps", trainer.state.global_step)

        adapter_path = f"{train_cfg.output_dir}/{train_cfg.adapter_subdir}"
        print(f"Saving LoRA adapters to {adapter_path}.")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        if wandb_cfg and wandb_cfg.enabled:
            import wandb
            wandb.finish()

        print("Training complete.")
