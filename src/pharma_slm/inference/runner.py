from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pharma_slm.config import InferenceConfig
from pharma_slm.synthesis.generator import TABLE_FIGURE_PROMPT
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)


def load_model(cfg: InferenceConfig):
    print(f"Loading model {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def run_inference(cfg: InferenceConfig, text: str) -> str:
    """Run extraction on a single pharmacopeia text chunk.
    """
    with tracer.start_as_current_span("inference.run") as span:
        span.set_attribute("inference.model", cfg.model_id)
        span.set_attribute("inference.input_length", len(text))

        model, tokenizer = load_model(cfg)

        user_content = TABLE_FIGURE_PROMPT.format(text=text)
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("Generating")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
        )

        # Decode only the newly generated tokens, skipping the prompt
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)

        span.set_attribute("inference.output_length", len(result))
        return result
