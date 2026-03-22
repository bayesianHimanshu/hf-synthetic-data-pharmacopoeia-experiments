from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pharma_slm.config import BayesianConfig, InferenceConfig
from pharma_slm.synthesis.generator import TABLE_FIGURE_PROMPT
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)


def calculate_shannon_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of a single token's probability distribution (in bits).
    """
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-10)
    entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
    return entropy.item()


def run_bayesian_inference(
    inference_cfg: InferenceConfig,
    bayesian_cfg: BayesianConfig,
    text: str,
) -> dict:
    """Generate a structured extraction and validate each token's uncertainty.
    """
    with tracer.start_as_current_span("bayesian.inference") as span:
        span.set_attribute("bayesian.model", inference_cfg.model_id)
        span.set_attribute("bayesian.entropy_threshold", bayesian_cfg.entropy_threshold)

        print(f"Loading {inference_cfg.model_id} for Bayesian inference ...")
        tokenizer = AutoTokenizer.from_pretrained(inference_cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            inference_cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        user_content = TABLE_FIGURE_PROMPT.format(text=text)
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("Generating and collecting token distributions")
        outputs = model.generate(
            **inputs,
            max_new_tokens=inference_cfg.max_new_tokens,
            temperature=inference_cfg.temperature,
            do_sample=inference_cfg.do_sample,
            output_scores=True,             # return raw logits per step
            return_dict_in_generate=True,   # structured output object
        )

        generated_sequence = outputs.sequences[0][inputs.input_ids.shape[1]:]
        scores = outputs.scores             # tuple of (vocab_size,) tensors, one per step

        skip_set = set(bayesian_cfg.skip_tokens)
        flagged: list[dict] = []
        print("[BAYESIAN VALIDATION: TOKEN-LEVEL UNCERTAINTY]")

        for i, token_id in enumerate(generated_sequence):
            token_text = tokenizer.decode(token_id)
            if token_text.strip() in skip_set:
                continue

            entropy = calculate_shannon_entropy(scores[i][0])
            has_number = any(ch.isdigit() for ch in token_text)

            if entropy > bayesian_cfg.entropy_threshold:
                marker = "[FLAGGED]  " if has_number else "   [Uncertain]"
                print(f"{marker} Token: {token_text!r:<12} | Entropy: {entropy:.3f}")
                flagged.append(
                    {"token": token_text, "entropy": entropy, "has_number": has_number}
                )

        output_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

        print("[FINAL TEXT OUTPUT]")
        print(output_text)

        validated = len(flagged) == 0
        if validated:
            print("SYSTEM RESULT: Zero critical uncertainty detected. Data Integrity Confirmed.")
        else:
            print(
                f"SYSTEM RESULT: High uncertainty detected in {len(flagged)} token(s). "
                "Routing to SME for manual review."
            )
        span.set_attribute("bayesian.flagged_tokens", len(flagged))
        span.set_attribute("bayesian.validated", validated)

        return {"flagged": flagged, "output_text": output_text, "validated": validated}
