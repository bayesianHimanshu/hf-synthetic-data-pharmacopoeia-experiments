from __future__ import annotations

import json
from pathlib import Path

from pharma_slm.config import SynthesisConfig
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)

# Single source of truth for the extraction prompt.
TABLE_FIGURE_PROMPT = """\
Rewrite the following Pharmacopeia document section to extract and structure all \
tabular data and visual references.
First, extract any limits, specifications, or data points and organize them into \
a clear Markdown table.
Second, if the text mentions any figures, diagrams, or chromatograms, write a \
standalone "Figure Description" paragraph detailing exactly what visual setup is required.
Finally, generate one analytical question that can be answered using the table or \
figure description, and provide the answer.
Output ONLY the table, the figure description (if any), and the Q&A pair.

Document:
{text}"""


def generate_synthetic_data(cfg: SynthesisConfig, raw_chunks: list[str]) -> None:
    """Run vLLM batch inference over raw_chunks and save results to cfg.output_path.
    """
    from vllm import LLM, SamplingParams

    with tracer.start_as_current_span("synthesis.generate") as span:
        span.set_attribute("synthesis.model", cfg.model_name)
        span.set_attribute("synthesis.num_chunks", len(raw_chunks))
        span.set_attribute("synthesis.tensor_parallel_size", cfg.tensor_parallel_size)

        formatted_prompts = [
            [{"role": "user", "content": TABLE_FIGURE_PROMPT.format(text=chunk)}]
            for chunk in raw_chunks
        ]

        print(f"Initialising vLLM with {cfg.model_name} (tensor_parallel_size={cfg.tensor_parallel_size})")
        llm = LLM(
            model=cfg.model_name,
            max_model_len=cfg.max_model_len,
            tensor_parallel_size=cfg.tensor_parallel_size,
        )

        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop_tokens,
        )

        print(f"Running batch generation over {len(raw_chunks)} chunks.")
        outputs = llm.chat(messages=formatted_prompts, sampling_params=sampling_params)

        results = [
            {
                "original_chunk": raw_chunks[i],
                "synthetic_table_figure": output.outputs[0].text,
            }
            for i, output in enumerate(outputs)
        ]

        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        span.set_attribute("synthesis.output_records", len(results))
        print(f"Saved {len(results)} synthetic records to {out_path}")
