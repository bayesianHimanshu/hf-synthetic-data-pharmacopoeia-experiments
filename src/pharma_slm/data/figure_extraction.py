"""
Two-pass figure extraction for pharmacopoeia PDFs.

Pass 1 (detect):  Use PyMuPDF to find pages that contain raster images or
                  significant vector drawings (chromatograms, apparatus, spectra).

Pass 2 (describe): Render each figure page to a full-resolution image and pass it
                   to a vision LLM (SmolVLM by default) for a detailed description.
                   Descriptions are then appended to the corresponding text chunk
                   BEFORE synthetic data generation, so the teacher model sees the
                   full content of each page.
"""
from __future__ import annotations

import gc
import io

import fitz  # PyMuPDF

from pharma_slm.config import FigureExtractionConfig
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)

# Prompt used for every figure page.
FIGURE_DESCRIPTION_PROMPT = (
    "You are analysing a page from the Indian Pharmacopoeia (a pharmaceutical regulatory document). "
    "Describe every scientific figure on this page in detail:\n"
    "- Chromatograms: list retention times, peak heights, USP tailing factors, and mobile phase if shown.\n"
    "- Apparatus diagrams: name components, note dimensions and labels.\n"
    "- UV/IR spectra: note wavelengths and absorbance peaks.\n"
    "- Chemical structures: name functional groups or substituents visible.\n"
    "- Tables or reference standards: list all numerical values.\n"
    "Include every number visible in the figure. "
    "If the page contains no meaningful scientific figure (only text or decorative lines), "
    "respond with exactly: NO_FIGURE"
)


def detect_figure_pages(pdf_path: str, cfg: FigureExtractionConfig) -> list[int]:
    """Return page indices (0-based) that contain images or significant vector drawings."""
    doc = fitz.open(pdf_path)
    figure_pages: list[int] = []

    with tracer.start_as_current_span("figure_extraction.detect_pages") as span:
        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # Raster images embedded in the page
            if page.get_images(full=False):
                figure_pages.append(page_idx)
                continue

            # Vector drawings (chromatograms, apparatus, spectra drawn as PDF paths)
            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect and (rect.width * rect.height) >= cfg.min_drawing_area:
                    figure_pages.append(page_idx)
                    break

        doc.close()
        span.set_attribute("figure_extraction.pages_with_figures", len(figure_pages))

    print(f"Detected {len(figure_pages)} pages with figures out of {len(doc)} total pages.")
    return figure_pages


def describe_figures(
    pdf_path: str,
    page_indices: list[int],
    cfg: FigureExtractionConfig,
) -> dict[int, str]:
    """Render figure pages and generate descriptions using a vision LLM.

    Returns a dict mapping page_index -> description string.
    Pages where the model responds NO_FIGURE are omitted.
    """
    if not page_indices:
        return {}

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    descriptions: dict[int, str] = {}

    with tracer.start_as_current_span("figure_extraction.describe_figures") as span:
        span.set_attribute("figure_extraction.vision_model", cfg.vision_model)
        span.set_attribute("figure_extraction.num_pages", len(page_indices))

        print(f"Loading vision model {cfg.vision_model}.")
        processor = AutoProcessor.from_pretrained(cfg.vision_model)
        vision_model = AutoModelForImageTextToText.from_pretrained(
            cfg.vision_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        doc = fitz.open(pdf_path)
        render_matrix = fitz.Matrix(cfg.dpi / 72, cfg.dpi / 72)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": FIGURE_DESCRIPTION_PROMPT},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        total = len(page_indices)
        for i, page_idx in enumerate(page_indices):
            # Render the page to a PIL image
            pixmap = doc[page_idx].get_pixmap(matrix=render_matrix)
            pil_image = Image.open(io.BytesIO(pixmap.tobytes("png")))

            inputs = processor(
                text=prompt_text,
                images=[pil_image],
                return_tensors="pt",
            ).to(vision_model.device)

            output_ids = vision_model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
            )

            # Decode only the newly generated tokens (skip the prompt)
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            description = processor.decode(new_tokens, skip_special_tokens=True).strip()

            if description and description != "NO_FIGURE":
                descriptions[page_idx] = description
                print(f"  [{i+1}/{total}] Page {page_idx+1}: described ({len(description)} chars)")
            else:
                print(f"  [{i+1}/{total}] Page {page_idx+1}: no meaningful figure, skipping")

        doc.close()

        # Free GPU memory so the synthesis model can load cleanly afterwards
        del vision_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        span.set_attribute("figure_extraction.descriptions_generated", len(descriptions))

    print(f"Figure extraction complete: {len(descriptions)} descriptions generated.")
    return descriptions
