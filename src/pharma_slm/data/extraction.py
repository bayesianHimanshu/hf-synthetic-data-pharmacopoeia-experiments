from __future__ import annotations

import urllib.request
from pathlib import Path

import pymupdf4llm
from datasets import Dataset

from pharma_slm.config import DataConfig
from pharma_slm.telemetry import get_tracer

tracer = get_tracer(__name__)


def download_pdf(cfg: DataConfig) -> Path:
    dest = Path(cfg.pdf_filename)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tracer.start_as_current_span("data.download_pdf") as span:
        span.set_attribute("pdf.url", cfg.pdf_url)
        span.set_attribute("pdf.dest", str(dest))
        if dest.exists():
            span.set_attribute("pdf.cached", True)
            print(f"PDF already exists at {dest}, skipping download.")
        else:
            print(f"Downloading PDF to {dest} ...")
            urllib.request.urlretrieve(cfg.pdf_url, dest)
            span.set_attribute("pdf.cached", False)
            print("Download complete.")

    return dest


def extract_and_chunk_pdf(cfg: DataConfig) -> list[str]:
    """Extract text from the PDF and chunk it into segments.

    When cfg.figure_extraction.enabled is True, runs a second pass using a vision
    LLM (SmolVLM by default) to describe every page that contains figures, diagrams,
    chromatograms, or spectra. The description is appended to that page's text
    *before* chunking so the synthetic data teacher model sees the full content.
    """
    fig_cfg = cfg.figure_extraction

    with tracer.start_as_current_span("data.extract_and_chunk") as span:
        span.set_attribute("pdf.filename", cfg.pdf_filename)
        span.set_attribute("chunk.size", cfg.chunk_size)
        span.set_attribute("figure_extraction.enabled", fig_cfg.enabled)

        if fig_cfg.enabled:
            page_texts = _extract_with_figures(cfg)
        else:
            print(f"Extracting markdown from {cfg.pdf_filename} ...")
            # Single-pass: full-document extraction (matches the original script behaviour)
            full_markdown = pymupdf4llm.to_markdown(cfg.pdf_filename)
            print(f"Extracted markdown: {len(full_markdown):,} characters.")
            page_texts = [full_markdown]

        # Join pages, then do character-level chunking
        full_text = "\n\n".join(page_texts)
        chunks = [
            full_text[i : i + cfg.chunk_size]
            for i in range(0, len(full_text), cfg.chunk_size)
        ]
        chunks = [c.strip() for c in chunks if len(c.strip()) > cfg.min_chunk_len]

        span.set_attribute("chunks.count", len(chunks))
        print(f"Created {len(chunks)} chunks.")

    return chunks


def _extract_with_figures(cfg: DataConfig) -> list[str]:
    """Two-pass extraction: full-doc text + vision LLM figure descriptions.

    Uses the same full-document pymupdf4llm extraction as the non-figure path so
    chunk counts stay consistent.  Figure descriptions are appended at the end of
    the document text rather than inserted mid-document.
    """
    # Lazy import — figure_extraction module only needed when enabled
    from pharma_slm.data.figure_extraction import detect_figure_pages, describe_figures
    import fitz  # PyMuPDF — already a hard dependency via figure_extraction

    fig_cfg = cfg.figure_extraction

    # Pass 1: full-document text extraction (same quality as the non-figure path)
    print(f"Extracting full markdown from {cfg.pdf_filename}.")
    full_markdown = pymupdf4llm.to_markdown(cfg.pdf_filename)

    with fitz.open(cfg.pdf_filename) as _doc:
        total_pages = len(_doc)

    import os
    file_size_mb = os.path.getsize(cfg.pdf_filename) / (1024 * 1024)
    print(f"PDF has {total_pages} pages, file size {file_size_mb:.1f} MB.")
    print(f"Extracted markdown: {len(full_markdown):,} characters ({len(full_markdown) // cfg.chunk_size} expected chunks before filtering).")

    # Pass 2: detect and describe figure pages
    figure_page_indices = detect_figure_pages(cfg.pdf_filename, fig_cfg)
    descriptions = describe_figures(cfg.pdf_filename, figure_page_indices, fig_cfg)

    # Append figure descriptions after the main text
    if descriptions:
        addendum = "\n\n".join(
            f"[FIGURE DESCRIPTION - Page {idx + 1}]: {desc}"
            for idx, desc in sorted(descriptions.items())
        )
        full_markdown += f"\n\n{addendum}"

    print(f"Enriched {len(descriptions)}/{total_pages} pages with figure descriptions.")
    return [full_markdown]


def save_chunks_jsonl(chunks: list[str], cfg: DataConfig) -> None:
    out = Path(cfg.raw_chunks_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tracer.start_as_current_span("data.save_chunks") as span:
        span.set_attribute("output.path", str(out))
        ds = Dataset.from_dict({"text": chunks})
        ds.to_json(str(out))
        print(f"Saved {len(chunks)} chunks to {out}")
