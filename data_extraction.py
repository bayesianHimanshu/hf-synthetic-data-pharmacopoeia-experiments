import os
import urllib.request
import pymupdf4llm
from datasets import Dataset

PDF_URL = "https://qps.nhsrcindia.org/sites/default/files/2022-01/INDIAN%20PHARMACOPOEIA%202010%20Volume%201.pdf"
PDF_FILENAME = "IP_2010_Vol1.pdf"

def download_pdf(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("File already exists.")

def extract_and_chunk_pdf(filename, chunk_size=3000):
    md_text = pymupdf4llm.to_markdown(filename)
    
    # Simple character level chunking
    chunks = [md_text[i:i + chunk_size] for i in range(0, len(md_text), chunk_size)]
    
    # Clean up any tiny residual chunks at the end
    chunks = [c.strip() for c in chunks if len(c.strip()) > 100]
    
    print(f"Created {len(chunks)} chunks.")
    return chunks

def main():
    download_pdf(PDF_URL, PDF_FILENAME)
    chunks = extract_and_chunk_pdf(PDF_FILENAME)
    
    # Structure it as a dictionary for the Hugging Face Datasets library
    dataset_dict = {"text": chunks}
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Save locally to verify
    hf_dataset.to_json("pharmacopeia_raw_chunks.jsonl")
    print("Saved to pharmacopeia_raw_chunks.jsonl. Ready for the Hub!")

if __name__ == "__main__":
    main()