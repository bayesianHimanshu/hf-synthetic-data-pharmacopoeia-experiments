import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "google/gemma-2b-it"
ADAPTER_DIR = "./gemma-pharmacopeia-slm/final_adapter"
NEW_REPO_ID = "EulerianKnight/gemma-2b-pharmacopeia-slm"

def main():
    print("Loading base model in bfloat16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print("Applying trained LoRA adapter to base model...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging adapter with base model... This might take a minute.")
    model = model.merge_and_unload()

    print(f"Pushing standalone model to {NEW_REPO_ID}...")
    model.push_to_hub(NEW_REPO_ID)
    tokenizer.push_to_hub(NEW_REPO_ID)
    print("Successfully pushed to the Hub! Your SLM is ready.")

if __name__ == "__main__":
    main()