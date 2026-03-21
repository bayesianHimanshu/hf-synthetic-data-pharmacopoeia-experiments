from datasets import load_dataset

# Load the local JSONL file
dataset = load_dataset("json", data_files="pharmacopeia_raw_chunks.jsonl", split="train")

# Push it to Hugging Face Hub account
repo_id = "EulerianKnight/pharmacopeia-raw-chunks"

print(f"Pushing dataset to {repo_id}...")
dataset.push_to_hub(repo_id)
print("Successfully pushed to the Hub!")