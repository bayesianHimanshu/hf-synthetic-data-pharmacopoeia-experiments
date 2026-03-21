import json
import random

def inspect_synthetic_data(filename="synthetic_pharmacopeia_tables.jsonl", num_samples=3):
    print(f"Loading {filename}...\n")
    
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Total synthetic records generated: {len(data)}\n")
    
    # Pick a few random samples to inspect
    samples = random.sample(data, min(num_samples, len(data)))
    
    for i, sample in enumerate(samples):
        print(f"SAMPLE {i+1}:")
        print("\n[ORIGINAL PHARMACOPEIA CHUNK]")
        original_text = sample.get('original_chunk', '')
        print(original_text[:500] + "...\n")
        
        print("\n[SYNTHETIC OUTPUT (Table, Figure, Q&A)]")
        print(sample.get('synthetic_table_figure', 'No output found.'))

if __name__ == "__main__":
    inspect_synthetic_data()