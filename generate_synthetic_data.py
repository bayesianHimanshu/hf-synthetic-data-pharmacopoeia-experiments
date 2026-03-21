from datasets import load_dataset
from vllm import LLM, SamplingParams
import json

# Configuration
DATASET_REPO = "EulerianKnight/pharmacopeia-raw-chunks"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Our custom prompt
TABLE_FIGURE_PROMPT = """Rewrite the following Pharmacopeia document section to extract and structure all tabular data and visual references. 
First, extract any limits, specifications, or data points and organize them into a clear Markdown table. 
Second, if the text mentions any figures, diagrams, or chromatograms, write a standalone "Figure Description" paragraph detailing exactly what visual setup is required.
Finally, generate one analytical question that can be answered using the table or figure description, and provide the answer.
Output ONLY the table, the figure description (if any), and the Q&A pair.

Document:
{text}"""

def main():
    print(f"Loading dataset from {DATASET_REPO}.")
    dataset = load_dataset(DATASET_REPO, split="train")
    raw_chunks = dataset["text"]
    
    print(f"Loaded {len(raw_chunks)} chunks. Formatting prompts.")
    
    # Wrap each chunk in the prompt template and apply the chat structure
    formatted_prompts = []
    for chunk in raw_chunks:
        user_message = TABLE_FIGURE_PROMPT.format(text=chunk)
        messages = [{"role": "user", "content": user_message}]
        formatted_prompts.append(messages)

    print(f"Initializing vLLM engine with {MODEL_NAME}.")
    # Load the model into the GPU. Set max_model_len to 8192 as per the playbook.
    llm = LLM(model=MODEL_NAME, max_model_len=8192, tensor_parallel_size=1)
    
    # Set generation parameters (Temperature 0 for factual extraction)
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=2048,
        # Stop tokens help prevent the model from rambling
        stop=["<|im_end|>"] 
    )
    
    print("Starting high-throughput generation.")
    # This is where the magic happens. vLLM will batch and process these incredibly fast.
    outputs = llm.chat(messages=formatted_prompts, sampling_params=sampling_params)
    
    print("Generation complete! Saving results.")
    
    # Save the original text alongside the synthetic output
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({
            "original_chunk": raw_chunks[i],
            "synthetic_table_figure": generated_text
        })
        
    with open("synthetic_pharmacopeia_tables.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Successfully saved to synthetic_pharmacopeia_tables.jsonl!")

if __name__ == "__main__":
    main()