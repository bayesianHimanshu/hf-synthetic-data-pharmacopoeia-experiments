import json
import os
from datasets import Dataset

TABLE_FIGURE_PROMPT = """Rewrite the following Pharmacopeia document section to extract and structure all tabular data and visual references. 
First, extract any limits, specifications, or data points and organize them into a clear Markdown table. 
Second, if the text mentions any figures, diagrams, or chromatograms, write a standalone "Figure Description" paragraph detailing exactly what visual setup is required.
Finally, generate one analytical question that can be answered using the table or figure description, and provide the answer.
Output ONLY the table, the figure description (if any), and the Q&A pair.

Document:
{text}"""

def prepare_and_push():
    print("Loading synthetic data...")
    formatted_data = []
    
    with open("synthetic_pharmacopeia_tables.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            original_text = record.get("original_chunk", "")
            synthetic_output = record.get("synthetic_table_figure", "")
            
            # Skip any empty outputs or catastrophic failures just in case
            if not synthetic_output or len(synthetic_output.strip()) < 10:
                continue
                
            # Format into the standard conversational style
            user_content = TABLE_FIGURE_PROMPT.format(text=original_text)
            
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": synthetic_output}
                ]
            })

    print(f"Prepared {len(formatted_data)} conversational records.")
    
    # Convert to a Hugging Face Dataset
    hf_dataset = Dataset.from_list(formatted_data)
    
    # Push to Hub
    repo_id = "EulerianKnight/pharmacopeia-synthetic-instruct"
    
    print(f"Pushing to Hub at {repo_id}...")
    hf_dataset.push_to_hub(repo_id, token=os.environ.get("HUGGINGFACE_API_KEY"))
    print("Done! Dataset is successfully packaged and ready for fine-tuning.")

if __name__ == "__main__":
    prepare_and_push()