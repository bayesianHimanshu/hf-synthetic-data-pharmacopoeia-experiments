import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EulerianKnight/gemma-2b-pharmacopeia-slm"

# The prompt template your model has mastered
TABLE_FIGURE_PROMPT = """Rewrite the following Pharmacopeia document section to extract and structure all tabular data and visual references. 
First, extract any limits, specifications, or data points and organize them into a clear Markdown table. 
Second, if the text mentions any figures, diagrams, or chromatograms, write a standalone "Figure Description" paragraph detailing exactly what visual setup is required.
Finally, generate one analytical question that can be answered using the table or figure description, and provide the answer.
Output ONLY the table, the figure description (if any), and the Q&A pair.

Document:
{text}"""

# A messy sample text simulating a Pharmacopeia PDF chunk
SAMPLE_TEXT = """
Ammonia-Ammonium Chloride Solution, Strong: A white or off-white crystals; mp, about 186°. 
Contains Ammonium Acetate, 0.1 M equivalent to 7.71 g. 
Ammonium Acetate Solution contains 150 g.
"""

def main():
    print(f"Downloading and loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Wrap the sample text in the prompt template
    user_content = TABLE_FIGURE_PROMPT.format(text=SAMPLE_TEXT.strip())
    
    # Apply Gemma's required conversational chat template
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating structural extraction...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )
    
    # Decode only the newly generated tokens, skipping the prompt
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("[SLM EXTRACTION OUTPUT]")
    print(generated_text)

if __name__ == "__main__":
    main()