import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
login()

# Configuration
MODEL_ID = "google/gemma-2b-it" 
DATASET_ID = "EulerianKnight/pharmacopeia-synthetic-instruct"
OUTPUT_DIR = "./gemma-pharmacopeia-slm"

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Gemma requires pad_token to be set for batching
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right"

    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")

    # Quantization Setup (QLoRA)
    # Load the base model in 4-bit precision to fit on the L4 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for gradient checkpointing to save more memory
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # Training Configuration
    # Using paged_adamw_8bit to page optimizer states to CPU if VRAM spikes
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="messages", # TRL expects the column name with the chat data
        max_length=2048, # Max context window for our training samples
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # Effective batch size = 8
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
    )

    # TRL SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Execute Training
    print("Starting training loop...")
    trainer.train()

    print("Training complete! Saving LoRA adapters...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print("Done!")

if __name__ == "__main__":
    main()