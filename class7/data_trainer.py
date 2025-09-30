from unsloth import FastLanguageModel#, SFTTrainer 
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset


#Load the base LLaMA 2 7B model in 4-bit mode (dynamic 4-bit quantization)
model_name = "unsloth/llama-2-7b-bnb-4bit" # or "unsloth/llama-3.1-7b-unsloth-bnb-4bit"
#model, tokenizer = FastLanguageModel.from_pretrained(model_name)#, AutoTokenizer.from_pretrained(model_name, use_fast=False) 
#trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=None)  # Placeholder for trainer initialization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters to make the model trainable
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,           # LoRA scaling parameter
    lora_dropout=0,          # Dropout for LoRA layers
    bias="none",             # Bias type
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load our synthetic Q&A dataset
dataset = load_dataset("json", data_files="synthetic_qa.jsonl", split="train")

# Initialize the trainer for Supervised Fine-Tuning (SFT)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        output_dir="llama3-7b-qlora-finetuned",
        per_device_train_batch_size=4,   # small batch size for Colab GPU
        gradient_accumulation_steps=4,   # accumulate gradients to simulate larger batch
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=50,
        save_strategy="epoch"
    )
)

trainer.train()



# Save the fine-tuned model
model.save_pretrained("llama2-7b-qlora-finetuned")
tokenizer.save_pretrained("llama2-7b-qlora-finetuned")

print("✅ Training completed! Model saved to llama2-7b-qlora-finetuned/")
