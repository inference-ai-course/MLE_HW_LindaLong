from unsloth import FastLanguageModel, SFTTrainer 
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset

# Load the base LLaMA 3 7B model in 4-bit mode (dynamic 4-bit quantization)
model_name = "unsloth/llama-3.1-7b-unsloth-bnb-4bit"
model = FastLanguageModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load our synthetic Q&A dataset
dataset = load_dataset("json", data_files="all_qa.jsonl", split="train")

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
        fp16=True,
        logging_steps=50,
        save_strategy="epoch"
    )
)

trainer.train()
model.save_pretrained("llama3-7b-qlora-finetuned")