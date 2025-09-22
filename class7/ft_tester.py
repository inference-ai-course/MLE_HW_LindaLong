from fastllm import FastLanguageModel  # Add this import, adjust module name if needed
from transformers import AutoTokenizer  # Add this import

# Define some test questions (ensure these were not exactly in training data)
test_questions = [
    "What is the main hypothesis proposed by the paper on quantum computing?",
    "How did the authors of the deep learning study evaluate their model's performance?",
    # ... (add total 10 questions)
]

# Load the base and fine-tuned models for inference
model_name = "llama3-7b"  # Define your base model name here
base_model = FastLanguageModel.from_pretrained(model_name)  # base 7B model
ft_model = FastLanguageModel.from_pretrained("llama3-7b-qlora-finetuned")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = ("You are a helpful assistant. Answer the question based on the context provided."
)


for q in test_questions:
    prompt_input = f"<|system|>{system_prompt}<|user|>{q}<|assistant|>"
    # Tokenize input and generate output with each model
    input_ids = tokenizer(prompt_input, return_tensors='pt').input_ids.cuda()
    base_output_ids = base_model.generate(input_ids, max_new_tokens=150)
    ft_output_ids  = ft_model.generate(input_ids, max_new_tokens=150)
    # Decode the outputs
    base_answer = tokenizer.decode(base_output_ids[0], skip_special_tokens=True)
    ft_answer   = tokenizer.decode(ft_output_ids[0], skip_special_tokens=True)
    # (Post-process to remove the prompt part if needed)
    base_answer = base_answer.split('<|assistant|>')[-1].strip()
    ft_answer   = ft_answer.split('<|assistant|>')[-1].strip()
    print(f"Q: {q}")
    print(f"Base Model Answer: {base_answer}")
    print(f"Fine-Tuned Model Answer: {ft_answer}")
    print("-" * 60)