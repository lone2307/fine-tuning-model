import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from settings import *

# Path to the fine-tuned model directory

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set padding token if needed
tokenizer.pad_token = tokenizer.eos_token

# Define your prompt
prompt = "### Prompt:\nHow does photosynthesis work?\n\n### Response:\n"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n===== Model Output =====\n")
print(response.split("### Response:")[-1].strip())