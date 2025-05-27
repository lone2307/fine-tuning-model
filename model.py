import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from settings import *
from dataloader import dataLoader

# Initialize wandb
wandb.init(project="qwen-lora", name="oasst-qwen-0.5B")

# Loading dataset and tokenizer
tokenized_dataset = dataLoader(dataset_name)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with LoRA config
print("Loading model with LoRA config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj"],  # Adjust as needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Set up training
training_args = TrainingArguments(
    output_dir="./lora-qwen-oasst",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=2,
    save_steps=200,
    learning_rate=2e-4,
    bf16=True,
    report_to="wandb",
    run_name="oasst-qwen-0.5b"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save
print("Saving model...")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)