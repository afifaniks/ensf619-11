import json
import os

import torch
import torch.nn.utils.prune as prune
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
MODEL = "google/codegemma-7b-it"
OUTPUT_DIR = "./pruned_finetuned_model"
GT_PATH = "data/data.json"
SPARSITY = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5


print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=SPARSITY)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_orig"):
        prune.remove(module, "weight")


if not os.path.exists(GT_PATH):
    raise FileNotFoundError(f"{GT_PATH} not found. Prepare data first.")

with open(GT_PATH, "r", encoding="utf-8") as f:
    distil_data = json.load(f)


dataset = Dataset.from_list(distil_data)


def tokenize_fn(batch):
    texts = [p + r for p, r in zip(batch["prompt"], batch["response"])]
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )


tokenized = dataset.map(
    tokenize_fn, batched=True, remove_columns=["prompt", "response"]
)
train_dataset = tokenized


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning pruned model...")
trainer.train()


os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Pruned and fine-tuned model saved to {OUTPUT_DIR}")
