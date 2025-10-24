import json
import os

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

TEACHER_MODEL = "google/codegemma-7b-it"
STUDENT_MODEL = "google/codegemma-1.1-2b"
OUTPUT_DIR = "./distilled"
DISTILLATION_DATA_PATH = "data/distillation_data.json"
DATASET_NAME = "mbpp"  # Mostly Basic Programming Problems
MAX_SAMPLES = 500  # adjust for scale
MAX_LENGTH = 512
TEMPERATURE = 0.3
USE_LORA = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512


print("Loading MBPP dataset...")

# Load MBPP dataset - has task descriptions that work as docstrings
dataset = load_dataset(DATASET_NAME, split=["train", "test"])
dataset = concatenate_datasets(dataset)


# Filter for valid examples
def is_valid_mbpp(example):
    code = example.get("code", "")
    text = example.get("text", "")

    # Keep examples with reasonable length and valid Python code
    has_code = len(code) > 20 and "def " in code
    has_description = len(text) > 10

    return has_code and has_description


filtered_dataset = dataset.filter(is_valid_mbpp)
dataset = filtered_dataset.select(range(min(MAX_SAMPLES, len(filtered_dataset))))

# Print sample to verify
print(f"\nDataset size: {len(dataset)}")
print("\nSample code:")
print(dataset[0]["code"][:300])
print("\nSample description:")
print(dataset[0]["text"][:200])


print("\n" + "=" * 80)
print("LOADING MODELS FOR PRE-DISTILLATION TESTING")
print("=" * 80)

print("\nLoading teacher model...")
teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

print("Loading student model...")
student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
student_tok.pad_token = student_tok.eos_token

# Copy chat template from teacher if student doesn't have one
if not student_tok.chat_template:
    print("Student model has no chat template. Copying from teacher model...")
    student_tok.chat_template = teacher_tok.chat_template

student_untrained = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
).eval()


print("\n" + "=" * 80)
print("TESTING MODELS BEFORE DISTILLATION")
print("=" * 80)

print("\n" + "=" * 80)
print("PRE-DISTILLATION TESTING COMPLETE - Starting training preparation...")
print("=" * 80 + "\n")

# Verify student tokenizer has chat template
if not student_tok.chat_template:
    raise ValueError(
        "Student tokenizer missing chat template! "
        "This should have been copied from teacher."
    )


print("Preparing training data from MBPP...")


def prepare_training_example(example, tokenizer):
    print(example)
    """Create prompt-response pairs from MBPP examples using chat template."""
    # code = example.get("code", "")
    # Create messages in chat format (CodeGemma style - no system message)
    user_content = (
        "Add a Google-style Python docstring to the following function. "
        "Include Args, Returns, and Examples sections. "
        "Output only the updated function code. No extra text or explanation."
        f"\n\n{example}"
    )
    msgs = [{"role": "user", "content": user_content}]

    # Apply chat template to create properly formatted prompt
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    teacher_prompt = teacher_tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(teacher_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        teacher_output = teacher.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
        )
    teacher_text = teacher_tok.decode(
        teacher_output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    print(f"Generated teacher response:\n{teacher_text}\n\n")

    return {"prompt": prompt, "response": teacher_text}


# Read when distillation data already prepared
if os.path.exists(DISTILLATION_DATA_PATH):
    print("Loading existing distillation data from distillation_data.json...")
    distil_data = json.load(open(DISTILLATION_DATA_PATH, "r", encoding="utf-8"))
else:
    # Prepare all training examples (pass tokenizer for chat template)
    distil_data = [
        prepare_training_example(ex, student_tok)
        for ex in tqdm(dataset["code"], desc="Preparing examples")
    ]
    json.dump(distil_data, open(DISTILLATION_DATA_PATH, "w", encoding="utf-8"))


print("\n" + "=" * 80)
print("SAMPLE TRAINING DATA")
print("=" * 80)

num_samples = min(3, len(distil_data))
for i in range(num_samples):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*80}")
    print(f"\nPROMPT:\n{distil_data[i]['prompt']}")
    print(f"\nTARGET RESPONSE:\n{distil_data[i]['response']}")
    print(f"\n{'='*80}")

print(f"\nTotal training examples: {len(distil_data)}")
print("=" * 80)


print("\nTokenizing distillation data...")

dataset = Dataset.from_list(distil_data)


print("\nReloading student model for training...")

# Clean up the untrained model
del student_untrained
torch.cuda.empty_cache()

# Reload for training
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)

if USE_LORA:
    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)


def tokenize_fn(batch):
    # Combine prompt and response for causal LM
    texts = [p + r for p, r in zip(batch["prompt"], batch["response"])]
    return student_tok(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )


tokenized = dataset.map(
    tokenize_fn, batched=True, remove_columns=["prompt", "response"]
)
dataset_dict = DatasetDict({"train": tokenized})

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    warmup_ratio=0.05,
    optim="adamw_torch",
    report_to="none",
)


print("Starting distillation training...")

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=student_tok,
    mlm=False,  # Causal LM, not masked LM
)

trainer = Trainer(
    model=student,
    args=args,
    train_dataset=dataset_dict["train"],
    data_collator=data_collator,
)
trainer.train()

print("Saving distilled model...")
trainer.save_model(OUTPUT_DIR)
student_tok.save_pretrained(OUTPUT_DIR)
print("Distillation complete. Saved to:", OUTPUT_DIR)
