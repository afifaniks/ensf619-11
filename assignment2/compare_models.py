import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------
TEACHER_MODEL = "google/codegemma-7b-it"
DISTILLED_MODEL_PATH = "./distilled-codegemma-docstring"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test function
TEST_FUNCTION = """def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count"""

PROMPT = (
    "Add a Google-style Python docstring to the following function:\n\n"
    f"{TEST_FUNCTION}\n\n###\n"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading teacher model...")
teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

print("Loading distilled student model...")
student_tok = AutoTokenizer.from_pretrained(DISTILLED_MODEL_PATH)
student = AutoModelForCausalLM.from_pretrained(
    DISTILLED_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

# -----------------------------
# COMPARISON
# -----------------------------
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(f"\nINPUT FUNCTION:\n{TEST_FUNCTION}\n")
print(f"PROMPT:\n{PROMPT}\n")

# Test Teacher Model
print("\n" + "-" * 80)
print("TEACHER MODEL (CodeGemma-7B)")
print("-" * 80)

inputs = teacher_tok(PROMPT, return_tensors="pt").to(DEVICE)
start_time = time.time()

with torch.no_grad():
    teacher_output = teacher.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        top_p=0.95,
    )

teacher_time = time.time() - start_time
teacher_text = teacher_tok.decode(
    teacher_output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
)

print(f"Generated Output:\n{teacher_text}")
print(f"\nInference Time: {teacher_time:.2f} seconds")
print(f"Model Size: ~7B parameters")

# Test Student Model
print("\n" + "-" * 80)
print("DISTILLED STUDENT MODEL (TinyLlama-1.1B)")
print("-" * 80)

inputs = student_tok(PROMPT, return_tensors="pt").to(DEVICE)
start_time = time.time()

with torch.no_grad():
    student_output = student.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        top_p=0.95,
        pad_token_id=student_tok.eos_token_id,
    )

student_time = time.time() - start_time
student_text = student_tok.decode(
    student_output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
)

print(f"Generated Output:\n{student_text}")
print(f"\nInference Time: {student_time:.2f} seconds")
print(f"Model Size: ~1.1B parameters")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Teacher inference time: {teacher_time:.2f}s")
print(f"Student inference time: {student_time:.2f}s")
print(f"Speedup: {teacher_time/student_time:.2f}x faster")
print(f"Model size reduction: ~6.4x smaller (7B → 1.1B parameters)")
print("\n✅ Comparison complete!")
