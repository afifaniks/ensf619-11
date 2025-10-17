import os
import time

import pandas as pd
import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./pruned_finetuned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RUNS = 5
CSV_FILE = "inference_metrics.csv"


def get_model_size(path: str) -> float:
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    return total_size / (1024**3)  # GB


def prepare_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        teacher_tok = AutoTokenizer.from_pretrained("google/codegemma-7b-it")
        tokenizer.chat_template = teacher_tok.chat_template
    return tokenizer


def run_inference_once(model, tokenizer, func: str):
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error")
    tracker.start()

    prompt = (
        "Add a Google-style Python docstring to the following function. "
        "Include Args, Returns, and Examples sections. "
        "Output only the updated function code.\n\n" + func
    )
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"Chat Prompt:\n{chat_prompt}\n")
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    emissions = tracker.stop()

    output_text = tokenizer.decode(
        gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()
    print(f"Output Text:\n{output_text}\n")

    return {"latency": latency, "emissions": emissions, "output": output_text}


def main():
    dataset = load_dataset("mbpp", split="validation")
    func = dataset[0]["code"]  # single example

    tokenizer = prepare_tokenizer(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model_size = get_model_size(MODEL_PATH)

    results = []
    for run in range(1, NUM_RUNS + 1):
        result = run_inference_once(model, tokenizer, func)
        result["run"] = run
        result["model_size"] = model_size
        results.append(result)
        print(
            f"Run {run}: latency={result['latency']:.3f}s, CO₂={result['emissions']:.6f}kg"
        )

    # Save to pandas dataframe
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

    print(f"\n✅ All results saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
