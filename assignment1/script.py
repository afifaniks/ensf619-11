import json
import os
import time
from datetime import datetime
from pprint import pprint

import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.strip().split(":")[1].strip()
    return "Unknown CPU"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Local (CPU)"
CPU_NAME = get_cpu_name()
RESULTS_FILE = "results_t0.5.csv"
PROMPTS_FILE = "prompts.json"
EPOCHS = 5

MODELS = [
    "google/codegemma-7b-it",
    "codellama/CodeLlama-7b-Instruct-hf",
    "Qwen/CodeQwen1.5-7B-Chat",
]

# Only apply quantization variations to the first model
QUANTIZATION_MODES = ["none", "8bit", "4bit"]

PROMPTS = {
    "P-1": (
        """You are an experienced Python software engineer.
        Please generate a complete and accurate Google-style docstring for the function below.
        The docstring must clearly describe the purpose of the function, its arguments,
        return value, and include at least one example.
        Do not add any explanations or commentary outside the function.
        Return only the function with the new docstring.
        
        Here's an example for you:
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.

            Args:
                a (int): The first integer.
                b (int): The second integer.

            Returns:
                int: The sum of the two integers.

            Examples:
                >>> add(2, 3)
                5
                >>> add(-1, 1)
                0
            \"\"\"
            return a + b

        Input:
        {code}
        Output:
        """
    ),
    "P-2": (
        """Generate a Google-style docstring for this Python function.
        Include Args, Returns, and Examples sections.
        Output only the updated function code. No extra text or explanation.
        
        Input:
        {code}
        Output:
        """
    ),
    "P-3": (
        """Add Google docstring and output only the updated function code.
        {code}."""
    ),
}


TEST_CODE = """
def connect_to_next_port(self, minimum: int) -> int:
    if minimum < 1024:
      raise ValueError(f'Min. port must be at least 1024, not {minimum}.')
    port = self._find_next_open_port(minimum)
    if port is None:
      raise ConnectionError(
          f'Could not connect to service on port {minimum} or higher.')
    assert port >= minimum, (
        f'Unexpected port {port} when minimum was {minimum}.')
    return port
"""


def get_quantization_config(mode: str):
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    return None


def get_gpu_usage():
    """Get current GPU memory usage and return as dictionary."""
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0,
            "utilization_percent": 0.0,
        }

    allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    utilization = (allocated / total * 100) if total > 0 else 0.0

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "utilization_percent": round(utilization, 2),
    }


def _load_model_and_tokenizer(model_id: str, quant_mode: str):
    print(f"\nLoading model: {model_id} ({quant_mode})")
    quant_config = get_quantization_config(quant_mode)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype="auto",
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model loaded successfully.")
    return model, tokenizer


def _build_chat_prompt(model_id: str, prompt_text: str, code_snippet: str, tokenizer):
    """
    Builds a proper chat-style prompt using the tokenizer’s chat template.
    """
    # Format user-facing content
    if "{code}" in prompt_text:
        prompt = prompt_text.format(code=code_snippet)
    else:
        prompt = f"{prompt_text}\n\n{code_snippet}"

    # Use the correct chat message structure per model type
    if "gemma" in model_id.lower():
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    # Apply chat template for correct tokenization
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


def _benchmark_inference(model, tokenizer, text, device):
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    Tin = model_inputs["input_ids"].shape[1]

    # Get GPU usage before inference
    gpu_before = get_gpu_usage()

    start_time = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            model_inputs.input_ids, max_new_tokens=512, temperature=0.5
        )
    end_time = time.perf_counter()

    # Get GPU usage after inference
    gpu_after = get_gpu_usage()

    total_tokens = generated.shape[1]
    Tout = total_tokens - Tin
    inference_time = end_time - start_time

    new_tokens = [
        out[len(inp) :] for inp, out in zip(model_inputs.input_ids, generated)
    ]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return Tin, Tout, Tin + Tout, inference_time, response, gpu_before, gpu_after


def _save_prompts():
    with open(PROMPTS_FILE, "w") as f:
        json.dump(PROMPTS, f, indent=4)
    print(f"Prompts saved to {PROMPTS_FILE}")


def _initialize_results_file():
    df = pd.DataFrame(
        columns=[
            "Timestamp",
            "Model",
            "Prompt",
            "Quantization",
            "Epoch",
            "Tin",
            "Tout",
            "Ttotal",
            "T(s)",
            "Accuracy",
            "GPU",
            "CPU",
            "Response",
            "GPU_Before_Allocated_GB",
            "GPU_Before_Utilization_%",
            "GPU_After_Allocated_GB",
            "GPU_After_Utilization_%",
        ]
    )
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Created {RESULTS_FILE}")


def _log_result(
    model_id,
    prompt_name,
    quant,
    epoch,
    Tin,
    Tout,
    Ttotal,
    Tsec,
    accuracy,
    response,
    gpu_before,
    gpu_after,
):
    df = pd.DataFrame(
        [
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": model_id,
                "Prompt": prompt_name,
                "Quantization": quant,
                "Epoch": epoch,
                "Tin": Tin,
                "Tout": Tout,
                "Ttotal": Ttotal,
                "T(s)": round(Tsec, 4),
                "Accuracy": accuracy,
                "GPU": GPU,
                "CPU": CPU_NAME,
                "Response": response,
                "GPU_Before_Allocated_GB": gpu_before["allocated_gb"],
                "GPU_Before_Utilization_%": gpu_before["utilization_percent"],
                "GPU_After_Allocated_GB": gpu_after["allocated_gb"],
                "GPU_After_Utilization_%": gpu_after["utilization_percent"],
            }
        ]
    )
    df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)


def _run():
    _initialize_results_file()
    _save_prompts()

    for model_id in MODELS:
        quant_modes = QUANTIZATION_MODES if model_id == MODELS[0] else ["none"]

        for quant in quant_modes:
            model, tokenizer = _load_model_and_tokenizer(model_id, quant)

            for prompt_name, prompt_text in PROMPTS.items():
                formatted_prompt = _build_chat_prompt(
                    model_id, prompt_text, TEST_CODE, tokenizer
                )

                for epoch in range(1, EPOCHS + 1):
                    print(
                        f"\n--- Running {model_id} | {prompt_name} | {quant} | Epoch {epoch} ---"
                    )

                    Tin, Tout, Ttotal, Tsec, response, gpu_before, gpu_after = (
                        _benchmark_inference(model, tokenizer, formatted_prompt, DEVICE)
                    )

                    print(
                        f"T_in={Tin}, T_out={Tout}, T_total={Ttotal}, T(s)={Tsec:.3f}"
                    )
                    print(
                        f"GPU Before: {gpu_before['allocated_gb']:.2f}GB ({gpu_before['utilization_percent']:.1f}%)"
                    )
                    print(
                        f"GPU After: {gpu_after['allocated_gb']:.2f}GB ({gpu_after['utilization_percent']:.1f}%)"
                    )
                    print("Output preview:")
                    pprint(response)
                    print("-" * 60)

                    _log_result(
                        model_id,
                        prompt_name,
                        quant,
                        epoch,
                        Tin,
                        Tout,
                        Ttotal,
                        Tsec,
                        "PENDING",  # Evaluation should be done manually
                        response,
                        gpu_before,
                        gpu_after,
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                del model
                del tokenizer

    print("\nAll experiments completed successfully.")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    print("Starting Green Prompt Engineering experiments...")
    print(f"Hardware: {GPU}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    _run()
