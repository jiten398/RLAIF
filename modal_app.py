"""
modal_app.py -- run panelforge's DPO training on a Modal GPU as a smoke test.

What this proves: the pipeline's last untested stage (panelforge.dpo_train ->
trl DPOTrainer -> LoRA adapter saved) actually runs on a real GPU with current
library versions, and that training visibly learns from a preference file.

Setup (once):
    pip install modal
    modal setup                      # authenticates YOUR machine; no token is shared with anyone

Run from the project root (the folder containing this file and ./panelforge):
    modal run modal_app.py                                   # tiny synthetic dataset, 0.5B model
    modal run modal_app.py --dataset prefs.jsonl             # your own panelforge-built dataset
    modal run modal_app.py --base-model Qwen/Qwen2.5-1.5B-Instruct --use-4bit

Pick a different GPU with an env var (default L4, which supports bf16):
    PANELFORGE_GPU=A10G modal run modal_app.py

Fetch trained adapters afterwards:
    modal volume get panelforge-outputs / ./modal-out

Notes:
- The default model is ungated, so no Hugging Face token is needed. For a gated
  model (e.g. Llama, Mistral), create a Modal secret named "huggingface" with
  HF_TOKEN and add `secrets=[modal.Secret.from_name("huggingface")]` to the
  @app.function decorator below.
- The synthetic dataset only tests the mechanics (does it run, does the loss
  move). It says nothing about the quality of a real judge panel's preferences.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("PANELFORGE_GPU", "L4")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.56.2,<5",
        "trl",
        "peft",
        "datasets>=4.7.0",
        "accelerate>=1.4.0",
        "bitsandbytes",
        "pyyaml",
    )
    .add_local_python_source("panelforge")  # keep last: ships ./panelforge into the container
)

app = modal.App("panelforge-dpo-smoketest", image=image)
outputs = modal.Volume.from_name("panelforge-outputs", create_if_missing=True)


def make_synthetic_dataset(n: int = 48) -> str:
    """Deterministic {prompt, chosen, rejected} JSONL with a clear, learnable
    signal: chosen = short and actionable, rejected = padded and vague."""
    tasks = [
        "reset my password", "cancel my subscription", "track my order",
        "update my billing details", "export my data", "turn on two-factor login",
        "change my email address", "download an invoice",
    ]
    openers = ["How do I", "Can you tell me how to", "I need help to", "What's the way to"]
    rows = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        opener = openers[(i // len(tasks)) % len(openers)]
        prompt = f"{opener} {task}?"
        chosen = (
            f"To {task}: open Settings, pick the matching option, and confirm. "
            "If it fails, contact support with your account email."
        )
        rejected = (
            "Great question!! I completely understand, and I'm so happy to help you today! "
            f"There are many things to consider when it comes to how to {task}, and honestly "
            "it depends on a lot of factors. Let me know if you would like me to say more!"
        )
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return "\n".join(json.dumps(r) for r in rows) + "\n"


@app.function(gpu=GPU, timeout=60 * 30, volumes={"/out": outputs})
def train(dataset_jsonl: str, base_model: str, epochs: int, lr: float,
          batch_size: int, grad_accum: int, use_4bit: bool) -> dict:
    import importlib.metadata as md

    import torch

    versions = {p: md.version(p) for p in
                ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]}
    print("versions:", versions)
    print("cuda available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    os.makedirs("/tmp/data", exist_ok=True)
    data_path = "/tmp/data/prefs.jsonl"
    with open(data_path, "w") as f:
        f.write(dataset_jsonl)
    n_rows = sum(1 for line in dataset_jsonl.splitlines() if line.strip())
    print(f"dataset: {n_rows} preference pairs")

    run_dir = f"/out/run-{int(time.time())}"
    cmd = [
        sys.executable, "-m", "panelforge.dpo_train",
        "--base-model", base_model,
        "--dataset", data_path,
        "--output-dir", run_dir,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
    ]
    if not use_4bit:
        cmd.append("--no-4bit")
    print("running:", " ".join(cmd))

    started = time.time()
    proc = subprocess.run(cmd)  # output streams straight to your `modal run` terminal
    elapsed = round(time.time() - started, 1)

    summary = {
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "versions": versions,
        "run_dir": run_dir,
        "base_model": base_model,
        "pairs": n_rows,
    }
    if proc.returncode != 0:
        outputs.commit()
        summary["verdict"] = "FAIL: training process exited non-zero (traceback is in the logs above)"
        return summary

    adapter_files = glob.glob(f"{run_dir}/**/adapter_model.safetensors", recursive=True)
    summary["adapter_saved"] = bool(adapter_files)

    states = sorted(glob.glob(f"{run_dir}/**/trainer_state.json", recursive=True))
    log = json.load(open(states[-1]))["log_history"] if states else []
    losses = [e["loss"] for e in log if "loss" in e]
    accs = [e["rewards/accuracies"] for e in log if "rewards/accuracies" in e]
    margins = [e["rewards/margins"] for e in log if "rewards/margins" in e]
    summary.update({
        "logged_steps": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "last_reward_accuracy": accs[-1] if accs else None,
        "first_reward_margin": margins[0] if margins else None,
        "last_reward_margin": margins[-1] if margins else None,
    })

    finite = bool(losses) and all(math.isfinite(x) for x in losses)
    learned = bool(margins) and margins[-1] > margins[0]
    if not summary["adapter_saved"]:
        summary["verdict"] = "FAIL: training finished but no adapter was saved"
    elif not finite:
        summary["verdict"] = "FAIL: loss was missing or non-finite"
    elif learned:
        summary["verdict"] = "PASS: trained, adapter saved, reward margin increased"
    else:
        summary["verdict"] = ("PARTIAL: trained and saved, but reward margin did not increase; "
                              "try more epochs or a higher --lr")
    outputs.commit()
    return summary


@app.local_entrypoint()
def main(dataset: str = "", base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
         epochs: int = 3, lr: float = 5e-5, batch_size: int = 2, grad_accum: int = 2,
         use_4bit: bool = False, n_synthetic: int = 48):
    if dataset:
        with open(dataset) as f:
            data = f.read()
        print(f"using {dataset}")
    else:
        data = make_synthetic_dataset(n_synthetic)
        print(f"using a synthetic dataset of {n_synthetic} pairs")

    result = train.remote(data, base_model, epochs, lr, batch_size, grad_accum, use_4bit)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print("\nFetch the adapter with:  modal volume get panelforge-outputs / ./modal-out")
