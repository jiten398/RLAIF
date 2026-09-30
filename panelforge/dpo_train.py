"""
panelforge.dpo_train
======================
Takes the JSONL preference dataset produced by dataset_builder.py and runs
Direct Preference Optimization on a local Hugging Face model, using LoRA +
optional 4-bit quantization so a single consumer GPU (e.g. one 24GB card) is
enough for 7-8B-parameter models -- no separate reward model, no PPO loop.

This module is intentionally NOT exercised by the test suite: it needs
`torch`, `transformers`, `trl`, `peft`, and (for 4-bit) `bitsandbytes`,
plus a real GPU and a downloaded base model, none of which belong in a CI
sandbox. Install with:

    pip install -r requirements-train.txt

Run with:

    python -m panelforge.dpo_train --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --dataset ./prefs.jsonl --output-dir ./out

The heavy imports are deferred into main() so the rest of panelforge (config,
judging, aggregation, safety checks) stays usable with only the lightweight
requirements.txt, e.g. in this repo's own test suite.
"""
from __future__ import annotations

import argparse

from .config import DPOTrainingConfig


def parse_args() -> DPOTrainingConfig:
    p = argparse.ArgumentParser(description="Fine-tune a base model with DPO on a panelforge preference dataset.")
    p.add_argument("--base-model", required=True, help="HF model id or local path, e.g. mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--dataset", required=True, help="Path to JSONL file with {prompt, chosen, rejected} rows")
    p.add_argument("--output-dir", default="./panelforge-dpo-out")
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA (needs much more VRAM)")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature; higher = stays closer to the base model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-length", type=int, default=1024)
    a = p.parse_args()
    return DPOTrainingConfig(
        base_model=a.base_model,
        dataset_path=a.dataset,
        output_dir=a.output_dir,
        use_lora=not a.no_lora,
        load_in_4bit=not a.no_4bit,
        learning_rate=a.lr,
        beta=a.beta,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        max_prompt_length=a.max_prompt_length,
        max_length=a.max_length,
    )


def main():
    cfg = parse_args()

    # Deferred, heavy imports -- see module docstring.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import DPOConfig, DPOTrainer

    print(f"[panelforge] Loading dataset from {cfg.dataset_path} ...")
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    print(f"[panelforge] {len(dataset)} preference pairs loaded.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[panelforge] Loading base model {cfg.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype if quantization_config is None else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        print("[panelforge] LoRA enabled: only adapter weights will be trained.")

    # bf16 needs Ampere-or-newer (A10/L4/A100/H100); older cards like the T4
    # fall back to fp16 (see use_bf16 above).
    config_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        max_length=cfg.max_length,
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
        report_to=[],
    )
    # trl's DPOConfig has changed across releases: recent versions dropped
    # `max_prompt_length`. Only pass it if this trl version still accepts it.
    import dataclasses
    if "max_prompt_length" in {f.name for f in dataclasses.fields(DPOConfig)}:
        config_kwargs["max_prompt_length"] = cfg.max_prompt_length
    training_args = DPOConfig(**config_kwargs)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("[panelforge] Starting DPO training ...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[panelforge] Done. Adapter/model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
