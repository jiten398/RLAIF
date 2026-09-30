"""
panelforge.dataset_builder
============================
Ties the pieces together: for each prompt, sample candidate responses from
the policy model (the model you're about to fine-tune), have the panel judge
them pairwise, keep only pairs the panel agreed on strongly enough, and write
a JSONL file in the {prompt, chosen, rejected} shape that HuggingFace `trl`'s
DPOTrainer expects out of the box.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from .model_adapters import ModelAdapter
from .orchestrator import Panel, ComparisonRecord


@dataclass
class BuildStats:
    prompts_processed: int
    pairs_judged: int
    pairs_accepted: int
    pairs_rejected_low_agreement: int


async def generate_candidates(policy_adapter: ModelAdapter, prompt: str, n: int = 2,
                               temperature: float = 0.9, max_tokens: int = 512) -> list[str]:
    """Sample n candidate responses from the policy model for one prompt.
    Higher temperature than you'd use in production is intentional -- you
    want genuinely different candidates for the panel to have an opinion about,
    not n near-identical greedy decodes."""
    import asyncio
    outs = await asyncio.gather(*[
        policy_adapter.complete(system="", user=prompt, temperature=temperature, max_tokens=max_tokens)
        for _ in range(n)
    ])
    return list(outs)


async def build_preference_dataset(
    prompts: list[str],
    panel: Panel,
    policy_adapter: ModelAdapter,
    out_path: str,
    candidates_per_prompt: int = 2,
    max_pairs_per_prompt: int = 1,
) -> tuple[BuildStats, list[ComparisonRecord]]:
    """
    For each prompt: sample `candidates_per_prompt` responses, form up to
    `max_pairs_per_prompt` pairs from them, judge each pair with the panel,
    and keep the ones that clear the panel's agreement_threshold. Every kept
    pair is written as one JSONL line: {"prompt": ..., "chosen": ..., "rejected": ...}.

    All records (accepted and rejected) are also returned so the caller can
    run safety_checks diagnostics and the human-audit sampler over the full
    judged set, not just what made it into the training file.
    """
    all_records: list[ComparisonRecord] = []
    accepted = 0
    rejected_low_agreement = 0
    pairs_judged = 0

    with open(out_path, "w") as f:
        for prompt in prompts:
            candidates = await generate_candidates(policy_adapter, prompt, n=candidates_per_prompt)
            pair_indices = list(itertools.combinations(range(len(candidates)), 2))[:max_pairs_per_prompt]

            batch_items = [(prompt, candidates[i], candidates[j]) for i, j in pair_indices]
            records = await panel.judge_batch(batch_items)
            all_records.extend(records)
            pairs_judged += len(records)

            for rec in records:
                if rec.passed_threshold:
                    chosen = rec.response_a if rec.result.winner == "A" else rec.response_b
                    rejected = rec.response_b if rec.result.winner == "A" else rec.response_a
                    f.write(json.dumps({"prompt": rec.prompt, "chosen": chosen, "rejected": rejected}) + "\n")
                    accepted += 1
                else:
                    rejected_low_agreement += 1

    stats = BuildStats(
        prompts_processed=len(prompts),
        pairs_judged=pairs_judged,
        pairs_accepted=accepted,
        pairs_rejected_low_agreement=rejected_low_agreement,
    )
    return stats, all_records
