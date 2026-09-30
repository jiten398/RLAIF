#!/usr/bin/env python3
"""
demo.py -- run the whole panelforge pipeline end to end with zero API keys
and zero network access, using MockAdapter for both the judge panel and the
"policy model" being fine-tuned.

This exists so you can verify the orchestration/aggregation/safety-check
logic actually works *before* spending real API budget on it. Swap
examples/agents.mock.yaml for examples/agents.yaml (real providers) and
policy_provider="mock" for a real one to go live.

Usage:
    python demo.py
"""
import asyncio
import json

from panelforge.config import PanelConfig
from panelforge.orchestrator import Panel
from panelforge.model_adapters import MockAdapter
from panelforge.dataset_builder import build_preference_dataset
from panelforge.safety_checks import run_diagnostics, sample_for_human_audit


PROMPTS = [
    "A customer says their order arrived damaged and wants a refund. Draft a reply.",
    "Explain why a SaaS subscription renewed automatically after the customer says they cancelled.",
    "A user is confused about how to reset their password. Walk them through it.",
    "Write a short apology for a delayed shipment, offering a discount.",
    "A customer is upset a feature they rely on was removed. Respond with empathy and next steps.",
]


class MockPolicyAdapter:
    """Stands in for 'the model you're fine-tuning'. Produces two visibly
    different candidate responses per prompt (one terse, one padded) so the
    panel's judgments -- including the intentional length/sycophancy biases
    baked into examples/agents.mock.yaml -- have something real to disagree
    about."""

    def __init__(self):
        self._call_count = 0

    async def complete(self, system: str, user: str, temperature: float = 0.9, max_tokens: int = 512) -> str:
        self._call_count += 1
        if self._call_count % 2 == 1:
            return f"Sure -- here's a direct fix: {user[:40]}... Done."
        else:
            return (
                f"Absolutely! Great question! Let me walk you through this step by step in detail. "
                f"First, I want to say I completely understand your situation regarding: {user[:40]}... "
                f"Here is a thorough, comprehensive explanation covering every angle of this issue, "
                f"so you have all the context you could possibly need going forward!"
            )


async def main():
    print("=" * 70)
    print("panelforge demo -- network-free run using MockAdapter")
    print("=" * 70)

    cfg = PanelConfig.from_yaml("examples/agents.mock.yaml")
    report = cfg.decorrelation_report()
    print("\n[1/4] Pre-flight decorrelation report:")
    print(json.dumps(report, indent=2))

    panel = Panel(cfg)
    policy_adapter = MockPolicyAdapter()

    print("\n[2/4] Generating candidates + judging with the panel ...")
    stats, records = await build_preference_dataset(
        prompts=PROMPTS,
        panel=panel,
        policy_adapter=policy_adapter,
        out_path="demo_preferences.jsonl",
        candidates_per_prompt=2,
        max_pairs_per_prompt=1,
    )
    print(json.dumps(stats.__dict__, indent=2))

    print("\n[3/4] Running safety diagnostics on the judged set ...")
    diagnostics = run_diagnostics(records)
    print(f"  length_bias_score: {diagnostics.length_bias_score:.2f}")
    print(f"  agent_agreement: {json.dumps(diagnostics.agent_agreement, indent=2)}")
    print(f"  position_consistency: {json.dumps(diagnostics.position_consistency, indent=2)}")
    if diagnostics.warnings:
        print("  warnings:")
        for w in diagnostics.warnings:
            print(f"    - {w}")
    else:
        print("  (no warnings)")

    print("\n[4/4] Sampling pairs for human audit ...")
    n = sample_for_human_audit(records, cfg.human_audit_rate, "demo_audit_sample.csv")
    print(f"  wrote {n} rows to demo_audit_sample.csv")

    print("\nDone. Preference dataset -> demo_preferences.jsonl")
    print("This file is in the exact {prompt, chosen, rejected} shape")
    print("panelforge.dpo_train / trl.DPOTrainer expect.")
    print("\nNote: this demo's panel (examples/agents.mock.yaml) is deliberately")
    print("imperfect -- it includes a length-biased and a sycophantic judge on")
    print("purpose, so the warnings above are the tool doing its job, not a bug.")


if __name__ == "__main__":
    asyncio.run(main())
