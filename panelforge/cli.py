"""
panelforge.cli
================
Command-line entrypoint. Kept thin on purpose -- all real logic lives in the
library modules so it's equally usable from a notebook or another script.

Subcommands:
  panelforge check-panel   --config agents.yaml
  panelforge build-dataset --config agents.yaml --prompts prompts.jsonl --policy-provider ... --out prefs.jsonl
  panelforge audit         --config agents.yaml --prompts prompts.jsonl --policy-provider ... --out-csv audit.csv
  panelforge train ...     (delegates to panelforge.dpo_train; run that module directly for the full flag set)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import PanelConfig, AgentSpec, Persona
from .orchestrator import Panel
from .model_adapters import build_adapter
from .dataset_builder import build_preference_dataset
from .safety_checks import run_diagnostics, sample_for_human_audit


def _load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj["prompt"] if isinstance(obj, dict) else obj)
            except json.JSONDecodeError:
                prompts.append(line)  # allow plain-text-per-line prompt files too
    return prompts


def _policy_adapter_from_args(args):
    # The policy model (the one being fine-tuned/evaluated) is configured the
    # same way a judge agent is, but standalone -- it's not part of the panel.
    spec = AgentSpec(
        persona=Persona(name="policy", system_prompt=""),
        provider=args.policy_provider,
        model=args.policy_model,
        base_url=getattr(args, "policy_base_url", None),
        api_key_env=getattr(args, "policy_api_key_env", None),
    )
    return build_adapter(spec)


def cmd_check_panel(args):
    cfg = PanelConfig.from_yaml(args.config)
    report = cfg.decorrelation_report()
    print(json.dumps(report, indent=2))
    if report["warnings"]:
        print("\nSee warnings above -- fix before spending API budget on a full run.", file=sys.stderr)
        sys.exit(1)


def cmd_build_dataset(args):
    cfg = PanelConfig.from_yaml(args.config)
    panel = Panel(cfg)
    prompts = _load_prompts(args.prompts)
    policy_adapter = _policy_adapter_from_args(args)

    stats, records = asyncio.run(build_preference_dataset(
        prompts=prompts, panel=panel, policy_adapter=policy_adapter, out_path=args.out,
        candidates_per_prompt=args.candidates_per_prompt, max_pairs_per_prompt=args.max_pairs_per_prompt,
    ))
    print(json.dumps(stats.__dict__, indent=2))

    diagnostics = run_diagnostics(records)
    for w in diagnostics.warnings:
        print(f"[warning] {w}", file=sys.stderr)

    if cfg.human_audit_rate > 0:
        audit_path = args.out.rsplit(".", 1)[0] + ".audit.csv"
        n = sample_for_human_audit(records, cfg.human_audit_rate, audit_path)
        print(f"Wrote {n} rows for human spot-check to {audit_path}")


def cmd_audit(args):
    cfg = PanelConfig.from_yaml(args.config)
    panel = Panel(cfg)
    prompts = _load_prompts(args.prompts)
    policy_adapter = _policy_adapter_from_args(args)

    _, records = asyncio.run(build_preference_dataset(
        prompts=prompts, panel=panel, policy_adapter=policy_adapter, out_path="/dev/null",
        candidates_per_prompt=args.candidates_per_prompt, max_pairs_per_prompt=1,
    ))
    diagnostics = run_diagnostics(records)
    print(json.dumps({
        "length_bias_score": diagnostics.length_bias_score,
        "agent_agreement": diagnostics.agent_agreement,
        "position_consistency": diagnostics.position_consistency,
        "warnings": diagnostics.warnings,
    }, indent=2))
    n = sample_for_human_audit(records, 1.0, args.out_csv)  # full audit run: sample everything
    print(f"Wrote {n} rows to {args.out_csv}")


def main():
    parser = argparse.ArgumentParser(prog="panelforge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-panel", help="Validate a panel config and print a decorrelation report")
    p_check.add_argument("--config", required=True)
    p_check.set_defaults(func=cmd_check_panel)

    def _add_common_build_args(sp):
        sp.add_argument("--config", required=True, help="Path to agents.yaml")
        sp.add_argument("--prompts", required=True, help="JSONL or plain-text file, one prompt per line")
        sp.add_argument("--policy-provider", required=True, choices=["anthropic", "openai_compatible", "mock"])
        sp.add_argument("--policy-model", required=True)
        sp.add_argument("--policy-base-url", default=None)
        sp.add_argument("--policy-api-key-env", default=None)
        sp.add_argument("--candidates-per-prompt", type=int, default=2)

    p_build = sub.add_parser("build-dataset", help="Generate candidates, judge them, write a DPO-ready JSONL")
    _add_common_build_args(p_build)
    p_build.add_argument("--max-pairs-per-prompt", type=int, default=1)
    p_build.add_argument("--out", required=True)
    p_build.set_defaults(func=cmd_build_dataset)

    p_audit = sub.add_parser("audit", help="Run diagnostics + full human-audit export without writing a training file")
    _add_common_build_args(p_audit)
    p_audit.add_argument("--out-csv", required=True)
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
