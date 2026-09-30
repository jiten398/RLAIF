"""
panelforge.safety_checks
=========================
Diagnostics that answer the question "is my panel actually giving clean
signal, or is my policy model about to learn to game it?"

These are deliberately cheap, statistical, and run on the panel's own
verdicts -- they add no extra API calls beyond what judging already made,
except the human-audit export which asks a human to look at a small sample.

None of this makes reward hacking impossible. It makes it visible.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass

from .orchestrator import ComparisonRecord


@dataclass
class DiagnosticsReport:
    length_bias_score: float           # -1..1, see length_bias_score() docstring
    agent_agreement: dict               # pairwise agreement rates between agents
    position_consistency: dict          # per-agent rate of agreeing with itself under A/B swap
    warnings: list[str]


def length_bias_score(records: list[ComparisonRecord]) -> float:
    """
    Correlation-flavored statistic: of the pairs the panel accepted (passed
    the agreement threshold), what fraction picked the *longer* response as
    the winner? 0.5 is what you'd expect from a length-blind panel purely by
    chance on realistic data; anything pushing toward 1.0 means the panel
    (and therefore your fine-tuned model) is learning "longer = better"
    rather than "more correct = better".

    Returns a score in [0, 1]. Values above ~0.65 are worth investigating --
    check individual rationales, and consider adding an explicit length
    penalty or a persona whose whole job is to penalize padding.
    """
    accepted = [r for r in records if r.passed_threshold]
    if not accepted:
        return 0.5
    longer_wins = 0
    for r in accepted:
        winner_text = r.response_a if r.result.winner == "A" else r.response_b
        loser_text = r.response_b if r.result.winner == "A" else r.response_a
        if len(winner_text) > len(loser_text):
            longer_wins += 1
    return longer_wins / len(accepted)


def agent_agreement_matrix(records: list[ComparisonRecord]) -> dict:
    """
    Pairwise agreement rate between every two agents on the panel, across all
    judged pairs. This is the direct, empirical version of the pre-flight
    `PanelConfig.decorrelation_report()`: if two agents agree on nearly
    everything, they are not contributing independent signal no matter how
    different their persona prompts look on paper -- collapse them into one
    vote (adjust weights) or swap one out for a different model family.
    """
    # collect per-agent winner for each record, using only the first verdict
    # if position-swap doubled entries (both entries share the same agent_id
    # after aggregation.py's collapse, so we just take unique agent ids seen)
    per_record_by_agent: list[dict[str, str]] = []
    for r in records:
        d = {}
        for v in r.result.per_agent:
            d[v.agent_id] = v.winner
        per_record_by_agent.append(d)

    agent_ids = sorted({aid for d in per_record_by_agent for aid in d})
    agreement: dict[str, dict[str, float]] = {a: {} for a in agent_ids}
    for i, a in enumerate(agent_ids):
        for b in agent_ids[i + 1:]:
            total, agree = 0, 0
            for d in per_record_by_agent:
                if a in d and b in d:
                    total += 1
                    if d[a] == d[b]:
                        agree += 1
            rate = (agree / total) if total else None
            agreement[a][b] = rate
            agreement[b][a] = rate
    return agreement


def position_consistency(records: list[ComparisonRecord]) -> dict:
    """
    Per-agent rate of *not* being flagged inconsistent by the position-swap
    collapse in aggregation.py (which marks a self-contradicting agent with
    rationale == 'inconsistent across position swap (position bias)' and
    confidence 0.1). Low consistency for an agent means its verdicts are
    closer to noise than judgment -- consider dropping it or lowering its
    weight rather than trusting its vote at face value.
    """
    total = defaultdict(int)
    consistent = defaultdict(int)
    for r in records:
        for v in r.result.per_agent:
            total[v.agent_id] += 1
            if v.rationale != "inconsistent across position swap (position bias)":
                consistent[v.agent_id] += 1
    return {aid: (consistent[aid] / total[aid] if total[aid] else None) for aid in total}


def run_diagnostics(records: list[ComparisonRecord], agreement_warn_threshold: float = 0.9,
                     length_bias_warn_threshold: float = 0.65,
                     position_consistency_warn_threshold: float = 0.7) -> DiagnosticsReport:
    lb = length_bias_score(records)
    agreement = agent_agreement_matrix(records)
    consistency = position_consistency(records)

    warnings = []
    if lb >= length_bias_warn_threshold:
        warnings.append(
            f"Length bias score is {lb:.2f} (>= {length_bias_warn_threshold}). The panel is favoring "
            "longer responses more often than chance would predict. Inspect rationales; consider a "
            "persona that explicitly penalizes verbosity/padding."
        )
    for a, row in agreement.items():
        for b, rate in row.items():
            if rate is not None and rate >= agreement_warn_threshold and a < b:
                warnings.append(
                    f"Agents '{a}' and '{b}' agree {rate:.0%} of the time -- they are likely correlated "
                    "rather than independent. Consider merging their weight into one vote or swapping "
                    "one for a different model family."
                )
    for aid, rate in consistency.items():
        if rate is not None and rate < position_consistency_warn_threshold:
            warnings.append(
                f"Agent '{aid}' is self-consistent under position swap only {rate:.0%} of the time -- "
                "its verdicts are close to noise. Consider lowering its weight or excluding it."
            )

    return DiagnosticsReport(
        length_bias_score=lb,
        agent_agreement=agreement,
        position_consistency=consistency,
        warnings=warnings,
    )


def sample_for_human_audit(records: list[ComparisonRecord], rate: float, out_csv: str, seed: int = 0) -> int:
    """
    Write a CSV of a random sample of accepted pairs for a human to spot-check.
    This is the "no free lunch without some human touch" recommendation made
    concrete: even 5% weekly review catches silent judge drift that a purely
    AI-feedback loop has no other way to notice. Returns the number of rows written.
    """
    accepted = [r for r in records if r.passed_threshold]
    n = max(1, int(len(accepted) * rate)) if accepted else 0
    sample = random.Random(seed).sample(accepted, min(n, len(accepted))) if accepted else []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response_a", "response_b", "panel_winner", "consensus_strength",
                          "human_agrees (fill in Y/N)", "human_notes"])
        for r in sample:
            writer.writerow([r.prompt, r.response_a, r.response_b, r.result.winner,
                              f"{r.result.strength:.2f}", "", ""])
    return len(sample)
