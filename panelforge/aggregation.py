"""
panelforge.aggregation
=======================
Turns N raw Verdicts (one per judge agent, possibly two per agent if
position-swap is enabled) into a single AggregatedResult: who won, how
strongly, and whether the panel actually agreed enough to trust this pair
as fine-tuning data.

Three strategies, in increasing cost:

- majority: every agent gets one vote (position-swap verdicts for the same
  agent are first collapsed to one vote by simple average).
- weighted: votes are weighted by persona weight * verdict confidence.
  This is the default -- it lets you say "the safety persona's vote counts
  more" without silencing the others.
- debate: after an initial weighted round, agents whose vote is in the
  minority are shown the majority's rationale (not identity) and get one
  chance to update. This is a hook, not magic: literature is mixed on
  whether debate improves accuracy or just manufactures false consensus,
  which is exactly why safety_checks.agent_agreement_matrix exists --
  use it to check whether debate mode is homogenizing your panel rather
  than improving it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import AggregationStrategy
from .judge_agent import Verdict


@dataclass
class AggregatedResult:
    winner: str                 # "A" | "B" | "tie"
    strength: float              # 0.0-1.0 consensus strength (not the same as any single confidence)
    per_agent: list[Verdict]     # every raw verdict that went into this decision
    passed_threshold: bool = False  # filled in by dataset_builder after comparing to PanelConfig.agreement_threshold


def _collapse_swapped_pairs(verdicts: list[Verdict]) -> list[Verdict]:
    """If an agent contributed two verdicts (position swap), collapse them into
    one: same winner in both orders -> keep, confidence averaged. Disagreement
    between the two orders (position bias) -> down-weight to a low-confidence
    tie, since the agent effectively contradicted itself."""
    by_agent: dict[str, list[Verdict]] = {}
    for v in verdicts:
        by_agent.setdefault(v.agent_id, []).append(v)

    collapsed = []
    for agent_id, vs in by_agent.items():
        if len(vs) == 1:
            collapsed.append(vs[0])
            continue
        winners = {v.winner for v in vs}
        avg_conf = sum(v.confidence for v in vs) / len(vs)
        if len(winners) == 1:
            collapsed.append(Verdict(agent_id=agent_id, winner=vs[0].winner, confidence=avg_conf,
                                      rationale=vs[0].rationale, raw=vs[0].raw))
        else:
            # inconsistent under position swap -> treat as unreliable this round
            collapsed.append(Verdict(agent_id=agent_id, winner="tie", confidence=0.1,
                                      rationale="inconsistent across position swap (position bias)", raw=""))
    return collapsed


def majority_vote(verdicts: list[Verdict]) -> AggregatedResult:
    verdicts = _collapse_swapped_pairs(verdicts)
    counts = {"A": 0, "B": 0, "tie": 0}
    for v in verdicts:
        counts[v.winner] += 1
    total = sum(counts.values()) or 1
    winner = max(counts, key=counts.get)
    strength = counts[winner] / total
    return AggregatedResult(winner=winner, strength=strength, per_agent=verdicts)


def weighted_vote(verdicts: list[Verdict], weights_by_agent: dict[str, float]) -> AggregatedResult:
    verdicts = _collapse_swapped_pairs(verdicts)
    scores = {"A": 0.0, "B": 0.0, "tie": 0.0}
    total_weight = 0.0
    for v in verdicts:
        w = weights_by_agent.get(v.agent_id, 1.0) * v.confidence
        scores[v.winner] += w
        total_weight += w
    if total_weight == 0:
        return AggregatedResult(winner="tie", strength=0.0, per_agent=verdicts)
    winner = max(scores, key=scores.get)
    strength = scores[winner] / total_weight
    return AggregatedResult(winner=winner, strength=strength, per_agent=verdicts)


async def debate_vote(verdicts: list[Verdict], weights_by_agent: dict[str, float], panel, prompt: str,
                       response_a: str, response_b: str) -> AggregatedResult:
    """One optional debate round on top of a weighted vote. `panel` is the
    Orchestrator (imported lazily to avoid a circular import), used to ask
    minority agents to reconsider. Kept intentionally simple: one round,
    no recursion, so cost stays bounded and predictable."""
    first_pass = weighted_vote(verdicts, weights_by_agent)
    if first_pass.winner == "tie" or first_pass.strength > 0.85:
        return first_pass  # already decisive or already a tie; debating won't help

    majority_side = first_pass.winner
    minority_verdicts = [v for v in first_pass.per_agent if v.winner != majority_side]
    if not minority_verdicts:
        return first_pass

    majority_rationales = [v.rationale for v in first_pass.per_agent if v.winner == majority_side]
    summary = " / ".join(majority_rationales[:3]) or "no rationale given"

    updated: list[Verdict] = [v for v in first_pass.per_agent if v.winner == majority_side]
    for v in minority_verdicts:
        agent = panel.agents_by_id.get(v.agent_id)
        if agent is None:
            updated.append(v)
            continue
        reconsider_prompt = (
            f"Other panelists leaned toward {'response ' + majority_side if majority_side != 'tie' else 'a tie'} "
            f"for this reason: {summary}\n\nRespond with only the JSON object, considering this input but "
            f"reaching your own independent judgment -- do not simply defer."
        )
        new_verdict = await agent.judge_pair(prompt + "\n\n" + reconsider_prompt, response_a, response_b)
        updated.append(new_verdict)

    return weighted_vote(updated, weights_by_agent)


async def aggregate(strategy: AggregationStrategy, verdicts: list[Verdict], weights_by_agent: dict[str, float],
                     panel=None, prompt: str = "", response_a: str = "", response_b: str = "") -> AggregatedResult:
    if strategy == AggregationStrategy.MAJORITY:
        return majority_vote(verdicts)
    if strategy == AggregationStrategy.WEIGHTED:
        return weighted_vote(verdicts, weights_by_agent)
    if strategy == AggregationStrategy.DEBATE:
        if panel is None:
            raise ValueError("debate aggregation requires a panel reference")
        return await debate_vote(verdicts, weights_by_agent, panel, prompt, response_a, response_b)
    raise ValueError(f"Unknown aggregation strategy: {strategy}")
