"""
panelforge.orchestrator
========================
The Panel class is the "any number of agents" piece: it owns every
JudgeAgent, fans a comparison out to all of them concurrently (asyncio, not
threads -- these are I/O-bound API calls), and hands the raw verdicts to
aggregation.py. This is the whole reason an indie dev can run this alone:
adding a tenth judge is a line in agents.yaml, not a hire.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .config import PanelConfig, AggregationStrategy
from .judge_agent import JudgeAgent, Verdict
from .model_adapters import build_adapter
from .aggregation import aggregate, AggregatedResult


@dataclass
class ComparisonRecord:
    """Full audit trail for one judged pair -- everything dataset_builder and
    safety_checks need, kept in one place so nothing has to be re-derived."""
    prompt: str
    response_a: str
    response_b: str
    result: AggregatedResult
    passed_threshold: bool


class Panel:
    def __init__(self, config: PanelConfig):
        self.config = config
        self.agents: list[JudgeAgent] = [JudgeAgent(spec, build_adapter(spec)) for spec in config.agents]
        self.agents_by_id: dict[str, JudgeAgent] = {a.id: a for a in self.agents}
        self.weights_by_agent: dict[str, float] = {a.id: a.spec.persona.weight for a in self.agents}

    async def judge_one(self, prompt: str, response_a: str, response_b: str) -> ComparisonRecord:
        if self.config.require_position_swap:
            pairs = await asyncio.gather(
                *[agent.judge_pair_both_orders(prompt, response_a, response_b) for agent in self.agents]
            )
            verdicts: list[Verdict] = [v for pair in pairs for v in pair]
        else:
            verdicts = await asyncio.gather(
                *[agent.judge_pair(prompt, response_a, response_b) for agent in self.agents]
            )

        result = await aggregate(
            self.config.aggregation, verdicts, self.weights_by_agent,
            panel=self, prompt=prompt, response_a=response_a, response_b=response_b,
        )
        passed = result.strength >= self.config.agreement_threshold and result.winner != "tie"
        result.passed_threshold = passed
        return ComparisonRecord(prompt=prompt, response_a=response_a, response_b=response_b,
                                 result=result, passed_threshold=passed)

    async def judge_batch(self, items: list[tuple[str, str, str]], concurrency: int = 5) -> list[ComparisonRecord]:
        """items: list of (prompt, response_a, response_b). `concurrency` caps
        how many *comparisons* run at once (each comparison already fans out
        to every agent internally) so a large prompt set doesn't blow past
        provider rate limits."""
        sem = asyncio.Semaphore(concurrency)

        async def _run(item):
            async with sem:
                return await self.judge_one(*item)

        return await asyncio.gather(*[_run(item) for item in items])
