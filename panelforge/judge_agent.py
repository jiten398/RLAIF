"""
panelforge.judge_agent
=======================
A JudgeAgent is one panelist: a Persona's rubric, applied by a specific
model, to a single pairwise comparison. It always asks for structured JSON
back so aggregation.py never has to parse prose.

Two anti-bias measures are baked in here rather than left as an exercise:

1. Explicit anti-length / anti-tone instructions in the rubric wrapper.
   This doesn't eliminate length or sycophancy bias (nothing fully does --
   that's why safety_checks.py exists) but a judge that's never told to
   watch for it will exhibit it more.
2. Position-swap support. `judge_pair` takes an `also_swapped` verdict from
   the orchestrator and JudgeAgent exposes `judge_pair_both_orders` so a
   single agent's *self-consistency* (does it pick the same winner when A
   and B are swapped?) can be measured. An agent that flips its answer
   under a no-op relabeling is telling you its verdicts are noise, not
   signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import AgentSpec
from .model_adapters import ModelAdapter


@dataclass
class Verdict:
    agent_id: str
    winner: str        # "A" | "B" | "tie"
    confidence: float  # 0.0 - 1.0
    rationale: str
    raw: str = ""       # original model output, kept for audit trails


RUBRIC_WRAPPER = """You are one member of a panel evaluating AI model outputs for fine-tuning \
data. Apply the following rubric strictly and consistently:

<rubric name="{persona_name}">
{persona_rubric}
</rubric>

Judging rules:
- Judge substance, correctness, and how well the response serves the rubric above.
- Do NOT prefer a response merely because it is longer, more confident-sounding, \
or more agreeable in tone. A shorter, more accurate response should beat a longer, \
padded one.
- If the two responses are genuinely comparable in quality, say "tie" rather than \
forcing a pick.
- Respond with ONLY a single JSON object, no other text: \
{{"winner": "A" | "B" | "tie", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""

PAIR_TEMPLATE = """<prompt>
{prompt}
</prompt>

<response_a>
{response_a}
</response_a>

<response_b>
{response_b}
</response_b>

Which response better satisfies the rubric? Respond with only the JSON object."""


class JudgeAgent:
    def __init__(self, spec: AgentSpec, adapter: ModelAdapter):
        self.spec = spec
        self.adapter = adapter

    @property
    def id(self) -> str:
        return self.spec.id

    def _system_prompt(self) -> str:
        return RUBRIC_WRAPPER.format(
            persona_name=self.spec.persona.name,
            persona_rubric=self.spec.persona.system_prompt,
        )

    async def judge_pair(self, prompt: str, response_a: str, response_b: str) -> Verdict:
        user = PAIR_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b)
        raw = await self.adapter.complete(system=self._system_prompt(), user=user, temperature=0.0, max_tokens=200)
        return self._parse(raw)

    async def judge_pair_both_orders(self, prompt: str, response_a: str, response_b: str) -> tuple[Verdict, Verdict]:
        """Judge once as (A, B) and once as (B, A). Used to measure position bias.
        The second verdict's winner is re-mapped back into the original A/B frame
        so callers never have to think about the swap."""
        v1 = await self.judge_pair(prompt, response_a, response_b)
        v2_swapped = await self.judge_pair(prompt, response_b, response_a)
        remap = {"A": "B", "B": "A", "tie": "tie"}
        v2 = Verdict(
            agent_id=v2_swapped.agent_id,
            winner=remap[v2_swapped.winner],
            confidence=v2_swapped.confidence,
            rationale=v2_swapped.rationale,
            raw=v2_swapped.raw,
        )
        return v1, v2

    def _parse(self, raw: str) -> Verdict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            # Fail safe rather than fail loud: an unparseable judge response
            # becomes a zero-confidence tie, which agreement_threshold
            # filtering will simply drop rather than poison the dataset.
            return Verdict(agent_id=self.id, winner="tie", confidence=0.0, rationale="unparseable response", raw=raw)
        try:
            obj = json.loads(match.group(0))
            winner = str(obj.get("winner", "tie")).strip().upper()
            if winner not in ("A", "B", "TIE"):
                winner = "tie"
            winner = "tie" if winner == "TIE" else winner
            confidence = float(obj.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            rationale = str(obj.get("rationale", ""))
            return Verdict(agent_id=self.id, winner=winner, confidence=confidence, rationale=rationale, raw=raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return Verdict(agent_id=self.id, winner="tie", confidence=0.0, rationale="malformed JSON", raw=raw)
