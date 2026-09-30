"""
panelforge.config
==================
Typed configuration objects for a judge panel. These are the knobs an indie
developer turns instead of hiring/managing a team of human annotators.

Design intent
-------------
A panel is a list of AgentSpecs. Each AgentSpec pairs a Persona (a rubric +
system prompt describing *what to look for*) with a provider/model (*who is
looking*). The whole point of decorrelation is that both axes should vary:
two agents with the same model but different personas are only partially
independent (they share the base model's blind spots); two agents with the
same persona but different model families are more independent still.
`PanelConfig.decorrelation_report()` gives you a cheap sanity check on this
before you spend money generating a full preference dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import itertools
import yaml


class AggregationStrategy(str, Enum):
    MAJORITY = "majority"       # simple majority vote across agents
    WEIGHTED = "weighted"       # confidence- and persona-weighted vote
    DEBATE = "debate"           # agents see each other's rationale, vote again


@dataclass
class Persona:
    """A rubric + point of view for one judge agent."""
    name: str
    system_prompt: str
    weight: float = 1.0

    @staticmethod
    def from_dict(d: dict) -> "Persona":
        return Persona(name=d["name"], system_prompt=d["system_prompt"], weight=float(d.get("weight", 1.0)))


@dataclass
class AgentSpec:
    """One seat on the panel: a persona, bound to a specific model/provider."""
    persona: Persona
    provider: str                       # "anthropic" | "openai_compatible" | "mock"
    model: str                          # e.g. "claude-sonnet-4-6", "gpt-4o-mini", "llama3-70b-local"
    base_url: Optional[str] = None      # required for openai_compatible (vLLM, Ollama, etc.)
    api_key_env: Optional[str] = None   # name of the env var holding the credential

    @property
    def id(self) -> str:
        return f"{self.persona.name}::{self.provider}:{self.model}"

    @staticmethod
    def from_dict(d: dict) -> "AgentSpec":
        return AgentSpec(
            persona=Persona.from_dict(d["persona"]),
            provider=d["provider"],
            model=d["model"],
            base_url=d.get("base_url"),
            api_key_env=d.get("api_key_env"),
        )


@dataclass
class PanelConfig:
    agents: list[AgentSpec]
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED
    agreement_threshold: float = 0.6     # min consensus strength to keep a pair as training data
    human_audit_rate: float = 0.05       # fraction of accepted pairs routed to a human-review export
    require_position_swap: bool = True   # ask each agent twice (A/B swapped) to catch position bias

    @staticmethod
    def from_yaml(path: str) -> "PanelConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        agents = [AgentSpec.from_dict(a) for a in raw["agents"]]
        return PanelConfig(
            agents=agents,
            aggregation=AggregationStrategy(raw.get("aggregation", "weighted")),
            agreement_threshold=float(raw.get("agreement_threshold", 0.6)),
            human_audit_rate=float(raw.get("human_audit_rate", 0.05)),
            require_position_swap=bool(raw.get("require_position_swap", True)),
        )

    def decorrelation_report(self) -> dict:
        """
        Cheap, pre-flight independence check. Not a substitute for the
        post-hoc agreement-matrix check in safety_checks.py (which uses
        actual verdicts) -- this just flags the obvious setup mistake of
        stacking near-identical agents.
        """
        model_counts: dict[str, int] = {}
        for a in self.agents:
            model_counts[a.model] = model_counts.get(a.model, 0) + 1

        dominant_model = max(model_counts.values()) if model_counts else 0
        warnings = []
        if len(self.agents) >= 2 and dominant_model == len(self.agents):
            warnings.append(
                "Every agent uses the same underlying model. Personas alone give you "
                "prompt diversity, not independent judgment -- shared blind spots will "
                "still pass by unanimous vote. Mix in at least one different model family."
            )
        n_unique_models = len(model_counts)
        for a, b in itertools.combinations(self.agents, 2):
            if a.persona.name == b.persona.name and a.model == b.model:
                warnings.append(f"Duplicate agent detected: '{a.id}' appears twice.")
        return {
            "num_agents": len(self.agents),
            "unique_models": n_unique_models,
            "model_counts": model_counts,
            "warnings": warnings,
        }


@dataclass
class DPOTrainingConfig:
    """Parameters handed to the (optional, heavy) DPO training script."""
    base_model: str
    dataset_path: str
    output_dir: str = "./panelforge-dpo-out"
    use_lora: bool = True
    load_in_4bit: bool = True
    learning_rate: float = 5e-6
    beta: float = 0.1                    # DPO temperature
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 512
    max_length: int = 1024
