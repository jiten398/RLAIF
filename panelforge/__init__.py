"""
panelforge -- multi-agent RLAIF orchestration for indie LLM developers.

An indie developer configures a *panel* of judge agents (any number, any mix
of providers/personas) instead of hiring human annotators. The panel judges
pairs of candidate responses from a policy model, disagreement-filtered
preference pairs are exported as JSONL, and panelforge.dpo_train fine-tunes
the policy model on that dataset with DPO (no separate reward model, no PPO
loop needed).

Quick start:
    from panelforge.config import PanelConfig
    from panelforge.orchestrator import Panel

    cfg = PanelConfig.from_yaml("agents.yaml")
    panel = Panel(cfg)
    # panel.judge_batch([(prompt, response_a, response_b), ...])

See README.md for the full pipeline and demo.py for a network-free walkthrough.
"""
from .config import PanelConfig, AgentSpec, Persona, AggregationStrategy, DPOTrainingConfig
from .orchestrator import Panel, ComparisonRecord
from .judge_agent import JudgeAgent, Verdict
from .aggregation import AggregatedResult

__all__ = [
    "PanelConfig", "AgentSpec", "Persona", "AggregationStrategy", "DPOTrainingConfig",
    "Panel", "ComparisonRecord", "JudgeAgent", "Verdict", "AggregatedResult",
]

__version__ = "0.1.0"
