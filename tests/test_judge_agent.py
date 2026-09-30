import pytest

from panelforge.config import AgentSpec, Persona
from panelforge.model_adapters import MockAdapter
from panelforge.judge_agent import JudgeAgent


def make_agent(bias="none", name="tester"):
    spec = AgentSpec(persona=Persona(name=name, system_prompt="rubric text"), provider="mock", model=bias)
    return JudgeAgent(spec, MockAdapter(name=name, bias=bias))


@pytest.mark.asyncio
async def test_judge_pair_returns_valid_verdict():
    agent = make_agent()
    verdict = await agent.judge_pair("prompt", "response one", "response two")
    assert verdict.winner in ("A", "B", "tie")
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.agent_id == agent.id


@pytest.mark.asyncio
async def test_judge_pair_is_deterministic_for_same_input():
    agent = make_agent()
    v1 = await agent.judge_pair("same prompt", "resp a", "resp b")
    v2 = await agent.judge_pair("same prompt", "resp a", "resp b")
    assert v1.winner == v2.winner
    assert v1.confidence == v2.confidence


@pytest.mark.asyncio
async def test_length_biased_mock_prefers_longer_response():
    agent = make_agent(bias="prefers_longer")
    short = "brief"
    long = "much longer response " * 20
    verdict = await agent.judge_pair("prompt", short, long)
    assert verdict.winner == "B"  # B is the longer one


@pytest.mark.asyncio
async def test_position_swap_consistency_for_clean_mock():
    # A deterministic, unbiased mock should (usually) agree with itself when
    # the two responses are swapped, since the hash swap changes inputs
    # consistently -- this test just exercises the swap plumbing end-to-end.
    agent = make_agent()
    v1, v2 = await agent.judge_pair_both_orders("prompt", "response one", "response two")
    assert v1.winner in ("A", "B", "tie")
    assert v2.winner in ("A", "B", "tie")


def test_parse_handles_malformed_json_gracefully():
    spec = AgentSpec(persona=Persona(name="x", system_prompt="r"), provider="mock", model="none")
    agent = JudgeAgent(spec, MockAdapter(name="x"))
    verdict = agent._parse("not json at all")
    assert verdict.winner == "tie"
    assert verdict.confidence == 0.0
