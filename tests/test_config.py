from panelforge.config import PanelConfig, AgentSpec, Persona, AggregationStrategy


def test_load_example_yaml():
    cfg = PanelConfig.from_yaml("examples/agents.yaml")
    assert len(cfg.agents) == 4
    assert cfg.aggregation == AggregationStrategy.WEIGHTED


def test_load_mock_yaml_for_demo():
    cfg = PanelConfig.from_yaml("examples/agents.mock.yaml")
    assert len(cfg.agents) == 4
    assert all(a.provider == "mock" for a in cfg.agents)


def test_decorrelation_report_warns_on_single_model():
    agents = [
        AgentSpec(persona=Persona(name="p1", system_prompt="a"), provider="anthropic", model="claude-haiku-4-5-20251001"),
        AgentSpec(persona=Persona(name="p2", system_prompt="b"), provider="anthropic", model="claude-haiku-4-5-20251001"),
    ]
    cfg = PanelConfig(agents=agents)
    report = cfg.decorrelation_report()
    assert report["unique_models"] == 1
    assert any("same underlying model" in w for w in report["warnings"])


def test_decorrelation_report_clean_for_mixed_panel():
    agents = [
        AgentSpec(persona=Persona(name="p1", system_prompt="a"), provider="anthropic", model="claude-haiku-4-5-20251001"),
        AgentSpec(persona=Persona(name="p2", system_prompt="b"), provider="openai_compatible", model="llama-3.1-8b", base_url="http://x"),
    ]
    cfg = PanelConfig(agents=agents)
    report = cfg.decorrelation_report()
    assert report["unique_models"] == 2
    assert report["warnings"] == []


def test_decorrelation_report_flags_exact_duplicate_agent():
    agents = [
        AgentSpec(persona=Persona(name="p1", system_prompt="a"), provider="anthropic", model="m"),
        AgentSpec(persona=Persona(name="p1", system_prompt="a"), provider="anthropic", model="m"),
    ]
    cfg = PanelConfig(agents=agents)
    report = cfg.decorrelation_report()
    assert any("Duplicate agent" in w for w in report["warnings"])
