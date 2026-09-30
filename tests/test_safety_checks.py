from panelforge.judge_agent import Verdict
from panelforge.aggregation import AggregatedResult
from panelforge.orchestrator import ComparisonRecord
from panelforge.safety_checks import length_bias_score, agent_agreement_matrix, position_consistency


def make_record(prompt, resp_a, resp_b, winner, per_agent_winners, strength=0.8, passed=True):
    per_agent = [Verdict(agent_id=aid, winner=w, confidence=0.8, rationale="") for aid, w in per_agent_winners.items()]
    result = AggregatedResult(winner=winner, strength=strength, per_agent=per_agent, passed_threshold=passed)
    return ComparisonRecord(prompt=prompt, response_a=resp_a, response_b=resp_b, result=result, passed_threshold=passed)


def test_length_bias_score_detects_systematic_length_preference():
    records = []
    for i in range(10):
        short = "ok"
        long = "a" * 500  # always the winner in this synthetic set
        records.append(make_record(f"p{i}", short, long, winner="B", per_agent_winners={"j1": "B"}))
    score = length_bias_score(records)
    assert score == 1.0  # every accepted win went to the longer response


def test_length_bias_score_is_neutral_when_balanced():
    records = []
    for i in range(10):
        a, b = "x" * 10, "y" * 500
        winner = "A" if i % 2 == 0 else "B"
        records.append(make_record(f"p{i}", a, b, winner=winner, per_agent_winners={"j1": winner}))
    score = length_bias_score(records)
    assert 0.4 <= score <= 0.6


def test_agent_agreement_matrix_flags_identical_agents():
    records = []
    for i in range(20):
        winner = "A" if i % 2 == 0 else "B"
        # agent_clone always agrees with agent_original
        records.append(make_record(f"p{i}", "resp a", "resp b", winner=winner,
                                    per_agent_winners={"agent_original": winner, "agent_clone": winner}))
    matrix = agent_agreement_matrix(records)
    assert matrix["agent_original"]["agent_clone"] == 1.0


def test_agent_agreement_matrix_lower_for_independent_agents():
    records = []
    for i in range(20):
        # agent_b disagrees with agent_a exactly half the time
        a_vote = "A" if i % 2 == 0 else "B"
        b_vote = "A" if i % 3 == 0 else "B"
        records.append(make_record(f"p{i}", "resp a", "resp b", winner=a_vote,
                                    per_agent_winners={"agent_a": a_vote, "agent_b": b_vote}))
    matrix = agent_agreement_matrix(records)
    assert matrix["agent_a"]["agent_b"] < 1.0


def test_position_consistency_reports_low_rate_for_flagged_agent():
    records = []
    for i in range(10):
        winner = "inconsistent across position swap (position bias)" if i < 6 else "A"
        rationale = "inconsistent across position swap (position bias)" if i < 6 else "fine"
        per_agent = [Verdict(agent_id="flaky", winner="tie" if i < 6 else "A", confidence=0.5, rationale=rationale)]
        result = AggregatedResult(winner="A", strength=0.5, per_agent=per_agent, passed_threshold=True)
        records.append(ComparisonRecord(prompt=f"p{i}", response_a="a", response_b="b", result=result, passed_threshold=True))
    consistency = position_consistency(records)
    assert consistency["flaky"] == 0.4  # 4 out of 10 were NOT flagged inconsistent
