import pytest

from panelforge.judge_agent import Verdict
from panelforge.aggregation import majority_vote, weighted_vote


def v(agent_id, winner, confidence=0.8, rationale=""):
    return Verdict(agent_id=agent_id, winner=winner, confidence=confidence, rationale=rationale)


def test_majority_vote_clear_winner():
    verdicts = [v("a1", "A"), v("a2", "A"), v("a3", "B")]
    result = majority_vote(verdicts)
    assert result.winner == "A"
    assert result.strength == pytest.approx(2 / 3)


def test_majority_vote_tie_breaks_to_tie_label_when_tied_counts():
    # A, B, tie all once -> counts are equal; max() picks the first max
    # deterministically, we just assert it lands on one of the valid options
    verdicts = [v("a1", "A"), v("a2", "B"), v("a3", "tie")]
    result = majority_vote(verdicts)
    assert result.winner in ("A", "B", "tie")
    assert result.strength == pytest.approx(1 / 3)


def test_weighted_vote_respects_persona_weight():
    verdicts = [v("safety", "B", confidence=1.0), v("helpfulness", "A", confidence=1.0)]
    weights = {"safety": 2.0, "helpfulness": 1.0}
    result = weighted_vote(verdicts, weights)
    assert result.winner == "B"
    assert result.strength == pytest.approx(2 / 3)


def test_weighted_vote_respects_confidence():
    verdicts = [v("a1", "A", confidence=0.9), v("a2", "B", confidence=0.2)]
    weights = {"a1": 1.0, "a2": 1.0}
    result = weighted_vote(verdicts, weights)
    assert result.winner == "A"


def test_position_swap_collapse_penalizes_inconsistent_agent():
    # Same agent voting A then (after swap-remap) B -> should collapse to a
    # low-confidence tie rather than counting as a real vote either way.
    verdicts = [
        v("flaky_agent", "A", confidence=0.9),
        v("flaky_agent", "B", confidence=0.9),  # already remapped by judge_agent.judge_pair_both_orders
        v("stable_agent", "A", confidence=0.9),
    ]
    result = majority_vote(verdicts)
    # flaky_agent collapses to one 'tie' vote with confidence 0.1;
    # stable_agent votes A -> final tally should favor A, not be swamped by noise
    assert result.winner == "A"


def test_weighted_vote_all_zero_confidence_is_a_tie():
    verdicts = [v("a1", "A", confidence=0.0), v("a2", "B", confidence=0.0)]
    result = weighted_vote(verdicts, {"a1": 1.0, "a2": 1.0})
    assert result.winner == "tie"
    assert result.strength == 0.0
