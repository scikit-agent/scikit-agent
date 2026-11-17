"""Test the payoff comparison behind the intention criterion.

An alternative must catch up when the original outcome is guaranteed, and the
reported guarantees must contain no unnecessary settings or outcomes.
"""

from collections.abc import Mapping
from typing import Any

import pytest

from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.ground import GroundedBlock
from skagent.intention import (
    compare_outcome_sets,
    compare_references,
    contextual_payoff,
    minimal_subsets,
)


def test_minimal_subsets_keeps_incomparable_sets_once():
    """Preserve every distinct minimal explanation of a payoff tie."""
    # The search passes a generator; consuming it twice would lose candidates.
    candidates = iter([{0, 1}, {0}, {1}, {0}, {1, 2}])
    assert minimal_subsets(candidates) == (frozenset({0}), frozenset({1}))
    assert minimal_subsets([set(), {0}]) == (frozenset(),)


@pytest.fixture
def farmer_harvest_comparison():
    """Isolate one setting where the alternative loses a valued outcome."""
    # Only bad weather makes rest worse, so only that setting needs a guarantee.
    block = DBlock(
        shocks={"Weather": (Bernoulli, {"p": 0.5})},
        dynamics={
            "D": Control([], agent="farmer"),
            "Harvest": lambda D, Weather: int(D or Weather),
            "Reward": lambda Harvest: Harvest,
        },
        reward={"Reward": "farmer"},
    )
    ground = GroundedBlock(block, {})
    work = {"D": lambda: 1}
    rest = {"D": lambda: 0}

    def setting_of(shocks: Mapping[str, Any]) -> int:
        return int(shocks["Weather"])

    return ground, work, rest, setting_of


def test_contextual_payoff(farmer_harvest_comparison):
    """A guarantee changes payoff only where the alternative falls short."""
    ground, work, rest, setting_of = farmer_harvest_comparison

    # Good weather checks that fixing an already achieved outcome adds no payoff.
    assert (
        float(contextual_payoff(ground, work, rest, "Harvest", (), setting_of)) == 0.5
    )
    assert float(contextual_payoff(ground, work, rest, "Harvest", {0}, setting_of)) == 1
    assert (
        float(contextual_payoff(ground, work, rest, "Harvest", {1}, setting_of)) == 0.5
    )

    # A decision is an action to compare, not an outcome to guarantee.
    with pytest.raises(ValueError, match="non-decision"):
        contextual_payoff(ground, work, rest, "D", {0}, setting_of)


def test_compare_references_finds_minimal_settings(farmer_harvest_comparison):
    """Keep only settings needed to erase the alternative's payoff gap."""
    ground, work, rest, setting_of = farmer_harvest_comparison

    # A one-shot outcome iterable must survive every subset and reference.
    results = compare_references(
        ground,
        work,
        {"rest": rest, "work": work},
        iter(["Harvest"]),
        (0, 1),
        setting_of,
    )
    assert float(results["rest"][0]) == 0.5
    # Minimality excludes good weather and needs nothing for a policy already tied.
    assert results["rest"][1] == (frozenset({0}),)
    assert results["work"][1] == (frozenset(),)

    # Repeated settings must not enter the subset search as separate choices.
    with pytest.raises(ValueError, match="distinct"):
        compare_references(ground, work, {"rest": rest}, "Harvest", (0, 0), setting_of)


def test_guaranteed_reward_counts_toward_payoff():
    """A reward guarantee must survive the block's reward recalculation."""
    block = DBlock(
        dynamics={"D": Control([], agent="farmer"), "Reward": lambda D: D},
        reward={"Reward": "farmer"},
    )
    ground = GroundedBlock(block, {})
    work = {"D": lambda: 1}
    rest = {"D": lambda: 0}

    def setting_of(_shocks: Mapping[str, Any]) -> str:
        return "only"

    assert (
        float(contextual_payoff(ground, work, rest, "Reward", {"only"}, setting_of))
        == 1
    )
    # A single fixed reward name must stay whole when passed to the block.
    assert block.calc_reward({"D": 0, "Reward": 1}, fixed="Reward") == {"Reward": 1}
    comparison = compare_references(
        ground, work, {"rest": rest}, "Reward", ("only",), setting_of
    )
    assert comparison["rest"][1] == (frozenset({"only"}),)
    assert compare_outcome_sets(
        ground, work, {"rest": rest}, "Reward", {"only"}, setting_of
    )["rest"] == (frozenset({"Reward"}),)


def test_guaranteed_reward_skips_its_formula():
    """A fixed reward must bypass a formula undefined for the alternative."""
    block = DBlock(
        dynamics={"D": Control([], agent="farmer"), "Reward": lambda D: 1 / D},
        reward={"Reward": "farmer"},
    )
    payoff = contextual_payoff(
        GroundedBlock(block, {}),
        {"D": lambda: 1},
        {"D": lambda: 0},
        "Reward",
        {"only"},
        lambda _shocks: "only",
        agent="farmer",
    )
    assert float(payoff) == 1


def test_comparison_returns_all_minimal_guarantees():
    """Keep distinct guarantees when either setting closes the gap."""
    # Equal weights make either setting sufficient, exposing incomparable minima.
    block = DBlock(
        shocks={"Weather": (Bernoulli, {"p": 0.5})},
        dynamics={
            "D": Control([], agent="farmer"),
            "Harvest": lambda D: D,
            "Reward": lambda D, Harvest: Harvest - 0.5 * D,
        },
        reward={"Reward": "farmer"},
    )
    ground = GroundedBlock(block, {})
    work = {"D": lambda: 1}
    rest = {"D": lambda: 0}
    results = compare_references(
        ground,
        work,
        {"rest": rest},
        "Harvest",
        (0, 1),
        lambda shocks: int(shocks["Weather"]),
    )

    # Neither singleton contains the other, so both are minimal.
    assert results["rest"][1] == (frozenset({0}), frozenset({1}))


def test_donation_needs_two_benefits_but_not_newsletter():
    """Joint benefits can explain a choice while an incidental outcome cannot."""
    # One benefit cannot repay the cost; any two can, while the newsletter cannot.
    block = DBlock(
        dynamics={
            "D": Control([], agent="alice"),
            "Helping": lambda D: D,
            "Reputation": lambda D: D,
            "Taxes": lambda D: D,
            "Newsletter": lambda D: D,
            "Cost": lambda D: 1.5 * D,
            "U": lambda Helping, Reputation, Taxes, Cost: (
                Helping + Reputation + Taxes - Cost
            ),
        },
        reward={"U": "alice"},
    )
    ground = GroundedBlock(block, {})
    donate = {"D": lambda: 1}
    skip = {"D": lambda: 0}

    def setting_of(_shocks: Mapping[str, Any]) -> str:
        return "only"

    assert (
        float(
            contextual_payoff(
                ground, donate, skip, ("Helping", "Reputation"), {"only"}, setting_of
            )
        )
        == 2
    )
    comparison = compare_outcome_sets(
        ground,
        donate,
        {"skip": skip, "donate": donate},
        ("Helping", "Reputation", "Taxes", "Newsletter"),
        {"only"},
        setting_of,
    )
    assert set(comparison["skip"]) == {
        frozenset({"Helping", "Reputation"}),
        frozenset({"Helping", "Taxes"}),
        frozenset({"Reputation", "Taxes"}),
    }
    assert comparison["donate"] == (frozenset(),)
