"""
Tests for skagent.algos.best_response.

The solved rules are asserted exactly. They are Monte Carlo estimates, so the
sample counts here are the smallest at which the reported actions are stable
across seeds; a solved action that moves is a signal, not noise to be papered
over with a looser assertion.
"""

import numpy as np
import pytest

from skagent.algos.best_response import TabularBestResponseSolver, TabulatedRule
from skagent.algos.vfi import get_action_rule
from skagent.block import Control, DBlock
from skagent.distributions import Uniform
import skagent.models.macid as macid

SHOCK_SAMPLES = 10_000


def solver(block, seed=0, **kwargs):
    return TabularBestResponseSolver(
        block, shock_samples=SHOCK_SAMPLES, rng=np.random.default_rng(seed), **kwargs
    )


@pytest.fixture
def tree_killer():
    return solver(macid.tree_killer_block)


# A cyclic relevance graph: two simultaneous decisions, each agent's payoff
# depending on both actions, neither observing the other.
simultaneous_block = DBlock(
    **{
        "name": "simultaneous",
        "dynamics": {
            "a1": Control([], agent="p1"),
            "a2": Control([], agent="p2"),
            "u1": lambda a1, a2: -((a1 - a2) ** 2),
            "u2": lambda a1, a2: -((a1 + a2 - 1.0) ** 2),
        },
        "reward": {"u1": "p1", "u2": "p2"},
    }
)


class TestSolve:
    def test_solution_of_a_sequential_game(self, tree_killer):
        """Each decision is solved conditional on what it observes."""
        policies = tree_killer.solve()

        # Bob treats a sick tree at full intensity, and otherwise buys only
        # enough treatment to rule out the baseline chance of death.
        assert policies["TDoc"].to_dict() == {(0.0,): 0.2, (1.0,): 1.0}

        # Alice builds the patio exactly when Bob called the doctor in earnest,
        # whatever she did with the poison: an inference about the tree, not a
        # response to her own decision.
        by_doctor = {
            (observed[1], action)
            for observed, action in policies["BP"].to_dict().items()
        }
        assert by_doctor == {(0.2, 0.0), (1.0, 1.0)}

        # And she does not poison.
        assert policies["PT"].to_dict() == {(): 0.0}

    def test_solve_accepts_starting_policies(self, tree_killer):
        """Rules given for decisions are replaced as those decisions are solved."""
        start = {"PT": get_action_rule(1.0), "TDoc": get_action_rule(1.0)}
        start["BP"] = tree_killer.mixed_rule()

        policies = tree_killer.solve(start)

        assert policies["TDoc"].to_dict() == {(0.0,): 0.2, (1.0,): 1.0}
        assert policies["PT"].to_dict() == {(): 0.0}

    def test_cyclic_relevance_graph_raises(self):
        """Decisions that rely on each other admit no one-at-a-time order."""
        assert not simultaneous_block.relevance_graph().is_acyclic()

        with pytest.raises(NotImplementedError) as excinfo:
            solver(simultaneous_block).solve()

        message = str(excinfo.value)
        assert "a1" in message and "a2" in message

    def test_prisoners_dilemma_defection_is_dominant(self):
        """Each player defects whether the other cooperates or defects."""
        game = solver(
            macid.prisoners_dilemma_block,
            actions=np.array([0.0, 0.5, 1.0]),
        )
        for opponent_action in [0.0, 1.0]:
            profile = {
                "D1": get_action_rule(opponent_action),
                "D2": get_action_rule(opponent_action),
            }
            for decision in ["D1", "D2"]:
                response = game.best_response(decision, profile)
                assert np.all(response.actions == 1.0)

    @pytest.mark.parametrize(
        "block",
        [
            macid.prisoners_dilemma_block,
            macid.iterated_prisoners_dilemma_block,
        ],
        ids=["one-shot", "iterated"],
    )
    def test_prisoners_dilemma_cyclic_component_is_refused(self, block):
        """Neither game admits the one-at-a-time order this solver requires."""
        with pytest.raises(NotImplementedError, match="solved jointly"):
            solver(block).solve()


class TestRelevanceOrder:
    """The solved rules honour what the relevance graph claims."""

    def test_best_response_ignores_a_policy_it_does_not_rely_on(self, tree_killer):
        """``TDoc`` relies on nothing, so opposed profiles give the same rule."""
        policies = tree_killer.solve()

        responses = [
            tree_killer.best_response(
                "TDoc",
                dict(
                    policies,
                    PT=get_action_rule(action),
                    BP=get_action_rule(action),
                ),
            ).to_dict()
            for action in (0.0, 1.0)
        ]

        assert responses[0] == responses[1] == {(0.0,): 0.2, (1.0,): 1.0}

    def test_best_response_follows_a_policy_it_relies_on(self, tree_killer):
        """``PT`` relies on ``TDoc``: replacing Bob's rule moves Alice's."""
        policies = tree_killer.solve()
        assert macid.tree_killer_block.relies_on("PT", "TDoc")

        passive = dict(policies, TDoc=get_action_rule(0.0))
        passive["BP"] = tree_killer.best_response(
            "BP", dict(passive, PT=tree_killer.mixed_rule())
        )

        assert policies["PT"].to_dict() == {(): 0.0}
        assert tree_killer.best_response("PT", passive).to_dict() == {(): 1.0}


class TestPayoffs:
    def test_payoff_sums_the_utilities_an_agent_owns(self, tree_killer):
        """An agent's payoff is the sum of their utility nodes."""
        block = macid.tree_killer_block
        policies = {
            "PT": get_action_rule(1.0),
            "TDoc": get_action_rule(0.0),
            "BP": get_action_rule(1.0),
        }
        vals = block.transition(dict(tree_killer.shocks), policies)

        alice = tree_killer.payoff(vals, "alice")
        expected = np.asarray(vals["E"]) + np.asarray(vals["V"])

        assert np.allclose(alice, expected)
        assert not np.allclose(alice, tree_killer.payoff(vals, "bob"))

    def test_conditional_payoffs_shape_and_support(self, tree_killer):
        """One expected payoff per information cell per candidate action."""
        payoffs = tree_killer.conditional_payoffs(
            "TDoc", tree_killer.initial_policies()
        )

        assert payoffs.cells.tolist() == [[0.0], [1.0]]
        assert payoffs.payoff.shape == (2, len(payoffs.actions))
        assert payoffs.counts.sum() == SHOCK_SAMPLES
        assert (payoffs.counts > 0).all()

    def test_empty_information_set_is_one_cell(self, tree_killer):
        payoffs = tree_killer.conditional_payoffs("PT", tree_killer.initial_policies())

        assert payoffs.cells.shape == (1, 0)
        assert payoffs.payoff.shape == (1, len(payoffs.actions))

    def test_unattributed_control_among_several_owners_raises(self):
        """Without agent attribution there is no telling whose payoff to use."""
        block = DBlock(
            **{
                "name": "unattributed",
                "dynamics": {
                    "a1": Control([]),
                    "a2": Control([], agent="p2"),
                    "u1": lambda a1: -a1,
                    "u2": lambda a2: -a2,
                },
                "reward": {"u1": "p1", "u2": "p2"},
            }
        )

        with pytest.raises(ValueError, match="no agent attribution"):
            solver(block).best_response("a1", solver(block).initial_policies())

    def test_agent_owning_no_utility_raises(self):
        block = DBlock(
            **{
                "name": "unowned",
                "dynamics": {
                    "a1": Control([], agent="nobody"),
                    "u1": lambda a1: -a1,
                },
                "reward": {"u1": "p1"},
            }
        )

        with pytest.raises(ValueError, match="owns no reward variable"):
            solver(block).solve()

    def test_too_many_information_cells_raises(self):
        """Conditioning by grouping samples needs observations that repeat."""
        block = DBlock(
            **{
                "name": "continuous_observation",
                "shocks": {"z": (Uniform, {"low": 0.0, "high": 1.0})},
                "dynamics": {
                    "a": Control(["z"], agent="p"),
                    "u": lambda a, z: -((a - z) ** 2),
                },
                "reward": {"u": "p"},
            }
        )

        with pytest.raises(ValueError, match="max_cells"):
            solver(block, max_cells=100).solve()


class TestMixedRule:
    def test_every_action_is_played(self, tree_killer):
        played = np.unique(tree_killer.mixed_rule()())

        assert np.allclose(played, tree_killer.actions)

    def test_weights_shift_the_shares(self, tree_killer):
        weights = np.ones_like(tree_killer.actions)
        weights[-1] = 100.0

        played = tree_killer.mixed_rule(weights)()

        assert np.allclose(np.unique(played), tree_killer.actions)
        assert (played == tree_killer.actions[-1]).mean() > 0.5

    def test_wrong_number_of_weights_raises(self, tree_killer):
        with pytest.raises(ValueError, match="weights"):
            tree_killer.mixed_rule([1.0, 1.0])


class TestTabulatedRule:
    def test_empty_information_set_returns_a_scalar(self):
        rule = TabulatedRule([], np.zeros((1, 0)), [0.25])

        assert rule() == 0.25

    def test_nearest_cell_answers_an_unseen_observation(self):
        rule = TabulatedRule(["x"], [[0.0], [1.0]], [0.2, 0.8])

        assert rule(0.4) == 0.2
        assert rule(0.6) == 0.8
        assert np.allclose(rule(np.array([0.0, 0.9])), [0.2, 0.8])

    def test_scalar_and_array_queries_agree(self):
        rule = TabulatedRule(["x", "y"], [[0.0, 0.0], [1.0, 1.0]], [0.1, 0.9])

        assert rule(1.0, 1.0) == 0.9
        assert np.allclose(rule(np.array([1.0]), np.array([1.0])), [0.9])

    def test_wrong_arity_raises(self):
        rule = TabulatedRule(["x"], [[0.0]], [0.5])

        with pytest.raises(TypeError, match="positional argument"):
            rule(0.0, 1.0)

    def test_action_per_cell_is_required(self):
        with pytest.raises(ValueError, match="one action per cell"):
            TabulatedRule(["x"], [[0.0], [1.0]], [0.5])

    def test_is_usable_as_a_decision_rule(self, tree_killer):
        """A solved rule drives ``Block.transition`` like any other rule."""
        policies = tree_killer.solve()
        vals = macid.tree_killer_block.transition(dict(tree_killer.shocks), policies)

        # The patio is built exactly when the doctor was called in earnest.
        assert np.allclose(np.asarray(vals["BP"]), np.asarray(vals["TDoc"]) == 1.0)


class TestSamplesDeprecation:
    """`samples` is accepted for one release, under a warning."""

    def test_sets_shock_samples(self):
        with pytest.warns(DeprecationWarning):
            solved = TabularBestResponseSolver(
                macid.tree_killer_block, samples=1_000, rng=np.random.default_rng(0)
            )

        assert solved.shock_samples == 1_000
        assert not hasattr(solved, "samples")
