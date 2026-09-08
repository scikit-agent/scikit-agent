from conftest import case_0
import numpy as np
import os
import skagent.ann as ann
import skagent.bellman as bellman
import skagent.block as block
import inspect
import skagent.grid as grid
from skagent.loss import (
    BellmanEquationLoss,
    CustomLoss,
    EstimatedDiscountedLifetimeRewardLoss,
    EulerEquationLoss,
    StaticRewardLoss,
    static_reward,
)
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_case_0(self):
        bp = case_0["bp"]

        cl = CustomLoss(
            static_reward,
            bp,
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["bp"], width=16)
        ann.train_block_nn(bpn, states_0_N, cl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )


# One agent owning two reward variables, and two agents owning one each. The
# first distinguishes an agent's whole payoff from its first reward symbol; the
# second checks the sum stays per-agent rather than becoming global.
def two_reward_period():
    blk = block.DBlock(
        name="two_rewards",
        dynamics={
            "c": block.Control([], agent="a"),
            "u": lambda c: 2.0 * c,
            "v": lambda c: 10.0 - c,
        },
        reward={"u": "a", "v": "a"},
    )
    return bellman.BellmanPeriod(blk, None, {})


def two_agent_period():
    blk = block.DBlock(
        name="two_agents",
        dynamics={
            "c": block.Control([], agent="a"),
            "u": lambda c: 2.0 * c,
            "v": lambda c: 10.0 - c,
        },
        reward={"u": "a", "v": "b"},
    )
    return bellman.BellmanPeriod(blk, None, {})


class TestStaticReward(unittest.TestCase):
    """An agent's payoff is every reward symbol it owns."""

    def test_an_agents_reward_symbols_are_summed(self):
        reward = static_reward(two_reward_period(), {"c": lambda: 3.0}, {})

        # u = 6, v = 7. Taking the first symbol alone would give 6.
        self.assertEqual(reward, 13.0)

    def test_naming_the_agent_gives_the_same_sum(self):
        reward = static_reward(two_reward_period(), {"c": lambda: 3.0}, {}, agent="a")

        self.assertEqual(reward, 13.0)

    def test_the_sum_is_per_agent_and_not_global(self):
        """Where two agents own one reward each, an agent gets only its own."""
        period = two_agent_period()

        self.assertEqual(static_reward(period, {"c": lambda: 3.0}, {}, agent="a"), 6.0)
        self.assertEqual(static_reward(period, {"c": lambda: 3.0}, {}, agent="b"), 7.0)

    def test_an_unnamed_agent_on_a_two_agent_block_sums_both(self):
        """Not any agent's objective; the caller has to name one to get one."""
        reward = static_reward(two_agent_period(), {"c": lambda: 3.0}, {})

        self.assertEqual(reward, 13.0)

    def test_a_nan_reward_raises_for_a_symbol_that_is_not_the_first(self):
        """The NaN check covers every symbol summed, not just the leading one."""
        blk = block.DBlock(
            name="nan_reward",
            dynamics={
                "c": block.Control([], agent="a"),
                "u": lambda c: np.array([2.0 * c]),
                "v": lambda c: np.array([np.nan]),
            },
            reward={"u": "a", "v": "a"},
        )
        period = bellman.BellmanPeriod(blk, None, {})

        with self.assertRaises(ValueError) as caught:
            static_reward(period, {"c": lambda: 3.0}, {})

        self.assertIn("v", str(caught.exception))


class TestStaticRewardLossAgent:
    """The loss maximizes the payoff of the agent it is given, and no other."""

    def period(self):
        blk = block.DBlock(
            name="two_agents",
            dynamics={
                "c": block.Control([], agent="a"),
                "u": lambda c: 2.0 * c,
                "v": lambda c: 10.0 - c,
            },
            reward={"u": "a", "v": "b"},
        )
        return bellman.BellmanPeriod(blk, None, {})

    def losses_at(self, agent):
        period = self.period()
        givens = grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 2}})
        loss = StaticRewardLoss(period, agent=agent)
        return float(torch.as_tensor(loss({"c": lambda: 3.0}, givens)).mean())

    def test_each_agent_gets_its_own_payoff(self):
        # u = 6 belongs to a, v = 7 belongs to b. The loss is the negative.
        assert self.losses_at("a") == -6.0
        assert self.losses_at("b") == -7.0

    def test_naming_no_agent_sums_both(self):
        """A planner's objective, and no player's -- kept explicit."""
        assert self.losses_at(None) == -13.0


#: Every loss in the module, so the uniformity tests cannot silently skip one.
LOSS_CLASSES = [
    CustomLoss,
    StaticRewardLoss,
    EstimatedDiscountedLifetimeRewardLoss,
    BellmanEquationLoss,
    EulerEquationLoss,
]


class TestTheLossApi:
    """One shape for every loss, so the family cannot drift apart again."""

    def test_the_period_is_the_only_positional_argument(self):
        """`CustomLoss` also takes the function it wraps, and nothing else is
        positional: a loss reads its calibration from the period."""
        for cls in LOSS_CLASSES:
            positional = [
                name
                for name, param in inspect.signature(cls).parameters.items()
                if param.kind is param.POSITIONAL_OR_KEYWORD
            ]
            expected = (
                ["loss_function", "bellman_period"]
                if cls is CustomLoss
                else ["bellman_period"]
            )
            assert positional == expected, cls.__name__

    def test_no_loss_takes_parameters(self):
        """The period carries the calibration; passing it twice let the two
        disagree."""
        for cls in LOSS_CLASSES:
            assert "parameters" not in inspect.signature(cls).parameters, cls.__name__

    def test_every_loss_is_told_whose_payoff_it_maximizes(self):
        for cls in LOSS_CLASSES:
            agent = inspect.signature(cls).parameters.get("agent")
            assert agent is not None, cls.__name__
            assert agent.kind is agent.KEYWORD_ONLY, cls.__name__

    def test_every_loss_reads_the_periods_calibration(self):
        blk = block.DBlock(
            name="scaled",
            dynamics={"c": block.Control([], agent="a"), "u": lambda c, k: k * c},
            reward={"u": "a"},
        )
        one = bellman.BellmanPeriod(blk, None, {"k": 1.0})
        two = bellman.BellmanPeriod(blk, None, {"k": 2.0})

        assert StaticRewardLoss(one).parameters == {"k": 1.0}
        assert StaticRewardLoss(two).parameters == {"k": 2.0}


class TestLifetimeRewardAgent:
    """The discounted lifetime objective is one agent's, not everyone's."""

    def losses_at(self, agent):
        period = bellman.BellmanPeriod(
            block.DBlock(
                name="two_agents",
                dynamics={
                    "c": block.Control([], agent="a"),
                    "u": lambda c: 2.0 * c,
                    "v": lambda c: 10.0 - c,
                },
                reward={"u": "a", "v": "b"},
            ),
            None,
            {},
        )
        givens = grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 2}})
        loss_fn = EstimatedDiscountedLifetimeRewardLoss(period, big_t=1, agent=agent)
        return float(torch.as_tensor(loss_fn({"c": lambda: 3.0}, givens)).mean())

    def test_each_agent_gets_its_own_payoff(self):
        # u = 6 belongs to a, v = 7 belongs to b, undiscounted over one period.
        assert self.losses_at("a") == -6.0
        assert self.losses_at("b") == -7.0

    def test_naming_no_agent_sums_both(self):
        """A planner's objective, and no player's -- kept explicit."""
        assert self.losses_at(None) == -13.0
