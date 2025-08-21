from conftest import case_1, case_2
import numpy as np
import skagent.bellman as bellman
from skagent.distributions import Normal
import skagent.model as model
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parameters = {"q": 1.1}

block_data = {
    "name": "test block - maliar",
    "dynamics": {
        "c": model.Control(["a"]),
        "a": lambda a, c, e, q: q * a - c + e,
        "e": lambda e: e,
        "u": lambda c: np.log(c),
    },
    "reward": {"u": "consumer"},
}

states_0 = {
    "a": 1,
    "e": 0.1,
}

# a dummy policy
decision_rules = {"c": lambda a: a / 2}
decisions = {"c": 0.5}


class TestSolverFunctions(unittest.TestCase):
    def setUp(self):
        self.block = model.DBlock(**block_data)

    def test_create_transition_function(self):
        transition_function = bellman.create_transition_function(self.block, ["a", "e"])

        states_1 = transition_function(states_0, {}, decisions, parameters=parameters)

        self.assertAlmostEqual(states_1["a"], 0.7)
        self.assertEqual(states_1["e"], 0.1)

    def test_create_decision_function(self):
        decision_function = bellman.create_decision_function(self.block, decision_rules)

        decisions_0 = decision_function(states_0, {}, parameters=parameters)

        self.assertEqual(decisions_0["c"], 0.5)

    def test_create_reward_function(self):
        reward_function = bellman.create_reward_function(self.block)

        reward_0 = reward_function(states_0, {}, decisions, parameters=parameters)

        self.assertAlmostEqual(reward_0["u"], -0.69314718)

    def test_estimate_discounted_lifetime_reward(self):
        dlr_0 = bellman.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            0,
            parameters=parameters,
        )

        self.assertEqual(dlr_0, 0)

        dlr_1 = bellman.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            1,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_1, -0.69314718)

        dlr_2 = bellman.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            2,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_2, -1.63798709)

    def test_estimate_bellman_residual(self):
        """Test the Bellman residual helper function."""

        # Create a simple value network with correct interface
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth  # Linear value function

        # Create a simple decision function
        def simple_decision_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.5 * wealth
            return {"consumption": consumption}

        # Create a simple test block
        test_block = model.DBlock(
            name="test_bellman_residual",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={
                "wealth": lambda wealth, income, consumption: wealth
                + income
                - consumption,
                "consumption": model.Control(iset=["wealth"], agent="consumer"),
                "utility": lambda consumption: torch.log(consumption + 1e-8),
            },
            reward={"utility": "consumer"},
        )
        test_block.construct_shocks({})

        # Test states and shocks - need combined object with both periods
        states_t = {"wealth": torch.tensor([2.0, 4.0])}
        shocks = {
            "income_0": torch.tensor([1.0, 1.0]),  # Period t
            "income_1": torch.tensor([1.2, 0.8]),  # Period t+1 (independent)
        }

        # Estimate Bellman residual
        residual = bellman.estimate_bellman_residual(
            test_block,
            0.95,  # discount factor
            simple_value_network,
            simple_decision_function,
            states_t,
            shocks,  # Now passing combined shock object
            parameters={},
        )

        # Check that we get a tensor with the right shape
        self.assertIsInstance(residual, torch.Tensor)
        self.assertEqual(residual.shape, (2,))  # Should match input batch size

        # Check that residuals are finite (not NaN or inf)
        self.assertTrue(torch.all(torch.isfinite(residual)))


class TestLifetimeReward(unittest.TestCase):
    """
    More tests of the lifetime reward function specifically.
    """

    def setUp(self):
        self.states_0 = {"a": 0}

    def test_block_1(self):
        dlr_1 = bellman.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            1,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0]]))},
        )

        self.assertEqual(dlr_1, 0)

        # big_t is 2
        dlr_1_2 = bellman.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            2,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0], [0]]))},
        )

        self.assertEqual(dlr_1_2, 0)

    def test_block_2(self):
        dlr_2 = bellman.estimate_discounted_lifetime_reward(
            case_2["block"],
            0.9,
            case_2["optimal_dr"],
            self.states_0,
            1,
            shocks_by_t={
                "theta": torch.FloatTensor(np.array([[0]])),
                "psi": torch.FloatTensor(np.array([[0]])),
            },
        )

        self.assertEqual(dlr_2, 0)
