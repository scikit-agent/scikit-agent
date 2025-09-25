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


class TestGradRewardFunction(unittest.TestCase):
    """
    Test suite for the get_grad_reward_function and grad_reward_function functionality.
    """

    def setUp(self):
        """Set up test blocks and common test data."""
        # Simple consumption block for testing gradients
        self.simple_block = model.DBlock(
            name="simple_consumption",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,  # Simple wealth dynamics
                "u": lambda c: torch.log(c),  # Log utility
            },
            reward={"u": "consumer"},
        )

        # Block with multiple rewards
        self.multi_reward_block = model.DBlock(
            name="multi_reward",
            dynamics={
                "c1": model.Control(["w"], agent="consumer1"),
                "c2": model.Control(["w"], agent="consumer2"),
                "w": lambda w, c1, c2: w - c1 - c2,  # Shared wealth
                "u1": lambda c1: torch.log(c1),  # Consumer 1 utility
                "u2": lambda c2: -0.5 * c2**2,  # Consumer 2 quadratic utility
            },
            reward={"u1": "consumer1", "u2": "consumer2"},
        )

        # Block with shocks and parameters
        self.shock_block = model.DBlock(
            name="consumption_with_shocks",
            shocks={"theta": Normal(mu=0.0, sigma=0.1)},
            dynamics={
                "c": model.Control(["a"], agent="consumer"),
                "a": lambda a, c, theta, gamma: gamma * a - c + theta,
                "u": lambda c, theta: torch.log(c + theta + 1e-8),
            },
            reward={"u": "consumer"},
        )
        self.shock_block.construct_shocks({})

    def test_get_grad_reward_function_basic(self):
        """Test basic functionality of get_grad_reward_function."""
        grad_reward_func = bellman.get_grad_reward_function(self.simple_block)

        # Create test inputs with requires_grad=True
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c}
        wrt = {"c": c}  # Compute gradient w.r.t. consumption

        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # For u = log(c), du/dc = 1/c = 1/0.5 = 2.0
        expected_grad = 1.0 / c

        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad, atol=1e-6))

    def test_get_grad_reward_function_multiple_variables(self):
        """Test gradients with respect to multiple variables."""
        grad_reward_func = bellman.get_grad_reward_function(self.simple_block)

        # Create test inputs
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c}
        wrt = {"c": c, "a": a}  # Compute gradients w.r.t. both variables

        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # For u = log(c), du/dc = 1/c, du/da = 0 (unused variable)
        expected_grad_c = 1.0 / c

        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        self.assertIn("a", gradients["u"])

        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad_c, atol=1e-6))
        self.assertIsNone(gradients["u"]["a"])  # Unused variable returns None

    def test_get_grad_reward_function_multiple_rewards(self):
        """Test gradients for multiple rewards."""
        grad_reward_func = bellman.get_grad_reward_function(self.multi_reward_block)

        # Create test inputs
        c1 = torch.tensor(0.3, requires_grad=True)
        c2 = torch.tensor(0.2, requires_grad=True)
        w = torch.tensor(1.0, requires_grad=True)

        states_t = {"w": w}
        controls_t = {"c1": c1, "c2": c2}
        wrt = {"c1": c1, "c2": c2}

        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # For u1 = log(c1), du1/dc1 = 1/c1, du1/dc2 = 0
        # For u2 = -0.5*c2^2, du2/dc1 = 0, du2/dc2 = -c2
        expected_grad_u1_c1 = 1.0 / c1
        expected_grad_u2_c2 = -c2

        self.assertIn("u1", gradients)
        self.assertIn("u2", gradients)

        self.assertTrue(
            torch.allclose(gradients["u1"]["c1"], expected_grad_u1_c1, atol=1e-6)
        )
        self.assertIsNone(gradients["u1"]["c2"])  # u1 doesn't depend on c2

        self.assertIsNone(gradients["u2"]["c1"])  # u2 doesn't depend on c1
        self.assertTrue(
            torch.allclose(gradients["u2"]["c2"], expected_grad_u2_c2, atol=1e-6)
        )

    def test_get_grad_reward_function_with_agent_filter(self):
        """Test agent filtering in grad_reward_function."""
        # Test with agent filter for consumer1 only
        grad_reward_func = bellman.get_grad_reward_function(
            self.multi_reward_block, agent="consumer1"
        )

        c1 = torch.tensor(0.3, requires_grad=True)
        c2 = torch.tensor(0.2, requires_grad=True)
        w = torch.tensor(1.0, requires_grad=True)

        states_t = {"w": w}
        controls_t = {"c1": c1, "c2": c2}
        wrt = {"c1": c1}

        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # Should only contain u1 (consumer1's reward)
        self.assertIn("u1", gradients)
        self.assertNotIn("u2", gradients)

        expected_grad = 1.0 / c1
        self.assertTrue(torch.allclose(gradients["u1"]["c1"], expected_grad, atol=1e-6))

    def test_get_grad_reward_function_with_shocks_and_parameters(self):
        """Test gradients with shocks and parameters."""
        grad_reward_func = bellman.get_grad_reward_function(self.shock_block)

        # Create test inputs
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        theta = torch.tensor(0.1, requires_grad=True)

        states_t = {"a": a}
        shocks_t = {"theta": theta}
        controls_t = {"c": c}
        parameters = {"gamma": 0.95}
        wrt = {"c": c, "theta": theta}

        gradients = grad_reward_func(states_t, shocks_t, controls_t, parameters, wrt)

        # For u = log(c + theta + eps), du/dc = 1/(c + theta + eps), du/dtheta = 1/(c + theta + eps)
        eps = 1e-8
        expected_grad = 1.0 / (c + theta + eps)

        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        self.assertIn("theta", gradients["u"])

        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad, atol=1e-6))
        self.assertTrue(
            torch.allclose(gradients["u"]["theta"], expected_grad, atol=1e-6)
        )

    def test_get_grad_reward_function_envelope_condition_example(self):
        """Test usage pattern for envelope condition in optimization."""
        # This test demonstrates how the function would be used for envelope conditions
        grad_reward_func = bellman.get_grad_reward_function(self.simple_block)

        # Simulate a batch of states and controls
        batch_size = 3
        c = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c}
        wrt = {"c": c, "a": a}  # Gradients needed for envelope condition

        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # Check that we get gradients for the full batch
        self.assertEqual(gradients["u"]["c"].shape, (batch_size,))
        self.assertIsNone(gradients["u"]["a"])  # u doesn't depend on a

        # For log utility, du/dc = 1/c
        expected_grad_c = 1.0 / c
        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad_c, atol=1e-6))

    def test_get_grad_reward_function_error_handling(self):
        """Test error handling and edge cases."""
        grad_reward_func = bellman.get_grad_reward_function(self.simple_block)

        # Test with variable that doesn't require gradients
        c_no_grad = torch.tensor(0.5, requires_grad=False)
        a = torch.tensor(1.0, requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c_no_grad}
        wrt = {"c": c_no_grad}  # This should handle gracefully

        # This should work but return None gradients
        gradients = grad_reward_func(states_t, {}, controls_t, {}, wrt)
        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        # Gradient should be None for tensor without requires_grad=True
        self.assertIsNone(gradients["u"]["c"])

    def test_get_grad_reward_function_consistency_across_calls(self):
        """Test that multiple calls with same inputs give consistent results."""
        grad_reward_func = bellman.get_grad_reward_function(self.simple_block)

        # Create test inputs
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c}
        wrt = {"c": c}

        # Call multiple times
        grad1 = grad_reward_func(states_t, {}, controls_t, {}, wrt)
        grad2 = grad_reward_func(states_t, {}, controls_t, {}, wrt)

        # Results should be identical
        self.assertTrue(torch.allclose(grad1["u"]["c"], grad2["u"]["c"], atol=1e-10))
