from conftest import case_1, case_2
import numpy as np
import skagent.bellman as bellman
from skagent.distributions import Normal
import skagent.block as model
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parameters = {"q": 1.1, "beta": 0.9}


def _make_consumption_savings_bp():
    """Consumption-savings block used by several test classes."""
    block = model.DBlock(
        name="consumption_savings",
        shocks={"income": Normal(mu=1.0, sigma=0.1)},
        dynamics={
            "wealth": lambda wealth, income, consumption: wealth + income - consumption,
            "consumption": model.Control(iset=["wealth"], agent="consumer"),
            "utility": lambda consumption: torch.log(consumption + 1e-8),
        },
        reward={"utility": "consumer"},
    )
    block.construct_shocks({})
    return block, bellman.BellmanPeriod(block, "beta", {"beta": 0.9})


def _make_multi_control_bp():
    """Two-control block used by Euler and FOC residual tests."""
    block = model.DBlock(
        name="multi_control",
        dynamics={
            "c1": model.Control(["a"]),
            "c2": model.Control(["a"]),
            "a": lambda a, c1, c2: a - c1 - c2,
            "u": lambda c1, c2: torch.log(c1 + 1e-8) + torch.log(c2 + 1e-8),
        },
        reward={"u": "consumer"},
    )
    return block, bellman.BellmanPeriod(block, "beta", {"beta": 0.9})


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


class TestBellmanPeriodFunctions(unittest.TestCase):
    def setUp(self):
        self.bp = bellman.BellmanPeriod(model.DBlock(**block_data), "beta", parameters)

    def test_transition_function(self):
        states_1 = self.bp.transition_function(states_0, decisions)

        self.assertAlmostEqual(states_1["a"], 0.7)
        self.assertEqual(states_1["e"], 0.1)

    def test_decision_function(self):
        decisions_0 = self.bp.decision_function(
            states_0, parameters=parameters, decision_rules=decision_rules
        )

        self.assertEqual(decisions_0["c"], 0.5)

    def test_reward_function(self):
        reward_0 = self.bp.reward_function(states_0, decisions, parameters=parameters)

        self.assertAlmostEqual(reward_0["u"], -0.69314718)

    def test_estimate_discounted_lifetime_reward(self):
        dlr_0 = bellman.estimate_discounted_lifetime_reward(
            self.bp,
            decision_rules,
            states_0,
            0,
            parameters=parameters,
        )

        self.assertEqual(dlr_0, 0)

        dlr_1 = bellman.estimate_discounted_lifetime_reward(
            self.bp,
            decision_rules,
            states_0,
            1,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_1, -0.69314718)

        dlr_2 = bellman.estimate_discounted_lifetime_reward(
            self.bp,
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

        _, test_bp = _make_consumption_savings_bp()

        # Test states and shocks - need combined object with both periods
        states_t = {"wealth": torch.tensor([2.0, 4.0])}
        shocks = {
            "income_0": torch.tensor([1.0, 1.0]),  # Period t
            "income_1": torch.tensor([1.2, 0.8]),  # Period t+1 (independent)
        }

        # Estimate Bellman residual
        residual = bellman.estimate_bellman_residual(
            test_bp,
            simple_value_network,
            simple_decision_function,
            states_t,
            shocks,  # Now passing combined shock object,
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
            bellman.BellmanPeriod(case_1["block"], "beta", {"beta": 0.9}),
            case_1["optimal_dr"],
            self.states_0,
            1,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0]]))},
        )

        self.assertEqual(dlr_1, 0)

        # big_t is 2
        dlr_1_2 = bellman.estimate_discounted_lifetime_reward(
            bellman.BellmanPeriod(case_1["block"], "beta", {"beta": 0.9}),
            case_1["optimal_dr"],
            self.states_0,
            2,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0], [0]]))},
        )

        self.assertEqual(dlr_1_2, 0)

    def test_block_2(self):
        dlr_2 = bellman.estimate_discounted_lifetime_reward(
            bellman.BellmanPeriod(case_2["block"], "beta", {"beta": 0.9}),
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

        self.simple_bp = bellman.BellmanPeriod(self.simple_block, "beta", {"beta": 0.9})

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

        self.multi_reward_bp = bellman.BellmanPeriod(
            self.multi_reward_block, "beta", {"beta": 0.9}
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

        self.shock_bp = bellman.BellmanPeriod(self.shock_block, "beta", {"beta": 0.9})

    def _simple_inputs(self):
        """Create standard (c, a) inputs for the simple block."""
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        states_t = {"a": a}
        controls_t = {"c": c}
        return states_t, controls_t, c, a

    def _multi_reward_inputs(self):
        """Create standard (c1, c2, w) inputs for the multi-reward block."""
        c1 = torch.tensor(0.3, requires_grad=True)
        c2 = torch.tensor(0.2, requires_grad=True)
        w = torch.tensor(1.0, requires_grad=True)
        states_t = {"w": w}
        controls_t = {"c1": c1, "c2": c2}
        return states_t, controls_t, c1, c2, w

    def test_get_grad_reward_function_basic(self):
        """Test basic functionality of get_grad_reward_function."""
        states_t, controls_t, c, a = self._simple_inputs()

        gradients = self.simple_bp.grad_reward_function(states_t, controls_t, {"c": c})

        # For u = log(c), du/dc = 1/c = 1/0.5 = 2.0
        expected_grad = 1.0 / c

        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad, atol=1e-6))

    def test_get_grad_reward_function_multiple_variables(self):
        """Test gradients with respect to multiple variables."""
        states_t, controls_t, c, a = self._simple_inputs()

        gradients = self.simple_bp.grad_reward_function(
            states_t, controls_t, {"c": c, "a": a}
        )

        # For u = log(c), du/dc = 1/c, du/da = 0 (unused variable)
        expected_grad_c = 1.0 / c

        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        self.assertIn("a", gradients["u"])

        self.assertTrue(torch.allclose(gradients["u"]["c"], expected_grad_c, atol=1e-6))
        self.assertIsNone(gradients["u"]["a"])  # Unused variable returns None

    def test_get_grad_reward_function_multiple_rewards(self):
        """Test gradients for multiple rewards."""
        states_t, controls_t, c1, c2, w = self._multi_reward_inputs()

        gradients = self.multi_reward_bp.grad_reward_function(
            states_t, controls_t, {"c1": c1, "c2": c2}
        )

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
        states_t, controls_t, c1, c2, w = self._multi_reward_inputs()

        gradients = self.multi_reward_bp.grad_reward_function(
            states_t, controls_t, {"c1": c1}, agent="consumer1"
        )

        # Should only contain u1 (consumer1's reward)
        self.assertIn("u1", gradients)
        self.assertNotIn("u2", gradients)

        expected_grad = 1.0 / c1
        self.assertTrue(torch.allclose(gradients["u1"]["c1"], expected_grad, atol=1e-6))

    def test_get_grad_reward_function_with_shocks_and_parameters(self):
        """Test gradients with shocks and parameters."""
        # Create test inputs
        c = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        theta = torch.tensor(0.1, requires_grad=True)

        states_t = {"a": a}
        shocks_t = {"theta": theta}
        controls_t = {"c": c}
        parameters = {"gamma": 0.95}
        wrt = {"c": c, "theta": theta}

        gradients = self.shock_bp.grad_reward_function(
            states_t, controls_t, wrt, shocks=shocks_t, parameters=parameters
        )

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

    def test_get_grad_reward_function_error_handling(self):
        """Test error handling and edge cases."""
        # Test with variable that doesn't require gradients
        c_no_grad = torch.tensor(0.5, requires_grad=False)
        a = torch.tensor(1.0, requires_grad=True)

        states_t = {"a": a}
        controls_t = {"c": c_no_grad}
        wrt = {"c": c_no_grad}  # This should handle gracefully

        # This should work but return None gradients
        gradients = self.simple_bp.grad_reward_function(states_t, controls_t, wrt)
        self.assertIn("u", gradients)
        self.assertIn("c", gradients["u"])
        # Gradient should be None for tensor without requires_grad=True
        self.assertIsNone(gradients["u"]["c"])


class TestGradTransitionFunction(unittest.TestCase):
    """Direct tests for grad_transition_function."""

    def test_transition_gradient_simple(self):
        """For a_next = a - c, ∂a_next/∂c should be -1."""
        block = model.DBlock(
            name="simple_savings",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
            },
            reward={"a": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})

        a = torch.tensor(2.0, requires_grad=True)
        c = torch.tensor(0.5, requires_grad=True)

        grads = bp.grad_transition_function(
            {"a": a}, {"c": c}, {"c": c}, create_graph=True
        )

        # ∂a_next/∂c = -1
        self.assertIn("a", grads)
        self.assertIn("c", grads["a"])
        self.assertTrue(torch.allclose(grads["a"]["c"], torch.tensor(-1.0), atol=1e-6))


class TestGradPreStateFunction(unittest.TestCase):
    """Direct tests for grad_pre_state_function."""

    def test_pre_state_gradient_simple(self):
        """For m = R*a + y, ∂m/∂a should be R."""
        R_val = 1.04
        block = model.DBlock(
            name="pre_state_test",
            dynamics={
                "m": lambda a, R: R * a,
                "c": model.Control(["m"]),
                "a": lambda m, c: m - c,
            },
            reward={"a": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9, "R": R_val})

        a = torch.tensor(2.0, requires_grad=True)

        grads = bp.grad_pre_state_function(
            {"a": a},
            {"a": a},
            parameters={"beta": 0.9, "R": R_val},
            control_sym="c",
            create_graph=True,
        )

        # ∂m/∂a = R = 1.04
        self.assertIn("m", grads)
        self.assertIn("a", grads["m"])
        self.assertTrue(torch.allclose(grads["m"]["a"], torch.tensor(R_val), atol=1e-6))

    def test_missing_iset_raises(self):
        """grad_pre_state_function should raise when control has no iset."""
        block = model.DBlock(
            name="no_iset",
            dynamics={
                "x": lambda a: a * 2,  # not a Control, no iset
            },
            reward={},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        a = torch.tensor(1.0, requires_grad=True)

        with self.assertRaises(ValueError, msg="No control with pre-state found"):
            bp.grad_pre_state_function({"a": a}, {"a": a})


class TestEulerResidualErrorHandling(unittest.TestCase):
    """Test error branches in estimate_euler_residual."""

    def setUp(self):
        self.block = model.DBlock(
            name="simple",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c),
            },
            reward={"u": "consumer"},
        )
        self.bp = bellman.BellmanPeriod(self.block, "beta", {"beta": 0.9})

    def test_multi_control_returns_dict(self):
        """Multiple controls should return a dict of residuals."""
        _, multi_bp = _make_multi_control_bp()

        def df(s, sh, p):
            a = s["a"]
            return {"c1": a * 0.3, "c2": a * 0.2}

        result = bellman.estimate_euler_residual(
            multi_bp,
            df,
            {"a": torch.tensor([1.0])},
            {},
            {"beta": 0.9},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("c1", result)
        self.assertIn("c2", result)
        for v in result.values():
            self.assertIsInstance(v, torch.Tensor)


class TestComputeControlsTypeError(unittest.TestCase):
    """Test compute_controls error handling."""

    def test_invalid_df_type_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={"c": model.Control(["a"]), "a": lambda a, c: a - c},
            reward={},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})

        with self.assertRaises(TypeError, msg="callable decision function or a dict"):
            bp.compute_controls(42, {"a": torch.tensor(1.0)})


class TestGetRewardSymsErrors(unittest.TestCase):
    """Test get_reward_syms error handling."""

    def test_invalid_agent_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={"u": lambda c: c},
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})

        with self.assertRaises(ValueError, msg="No reward variables found"):
            bp.get_reward_syms(agent="nonexistent_agent")

    def test_no_rewards_raises(self):
        block = model.DBlock(name="test", dynamics={"x": lambda a: a}, reward={})
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})

        with self.assertRaises(ValueError, msg="No reward variables found"):
            bp.get_reward_syms()


class TestExtractPeriodShocksErrors(unittest.TestCase):
    """Test _extract_period_shocks error handling."""

    def setUp(self):
        self.block = model.DBlock(
            name="with_shocks",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c, income: a - c + income,
                "u": lambda c: torch.log(c),
            },
            reward={"u": "consumer"},
        )
        self.block.construct_shocks({})
        self.bp = bellman.BellmanPeriod(self.block, "beta", {"beta": 0.9})

    def test_missing_period_0_shock_raises(self):
        """Missing _0 shock key should raise KeyError."""
        shocks = {"income_1": torch.tensor([1.0])}
        with self.assertRaises(KeyError, msg="income_0"):
            bellman._extract_period_shocks(self.bp, shocks)

    def test_missing_period_1_shock_raises(self):
        """Missing _1 shock key should raise KeyError."""
        shocks = {"income_0": torch.tensor([1.0])}
        with self.assertRaises(KeyError, msg="income_1"):
            bellman._extract_period_shocks(self.bp, shocks)


class TestFischerBurmeister(unittest.TestCase):
    """Test the Fischer-Burmeister complementarity function."""

    def test_both_zero(self):
        """FB(0, 0) = 0."""
        a = torch.tensor(0.0)
        h = torch.tensor(0.0)
        result = bellman.fischer_burmeister(a, h)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_complementary_slackness(self):
        """FB(0, s) ≈ 0 for s > 0 and FB(f, 0) ≈ 0 for f > 0."""
        # When one is zero and the other is positive, FB should be ≈ 0
        s = torch.tensor(2.0)
        result = bellman.fischer_burmeister(torch.tensor(0.0), s)
        self.assertAlmostEqual(result.item(), 0.0, places=4)

        f = torch.tensor(3.0)
        result = bellman.fischer_burmeister(f, torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.0, places=4)

    def test_violation_nonzero(self):
        """FB(a, h) != 0 when both a > 0 and h > 0."""
        a = torch.tensor(1.0)
        h = torch.tensor(1.0)
        result = bellman.fischer_burmeister(a, h)
        self.assertNotAlmostEqual(result.item(), 0.0, places=2)

    def test_differentiable(self):
        """FB is differentiable through autograd."""
        a = torch.tensor(1.0, requires_grad=True)
        h = torch.tensor(2.0, requires_grad=True)
        result = bellman.fischer_burmeister(a, h)
        result.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(a.grad))
        self.assertTrue(torch.isfinite(h.grad))

    def test_differentiable_at_zero(self):
        """FB gradient is finite near zero due to epsilon safeguard."""
        a = torch.tensor(0.0, requires_grad=True)
        h = torch.tensor(0.0, requires_grad=True)
        result = bellman.fischer_burmeister(a, h)
        result.backward()
        self.assertTrue(torch.isfinite(a.grad))
        self.assertTrue(torch.isfinite(h.grad))


class TestEstimateBellmanFocResidual(unittest.TestCase):
    """Test estimate_bellman_foc_residual."""

    def setUp(self):
        self.block, self.bp = _make_consumption_savings_bp()

    def test_returns_finite_tensor_with_correct_shape(self):
        """FOC residual should match batch size and change with different policies."""

        def value_fn(states, shocks, params):
            return 10.0 * states["wealth"]

        def decision_fn_half(states, shocks, params):
            return {"consumption": 0.5 * states["wealth"]}

        def decision_fn_quarter(states, shocks, params):
            return {"consumption": 0.25 * states["wealth"]}

        states_t = {"wealth": torch.tensor([2.0, 4.0])}
        shocks = {
            "income_0": torch.tensor([1.0, 1.0]),
            "income_1": torch.tensor([1.2, 0.8]),
        }

        residual_half = bellman.estimate_bellman_foc_residual(
            self.bp, value_fn, decision_fn_half, states_t, shocks
        )
        residual_quarter = bellman.estimate_bellman_foc_residual(
            self.bp, value_fn, decision_fn_quarter, states_t, shocks
        )
        self.assertEqual(residual_half.shape, (2,))
        self.assertTrue(torch.all(torch.isfinite(residual_half)))
        self.assertTrue(torch.all(torch.isfinite(residual_quarter)))
        # Different policies should produce different FOC residuals
        self.assertFalse(
            torch.allclose(residual_half, residual_quarter),
            "Different policies should produce different FOC residuals",
        )

    def test_multi_control_returns_dict(self):
        """Multi-control model should return a dict of finite residuals per control."""
        _, multi_bp = _make_multi_control_bp()

        def value_fn(states, shocks, params):
            return 5.0 * states["a"]

        def df(s, sh, p):
            return {"c1": s["a"] * 0.3, "c2": s["a"] * 0.2}

        result = bellman.estimate_bellman_foc_residual(
            multi_bp,
            value_fn,
            df,
            {"a": torch.tensor([1.0, 2.0])},
            {},
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"c1", "c2"})
        for key, residual in result.items():
            self.assertIsInstance(
                residual, torch.Tensor, f"residual[{key}] not a tensor"
            )
            self.assertEqual(residual.shape, (2,), f"residual[{key}] wrong shape")
            self.assertTrue(
                torch.all(torch.isfinite(residual)),
                f"residual[{key}] contains non-finite values",
            )


class TestEnsureGrad(unittest.TestCase):
    """Test _ensure_grad helper."""

    def test_non_tensor_raises(self):
        """Non-tensor values should raise TypeError."""
        with self.assertRaises(TypeError, msg="must be a torch.Tensor"):
            bellman._ensure_grad({"c": 0.5}, "c")

    def test_already_has_grad(self):
        """Tensor with requires_grad should be returned unchanged."""
        c = torch.tensor(0.5, requires_grad=True)
        result, controls = bellman._ensure_grad({"c": c}, "c")
        self.assertIs(result, c)

    def test_adds_grad(self):
        """Tensor without requires_grad should be detached and reattached."""
        c = torch.tensor(0.5, requires_grad=False)
        result, controls = bellman._ensure_grad({"c": c}, "c")
        self.assertTrue(result.requires_grad)
