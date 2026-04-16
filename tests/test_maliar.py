from conftest import case_1, case_3, case_4
import numpy as np
import skagent.algos.maliar as maliar
import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
import skagent.block as model
import torch
import unittest
from skagent.distributions import Normal
from skagent.models.benchmarks import (
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
)

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


class TestGridManipulations(unittest.TestCase):
    def setUp(self):
        pass

    def test_givens_case_1(self):
        # TODO: we're going to need to build the blocks in the test, because of this mutation,
        #       or else make this return a copy.
        block = case_1["block"]
        block.construct_shocks(
            case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        state_grid = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 7},
            }
        )

        full_grid = maliar.generate_givens_from_states(state_grid, block, 1)

        self.assertEqual(full_grid["theta_0"].shape.numel(), 7)

    def test_givens_case_3(self):
        block = case_3["block"]
        block.construct_shocks(
            case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        state_grid = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 1, "count": 7},
            }
        )

        full_grid = maliar.generate_givens_from_states(state_grid, block, 2)

        self.assertEqual(len(full_grid["psi_0"]), 7)


class TestBellmanLossFunctions(unittest.TestCase):
    """Test the Bellman equation loss functions for the Maliar method."""

    def setUp(self):
        """Set up a simple consumption-savings model for testing."""
        # Create a simple consumption-savings model
        self.block = model.DBlock(
            name="consumption_savings",
            description="Simple consumption-savings model",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={
                "consumption": model.Control(
                    iset=["wealth"],
                    lower_bound=lambda wealth: 0.0,
                    upper_bound=lambda wealth: wealth,
                    agent="consumer",
                ),
                "wealth": lambda wealth, income, consumption: wealth
                + income
                - consumption,
                "utility": lambda consumption: torch.log(
                    consumption + 1e-8
                ),  # Add small constant to avoid log(0)
            },
            reward={"utility": "consumer"},
        )

        # Construct shocks
        self.block.construct_shocks({})

        # Parameters
        self.parameters = {"beta": 0.95}
        self.bp = bellman.BellmanPeriod(self.block, "beta", self.parameters)
        self.state_variables = ["wealth"]  # Endogenous state variables

        # Create a simple decision function for testing
        def simple_decision_function(states_t, shocks_t, parameters):
            # Simple consumption rule: consume half of wealth
            wealth = states_t["wealth"]
            consumption = 0.5 * wealth
            return {"consumption": consumption}

        self.decision_function = simple_decision_function

        # Create test grid with two independent shock realizations
        # This matches the new 2-shock requirement for Bellman equation
        wealth_values = torch.linspace(0.1, 10.0, 5)
        income_0_values = torch.linspace(0.8, 1.2, 5)  # Period t shocks
        income_1_values = torch.linspace(0.9, 1.1, 5)  # Period t+1 shocks (independent)
        self.test_grid = grid.Grid.from_dict(
            {
                "wealth": wealth_values,
                "income_0": income_0_values,
                "income_1": income_1_values,
            }
        )

    def test_bellman_loss_function_components(self):
        """Test that the Bellman loss function components work correctly."""
        # Test transition function
        states_t = {"wealth": torch.tensor([1.0, 2.0])}
        shocks_t = {"income": torch.tensor([1.0, 1.0])}
        controls_t = {"consumption": torch.tensor([0.5, 1.0])}

        next_states = self.bp.transition_function(
            states_t, controls_t, shocks=shocks_t, parameters=self.parameters
        )
        self.assertIn("wealth", next_states)
        self.assertTrue(torch.allclose(next_states["wealth"], torch.tensor([1.5, 2.0])))

        # Test reward function
        reward = self.bp.reward_function(
            states_t,
            controls_t,
            shocks=shocks_t,
            parameters=self.parameters,
            agent="consumer",
        )
        self.assertIn("utility", reward)
        # u = log(c + 1e-8) for c = [0.5, 1.0]
        expected_utility = torch.log(controls_t["consumption"] + 1e-8)
        self.assertTrue(
            torch.allclose(reward["utility"], expected_utility, atol=1e-6),
            f"Reward should be log(c). Got {reward['utility']}, expected {expected_utility}",
        )

    def test_bellman_loss_function_error_handling(self):
        """Test error handling in Bellman loss functions."""
        # Test with block that has no controls
        no_control_block = model.DBlock(
            name="no_control",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={"wealth": lambda wealth, income: wealth + income},
            reward={},
        )
        no_control_block.construct_shocks({})

        # Create a dummy value network for error testing
        def dummy_value_network(wealth):
            return 10.0 * wealth

        # Test with block that has no rewards
        no_reward_block = model.DBlock(
            name="no_reward",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={
                "wealth": lambda wealth, income: wealth + income,
                "consumption": model.Control(iset=["wealth"], agent="consumer"),
            },
            reward={},
        )
        no_reward_block.construct_shocks({})

        nrbp = bellman.BellmanPeriod(no_reward_block, "beta", {"beta": 0.95})

        with self.assertRaises(Exception) as context:
            loss.BellmanEquationLoss(nrbp, dummy_value_network)
        self.assertIn("No reward variables found in block", str(context.exception))

    def test_bellman_loss_function_integration(self):
        """Test integration with the Maliar training loop components."""

        # Create a more realistic decision function
        def learned_decision_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            # More sophisticated consumption rule
            consumption = 0.3 * wealth + 0.1
            # Ensure consumption is positive and not more than wealth
            consumption = torch.maximum(consumption, torch.tensor(0.01))
            consumption = torch.minimum(consumption, 0.9 * wealth)
            return {"consumption": consumption}

        # Create a simple value network with correct interface
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth  # Linear value function

        loss_function = loss.BellmanEquationLoss(
            self.bp,
            simple_value_network,
            self.parameters,
        )

        # Test with the learned decision function
        losses = loss_function(learned_decision_function, self.test_grid)

        self.assertIsInstance(losses, torch.Tensor)
        self.assertEqual(losses.shape, (5,))
        self.assertTrue(torch.all(losses >= 0))

        # Test that loss changes with different decision functions
        def different_decision_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.8 * wealth  # Different consumption rule
            return {"consumption": consumption}

        loss2 = loss_function(different_decision_function, self.test_grid)

        # Losses should be different for different decision functions
        self.assertFalse(torch.allclose(losses, loss2))

    def test_shock_independence_in_bellman_residual(self):
        """Test that independent shock realizations produce different results than identical shocks."""

        # Create a simple value network that depends on income
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            income = shocks_t["income"]
            return (
                10.0 * wealth + 5.0 * income
            )  # Value depends on both wealth and income

        states_t = {"wealth": torch.tensor([2.0, 4.0])}

        # Test 1: Identical shocks for both periods (should be like old behavior)
        shocks_identical = {
            "income_0": torch.tensor([1.0, 1.0]),  # Period t
            "income_1": torch.tensor([1.0, 1.0]),  # Period t+1 (same as t)
        }

        # Test 2: Independent shocks for the two periods
        shocks_independent = {
            "income_0": torch.tensor([1.0, 1.0]),  # Period t
            "income_1": torch.tensor([1.5, 0.5]),  # Period t+1 (different from t)
        }

        residual_identical = bellman.estimate_bellman_residual(
            self.bp,
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters=None,
        )

        residual_independent = bellman.estimate_bellman_residual(
            self.bp,
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_independent,
            parameters=None,
        )

        # Results should be different when using independent vs identical shocks
        self.assertFalse(torch.allclose(residual_identical, residual_independent))

        # Both should be finite
        self.assertTrue(torch.all(torch.isfinite(residual_identical)))
        self.assertTrue(torch.all(torch.isfinite(residual_independent)))

    def test_bellman_residual_error_handling(self):
        """Test error handling in the refactored Bellman residual function."""

        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth

        states_t = {"wealth": torch.tensor([2.0, 4.0])}

        # Test with missing shock periods
        shocks_missing_t1 = {
            "income_0": torch.tensor([1.0, 1.0]),
            # Missing "income_1"
        }

        with self.assertRaises(KeyError):
            bellman.estimate_bellman_residual(
                self.bp,
                simple_value_network,
                self.decision_function,
                states_t,
                shocks_missing_t1,
                parameters=None,
            )


def test_get_euler_residual_loss():
    """Test Euler equation loss function using D-2 benchmark with analytical policy."""
    # Use D-2 benchmark: Infinite horizon CRRA perfect foresight
    d2_block = get_benchmark_model("D-2")
    d2_calibration = get_benchmark_calibration("D-2")
    d2_policy = get_analytical_policy("D-2")

    # Construct shocks for the block (D-2 is deterministic, so this is a no-op)
    d2_block.construct_shocks(d2_calibration)

    # Create BellmanPeriod
    test_bp = bellman.BellmanPeriod(d2_block, "DiscFac", d2_calibration)

    # Create input grid with arrival states
    # D-2 uses: a (arrival assets)
    # D-2 is deterministic, so no shocks needed in grid
    n_points = 10
    input_grid = grid.Grid.from_dict(
        {
            "a": torch.linspace(0.5, 5.0, n_points),
        }
    )

    # Create Euler equation loss function
    loss_fn = loss.EulerEquationLoss(test_bp, parameters=d2_calibration)

    # Test that loss function works with the analytical optimal policy
    losses = loss_fn(d2_policy, input_grid)

    # Check that we get per-sample losses
    assert isinstance(losses, torch.Tensor)
    assert losses.shape[0] == n_points  # One loss per grid point
    assert torch.all(losses >= 0)  # Squared residuals should be non-negative

    # For the analytical optimal policy, Euler residuals should be near zero
    # (within numerical tolerance for perfect foresight model)
    mean_loss = torch.mean(losses).item()
    assert mean_loss < 1e-8, (
        f"Analytical optimal policy should have near-zero Euler loss. Got mean loss: {mean_loss:.6e}"
    )


class TestOneSidedEulerLoss(unittest.TestCase):
    """Test the one-sided Euler loss formula for borrowing-constrained models."""

    def test_one_sided_loss_formula_math(self):
        """Test the mathematical correctness of one-sided Euler loss.

        The constrained=True path uses: loss = torch.relu(-residual)**2
        - When residual > 0 (constraint binding): loss = 0
        - When residual < 0 (undersaving): loss = residual**2
        """
        # Test with positive residuals (constraint binding - should produce zero loss)
        positive_residual = torch.tensor([0.1, 0.5, 1.0, 2.0])
        constrained_loss_positive = torch.relu(-positive_residual) ** 2

        self.assertTrue(
            torch.allclose(constrained_loss_positive, torch.zeros(4)),
            f"Positive residuals should produce zero constrained loss. "
            f"Got: {constrained_loss_positive}",
        )

        # Test with negative residuals (undersaving - should produce squared loss)
        negative_residual = torch.tensor([-0.1, -0.5, -1.0, -2.0])
        constrained_loss_negative = torch.relu(-negative_residual) ** 2
        expected_loss_negative = negative_residual**2

        self.assertTrue(
            torch.allclose(constrained_loss_negative, expected_loss_negative),
            f"Negative residuals should have squared loss. "
            f"Got: {constrained_loss_negative}, expected: {expected_loss_negative}",
        )

        # Test with mixed residuals
        mixed_residual = torch.tensor([0.5, -0.3, 1.0, -0.8])
        constrained_loss_mixed = torch.relu(-mixed_residual) ** 2
        expected_mixed = torch.tensor([0.0, 0.09, 0.0, 0.64])

        self.assertTrue(
            torch.allclose(constrained_loss_mixed, expected_mixed, atol=1e-6),
            f"Mixed residuals not handled correctly. "
            f"Got: {constrained_loss_mixed}, expected: {expected_mixed}",
        )


class TestU2BorrowingAgainstHumanWealth(unittest.TestCase):
    """Test that U-2 allows borrowing against human wealth (c > m)."""

    def test_u2_allows_borrowing_at_low_wealth(self):
        """Test U-2: At low wealth, consumption can exceed cash-on-hand.

        The key feature of U-2 (vs U-3) is that the agent can borrow against
        human wealth h = 1/r. At m = 0, the analytical solution is:
        c = (1-β)/r ≈ 1.33 > 0 = m

        This test verifies that the analytical policy correctly produces
        consumption exceeding cash-on-hand at low asset levels.
        """
        calibration = get_benchmark_calibration("U-2")
        policy = get_analytical_policy("U-2")

        beta = calibration["DiscFac"]
        R = calibration["R"]
        r = R - 1
        h = 1 / r  # Normalized human wealth ≈ 33.33

        # Test at very low arrival assets where borrowing is needed
        test_states = {"a": torch.tensor([0.0, 0.1, 0.5])}
        test_shocks = {"psi": torch.ones(3)}  # Mean shock = 1

        result = policy(test_states, test_shocks, calibration)

        # Compute m for these states
        m = R * test_states["a"] / test_shocks["psi"] + 1

        # At a = 0: m = 1.0, analytical c = (1-0.96)*(1.0 + 33.33) ≈ 1.37
        # Since 1.37 > 1.0, this demonstrates borrowing against human wealth
        expected_c = (1 - beta) * (m + h)

        # Verify analytical formula is correct
        self.assertTrue(
            torch.allclose(result["c"], expected_c, atol=1e-6),
            f"Analytical policy should match formula. Got: {result['c']}, "
            f"expected: {expected_c}",
        )

        # The key test: at low wealth, consumption EXCEEDS cash-on-hand m
        # This is only possible by borrowing against human wealth
        self.assertTrue(
            torch.any(result["c"] > m),
            f"U-2 should allow borrowing: c > m for low wealth. "
            f"Got c = {result['c']}, m = {m}. "
            f"The agent should borrow against human wealth h = {h:.2f}",
        )

        # Specifically check at a = 0 where m = 1.0
        c_at_zero_assets = result["c"][0].item()
        m_at_zero_assets = m[0].item()

        self.assertGreater(
            c_at_zero_assets,
            m_at_zero_assets,
            f"At zero assets, consumption should exceed cash-on-hand. "
            f"c = {c_at_zero_assets:.4f}, m = {m_at_zero_assets:.4f}",
        )


class TestEulerLossConstrainedIntegration(unittest.TestCase):
    """Integration tests for EulerEquationLoss with constrained=True."""

    def test_constrained_loss_integration_with_model(self):
        """Test that EulerEquationLoss(constrained=True) produces correct loss values.

        This test verifies that the constrained flag is correctly wired through
        the entire loss computation pipeline, not just the formula in isolation.
        """
        # Use U-3 buffer stock model
        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")

        bp = bellman.BellmanPeriod(u3_block, "DiscFac", u3_calibration)

        # Create loss functions with both settings
        loss_unconstrained = loss.EulerEquationLoss(
            bp,
            parameters=u3_calibration,
            constrained=False,
        )
        loss_constrained = loss.EulerEquationLoss(
            bp,
            parameters=u3_calibration,
            constrained=True,
        )

        # Create a simple test grid (small to avoid OOM)
        test_grid = grid.Grid.from_config(
            {
                "a": {"min": 0.5, "max": 5.0, "count": 5},
                "psi_0": {"min": 1.0, "max": 1.0, "count": 5},
                "psi_1": {"min": 1.0, "max": 1.0, "count": 5},
                "theta_0": {"min": 1.0, "max": 1.0, "count": 5},
                "theta_1": {"min": 1.0, "max": 1.0, "count": 5},
            }
        )

        # Create a simple decision function that consumes a fraction of m
        def simple_policy(states, shocks, parameters):
            R = parameters["R"]
            a = states["a"]
            psi = shocks.get("psi", torch.ones_like(a))
            theta = shocks.get("theta", torch.ones_like(a))
            m = R * a / psi + theta
            # Consume 80% of cash-on-hand (likely constrained at low wealth)
            c = 0.8 * m
            return {"c": c}

        # Compute losses
        unconstrained_loss = loss_unconstrained(simple_policy, test_grid)
        constrained_loss = loss_constrained(simple_policy, test_grid)

        # Both should produce valid tensor outputs
        self.assertIsInstance(unconstrained_loss, torch.Tensor)
        self.assertIsInstance(constrained_loss, torch.Tensor)
        self.assertEqual(unconstrained_loss.shape, constrained_loss.shape)

        # Both losses should be finite and non-negative
        self.assertTrue(
            torch.isfinite(unconstrained_loss).all(),
            "Unconstrained loss should be finite",
        )
        self.assertTrue(
            torch.isfinite(constrained_loss).all(),
            "Constrained loss (Fischer-Burmeister) should be finite",
        )
        self.assertGreaterEqual(
            constrained_loss.mean().item(),
            0.0,
            "Constrained loss should be non-negative",
        )

        # Constrained (Fischer-Burmeister) and unconstrained (squared residual)
        # use different formulations, so they should produce different losses
        # for the same suboptimal policy
        self.assertFalse(
            torch.allclose(unconstrained_loss, constrained_loss),
            "Constrained and unconstrained losses should differ "
            "(Fischer-Burmeister vs squared residual)",
        )
        # Both should be strictly positive for a suboptimal policy
        self.assertGreater(
            unconstrained_loss.mean().item(),
            1e-8,
            "Unconstrained loss should be positive for suboptimal policy",
        )
        self.assertGreater(
            constrained_loss.mean().item(),
            1e-8,
            "Constrained loss should be positive for suboptimal policy",
        )


class TestConstrainedWarning(unittest.TestCase):
    """Test that EulerEquationLoss warns on misuse of constrained=True."""

    def test_constrained_warns_without_upper_bound(self):
        """constrained=True should warn when no Control has an upper_bound."""
        import logging

        # D-1 has a Control with upper_bound, but build a minimal block without one
        no_bound_block = model.DBlock(
            **{
                "name": "test_no_upper_bound",
                "shocks": {},
                "dynamics": {
                    "c": model.Control(["a"]),  # No upper_bound
                    "a": lambda a, c: a - c,
                    "u": lambda c: torch.log(c),
                },
                "reward": {"u": "consumer"},
            }
        )
        calibration = {"DiscFac": 0.96}
        bp = bellman.BellmanPeriod(no_bound_block, "DiscFac", calibration)

        with self.assertLogs("skagent.loss", level=logging.WARNING) as cm:
            loss.EulerEquationLoss(
                bp,
                parameters=calibration,
                constrained=True,
            )

        self.assertTrue(
            any("constrained=True but no Control" in msg for msg in cm.output),
            f"Expected warning about constrained=True, got: {cm.output}",
        )

    def test_constrained_no_warn_with_upper_bound(self):
        """constrained=True should NOT warn when a Control has upper_bound."""
        import logging

        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")
        bp = bellman.BellmanPeriod(u3_block, "DiscFac", u3_calibration)

        logger = logging.getLogger("skagent.loss")
        with self.assertNoLogs(logger, level=logging.WARNING):
            loss.EulerEquationLoss(
                bp,
                parameters=u3_calibration,
                constrained=True,
            )


class TestSimulateForwardValidation(unittest.TestCase):
    """Test input validation in simulate_forward."""

    def setUp(self):
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)
        self.bp = case_4["bp"]
        self.policy = lambda s, sh, p: {
            "c": s.get("m", s.get("a", next(iter(s.values())))) * 0.5
        }

    def test_negative_big_t_raises(self):
        states = {"m": torch.tensor([1.0, 2.0])}
        with self.assertRaises(ValueError, msg="big_t must be non-negative"):
            maliar.simulate_forward(states, self.bp, self.policy, {}, big_t=-1)

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError, msg="states_t cannot be an empty dict"):
            maliar.simulate_forward({}, self.bp, self.policy, {}, big_t=5)

    def test_big_t_zero_returns_unchanged(self):
        states = {"m": torch.tensor([1.0, 2.0])}
        result = maliar.simulate_forward(states, self.bp, self.policy, {}, big_t=0)
        self.assertIs(result, states)

    def test_big_t_zero_with_grid_returns_dict(self):
        states = grid.Grid.from_dict({"m": torch.tensor([1.0, 2.0])})
        result = maliar.simulate_forward(states, self.bp, self.policy, {}, big_t=0)
        self.assertIsInstance(result, dict)
        self.assertTrue(torch.allclose(result["m"], torch.tensor([1.0, 2.0])))


class TestSimulateForwardHappyPath(unittest.TestCase):
    """Test simulate_forward simulation loop for big_t >= 1."""

    def setUp(self):
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)
        self.bp = case_4["bp"]
        # case_4 Control(["g", "m"]) — policy returns c given g, m
        self.policy = lambda s, sh, p: {"c": s["m"] * 0.5 + s["g"] * 0.0}
        self.states = {
            "m": torch.tensor([1.0, 2.0, 3.0]),
            "g": torch.tensor([0.5, 0.5, 0.5]),
        }

    def test_big_t_one_returns_dict_with_same_keys(self):
        """simulate_forward with big_t=1 returns a dict with the same state keys."""
        result = maliar.simulate_forward(self.states, self.bp, self.policy, {}, big_t=1)
        self.assertIsInstance(result, dict)
        self.assertIn("m", result)
        self.assertIn("g", result)

    def test_big_t_one_output_shape_matches_input(self):
        """Output tensor shape should match input tensor shape after one step."""
        result = maliar.simulate_forward(self.states, self.bp, self.policy, {}, big_t=1)
        self.assertEqual(result["m"].shape, self.states["m"].shape)
        self.assertEqual(result["g"].shape, self.states["g"].shape)


class TestComputeSlack(unittest.TestCase):
    """Test EulerEquationLoss._compute_slack."""

    def setUp(self):
        self.block = model.DBlock(
            name="slack_test",
            dynamics={
                "m": lambda a, R: R * a,
                "c": model.Control(
                    iset=["m"],
                    upper_bound=lambda m: m,
                    agent="consumer",
                ),
                "a": lambda m, c: m - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        self.bp = bellman.BellmanPeriod(self.block, "beta", {"beta": 0.95, "R": 1.04})
        self.loss_fn = loss.EulerEquationLoss(
            self.bp,
            parameters={"beta": 0.95, "R": 1.04},
            constrained=True,
        )

    def test_slack_positive_when_not_binding(self):
        """Slack > 0 when control is below upper bound."""
        states_t = {"a": torch.tensor([2.0])}
        shocks_t = {}
        controls_t = {"c": torch.tensor([1.0])}  # c < m = R*a = 2.08
        slack = self.loss_fn._compute_slack("c", controls_t, states_t, shocks_t)
        self.assertIsNotNone(slack)
        self.assertTrue((slack > 0).all())

    def test_slack_zero_when_binding(self):
        """Slack ≈ 0 when control equals upper bound."""
        states_t = {"a": torch.tensor([2.0])}
        shocks_t = {}
        ub = 1.04 * 2.0  # m = R*a
        controls_t = {"c": torch.tensor([ub])}
        slack = self.loss_fn._compute_slack("c", controls_t, states_t, shocks_t)
        self.assertIsNotNone(slack)
        self.assertAlmostEqual(slack.item(), 0.0, places=4)

    def test_no_upper_bound_returns_none(self):
        """Control without upper_bound returns None."""
        block_no_ub = model.DBlock(
            name="no_ub",
            dynamics={
                "c": model.Control(iset=["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp_no_ub = bellman.BellmanPeriod(block_no_ub, "beta", {"beta": 0.95})
        loss_fn = loss.EulerEquationLoss(bp_no_ub, parameters={"beta": 0.95})
        slack = loss_fn._compute_slack(
            "c", {"c": torch.tensor([1.0])}, {"a": torch.tensor([2.0])}, {}
        )
        self.assertIsNone(slack)


class TestMultiControlConstrainedLoss(unittest.TestCase):
    """Test constrained Euler loss with multi-control model (S6)."""

    def test_multi_control_constrained_produces_finite_loss(self):
        """Fischer-Burmeister works on a multi-control model with upper bounds."""
        block = model.DBlock(
            name="multi_ctrl_constrained",
            dynamics={
                "c1": model.Control(["a"], upper_bound=lambda a: a),
                "c2": model.Control(["a"], upper_bound=lambda a: a),
                "a": lambda a, c1, c2: a - c1 - c2,
                "u": lambda c1, c2: torch.log(c1 + 1e-8) + torch.log(c2 + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})

        loss_fn = loss.EulerEquationLoss(bp, parameters={"beta": 0.9}, constrained=True)

        input_grid = grid.Grid.from_dict({"a": torch.linspace(1.0, 5.0, 5)})

        def df(states, shocks, params):
            a = states["a"]
            return {"c1": a * 0.2, "c2": a * 0.1}

        result = loss_fn(df, input_grid)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.all(torch.isfinite(result)))
        self.assertTrue(torch.all(result >= 0))


class TestEulerEquationLossWeightValidation(unittest.TestCase):
    """Test weight validation for EulerEquationLoss (S2)."""

    def test_zero_weight_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        with self.assertRaises(ValueError, msg="weight must be > 0"):
            loss.EulerEquationLoss(bp, parameters={"beta": 0.9}, weight=0.0)

    def test_negative_weight_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        with self.assertRaises(ValueError, msg="weight must be > 0"):
            loss.EulerEquationLoss(bp, parameters={"beta": 0.9}, weight=-1.0)


class TestBellmanEquationLossValidation(unittest.TestCase):
    """Test validation for BellmanEquationLoss (S2)."""

    def test_non_callable_value_network_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        with self.assertRaises(TypeError, msg="value_network must be callable"):
            loss.BellmanEquationLoss(bp, value_network="not_callable")

    def test_negative_foc_weight_raises(self):
        block = model.DBlock(
            name="test",
            dynamics={
                "c": model.Control(["a"]),
                "a": lambda a, c: a - c,
                "u": lambda c: torch.log(c + 1e-8),
            },
            reward={"u": "consumer"},
        )
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        with self.assertRaises(ValueError, msg="foc_weight must be >= 0"):
            loss.BellmanEquationLoss(
                bp, value_network=lambda s, sh, p: s["a"], foc_weight=-0.5
            )
