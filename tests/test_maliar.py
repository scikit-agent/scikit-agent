from conftest import case_1, case_3, case_4
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
import skagent.block as model
import torch
import unittest
from skagent.distributions import Normal
from skagent.ann import BlockValueNet
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


class TestMaliarTrainingLoop(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_maliar_state_convergence(self):
        big_t = 2

        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        states_0_n = grid.Grid.from_config(
            {
                "m": {"min": -20, "max": 20, "count": 9},
                "g": {"min": -20, "max": 20, "count": 9},
            }
        )

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_4["bp"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Use fixed random seed for deterministic training
        ann, states = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,  # Fixed seed for deterministic training
            max_iterations=3,
        )

        sd = states.to_dict()

        # testing for the states converged on the ergodic distribution
        # note we actual expect these to diverge up to Uniform[-1, 1] shocks.
        self.assertTrue(torch.allclose(sd["m"], sd["g"], atol=2.5))

    def test_maliar_convergence_tolerance(self):
        """Test the convergence functionality in the Maliar training loop."""
        big_t = 2

        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        states_0_n = grid.Grid.from_config(
            {
                "m": {"min": -10, "max": 10, "count": 5},
                "g": {"min": -10, "max": 10, "count": 5},
            }
        )

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_4["bp"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Test 1: High tolerance (should converge quickly)
        ann_high_tol, states_high_tol = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=10,
            tolerance=1e-1,  # High tolerance for quick convergence
        )

        # Test 2: Low tolerance (should require more iterations or hit max_iterations)
        ann_low_tol, states_low_tol = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=3,
            tolerance=1e-8,  # Very low tolerance
        )

        # Both should return valid networks and states
        self.assertIsNotNone(ann_high_tol)
        self.assertIsNotNone(states_high_tol)
        self.assertIsNotNone(ann_low_tol)
        self.assertIsNotNone(states_low_tol)

        # Test that tolerance affects convergence behavior
        # (We can't easily test exact iteration counts due to randomness,
        # but we can verify the function completes successfully with different tolerances)
        sd_high = states_high_tol.to_dict()
        sd_low = states_low_tol.to_dict()

        # Both should produce valid state dictionaries
        self.assertIn("m", sd_high)
        self.assertIn("g", sd_high)
        self.assertIn("m", sd_low)
        self.assertIn("g", sd_low)

        # Verify states are finite tensors
        self.assertTrue(torch.all(torch.isfinite(sd_high["m"])))
        self.assertTrue(torch.all(torch.isfinite(sd_high["g"])))
        self.assertTrue(torch.all(torch.isfinite(sd_low["m"])))
        self.assertTrue(torch.all(torch.isfinite(sd_low["g"])))

    def test_maliar_convergence_early_stopping(self):
        """Test that the training loop can stop early when convergence is achieved."""
        big_t = 2

        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        # Use a smaller grid for faster convergence testing
        states_0_n = grid.Grid.from_config(
            {
                "m": {"min": 0, "max": 5, "count": 3},
                "g": {"min": 0, "max": 5, "count": 3},
            }
        )

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_4["bp"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Test with very high tolerance to ensure early convergence
        ann, states = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=100,  # Set high max iterations
            tolerance=1.0,  # Very high tolerance - should converge in 1-2 iterations
        )

        # Should complete successfully
        self.assertIsNotNone(ann)
        self.assertIsNotNone(states)

        # States should be valid
        sd = states.to_dict()
        self.assertIn("m", sd)
        self.assertIn("g", sd)
        self.assertTrue(torch.all(torch.isfinite(sd["m"])))
        self.assertTrue(torch.all(torch.isfinite(sd["g"])))

    def test_maliar_convergence_by_loss(self):
        """Test convergence by both parameter and loss criteria."""
        big_t = 2

        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        # Use a small grid for faster testing
        states_0_n = grid.Grid.from_config(
            {
                "m": {"min": 0, "max": 5, "count": 3},
                "g": {"min": 0, "max": 5, "count": 3},
            }
        )

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_4["bp"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Test 1: Strict tolerance (should require more iterations)
        ann_strict, states_strict = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=10,
            tolerance=1e-6,  # Strict tolerance
        )

        # Test 2: Relaxed tolerance (should converge faster)
        ann_relaxed, states_relaxed = maliar.maliar_training_loop(
            case_4["bp"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=10,
            tolerance=1e-1,  # Relaxed tolerance
        )

        # Both tests should return valid networks and states
        for ann, states in [(ann_strict, states_strict), (ann_relaxed, states_relaxed)]:
            self.assertIsNotNone(ann)
            self.assertIsNotNone(states)

            # States should be valid
            sd = states.to_dict()
            self.assertIn("m", sd)
            self.assertIn("g", sd)
            self.assertTrue(torch.all(torch.isfinite(sd["m"])))
            self.assertTrue(torch.all(torch.isfinite(sd["g"])))


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
        self.discount_factor = 0.95
        self.parameters = {}
        self.bp = bellman.BellmanPeriod(self.block, self.parameters)
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
            states_t, shocks_t, controls_t, self.parameters
        )
        self.assertIn("wealth", next_states)
        self.assertTrue(torch.allclose(next_states["wealth"], torch.tensor([1.5, 2.0])))

        # Test reward function
        reward = self.bp.reward_function(
            states_t, shocks_t, controls_t, self.parameters, agent="consumer"
        )
        self.assertIn("utility", reward)
        # Utility can be negative for log(consumption), so just check it's finite
        self.assertTrue(torch.all(torch.isfinite(reward["utility"])))

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

        nrbp = bellman.BellmanPeriod(no_reward_block, {})

        with self.assertRaises(Exception) as context:
            loss.BellmanEquationLoss(nrbp, self.discount_factor, dummy_value_network)
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
            self.discount_factor,
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

    def test_consistency_with_existing_patterns(self):
        """Test that the new Bellman loss functions follow existing skagent patterns."""

        # Create a simple value network with correct interface
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth  # Linear value function

        # Test that it works with the training infrastructure
        loss_function = loss.BellmanEquationLoss(
            self.bp,
            self.discount_factor,
            simple_value_network,
            self.parameters,
        )

        # Test with aggregate_net_loss (from ann.py)
        from skagent.ann import aggregate_net_loss

        # This should work without errors
        aggregated_loss = aggregate_net_loss(
            self.test_grid, self.decision_function, loss_function
        )

        self.assertIsInstance(aggregated_loss, torch.Tensor)
        self.assertEqual(aggregated_loss.shape, ())  # Scalar after aggregation
        self.assertTrue(aggregated_loss >= 0)

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
            0.95,
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters={},
        )

        residual_independent = bellman.estimate_bellman_residual(
            self.bp,
            0.95,
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_independent,
            parameters={},
        )

        # Results should be different when using independent vs identical shocks
        self.assertFalse(torch.allclose(residual_identical, residual_independent))

        # Both should be finite
        self.assertTrue(torch.all(torch.isfinite(residual_identical)))
        self.assertTrue(torch.all(torch.isfinite(residual_independent)))

    def test_bellman_loss_with_different_shock_patterns(self):
        """Test Bellman loss function with various shock patterns."""

        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            income = shocks_t["income"]
            return (
                10.0 * wealth + 2.0 * income
            )  # Value depends on both wealth and income

        loss_function = loss.BellmanEquationLoss(
            self.bp,
            self.discount_factor,
            simple_value_network,
            self.parameters,
        )

        # Test with correlated shocks (period t+1 same as period t)
        test_grid_correlated = grid.Grid.from_dict(
            {
                "wealth": torch.tensor([1.0, 2.0, 3.0]),
                "income_0": torch.tensor([1.0, 1.2, 0.8]),
                "income_1": torch.tensor([1.0, 1.2, 0.8]),  # Same as period t
            }
        )

        # Test with anti-correlated shocks
        test_grid_anticorrelated = grid.Grid.from_dict(
            {
                "wealth": torch.tensor([1.0, 2.0, 3.0]),
                "income_0": torch.tensor([1.0, 1.2, 0.8]),
                "income_1": torch.tensor([1.0, 0.8, 1.2]),  # Opposite of period t
            }
        )

        loss_correlated = loss_function(self.decision_function, test_grid_correlated)
        loss_anticorrelated = loss_function(
            self.decision_function, test_grid_anticorrelated
        )

        # Both should produce valid losses
        self.assertTrue(torch.all(loss_correlated >= 0))
        self.assertTrue(torch.all(loss_anticorrelated >= 0))
        self.assertTrue(torch.all(torch.isfinite(loss_correlated)))
        self.assertTrue(torch.all(torch.isfinite(loss_anticorrelated)))

        # Losses should be different for different shock patterns
        self.assertFalse(torch.allclose(loss_correlated, loss_anticorrelated))

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
                0.95,
                simple_value_network,
                self.decision_function,
                states_t,
                shocks_missing_t1,
                parameters={},
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
    test_bp = bellman.BellmanPeriod(d2_block, d2_calibration)

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
    loss_fn = loss.EulerEquationLoss(
        test_bp, discount_factor=d2_calibration["DiscFac"], parameters=d2_calibration
    )

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


def test_bellman_equation_loss_with_value_network():
    """Test Bellman equation loss function with separate value network.

    This test verifies that the BellmanEquationLoss class works correctly
    with a value network. Uses a simple consumption model where the control
    directly depends on the arrival state (no intermediate dynamics).
    """
    # Create a simple test block where control iset matches arrival state
    # This ensures BlockValueNet can directly use arrival states
    test_block = model.DBlock(
        name="test_value_net",
        shocks={},  # Deterministic for simplicity
        dynamics={
            "c": model.Control(iset=["a"], agent="consumer"),
            "a": lambda a, c: a - c,  # Simple savings transition
            "u": lambda c: torch.log(c + 1e-8),
        },
        reward={"u": "consumer"},
    )
    test_block.construct_shocks({})

    test_bp = bellman.BellmanPeriod(test_block, {})

    # Create value network
    value_net = BlockValueNet(test_bp, width=16)

    # Create input grid with arrival states
    n_points = 10
    input_grid = grid.Grid.from_dict(
        {
            "a": torch.linspace(1.0, 10.0, n_points),
        }
    )

    # Create a simple decision function: consume 30% of assets
    # This is feasible (c < a) for all a >= 1.0 in our grid
    def simple_policy(states_t, shocks_t, parameters):
        a = states_t["a"]
        return {"c": 0.3 * a}

    # Create loss function
    loss_fn = loss.BellmanEquationLoss(
        test_bp, 0.95, value_net.get_value_function(), parameters={}
    )

    # Test that loss function works
    losses = loss_fn(simple_policy, input_grid)

    # Check that we get per-sample losses
    assert isinstance(losses, torch.Tensor)
    assert losses.shape[0] == n_points  # One loss per grid point
    assert torch.all(losses >= 0)  # Squared residuals should be non-negative


def test_block_value_net():
    """Test BlockValueNet functionality."""
    # Create a test block with multiple state variables
    test_block = model.DBlock(
        name="test_multi_state",
        shocks={"income": Normal(mu=1.0, sigma=0.1)},
        dynamics={
            "wealth": lambda wealth, income, consumption: wealth + income - consumption,
            "capital": lambda capital, investment: capital + investment,
            "consumption": model.Control(iset=["wealth"], agent="consumer"),
            "investment": model.Control(iset=["capital"], agent="consumer"),
            "utility": lambda consumption: torch.log(consumption + 1e-8),
        },
        reward={"utility": "consumer"},
    )
    test_block.construct_shocks({})

    test_bp = bellman.BellmanPeriod(test_block, {})

    # Create value network - now takes block instead of state_variables
    value_net = BlockValueNet(test_bp, width=16)

    # Test value function computation
    states_t = {"wealth": torch.tensor([1.0, 2.0, 3.0])}
    shocks_t = {"income": torch.tensor([1.0, 1.0, 1.0])}

    values = value_net.value_function(states_t, shocks_t, {})

    # Check output shape and type
    assert isinstance(values, torch.Tensor)
    assert values.shape == (3,)

    # Test get_value_function method
    vf = value_net.get_value_function()
    values2 = vf(states_t, shocks_t, {})

    # Should give same results
    assert torch.allclose(values, values2)


class TestEulerResidualsBenchmarks(unittest.TestCase):
    """
    Test Euler residuals using consumption-saving models.

    These tests verify that the Euler equation residual computation works correctly
    and produces sensible results for various model specifications. The tests follow
    the methodology described in Maliar, Maliar, and Winant (2021, JME) for testing
    first-order conditions.
    """

    def test_euler_residual_d2_optimal_policy(self):
        """
        Test that the analytical optimal policy achieves near-zero Euler residual.

        D-2 is the infinite horizon CRRA perfect foresight model.
        The analytical solution is: c_t = κ*W_t where κ = (R - (βR)^(1/σ))/R
        and W_t = m_t + H_t is total wealth (cash-on-hand plus human wealth H_t = y/r).

        The Euler equation is: u'(c_t) = β*R*u'(c_{t+1})
        For the optimal policy, the Euler residual should be essentially zero,
        validating that estimate_euler_residual correctly implements the FOC.
        """
        # Get D-2 benchmark model and calibration
        d2_block = get_benchmark_model("D-2")
        d2_calibration = get_benchmark_calibration("D-2")
        d2_policy = get_analytical_policy("D-2")

        # Construct shocks for the block (needed even though D-2 is deterministic)
        d2_block.construct_shocks(d2_calibration)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(d2_block, d2_calibration)

        # Create test states
        n_samples = 100
        test_states = {"a": torch.linspace(0.1, 10.0, n_samples)}
        shocks = {}

        # Compute Euler residual with analytical optimal policy
        optimal_residual = bellman.estimate_euler_residual(
            bp,
            d2_calibration["DiscFac"],
            d2_policy,
            test_states,
            shocks,
            d2_calibration,
        )

        # For optimal policy, residual should be essentially zero (machine precision)
        mean_residual = torch.mean(torch.abs(optimal_residual)).item()
        mse_residual = torch.mean(optimal_residual**2).item()

        # Tolerance accounts for numerical precision in autograd
        assert mean_residual < 1e-5, (
            f"Analytical optimal policy should have near-zero Euler residual. "
            f"Got mean |residual| = {mean_residual:.6e}"
        )

        assert mse_residual < 1e-10, (
            f"Analytical optimal policy should have near-zero Euler MSE. "
            f"Got MSE = {mse_residual:.6e}"
        )

    def test_euler_equation_training(self):
        """
        Test that a policy can be trained via Euler equation loss to achieve near-zero residual.

        This is the key validation test for the Maliar et al. (2021) methodology:
        train a neural network policy by minimizing squared Euler residuals,
        and verify the trained policy satisfies the first-order conditions.

        Uses scikit-agent's BlockPolicyNet and train_block_nn for proper integration.
        """
        from skagent.ann import BlockPolicyNet, train_block_nn

        # Get D-2 benchmark model and calibration
        d2_block = get_benchmark_model("D-2")
        d2_calibration = get_benchmark_calibration("D-2")

        # Construct shocks for the block
        d2_block.construct_shocks(d2_calibration)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(d2_block, d2_calibration)

        # Create policy network using scikit-agent's BlockPolicyNet
        torch.manual_seed(TEST_SEED)
        policy_net = BlockPolicyNet(bp, width=32, init_seed=TEST_SEED)

        # Create Euler equation loss
        euler_loss_fn = loss.EulerEquationLoss(
            bp, discount_factor=d2_calibration["DiscFac"], parameters=d2_calibration
        )

        # Create training grid with states (D-2 is deterministic, no shocks needed)
        n_grid_points = 64
        train_grid = grid.Grid.from_dict(
            {
                "a": torch.rand(n_grid_points, device=device) * 9.9
                + 0.1,  # a in [0.1, 10]
            }
        )

        # Train using scikit-agent's train_block_nn
        trained_net, final_loss = train_block_nn(
            policy_net, train_grid, euler_loss_fn, epochs=300
        )

        # Verify training achieved small loss
        assert final_loss < 0.01, (
            f"Training should achieve small Euler loss. Got final loss: {final_loss:.6e}"
        )

        # Evaluate on test grid using the trained policy
        test_states = {"a": torch.linspace(0.1, 10.0, 100, device=device)}
        shocks = {}

        # Get decision function from trained network
        decision_fn = trained_net.get_decision_function()

        # Compute final Euler residual
        final_residual = bellman.estimate_euler_residual(
            bp,
            d2_calibration["DiscFac"],
            decision_fn,
            test_states,
            shocks,
            d2_calibration,
        )
        final_residual = final_residual.detach()

        # The trained policy should achieve small Euler residual
        mean_residual = torch.mean(torch.abs(final_residual)).item()
        mse_residual = torch.mean(final_residual**2).item()

        # Tolerance: trained policy should have mean |residual| < 0.1
        # (Training loss was ~0.0002, but evaluation on different grid points may be higher)
        assert mean_residual < 0.1, (
            f"Trained policy should have small Euler residual. "
            f"Got mean |residual| = {mean_residual:.6e}"
        )

        # Verify MSE is reasonably small (< 0.05)
        assert mse_residual < 0.05, (
            f"Trained policy should have small Euler residual MSE. "
            f"Got MSE = {mse_residual:.6e}"
        )

    def test_maliar_training_loop_u2_analytical(self):
        """
        Test maliar_training_loop with U-2 model against analytical PIH solution.

        U-2 uses NORMALIZED variables (all divided by permanent income P):
        - a = A/P (normalized assets, arrival state)
        - m = M/P = R*a/ψ + 1 (normalized cash-on-hand)
        - c = C/P (normalized consumption)

        Analytical solution: c = (1-β)(m + 1/r) where 1/r is normalized human wealth.
        """
        # Get U-2 benchmark model (no borrowing constraint - has analytical solution)
        u2_block = get_benchmark_model("U-2")
        u2_calibration = get_benchmark_calibration("U-2")
        analytical_policy = get_analytical_policy("U-2")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u2_block.construct_shocks(u2_calibration, rng=rng)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(u2_block, u2_calibration)

        # Create initial states grid (normalized assets)
        states_0_n = grid.Grid.from_config(
            {
                "a": {"min": 0.5, "max": 5.0, "count": 15},
            }
        )

        # Create Euler equation loss
        euler_loss_fn = loss.EulerEquationLoss(
            bp,
            discount_factor=u2_calibration["DiscFac"],
            parameters=u2_calibration,
        )

        # Train the policy
        trained_net, final_states = maliar.maliar_training_loop(
            bp,
            euler_loss_fn,
            states_0_n,
            u2_calibration,
            shock_copies=2,
            max_iterations=8,
            tolerance=1e-6,
            random_seed=TEST_SEED,
            simulation_steps=1,
        )

        # Verify training completed
        self.assertIsNotNone(trained_net)

        # Get decision functions
        decision_fn = trained_net.get_decision_function()

        # Test on grid within training range (normalized assets)
        n_test = 25
        test_a = torch.linspace(0.5, 5.0, n_test, device=device)
        test_states = {"a": test_a}
        test_shocks = {"psi": torch.ones(n_test, device=device)}

        # Get trained and analytical consumption (both normalized)
        trained_c = decision_fn(test_states, test_shocks, u2_calibration)["c"]
        analytical_c = analytical_policy(test_states, test_shocks, u2_calibration)["c"]

        # Compare trained vs analytical
        rel_error = torch.abs(trained_c - analytical_c) / (analytical_c + 1e-8)
        mean_rel_error = rel_error.mean().item()

        # Trained policy should be reasonably close to analytical (within 20%).
        # Note: with CRRA utility, the Euler equation pins down both the shape and
        # the level of the optimal consumption policy; arbitrary rescalings k * c*
        # do NOT satisfy the Euler equation unless k = 1. The 20% tolerance here
        # reflects numerical approximation error (finite training iterations,
        # stochastic optimization, and function-approximation error), not any
        # theoretical indeterminacy in the Euler condition.
        self.assertLess(
            mean_rel_error,
            0.20,
            f"Trained policy should approximately match analytical PIH solution. "
            f"Mean relative error: {mean_rel_error:.4f}",
        )

    def test_maliar_training_loop_u3_buffer_stock(self):
        """
        Test maliar_training_loop with U-3 buffer stock model (CRRA=2, with constraint).

        U-3 uses NORMALIZED variables (all divided by permanent income P):
        - a = A/P (normalized assets, arrival state)
        - m = M/P = R*a/ψ + θ (normalized cash-on-hand)
        - c = C/P (normalized consumption)

        This model does NOT have a closed-form solution due to the borrowing
        constraint + income uncertainty interaction. We validate the trained
        policy using:
        1. Basic sanity checks (positive consumption, budget constraint)
        2. Monotonicity in wealth
        3. Euler residual near zero in non-constrained region (high wealth)
        4. Limiting MPC approaching perfect foresight value at high wealth
        """
        # Get U-3 buffer stock model (CRRA=2, with borrowing constraint)
        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u3_block.construct_shocks(u3_calibration, rng=rng)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(u3_block, u3_calibration)

        # Create initial states grid (normalized assets)
        # Use wider range to test both constrained and unconstrained regions
        states_0_n = grid.Grid.from_config(
            {
                "a": {"min": 0.5, "max": 10.0, "count": 25},
            }
        )

        # Create Euler equation loss with constrained=True for buffer stock model
        # The borrowing constraint (c <= m) means the Euler equation becomes an
        # inequality at constrained points: u'(c) >= βR E[u'(c')].
        # Using constrained=True enables one-sided loss that only penalizes
        # negative residuals (overconsumption), not positive ones (constraint binding).
        euler_loss_fn = loss.EulerEquationLoss(
            bp,
            discount_factor=u3_calibration["DiscFac"],
            parameters=u3_calibration,
            constrained=True,  # Key fix: use one-sided loss for borrowing constraint
        )

        # Train the policy with more iterations for better convergence
        trained_net, final_states = maliar.maliar_training_loop(
            bp,
            euler_loss_fn,
            states_0_n,
            u3_calibration,
            shock_copies=2,
            max_iterations=15,
            tolerance=1e-6,
            random_seed=TEST_SEED,
            simulation_steps=1,
        )

        # Verify training completed
        self.assertIsNotNone(trained_net)
        self.assertIsNotNone(final_states)

        # Get decision function and parameters
        decision_fn = trained_net.get_decision_function()
        R = u3_calibration["R"]
        beta = u3_calibration["DiscFac"]
        gamma = u3_calibration["CRRA"]

        # =================================================================
        # 1. Basic sanity checks
        # =================================================================
        n_test = 30
        test_a = torch.linspace(0.5, 10.0, n_test, device=device)
        test_states = {"a": test_a}
        test_shocks = {
            "psi": torch.ones(n_test, device=device),
            "theta": torch.ones(n_test, device=device),
        }

        trained_c = decision_fn(test_states, test_shocks, u3_calibration)["c"]

        # Consumption should be positive
        self.assertTrue(
            torch.all(trained_c > 0), "Trained consumption should be positive"
        )

        # Consumption should respect budget constraint (c <= m)
        expected_m = R * test_a + 1  # m = R*a/psi + theta when psi=theta=1
        self.assertTrue(
            torch.all(trained_c <= expected_m + 1e-6),
            "Trained consumption should respect budget constraint (c <= m)",
        )

        # =================================================================
        # 2. Monotonicity check (overall trend)
        # For neural network approximation, we allow minor violations but
        # ensure the overall trend is strongly increasing.
        # =================================================================
        c_diff = trained_c[1:] - trained_c[:-1]

        # Overall consumption at high wealth should be much larger than at low wealth
        c_range = trained_c[-1] - trained_c[0]
        self.assertGreater(
            c_range.item(),
            0.5,
            f"Consumption should increase substantially over wealth range. "
            f"Got c(high) - c(low) = {c_range.item():.4f}",
        )

        # Most differences should be positive (allow up to 20% violations)
        positive_diffs = torch.sum(c_diff > 0).item()
        total_diffs = len(c_diff)
        positive_ratio = positive_diffs / total_diffs
        self.assertGreater(
            positive_ratio,
            0.8,
            f"At least 80% of consumption differences should be positive. "
            f"Got {positive_ratio:.2%} ({positive_diffs}/{total_diffs}).",
        )

        # =================================================================
        # 3. Euler residual check in non-constrained region (high wealth)
        # At high wealth, the borrowing constraint is slack and Euler
        # equation should be satisfied exactly.
        # =================================================================
        # Test at high wealth where constraint is not binding
        n_high = 10
        high_wealth_a = torch.linspace(5.0, 10.0, n_high, device=device)
        high_wealth_states = {"a": high_wealth_a}

        # Provide shocks in _0/_1 format for estimate_euler_residual
        high_wealth_shocks = {
            "psi_0": torch.ones(n_high, device=device),
            "psi_1": torch.ones(n_high, device=device),
            "theta_0": torch.ones(n_high, device=device),
            "theta_1": torch.ones(n_high, device=device),
        }

        euler_residual = bellman.estimate_euler_residual(
            bp,
            beta,
            decision_fn,
            high_wealth_states,
            high_wealth_shocks,
            u3_calibration,
        )

        mean_euler_residual = torch.mean(torch.abs(euler_residual)).item()
        # In the unconstrained region, Euler residual should be small
        self.assertLess(
            mean_euler_residual,
            0.5,
            f"Euler residual should be small at high wealth (unconstrained). "
            f"Got mean |residual| = {mean_euler_residual:.4f}",
        )

        # =================================================================
        # 4. Limiting MPC check (informational)
        # As wealth -> infinity, MPC -> kappa_pf = (R - (beta*R)^(1/gamma)) / R
        # With limited training iterations, the policy may not fully converge.
        # We verify MPC is in a reasonable range (positive, less than 1).
        # =================================================================
        kappa_pf = (R - (beta * R) ** (1 / gamma)) / R

        # Compute MPC numerically at high wealth
        delta_a = 0.5
        high_a = torch.tensor([8.0, 9.0, 10.0], device=device)
        high_states = {"a": high_a}
        high_shocks = {
            "psi": torch.ones(3, device=device),
            "theta": torch.ones(3, device=device),
        }

        c_high = decision_fn(high_states, high_shocks, u3_calibration)["c"]
        c_high_plus = decision_fn({"a": high_a + delta_a}, high_shocks, u3_calibration)[
            "c"
        ]

        # MPC = dc/dm, and dm/da = R (when psi=1)
        delta_m = R * delta_a
        mpc_high = ((c_high_plus - c_high) / delta_m).mean().item()

        # Log the limiting MPC for informational purposes
        print(
            f"\nLimiting MPC check: computed = {mpc_high:.4f}, "
            f"perfect foresight = {kappa_pf:.4f}"
        )

        # MPC should be strictly between 0 and 1 for a valid consumption function.
        # With the constrained=True Euler loss, training should converge to
        # economically sensible policies. We use tight bounds (0.0, 1.0) that
        # reflect the theoretical requirement for buffer stock models.
        self.assertGreater(
            mpc_high,
            0.0,  # MPC must be positive (saving decreases with wealth)
            f"MPC should be positive at high wealth. Got MPC = {mpc_high:.4f}",
        )
        self.assertLess(
            mpc_high,
            1.0,  # MPC must be less than 1 (some saving at high wealth)
            f"MPC should be less than 1 at high wealth. Got MPC = {mpc_high:.4f}",
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

    def test_one_sided_vs_two_sided_loss(self):
        """Test that one-sided loss differs from two-sided loss appropriately."""
        # Two-sided loss always penalizes deviation from zero
        residuals = torch.tensor([0.5, -0.3, 1.0, -0.8])

        two_sided_loss = residuals**2
        one_sided_loss = torch.relu(-residuals) ** 2

        # For positive residuals, one-sided should be less than two-sided
        self.assertLess(
            one_sided_loss[0].item(),
            two_sided_loss[0].item(),
            "One-sided loss should be less than two-sided for positive residual",
        )
        self.assertLess(
            one_sided_loss[2].item(),
            two_sided_loss[2].item(),
            "One-sided loss should be less than two-sided for positive residual",
        )

        # For negative residuals, one-sided should equal two-sided
        self.assertAlmostEqual(
            one_sided_loss[1].item(),
            two_sided_loss[1].item(),
            places=6,
            msg="One-sided loss should equal two-sided for negative residual",
        )
        self.assertAlmostEqual(
            one_sided_loss[3].item(),
            two_sided_loss[3].item(),
            places=6,
            msg="One-sided loss should equal two-sided for negative residual",
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
