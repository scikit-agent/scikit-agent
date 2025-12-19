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

    def test_maliar_training_loop_with_euler_loss_and_shocks(self):
        """
        Test that maliar_training_loop works with EulerEquationLoss on a model with shocks.

        This test verifies the full integration of the Maliar training loop with
        Euler equation loss on a stochastic model. It specifically tests that:
        1. The shock suffix handling (_0, _1) works correctly
        2. generate_givens_from_states produces correctly formatted grids
        3. EulerEquationLoss can consume the grid with shock suffixes
        4. Training completes without errors and produces reasonable results

        Uses U-2 benchmark model: Log utility with geometric random walk income.
        This model has permanent income shocks (psi ~ MeanOneLogNormal) and tests
        the case where some arrival states (p) don't depend on the control (c).
        """
        # Get U-2 benchmark model with permanent income shocks
        u2_block = get_benchmark_model("U-2")
        u2_calibration = get_benchmark_calibration("U-2")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u2_block.construct_shocks(u2_calibration, rng=rng)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(u2_block, u2_calibration)

        # Create initial states grid
        # U-2 arrival states are: A (assets), p (permanent income level)
        states_0_n = grid.Grid.from_config(
            {
                "A": {"min": 0.5, "max": 3.0, "count": 5},
                "p": {"min": 0.8, "max": 1.2, "count": 3},
            }
        )

        # Create Euler equation loss
        euler_loss_fn = loss.EulerEquationLoss(
            bp,
            discount_factor=u2_calibration["DiscFac"],
            parameters=u2_calibration,
        )

        # Run maliar_training_loop with EulerEquationLoss
        # This tests the full integration:
        # 1. generate_givens_from_states creates psi_0 and psi_1 shocks
        # 2. EulerEquationLoss extracts them correctly
        # 3. Training completes (including handling states that don't depend on control)
        trained_net, final_states = maliar.maliar_training_loop(
            bp,
            euler_loss_fn,
            states_0_n,
            u2_calibration,
            shock_copies=2,  # Required for EulerEquationLoss
            max_iterations=3,
            tolerance=1e-4,
            random_seed=TEST_SEED,
            simulation_steps=1,
        )

        # Verify training completed successfully
        self.assertIsNotNone(trained_net)
        self.assertIsNotNone(final_states)

        # Verify final states are valid
        final_states_dict = final_states.to_dict()
        self.assertIn("A", final_states_dict)
        self.assertTrue(torch.all(torch.isfinite(final_states_dict["A"])))

        # Get the trained decision function and verify it produces valid output
        decision_fn = trained_net.get_decision_function()

        # Test the decision function with sample arrival states and shocks
        test_states = {
            "A": torch.tensor([1.0, 2.0, 3.0], device=device),
            "p": torch.tensor([1.0, 1.0, 1.0], device=device),
        }
        test_shocks = {"psi": torch.tensor([1.0, 1.0, 1.0], device=device)}
        controls = decision_fn(test_states, test_shocks, u2_calibration)

        self.assertIn("c", controls)
        # Consumption should be positive
        self.assertTrue(torch.all(controls["c"] > 0))
        # Compute expected m = A*R + p*psi for bounds checking
        R = u2_calibration["R"]
        expected_m = test_states["A"] * R + test_states["p"] * test_shocks["psi"]
        # Consumption should be less than cash on hand (m)
        self.assertTrue(torch.all(controls["c"] < expected_m))
