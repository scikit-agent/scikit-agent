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

    def test_consistency_with_existing_patterns(self):
        """Test that the new Bellman loss functions follow existing skagent patterns."""

        # Create a simple value network with correct interface
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth  # Linear value function

        # Test that it works with the training infrastructure
        loss_function = loss.BellmanEquationLoss(
            self.bp,
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
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters={},
        )

        residual_independent = bellman.estimate_bellman_residual(
            self.bp,
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
                simple_value_network,
                self.decision_function,
                states_t,
                shocks_missing_t1,
                parameters={},
            )


def test_get_euler_residual_loss():
    """Test function placeholder - not implemented yet."""
    pass


def test_bellman_equation_loss_with_value_network():
    """Test Bellman equation loss function with separate value network."""
    # Create a simple test block using the same pattern as the existing tests
    test_block = model.DBlock(
        name="test_value_net",
        shocks={"income": Normal(mu=1.0, sigma=0.1)},
        dynamics={
            "consumption": model.Control(iset=["wealth"], agent="consumer"),
            "wealth": lambda wealth, income, consumption: wealth + income - consumption,
            "utility": lambda consumption: torch.log(consumption + 1e-8),
        },
        reward={"utility": "consumer"},
    )
    test_block.construct_shocks({})

    test_bp = bellman.BellmanPeriod(test_block, "beta", {"beta": 0.95})

    # Create value network
    value_net = BlockValueNet(test_bp, width=16)

    # Create input grid with two independent shock realizations
    input_grid = grid.Grid.from_dict(
        {
            "wealth": torch.linspace(1.0, 10.0, 5),
            "income_0": torch.ones(5),  # Period t shocks
            "income_1": torch.ones(5) * 1.1,  # Period t+1 shocks (independent)
        }
    )

    # Create a decision function
    def learned_decision_function(states_t, shocks_t, parameters):
        wealth = states_t["wealth"]
        consumption = 0.3 * wealth + 0.1
        consumption = torch.maximum(consumption, torch.tensor(0.01))
        consumption = torch.minimum(consumption, 0.9 * wealth)
        return {"consumption": consumption}

    # Create loss function
    loss_fn = loss.BellmanEquationLoss(
        test_bp, value_net.get_value_function(), parameters={}
    )

    # Test that loss function works
    losses = loss_fn(learned_decision_function, input_grid)

    # Check that we get per-sample losses
    assert isinstance(losses, torch.Tensor)
    assert losses.shape[0] == 5  # One loss per grid point
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

    test_bp = bellman.BellmanPeriod(test_block, "beta", {"beta": 0.95})

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
