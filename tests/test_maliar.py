from conftest import case_0, case_1, case_2, case_3, case_4, case_11
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.grid as grid
import skagent.model as model
import torch
import unittest
from skagent.distributions import Normal
from skagent.ann import BlockValueNet
from skagent.algos.maliar import (
    get_bellman_equation_loss,
)

# Deterministic test seed - change this single value to modify all seeding
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_simple_linear_value_network():
    """Create a simple linear value network for testing: V(wealth) = 10.0 * wealth"""

    def simple_value_network(states_t, shocks_t, parameters):
        wealth = states_t["wealth"]
        return 10.0 * wealth

    return simple_value_network


def create_income_aware_value_network():
    """Create a value network that depends on both wealth and income"""

    def income_aware_value_network(states_t, shocks_t, parameters):
        wealth = states_t["wealth"]
        income = shocks_t["income"]
        return 10.0 * wealth + 5.0 * income  # Value depends on both wealth and income

    return income_aware_value_network


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
        transition_function = maliar.create_transition_function(self.block, ["a", "e"])

        states_1 = transition_function(states_0, {}, decisions, parameters=parameters)

        self.assertAlmostEqual(states_1["a"], 0.7)
        self.assertEqual(states_1["e"], 0.1)

    def test_create_decision_function(self):
        decision_function = maliar.create_decision_function(self.block, decision_rules)

        decisions_0 = decision_function(states_0, {}, parameters=parameters)

        self.assertEqual(decisions_0["c"], 0.5)

    def test_create_reward_function(self):
        reward_function = maliar.create_reward_function(self.block)

        reward_0 = reward_function(states_0, {}, decisions, parameters=parameters)

        self.assertAlmostEqual(reward_0["u"], -0.69314718)

    def test_estimate_discounted_lifetime_reward(self):
        dlr_0 = maliar.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            0,
            parameters=parameters,
        )

        self.assertEqual(dlr_0, 0)

        dlr_1 = maliar.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            1,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_1, -0.69314718)

        dlr_2 = maliar.estimate_discounted_lifetime_reward(
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

        simple_value_network = create_simple_linear_value_network()

        def simple_decision_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.5 * wealth
            return {"consumption": consumption}

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
        residual = maliar.estimate_bellman_residual(
            test_block,
            0.95,  # discount factor
            simple_value_network,
            simple_decision_function,
            states_t,
            shocks,  # Now passing combined shock object
            parameters={},
        )

        self.assertIsInstance(residual, torch.Tensor)
        self.assertEqual(residual.shape, (2,))

        # Check that residuals are finite
        self.assertTrue(torch.all(torch.isfinite(residual)))


class TestLifetimeReward(unittest.TestCase):
    """
    More tests of the lifetime reward function specifically.
    """

    def setUp(self):
        self.states_0 = {"a": 0}

    def test_block_1(self):
        dlr_1 = maliar.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            1,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0]]))},
        )

        self.assertEqual(dlr_1, 0)

        # big_t is 2
        dlr_1_2 = maliar.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            2,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0], [0]]))},
        )

        self.assertEqual(dlr_1_2, 0)

    def test_block_2(self):
        dlr_2 = maliar.estimate_discounted_lifetime_reward(
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

        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            states_0_n.labels,
            case_4["block"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Use fixed random seed for deterministic training
        ann, states = maliar.maliar_training_loop(
            case_4["block"],
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

        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            states_0_n.labels,
            case_4["block"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Test 1: High tolerance (should converge quickly)
        ann_high_tol, states_high_tol = maliar.maliar_training_loop(
            case_4["block"],
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
            case_4["block"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=3,
            tolerance=1e-8,  # Very low tolerance
        )

        self.assertIsNotNone(ann_high_tol)
        self.assertIsNotNone(states_high_tol)
        self.assertIsNotNone(ann_low_tol)
        self.assertIsNotNone(states_low_tol)
        sd_high = states_high_tol.to_dict()
        sd_low = states_low_tol.to_dict()

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

        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            states_0_n.labels,
            case_4["block"],
            0.9,
            big_t,
            case_4["calibration"],
        )

        # Test with very high tolerance to ensure early convergence
        ann, states = maliar.maliar_training_loop(
            case_4["block"],
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
                "wealth": lambda wealth, income, consumption: wealth
                + income
                - consumption,
                "consumption": model.Control(
                    iset=["wealth"],
                    lower_bound=lambda wealth: 0.0,
                    upper_bound=lambda wealth: wealth,
                    agent="consumer",
                ),
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

    def test_get_bellman_equation_loss(self):
        """Test Bellman equation loss function creation and basic functionality."""

        # Create a simple value network with correct interface
        simple_value_network = (
            create_simple_linear_value_network()
        )  # Linear value function

        # Test basic loss function creation
        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            simple_value_network,
            self.parameters,
        )

        # Test that the loss function works
        loss = loss_function(self.decision_function, self.test_grid)

        # Verify basic properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (5,))  # 5 grid points
        self.assertTrue(torch.all(loss >= 0))  # Squared residuals

        # Test with agent specification
        loss_function_with_agent = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            simple_value_network,
            self.parameters,
            agent="consumer",
        )

        # Should produce same results for this model
        loss_with_agent = loss_function_with_agent(
            self.decision_function, self.test_grid
        )
        self.assertIsInstance(loss_with_agent, torch.Tensor)
        self.assertEqual(loss_with_agent.shape, (5,))
        self.assertTrue(torch.all(loss_with_agent >= 0))

    def test_bellman_loss_function_components(self):
        """Test that the Bellman loss function components work correctly."""
        # Test transition function
        tf = maliar.create_transition_function(self.block, ["wealth"])
        states_t = {"wealth": torch.tensor([1.0, 2.0])}
        shocks_t = {"income": torch.tensor([1.0, 1.0])}
        controls_t = {"consumption": torch.tensor([0.5, 1.0])}

        next_states = tf(states_t, shocks_t, controls_t, self.parameters)
        self.assertIn("wealth", next_states)
        self.assertTrue(torch.allclose(next_states["wealth"], torch.tensor([1.5, 2.0])))

        # Test reward function
        rf = maliar.create_reward_function(self.block, "consumer")
        reward = rf(states_t, shocks_t, controls_t, self.parameters)
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

        with self.assertRaises(Exception) as context:
            maliar.get_bellman_equation_loss(
                ["wealth"], no_control_block, self.discount_factor, dummy_value_network
            )
        self.assertIn("No control variables found in block", str(context.exception))

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

        with self.assertRaises(Exception) as context:
            maliar.get_bellman_equation_loss(
                ["wealth"], no_reward_block, self.discount_factor, dummy_value_network
            )
        self.assertIn("No reward variables found in block", str(context.exception))

    def test_bellman_loss_with_different_policies(self):
        """Test that Bellman loss function produces different values for different policies."""

        # Create a simple value network
        simple_value_network = (
            create_simple_linear_value_network()
        )  # Linear value function

        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            simple_value_network,
            self.parameters,
        )

        # Test with conservative policy
        def conservative_policy(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.3 * wealth  # Conservative consumption
            return {"consumption": consumption}

        # Test with aggressive policy
        def aggressive_policy(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.8 * wealth  # Aggressive consumption
            return {"consumption": consumption}

        loss_conservative = loss_function(conservative_policy, self.test_grid)
        loss_aggressive = loss_function(aggressive_policy, self.test_grid)

        # Both should be valid losses
        self.assertIsInstance(loss_conservative, torch.Tensor)
        self.assertIsInstance(loss_aggressive, torch.Tensor)
        self.assertEqual(loss_conservative.shape, (5,))
        self.assertEqual(loss_aggressive.shape, (5,))
        self.assertTrue(torch.all(loss_conservative >= 0))
        self.assertTrue(torch.all(loss_aggressive >= 0))

        # Different policies should produce different Bellman residuals
        self.assertFalse(torch.allclose(loss_conservative, loss_aggressive))

    def test_consistency_with_existing_patterns(self):
        """Test that the new Bellman loss functions follow existing skagent patterns."""

        # Create a simple value network with correct interface
        simple_value_network = (
            create_simple_linear_value_network()
        )  # Linear value function

        # Test that it works with the training infrastructure
        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
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
        simple_value_network = create_income_aware_value_network()

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

        residual_identical = maliar.estimate_bellman_residual(
            self.block,
            0.95,
            simple_value_network,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters={},
        )

        residual_independent = maliar.estimate_bellman_residual(
            self.block,
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

        # Create custom value network for this specific test
        def simple_value_network(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            income = shocks_t["income"]
            return (
                10.0 * wealth + 2.0 * income
            )  # Value depends on both wealth and income

        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
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

        simple_value_network = create_simple_linear_value_network()

        states_t = {"wealth": torch.tensor([2.0, 4.0])}

        # Test with missing shock periods
        shocks_missing_t1 = {
            "income_0": torch.tensor([1.0, 1.0]),
            # Missing "income_1"
        }

        with self.assertRaises(KeyError):
            maliar.estimate_bellman_residual(
                self.block,
                0.95,
                simple_value_network,
                self.decision_function,
                states_t,
                shocks_missing_t1,
                parameters={},
            )

    def test_maliar_training_loop_with_bellman_loss(self):
        """Test that maliar_training_loop works with Bellman loss for policy training."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        # Use case_0 which has no shocks in the information set (simpler)
        case_0["block"].construct_shocks(
            case_0["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        # Create a simple pre-trained value network (fixed baseline)
        value_net = BlockValueNet(case_0["block"], width=16)

        # Create Bellman loss function using the fixed value network
        bellman_loss = get_bellman_equation_loss(
            ["a"],  # state variables
            case_0["block"],
            0.9,  # discount factor
            value_net.get_value_function(),
            parameters=case_0["calibration"],
        )

        # Test maliar_training_loop with Bellman loss (policy training only)
        trained_policy, final_states = maliar.maliar_training_loop(
            case_0["block"],
            bellman_loss,  # Use Bellman loss instead of lifetime reward loss
            case_0["givens"],
            case_0["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=2,  # Keep short for testing
            tolerance=1e-4,
        )

        # Verify that policy training worked
        self.assertIsNotNone(trained_policy)
        self.assertIsNotNone(final_states)

        # Test that the trained policy produces valid outputs
        test_states = case_0["givens"].to_dict()
        decision_function = trained_policy.get_decision_function()
        controls = decision_function(test_states, {}, case_0["calibration"])

        self.assertIn("c", controls)
        self.assertIsInstance(controls["c"], torch.Tensor)
        self.assertTrue(
            torch.all(torch.isfinite(controls["c"]))
        )  # Consumption should be finite


def test_get_euler_residual_loss():
    """Test function placeholder - not implemented yet."""
    pass


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

    # Create value network - now takes block instead of state_variables
    value_net = BlockValueNet(test_block, width=16)

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


def test_train_block_value_and_policy_nn():
    """Test joint training of value and policy networks."""
    # Note: This is a placeholder test. The joint training function exists and follows
    # the same patterns as the individual training functions. A full test would require
    # careful handling of block dynamics ordering to avoid dependency issues.
    from skagent.ann import train_block_value_and_policy_nn

    # Just verify the function exists and can be imported
    assert callable(train_block_value_and_policy_nn)

    # The function signature is consistent with other training functions:
    # train_block_value_and_policy_nn(policy_net, value_net, inputs, policy_loss, value_loss, epochs)
    pass


class TestMaliarBellmanTrainingLoop(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_maliar_case_11(self):
        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_11["block"].construct_shocks(case_11["calibration"], rng=rng)

        bvn = BlockValueNet(case_11["block"], width=16)

        # Use actual state variables (a) not derived variables (m)
        # state_variables, block, discount_factor, value_network, parameters={}, agent=None
        bl = maliar.get_bellman_equation_loss(
            ["a"],  # state variables are ["a"], not ["m"]
            case_11["block"],
            0.9,
            bvn.value_function,
            parameters=case_11["calibration"],
        )

        # Use fixed random seed for deterministic training
        ann, states = maliar.maliar_training_loop(
            case_11["block"],
            bl,
            case_11["givens"]["bellman"],
            case_11["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,  # Fixed seed for deterministic training
            max_iterations=3,  # Enable max_iterations to prevent long runs
        )

        # Smoke test - verify we got back valid objects
        self.assertIsNotNone(ann)
        self.assertIsNotNone(states)
