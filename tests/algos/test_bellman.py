from conftest import case_0, case_11
import conftest
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.grid as grid
import torch
import unittest
from skagent.ann import BlockValueNet
from skagent.algos.maliar import (
    get_bellman_equation_loss,
)
from skagent.models.benchmarks import d3_block, d3_calibration, d3_analytical_policy

from algos.test_maliar import create_simple_linear_value_network

# Deterministic test seed - change this single value to modify all seeding
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBellmanLossFunctions(unittest.TestCase):
    """Test the Bellman equation loss functions for the Maliar method."""

    def setUp(self):
        """Set up using case_11 from conftest - simple consumer problem."""
        # Use case_11 from conftest - simple consumer problem with Bellman grid

        self.case = conftest.case_11
        self.block = self.case["block"]
        self.block.construct_shocks(self.case["calibration"])

        # Parameters from case_11
        self.discount_factor = 0.95
        self.parameters = self.case["calibration"]
        self.state_variables = ["a"]  # Asset state variable from case_11

        # Create a simple decision function for testing
        def simple_decision_function(states_t, shocks_t, parameters):
            # Simple consumption rule: consume half of m (income)
            a = states_t["a"]
            r = parameters.get("r", torch.tensor(1.1))
            theta = shocks_t["theta"]
            m = a * r + torch.exp(theta)  # Follow case_11 dynamics
            consumption = 0.5 * m
            return {"c": consumption}

        self.decision_function = simple_decision_function

        # Use the Bellman grid from case_11 (already has theta_0 and theta_1)
        self.test_grid = self.case["givens"]["bellman"]

    def test_get_bellman_equation_loss(self):
        """Test Bellman equation loss function creation and basic functionality."""

        # Create a simple value network with correct interface
        simple_value_network = (
            create_simple_linear_value_network()
        )  # Linear value function

        # Test basic loss function creation (unified MMW Definition 2.10)
        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            parameters=self.parameters,
        )

        # Test that the loss function works with value function first, then decision function
        # simple_value_network is already a value function
        loss = loss_function(
            simple_value_network, self.decision_function, self.test_grid
        )

        # Verify basic properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(
            loss.shape, (539,)
        )  # 11 * 7 * 7 grid points from case_11 bellman grid
        self.assertTrue(
            torch.all(torch.isfinite(loss))
        )  # Should be finite (not NaN/inf)
        # Note: Complete MMW Definition 2.10 can produce negative values due to product of residuals
        # Note: requires_grad depends on input tensor gradients

        # Test with agent specification
        loss_function_with_agent = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            parameters=self.parameters,
            agent="agent",  # case_11 uses "agent" not "consumer"
        )

        # Should produce same results for this model
        loss_with_agent = loss_function_with_agent(
            simple_value_network, self.decision_function, self.test_grid
        )
        self.assertIsInstance(loss_with_agent, torch.Tensor)
        self.assertEqual(loss_with_agent.shape, (539,))  # Same grid size
        self.assertTrue(
            torch.all(torch.isfinite(loss_with_agent))
        )  # Should be finite (not NaN/inf)

    def test_bellman_equation_loss_function_components(self):
        """Test that the Bellman loss function components work correctly."""
        # Test transition function - use case_11 variables
        tf = maliar.create_transition_function(self.block, ["a"])
        states_t = {"a": torch.tensor([1.0, 2.0])}
        shocks_t = {"theta": torch.tensor([0.5, 0.5])}
        controls_t = {"c": torch.tensor([0.5, 1.0])}

        next_states = tf(states_t, shocks_t, controls_t, self.parameters)
        self.assertIn("a", next_states)
        # a = m - c, where m = a * r + exp(theta)
        # For a=1.0, r=1.1, theta=0.5, c=0.5: m = 1.0*1.1 + exp(0.5) ≈ 2.749, next_a = 2.749 - 0.5 = 2.249
        # For a=2.0, r=1.1, theta=0.5, c=1.0: m = 2.0*1.1 + exp(0.5) ≈ 3.849, next_a = 3.849 - 1.0 = 2.849
        self.assertTrue(torch.all(next_states["a"] > 0))  # Should be positive

        # Test reward function
        rf = maliar.create_reward_function(self.block, "agent")  # case_11 uses "agent"
        reward = rf(states_t, shocks_t, controls_t, self.parameters)
        self.assertIn("u", reward)  # case_11 reward variable is "u"
        # Utility can be negative for log(consumption), so just check it's finite
        self.assertTrue(torch.all(torch.isfinite(reward["u"])))

    def test_bellman_equation_loss_with_different_policies(self):
        """Test that Bellman loss function produces different values for different policies."""

        # Create a simple value network
        simple_value_network = (
            create_simple_linear_value_network()
        )  # Linear value function

        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            parameters=self.parameters,
        )

        # Test with conservative policy
        def conservative_policy(states_t, shocks_t, parameters):
            a = states_t["a"]  # Use case_11 state variable
            r = parameters.get("r", torch.tensor(1.1))
            theta = shocks_t["theta"]
            m = a * r + torch.exp(theta)  # Follow case_11 dynamics
            consumption = 0.3 * m  # Conservative consumption
            return {"c": consumption}

        # Test with aggressive policy
        def aggressive_policy(states_t, shocks_t, parameters):
            a = states_t["a"]  # Use case_11 state variable
            r = parameters.get("r", torch.tensor(1.1))
            theta = shocks_t["theta"]
            m = a * r + torch.exp(theta)  # Follow case_11 dynamics
            consumption = 0.8 * m  # Aggressive consumption
            return {"c": consumption}

        loss_conservative = loss_function(
            simple_value_network, conservative_policy, self.test_grid
        )
        loss_aggressive = loss_function(
            simple_value_network, aggressive_policy, self.test_grid
        )

        # Both should be valid losses
        self.assertIsInstance(loss_conservative, torch.Tensor)
        self.assertIsInstance(loss_aggressive, torch.Tensor)
        self.assertEqual(loss_conservative.shape, (539,))  # case_11 bellman grid size
        self.assertEqual(loss_aggressive.shape, (539,))
        self.assertTrue(
            torch.all(torch.isfinite(loss_conservative))
        )  # Should be finite (not NaN/inf)
        self.assertTrue(
            torch.all(torch.isfinite(loss_aggressive))
        )  # Should be finite (not NaN/inf)

        # Different policies should produce different Bellman residuals (now properly implemented)
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
            parameters=self.parameters,
        )

        # Test with aggregate_net_loss (from ann.py) - not directly compatible with 3-parameter loss
        # The 3-parameter Bellman loss is designed for train_block_value_and_policy_nn
        # This test verifies our loss function structure is sound

        # Test direct usage instead
        test_loss = loss_function(
            simple_value_network, self.decision_function, self.test_grid
        )

        self.assertIsInstance(test_loss, torch.Tensor)
        self.assertEqual(test_loss.shape, (539,))  # case_11 bellman grid size
        self.assertTrue(
            torch.all(torch.isfinite(test_loss))
        )  # Should be finite (not NaN/inf)

    def test_shock_independence_in_bellman_residual(self):
        """Test that independent shock realizations produce different results than identical shocks."""

        # Create a value network that depends on both state and shocks for this test
        def shock_sensitive_value_network(states_t, shocks_t, parameters):
            if "a" in states_t:
                state_val = states_t["a"]
            else:
                state_val = states_t["wealth"]

            if "theta" in shocks_t:
                shock_val = shocks_t["theta"]
            else:
                shock_val = shocks_t.get("income", torch.tensor(0.0))

            return (
                10.0 * state_val + 2.0 * shock_val
            )  # Value depends on both state and shock

        states_t = {"a": torch.tensor([2.0, 4.0])}  # Use case_11 state variable

        # Test 1: Identical shocks for both periods (should be like old behavior)
        shocks_identical = {
            "theta_0": torch.tensor([1.0, 1.0]),  # Period t
            "theta_1": torch.tensor([1.0, 1.0]),  # Period t+1 (same as t)
        }

        # Test 2: Independent shocks for the two periods
        shocks_independent = {
            "theta_0": torch.tensor([1.0, 1.0]),  # Period t
            "theta_1": torch.tensor([1.5, 0.5]),  # Period t+1 (different from t)
        }

        residual_identical = maliar.estimate_bellman_residual(
            self.block,
            0.95,
            shock_sensitive_value_network,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters=self.parameters,  # Use case_11 parameters which include 'r'
        )

        residual_independent = maliar.estimate_bellman_residual(
            self.block,
            0.95,
            shock_sensitive_value_network,
            self.decision_function,
            states_t,
            shocks_independent,
            parameters=self.parameters,  # Use case_11 parameters which include 'r'
        )

        # Results should be different when using independent vs identical shocks
        self.assertFalse(torch.allclose(residual_identical, residual_independent))

        # Both should be finite
        self.assertTrue(torch.all(torch.isfinite(residual_identical)))
        self.assertTrue(torch.all(torch.isfinite(residual_independent)))

    def test_bellman_equation_loss_with_different_shock_patterns(self):
        """Test Bellman loss function with various shock patterns."""

        # Create custom value network for this specific test
        def simple_value_network(states_t, shocks_t, parameters):
            a = states_t["a"]  # Use case_11 state variable
            theta = shocks_t["theta"]  # Use case_11 shock variable
            return 10.0 * a + 2.0 * theta  # Value depends on both assets and shock

        loss_function = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.block,
            self.discount_factor,
            parameters=self.parameters,
        )

        # Test with correlated shocks (period t+1 same as period t)
        test_grid_correlated = grid.Grid.from_dict(
            {
                "a": torch.tensor([1.0, 2.0, 3.0]),  # Use case_11 state variable
                "theta_0": torch.tensor([1.0, 1.2, 0.8]),
                "theta_1": torch.tensor([1.0, 1.2, 0.8]),  # Same as period t
            }
        )

        # Test with anti-correlated shocks
        test_grid_anticorrelated = grid.Grid.from_dict(
            {
                "a": torch.tensor([1.0, 2.0, 3.0]),  # Use case_11 state variable
                "theta_0": torch.tensor([1.0, 1.2, 0.8]),
                "theta_1": torch.tensor([1.0, 0.8, 1.2]),  # Opposite of period t
            }
        )

        loss_correlated = loss_function(
            simple_value_network, self.decision_function, test_grid_correlated
        )
        loss_anticorrelated = loss_function(
            simple_value_network, self.decision_function, test_grid_anticorrelated
        )

        # Both should produce valid losses
        self.assertTrue(
            torch.all(torch.isfinite(loss_correlated))
        )  # Should be finite (not NaN/inf)
        self.assertTrue(
            torch.all(torch.isfinite(loss_anticorrelated))
        )  # Should be finite (not NaN/inf)

        # Losses should be different for different shock patterns (now properly implemented)
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

    def test_bellman_training_loop_with_bellman_equation_loss(self):
        """Test that bellman_training_loop works with Bellman loss for joint training."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        # Use case_0 which has no shocks in the information set (simpler)
        case_0["block"].construct_shocks(
            case_0["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        # Create Bellman loss function (unified MMW Definition 2.10)
        bellman_equation_loss = get_bellman_equation_loss(
            ["a"],  # state variables
            case_0["block"],
            0.9,  # discount factor
            parameters=case_0["calibration"],
        )

        # Test bellman_training_loop with Bellman loss (joint training)
        trained_policy, trained_value, final_states = maliar.bellman_training_loop(
            case_0["block"],
            bellman_equation_loss,  # Use Bellman loss for joint training
            case_0["givens"],
            case_0["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=2,  # Keep short for testing
            tolerance=1e-4,
        )

        # Verify that joint training worked - test meaningful properties
        self.assertIsInstance(trained_policy, ann.BlockPolicyNet)
        self.assertIsInstance(trained_value, ann.BlockValueNet)
        self.assertIsInstance(final_states, grid.Grid)

        # Test that the trained policy produces valid outputs
        test_states = case_0["givens"].to_dict()
        decision_function = trained_policy.get_decision_function()
        controls = decision_function(test_states, {}, case_0["calibration"])

        self.assertIn("c", controls)
        self.assertIsInstance(controls["c"], torch.Tensor)
        self.assertTrue(
            torch.all(torch.isfinite(controls["c"]))
        )  # Consumption should be finite


class TestBlockValueNet(unittest.TestCase):
    def setUp(self):
        self.case = conftest.case_11
        self.test_block = self.case["block"]
        self.test_block.construct_shocks(self.case["calibration"])
        self.value_net = BlockValueNet(self.test_block)

    def test_block_value_net_functionality(self):
        """Test BlockValueNet functionality using conftest case."""
        # Test value function computation using case_11 variables
        states_t = {"a": torch.tensor([1.0, 2.0, 3.0])}  # Use case_11 state variable
        shocks_t = {
            "theta": torch.tensor([0.5, 0.5, 0.5])
        }  # Use case_11 shock variable

        values = self.value_net.value_function(
            states_t, shocks_t, self.case["calibration"]
        )

        # Check output shape and type
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (3,))

        # Test get_value_function method
        vf = self.value_net.get_value_function()
        values2 = vf(states_t, shocks_t, self.case["calibration"])

        # Should give same results
        self.assertTrue(torch.allclose(values, values2))


class TestBellmanJointTrainingLoop(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_bellman_case_11(self):
        # Use deterministic RNG for shock construction
        rng = np.random.default_rng(TEST_SEED)
        case_11["block"].construct_shocks(case_11["calibration"], rng=rng)

        # Create Bellman loss function for joint training
        bl = maliar.get_bellman_equation_loss(
            case_11["givens"]["bellman"].labels,
            case_11["block"],
            0.9,
            parameters=case_11["calibration"],
        )

        # Use bellman_training_loop for joint policy+value training (not maliar_training_loop)
        policy_net, value_net, states = maliar.bellman_training_loop(
            case_11["block"],
            bl,
            case_11["givens"]["bellman"],
            case_11["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,  # Fixed seed for deterministic training
            max_iterations=2,  # Keep short for testing
        )

        # Verify both networks were trained - test meaningful properties
        self.assertIsInstance(policy_net, ann.BlockPolicyNet)
        self.assertIsInstance(value_net, ann.BlockValueNet)

        # Test that networks can produce outputs (basic functionality)
        test_states = {"a": torch.tensor([1.0])}
        test_shocks = {"theta": torch.tensor([0.5])}

        policy_output = policy_net.decision_function(
            test_states, test_shocks, case_11["calibration"]
        )
        value_output = value_net.value_function(
            test_states, test_shocks, case_11["calibration"]
        )

        self.assertIsInstance(policy_output, dict)
        self.assertIsInstance(value_output, torch.Tensor)
        self.assertTrue(torch.isfinite(value_output).all())

        # Verify final states were updated
        self.assertIsInstance(states, grid.Grid)

    def test_bellman_convergence_tolerance(self):
        """Test the convergence functionality in the Bellman training loop (parallel to maliar test)."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        # Use case_0 for simpler testing
        case_0["block"].construct_shocks(
            case_0["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        bellman_equation_loss = get_bellman_equation_loss(
            ["a"],  # state variables
            case_0["block"],
            0.9,  # discount factor
            parameters=case_0["calibration"],
        )

        # Test 1: High tolerance (should converge quickly)
        policy_high_tol, value_high_tol, states_high_tol = maliar.bellman_training_loop(
            case_0["block"],
            bellman_equation_loss,
            case_0["givens"],
            case_0["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=10,
            tolerance=1e-2,  # High tolerance
        )

        # Test 2: Low tolerance (should require more iterations or hit max_iterations)
        policy_low_tol, value_low_tol, states_low_tol = maliar.bellman_training_loop(
            case_0["block"],
            bellman_equation_loss,
            case_0["givens"],
            case_0["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=3,
            tolerance=1e-8,  # Very low tolerance
        )

        # Both should produce valid results - test meaningful properties
        self.assertIsInstance(policy_high_tol, ann.BlockPolicyNet)
        self.assertIsInstance(value_high_tol, ann.BlockValueNet)
        self.assertIsInstance(policy_low_tol, ann.BlockPolicyNet)
        self.assertIsInstance(value_low_tol, ann.BlockValueNet)

        # Test that both networks can produce outputs
        test_states = {"a": torch.tensor([1.0])}
        test_shocks = {}  # case_0 has no shocks

        policy_high_output = policy_high_tol.decision_function(
            test_states, test_shocks, case_0["calibration"]
        )
        policy_low_output = policy_low_tol.decision_function(
            test_states, test_shocks, case_0["calibration"]
        )

        self.assertIsInstance(policy_high_output, dict)
        self.assertIsInstance(policy_low_output, dict)

    def test_bellman_convergence_early_stopping(self):
        """Test that the Bellman training loop can stop early when convergence is achieved (parallel to maliar test)."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        # Use case_0 for simpler testing
        case_0["block"].construct_shocks(
            case_0["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        bellman_equation_loss = get_bellman_equation_loss(
            ["a"],  # state variables
            case_0["block"],
            0.9,  # discount factor
            parameters=case_0["calibration"],
        )

        # Test with very high tolerance to ensure early convergence
        policy, value, states = maliar.bellman_training_loop(
            case_0["block"],
            bellman_equation_loss,
            case_0["givens"],
            case_0["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=10,  # Allow many iterations
            tolerance=1.0,  # Very high tolerance - should converge immediately
        )

        # Should have converged early - test meaningful properties
        self.assertIsInstance(policy, ann.BlockPolicyNet)
        self.assertIsInstance(value, ann.BlockValueNet)
        self.assertIsInstance(states, grid.Grid)

        # Test that networks can produce outputs after training
        test_states = {"a": torch.tensor([1.0])}
        test_shocks = {}  # case_0 has no shocks

        policy_output = policy.decision_function(
            test_states, test_shocks, case_0["calibration"]
        )
        value_output = value.value_function(
            test_states, test_shocks, case_0["calibration"]
        )

        self.assertIsInstance(policy_output, dict)
        self.assertIsInstance(value_output, torch.Tensor)
        self.assertTrue(torch.isfinite(value_output).all())


class TestD3AnalyticalApproximation(unittest.TestCase):
    """Test that neural networks can approximate the D-3 analytical policy solution."""

    def test_d3_analytical_policy_approximation(self):
        """Test that Bellman training can approximate the D-3 analytical policy solution."""

        # Set up the D-3 model with tensor parameters
        d3_calibration_tensors = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            if isinstance(v, (int, float))
            else v
            for k, v in d3_calibration.items()
        }
        d3_block.construct_shocks(d3_calibration_tensors)

        # Get the analytical policy
        analytical_policy = d3_analytical_policy(d3_calibration)

        # Create test grid for D-3 infinite horizon model
        # Need both m (current market resources) and a (previous assets that generated m)
        # Relationship: m = a * R + y, so a = (m - y) / R
        R = d3_calibration_tensors["R"]
        y = d3_calibration_tensors["y"]

        # D-3 training grid should provide pre-decision assets 'a'
        # The model will compute m = a * R + y internally
        a_min, a_max = 0.0, 8.0  # Assets range that will generate reasonable m values

        test_grid = grid.Grid.from_config(
            {"a": {"min": a_min, "max": a_max, "count": 25}}
        )

        # Create Bellman loss function
        bellman_loss = maliar.get_bellman_equation_loss(
            test_grid.labels,
            d3_block,
            d3_calibration_tensors["DiscFac"],
            parameters=d3_calibration_tensors,
        )

        # Train neural networks
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        policy_net, value_net, final_states = maliar.bellman_training_loop(
            d3_block,
            bellman_loss,
            test_grid,
            d3_calibration_tensors,
            max_iterations=10,
            epochs=500,  # Increased for better convergence
            tolerance=1e-4,
            random_seed=TEST_SEED,
        )

        # Test approximation quality - provide pre-decision assets
        test_a = torch.tensor([1.0, 3.0, 6.0], device=device)
        test_states = {"a": test_a}  # D-3 pre-decision state
        test_shocks = {}  # D-3 has no shocks

        # Get neural network policy
        nn_policy_output = policy_net.decision_function(
            test_states, test_shocks, d3_calibration_tensors
        )

        # Get analytical policy - it expects m (cash-on-hand) as input
        # Compute m from the pre-decision assets a
        test_m = test_a * R.cpu().item() + y.cpu().item()
        test_states_for_analytical = {"m": test_m}
        analytical_output = analytical_policy(
            test_states_for_analytical, test_shocks, d3_calibration
        )

        # Compare policies - should be close (ensure same device)
        nn_consumption = nn_policy_output["c"]
        analytical_consumption = analytical_output["c"].to(nn_consumption.device)

        # Test that neural network approximation is reasonably close to analytical solution
        relative_error = torch.abs(
            (nn_consumption - analytical_consumption) / analytical_consumption
        )
        max_relative_error = torch.max(relative_error).item()

        # Allow up to 10% relative error for neural network approximation
        self.assertLess(
            max_relative_error,
            0.10,
            f"Neural network policy deviates too much from analytical solution. "
            f"Max relative error: {max_relative_error:.4f}",
        )

        # Test that both produce finite, positive consumption
        self.assertTrue(torch.all(nn_consumption > 0))
        self.assertTrue(torch.all(torch.isfinite(nn_consumption)))
        self.assertTrue(torch.all(analytical_consumption > 0))
        self.assertTrue(torch.all(torch.isfinite(analytical_consumption)))
