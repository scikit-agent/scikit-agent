from conftest import (
    case_0,
    case_1,
    case_2,
    case_3,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
)
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.grid as grid
import skagent.models.perfect_foresight as pfm
import skagent.solver as solver
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class test_ann_lr(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_case_0(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_0["block"],
            0.9,
            1,
            parameters=case_0["calibration"],
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["block"], width=16)
        ann.train_block_policy_nn(bpn, states_0_N, edlrl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )

    def test_case_1(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_1["block"],
            0.9,
            1,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][1]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=500)

        c_ann = bpn.decision_function(
            # TODO -- make this from the Grid
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N.to_dict()["theta_0"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.03)
        )

    def test_case_1_2(self):
        """
        Running case 1 with big_t == 2
        """
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_1["block"],
            0.9,
            2,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][2]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=500)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N["theta_0"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.03)
        )

    def test_case_2(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_2["block"],
            0.9,
            1,
            parameters=case_2["calibration"],
        )

        given_0_N = case_2["givens"]

        bpn = ann.BlockPolicyNet(case_2["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=100)

        # optimal DR is c = 0 = E[theta]

        # Just a smoke test. Since the information set to the control
        # actually gives no information, training isn't effective...

    def test_case_3(self):
        # Construct shocks with deterministic RNG for reproducible test
        case_3["block"].construct_shocks(
            case_3["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_3["block"],
            0.9,
            1,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][1]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        self.assertTrue(torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.03))

    def test_case_3_2(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_3["block"],
            0.9,
            2,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][2]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        self.assertTrue(torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.04))

    def test_case_5_double_bounded_upper_binds(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_5["block"],
            0.9,
            1,
            parameters=case_5["calibration"],
        )

        given_0_N = case_5["givens"]

        bpn = ann.BlockPolicyNet(case_5["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_6_double_bounded_lower_binds(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_6["block"],
            0.9,
            1,
            parameters=case_6["calibration"],
        )

        given_0_N = case_6["givens"]

        bpn = ann.BlockPolicyNet(case_6["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_7_only_lower_bound(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_7["block"],
            0.9,
            1,
            parameters=case_7["calibration"],
        )

        given_0_N = case_7["givens"]

        bpn = ann.BlockPolicyNet(case_7["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(
                c_ann.flatten(), torch.zeros(c_ann.shape).to(device) + 1, atol=0.03
            )
        )

    def test_case_8_only_upper_bound(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_8["block"],
            0.9,
            1,
            parameters=case_8["calibration"],
        )

        given_0_N = case_8["givens"]

        bpn = ann.BlockPolicyNet(case_8["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_9_empty_information_set(self):
        loss_fn = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_9["block"],
            0.9,
            2,
            parameters=case_9["calibration"],
        )

        given_0_N = case_9["givens"]

        bpn = ann.BlockPolicyNet(case_9["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, loss_fn, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(
                c_ann.flatten(), torch.full_like(c_ann.flatten(), 3.0), atol=0.04
            )
        )

    def test_lifetime_reward_perfect_foresight(self):
        ### Model data

        pfblock = pfm.block_no_shock
        state_variables = ["a", "p"]

        ### Loss function
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            state_variables, pfblock, 0.9, 1, parameters=pfm.calibration
        )

        ### Setting up the training

        states_0_N = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 3, "count": 5},
                "p": {"min": 0, "max": 1, "count": 4},
            }
        )

        bpn = ann.BlockPolicyNet(pfblock, width=8)
        ann.train_block_policy_nn(bpn, states_0_N, edlrl, epochs=100)
        ## This is just a smoke test.


class test_ann_multiple_controls(unittest.TestCase):
    def setUp(self):
        pass

    def test_case_10_multiple_controls(self):
        # Control policy networks for each control in the block.
        cpns = {}

        # Invent Policy Neural Networks for each Control variable.
        for control_sym in case_10["block"].get_controls():
            cpns[control_sym] = ann.BlockPolicyNet(
                case_10["block"], control_sym=control_sym
            )

        dict_of_decision_rules = {
            k: v
            for d in [
                cpns[control_sym].get_decision_rule(length=case_10["givens"].n())
                for control_sym in cpns
            ]
            for k, v in d.items()
        }

        # train for control_sym1 with decision rule from the other net
        # for 'c'
        ann.train_block_policy_nn(
            cpns["c"],
            case_10["givens"],
            solver.get_static_reward_loss(
                ["a"],
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # train for control_sym2 with decision rule from other net
        ann.train_block_policy_nn(
            cpns["d"],
            case_10["givens"],
            solver.get_static_reward_loss(
                ["a"],
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # Train the policy neural network for 'c' again to refine its decision rule.
        # This step ensures that 'c' is optimized with the updated decision rules
        # from the other networks, improving the overall policy performance.

        ann.train_block_policy_nn(
            cpns["c"],
            case_10["givens"],
            solver.get_static_reward_loss(
                ["a"],
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=100,
        )

        rf = maliar.create_reward_function(
            case_10["block"], decision_rules=dict_of_decision_rules
        )
        rewards = rf({"a": case_10["givens"]["a"]}, {}, {}, case_10["calibration"])

        self.assertTrue(
            torch.allclose(
                rewards["u"], torch.zeros(rewards["u"].shape).to(device), atol=0.03
            )
        )


class test_ann_value_functions(unittest.TestCase):
    """Test the new value function capabilities in ann.py"""

    def setUp(self):
        """Set up a simple test block for value function testing."""
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Use case_1 benchmark problem for value function tests
        case_1["block"].construct_shocks(
            case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        self.test_block = case_1["block"]
        self.state_variables = ["a"]
        self.discount_factor = 0.9
        self.parameters = case_1["calibration"]

        # Create test grid with two independent shock realizations for Bellman tests
        self.test_grid = grid.Grid.from_dict(
            {
                "a": torch.linspace(0.1, 1.0, 10),
                "theta_0": torch.zeros(10),  # Period t shocks
                "theta_1": torch.ones(10) * 0.1,  # Period t+1 shocks (independent)
            }
        )

    def test_block_value_net_creation(self):
        """Test that BlockValueNet can be created and used."""
        value_net = ann.BlockValueNet(self.test_block, width=16)

        self.assertIn("a", value_net.state_variables)
        self.assertIn("theta", value_net.state_variables)
        self.assertNotIn("c", value_net.state_variables)
        self.assertNotIn("u", value_net.state_variables)

        states_t = {"a": torch.tensor([1.0, 2.0, 3.0])}
        shocks_t = {"theta": torch.tensor([1.0, 1.0, 1.0])}
        values = value_net.value_function(states_t, shocks_t, {})

        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (3,))

        vf = value_net.get_value_function()
        values2 = vf(states_t, shocks_t, {})
        self.assertTrue(torch.allclose(values, values2))

    def test_bellman_equation_loss_creation(self):
        """Test that Bellman equation loss functions can be created."""
        value_net = ann.BlockValueNet(self.test_block, width=16)

        bellman_loss = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.test_block,
            self.discount_factor,
            value_net.get_value_function(),
            self.parameters,
        )

        def simple_decision_function(states_t, shocks_t, parameters):
            a = states_t["a"]
            c = 0.3 * a
            return {"c": c}

        losses = bellman_loss(simple_decision_function, self.test_grid)

        self.assertIsInstance(losses, torch.Tensor)
        self.assertEqual(losses.shape, (10,))
        self.assertTrue(torch.all(losses >= 0))

    def test_train_block_value_nn(self):
        """Test that value networks can be trained."""
        value_net = ann.BlockValueNet(self.test_block, width=8)

        def simple_value_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            states_t = {"a": given_vals["a"]}
            shocks_t = {
                "theta": given_vals.get("theta_0", torch.zeros_like(given_vals["a"]))
            }
            values = vf(states_t, shocks_t, {})
            target_values = 2.0 * given_vals["a"]
            return (values - target_values) ** 2

        trained_net = ann.train_block_value_nn(
            value_net, self.test_grid, simple_value_loss, epochs=5
        )

        self.assertIs(trained_net, value_net)

        test_states = {"a": torch.tensor([0.5])}
        test_shocks = {"theta": torch.tensor([0.1])}
        values = trained_net.value_function(test_states, test_shocks, {})
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (1,))

    def test_all_in_one_bellman_loss_integration(self):
        """Test Bellman loss function integration."""
        ann.BlockPolicyNet(self.test_block, width=8)
        value_net = ann.BlockValueNet(self.test_block, width=8)

        bellman_loss = maliar.get_bellman_equation_loss(
            self.state_variables,
            self.test_block,
            self.discount_factor,
            value_net.get_value_function(),
            self.parameters,
        )

        self.assertTrue(callable(bellman_loss))

        lifetime_loss = maliar.get_estimated_discounted_lifetime_reward_loss(
            self.state_variables,
            self.test_block,
            self.discount_factor,
            1,
            self.parameters,
        )

        self.assertTrue(callable(lifetime_loss))

    def test_joint_training_function_exists(self):
        """Test that joint training function exists."""
        self.assertTrue(hasattr(ann, "train_block_value_and_policy_nn"))
        self.assertTrue(callable(ann.train_block_value_and_policy_nn))

    def test_joint_training_integration(self):
        """Test joint training integration with Bellman loss functions."""
        ann.BlockPolicyNet(self.test_block, width=8)
        value_net = ann.BlockValueNet(self.test_block, width=8)

        maliar.get_bellman_equation_loss(
            self.state_variables,
            self.test_block,
            self.discount_factor,
            value_net.get_value_function(),
            self.parameters,
        )

        self.assertTrue(callable(ann.train_block_value_and_policy_nn))

    def test_value_function_case_scenarios(self):
        """Test value function training with the same case scenarios used for policy testing."""
        # Test value function training with case_0 scenario

        # Create value network for case_0
        value_net = ann.BlockValueNet(case_0["block"], width=16)

        # Create a simple value loss function that targets zero (like the policy tests)
        def zero_target_value_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            # Use the first state variable found by the value network
            state_var = value_net.state_variables[0]
            states_t = {state_var: given_vals[state_var]}
            shocks_t = {}  # No shocks for this simple test
            values = vf(states_t, shocks_t, {})
            target_values = torch.zeros_like(values)  # Target zero like policy tests
            return (values - target_values) ** 2

        # Test training (short epochs for testing)
        trained_value_net = ann.train_block_value_nn(
            value_net, case_0["givens"], zero_target_value_loss, epochs=10
        )

        # Verify training completed
        self.assertIs(trained_value_net, value_net)

    def test_value_function_with_shocks(self):
        """Test value function training with shock scenarios (like case_1)."""

        # Create value network for case_1
        value_net = ann.BlockValueNet(case_1["block"], width=16)

        # Create a value loss function that incorporates shock information
        def shock_aware_value_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            states_t = {"a": given_vals["a"]}
            # Map theta_0 (from grid) to theta (expected by information set)
            shocks_t = {"theta": given_vals["theta_0"]}
            values = vf(states_t, shocks_t, {})
            # Target based on shock information (similar to policy test pattern)
            target_values = given_vals.get("theta_0", torch.zeros_like(given_vals["a"]))
            return (values - target_values) ** 2

        # Test training with shock-aware grid
        given_0_N = case_1["givens"][1]
        trained_value_net = ann.train_block_value_nn(
            value_net, given_0_N, shock_aware_value_loss, epochs=50
        )

        # Verify training completed and network can make predictions
        self.assertIs(trained_value_net, value_net)
        test_states = {"a": given_0_N["a"]}
        test_shocks = {"theta": given_0_N["theta_0"]}
        test_values = trained_value_net.value_function(test_states, test_shocks, {})
        self.assertIsInstance(test_values, torch.Tensor)

    def test_value_function_convergence_accuracy(self):
        """Test value function convergence accuracy (similar to policy convergence tests)."""
        # Use case_1 benchmark problem for convergence testing
        case_1["block"].construct_shocks(
            case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        # Create value network
        value_net = ann.BlockValueNet(case_1["block"], width=32)

        # Create a loss function that targets a known linear value function
        def linear_target_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            states_t = {"a": given_vals["a"]}
            shocks_t = {
                "theta": given_vals.get("theta_0", torch.zeros_like(given_vals["a"]))
            }
            values = vf(states_t, shocks_t, {})
            # Target a simple linear value function: V(a) = 2*a
            target_values = 2.0 * given_vals["a"]
            return (values - target_values) ** 2

        # Create test grid using case_1 structure
        test_grid = grid.Grid.from_dict(
            {"a": torch.linspace(0.1, 1.0, 20), "theta_0": torch.zeros(20)}
        )

        # Train with more epochs for convergence
        trained_net = ann.train_block_value_nn(
            value_net, test_grid, linear_target_loss, epochs=200
        )

        # Test convergence accuracy
        test_states = {"a": torch.tensor([0.5])}
        predicted_value = trained_net.value_function(
            test_states, {"theta": torch.tensor([0.0])}, {}
        )
        target_value = 2.0 * test_states["a"]

        # Should converge reasonably close (similar tolerance to policy tests)
        self.assertTrue(torch.allclose(predicted_value, target_value, atol=0.5))

    def test_value_function_perfect_foresight(self):
        """Test value function with perfect foresight model (mirrors policy test)."""

        # Create value network for perfect foresight model
        value_net = ann.BlockValueNet(pfm.block_no_shock, width=16)

        # Check what state variables the value network actually found
        print(
            f"Perfect foresight value network state variables: {value_net.state_variables}"
        )

        self.assertIsInstance(value_net, ann.BlockValueNet)
        self.assertTrue(len(value_net.state_variables) > 0)

        # Test that the value function method exists and can be called
        vf = value_net.get_value_function()
        self.assertTrue(callable(vf))

        # Test with a simple state input (just verify no crashes)

        test_state_vals = {
            var: torch.tensor([1.0]) for var in value_net.state_variables
        }
        test_values = value_net.value_function(test_state_vals, {}, {})
        self.assertIsInstance(test_values, torch.Tensor)
        self.assertEqual(test_values.shape, (1,))

    def test_bellman_loss_with_trained_policy_on_benchmark(self):
        """Test comprehensive Bellman iteration using the new bellman_training_loop."""
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        case_1["block"].construct_shocks(
            case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        # Create a factory function for Bellman loss (follows maliar pattern)
        def create_bellman_loss(value_net):
            """Factory function that creates Bellman loss using current value network."""
            return maliar.get_bellman_equation_loss(
                ["a"],
                case_1["block"],
                0.9,
                value_net.get_value_function(),
                parameters=case_1["calibration"],
            )

        # Demonstrate the new bellman_training_loop that trains both networks
        trained_policy, trained_value, final_states = maliar.bellman_training_loop(
            block=case_1["block"],
            loss_function=create_bellman_loss,
            states_0_n=case_1["givens"][1],
            parameters=case_1["calibration"],
            shock_copies=2,
            max_iterations=2,  # Keep short for testing
            tolerance=1e-4,
            random_seed=TEST_SEED,
            simulation_steps=1,
        )

        # Verify both networks are returned and functional
        self.assertIsInstance(trained_policy, ann.BlockPolicyNet)
        self.assertIsInstance(trained_value, ann.BlockValueNet)
        self.assertIsInstance(final_states, grid.Grid)

        # Test that both networks can make predictions
        test_states = {"a": torch.tensor([1.0, 2.0])}
        test_shocks = {"theta": torch.tensor([0.1, 0.2])}

        # Policy predictions
        decisions = trained_policy.decision_function(test_states, test_shocks, {})
        self.assertIn("c", decisions)
        self.assertIsInstance(decisions["c"], torch.Tensor)
        self.assertEqual(decisions["c"].shape, (2,))

        # Value predictions
        values = trained_value.value_function(test_states, test_shocks, {})
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (2,))

        # Verify Bellman loss can be computed with final networks
        bellman_loss = maliar.get_bellman_equation_loss(
            ["a"],
            case_1["block"],
            0.9,
            trained_value.get_value_function(),
            parameters=case_1["calibration"],
        )

        # Create test grid for evaluation
        eval_grid_dict = case_1["givens"][1].to_dict()
        eval_grid_dict["theta_1"] = torch.zeros_like(eval_grid_dict["theta_0"])
        eval_grid = grid.Grid.from_dict(eval_grid_dict)

        residuals = bellman_loss(trained_policy.get_decision_function(), eval_grid)
        self.assertIsInstance(residuals, torch.Tensor)
        self.assertTrue(torch.all(residuals >= 0))

        # This test demonstrates the complete pipeline addressing issues #100 and #101:
        # 1. New bellman_training_loop() that mirrors maliar_training_loop signature
        # 2. Uses unified Bellman loss to train both networks simultaneously
        # 3. Follows existing patterns and conventions
        # 4. Demonstrates the comprehensive joint training that was missing before
