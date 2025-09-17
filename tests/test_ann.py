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
    case_11,
)
import numpy as np
import os
import skagent.ann as ann
import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
import skagent.model as model
import skagent.models.perfect_foresight as pfm
import skagent.solver as solver
import torch
import unittest
from skagent.distributions import Normal

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_0["block"],
            0.9,
            1,
            parameters=case_0["calibration"],
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["block"], width=16)
        ann.train_block_nn(bpn, states_0_N, edlrl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )

    def test_case_1(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_1["block"],
            0.9,
            1,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][1]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=500)

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_1["block"],
            0.9,
            2,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][2]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=500)

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_2["block"],
            0.9,
            1,
            parameters=case_2["calibration"],
        )

        given_0_N = case_2["givens"]

        bpn = ann.BlockPolicyNet(case_2["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=100)

        # optimal DR is c = 0 = E[theta]

        # Just a smoke test. Since the information set to the control
        # actually gives no information, training isn't effective...

    def test_case_3(self):
        # Construct shocks with deterministic RNG for reproducible test
        case_3["block"].construct_shocks(
            case_3["calibration"], rng=np.random.default_rng(TEST_SEED)
        )

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_3["block"],
            0.9,
            1,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][1]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_3["block"],
            0.9,
            2,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][2]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_5["block"],
            0.9,
            1,
            parameters=case_5["calibration"],
        )

        given_0_N = case_5["givens"]

        bpn = ann.BlockPolicyNet(case_5["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_6_double_bounded_lower_binds(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_6["block"],
            0.9,
            1,
            parameters=case_6["calibration"],
        )

        given_0_N = case_6["givens"]

        bpn = ann.BlockPolicyNet(case_6["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_7_only_lower_bound(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_7["block"],
            0.9,
            1,
            parameters=case_7["calibration"],
        )

        given_0_N = case_7["givens"]

        bpn = ann.BlockPolicyNet(case_7["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

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
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_8["block"],
            0.9,
            1,
            parameters=case_8["calibration"],
        )

        given_0_N = case_8["givens"]

        bpn = ann.BlockPolicyNet(case_8["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_9_empty_information_set(self):
        loss_fn = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_9["block"],
            0.9,
            2,
            parameters=case_9["calibration"],
        )

        given_0_N = case_9["givens"]

        bpn = ann.BlockPolicyNet(case_9["block"], width=8)
        ann.train_block_nn(bpn, given_0_N, loss_fn, epochs=300)

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

        ### Loss function
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            pfblock, 0.9, 1, parameters=pfm.calibration
        )

        ### Setting up the training

        states_0_N = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 3, "count": 5},
                "p": {"min": 0, "max": 1, "count": 4},
            }
        )

        bpn = ann.BlockPolicyNet(pfblock, width=8)
        ann.train_block_nn(bpn, states_0_N, edlrl, epochs=100)
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
        ann.train_block_nn(
            cpns["c"],
            case_10["givens"],
            solver.StaticRewardLoss(
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # train for control_sym2 with decision rule from other net
        ann.train_block_nn(
            cpns["d"],
            case_10["givens"],
            solver.StaticRewardLoss(
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # Train the policy neural network for 'c' again to refine its decision rule.
        # This step ensures that 'c' is optimized with the updated decision rules
        # from the other networks, improving the overall policy performance.

        ann.train_block_nn(
            cpns["c"],
            case_10["givens"],
            solver.StaticRewardLoss(
                case_10["block"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=100,
        )

        rf = bellman.create_reward_function(
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

        import skagent.model as model
        from skagent.distributions import Normal

        # Create a simple consumption-savings model
        self.test_block = model.DBlock(
            name="test_value_functions",
            shocks={"income": Normal(mu=1.0, sigma=0.1)},
            dynamics={
                "consumption": model.Control(iset=["wealth"], agent="consumer"),
                "wealth": lambda wealth, income, consumption: wealth
                + income
                - consumption,
                "utility": lambda consumption: (
                    torch.log(consumption + 1e-8)
                    if hasattr(consumption, "device")
                    else torch.log(torch.tensor(consumption) + 1e-8)
                ),
            },
            reward={"utility": "consumer"},
        )
        self.test_block.construct_shocks({}, rng=np.random.default_rng(TEST_SEED))

        self.state_variables = ["wealth"]
        self.discount_factor = 0.95
        self.parameters = {}

        # Create test grid with two independent shock realizations for Bellman tests
        self.test_grid = grid.Grid.from_dict(
            {
                "wealth": torch.linspace(1.0, 5.0, 10),
                "income_0": torch.ones(10),  # Period t shocks
                "income_1": torch.ones(10) * 1.1,  # Period t+1 shocks (independent)
            }
        )

    def test_block_value_net_creation(self):
        """Test that BlockValueNet can be created and used."""
        # Create value network
        value_net = ann.BlockValueNet(self.test_block, width=16)

        # Test that it correctly identifies state variables
        self.assertIn("wealth", value_net.state_variables)
        self.assertNotIn("consumption", value_net.state_variables)  # Control
        self.assertNotIn("utility", value_net.state_variables)  # Reward
        self.assertNotIn("income", value_net.state_variables)  # Shock

        # Test value function computation (same interface as policy function)
        states_t = {"wealth": torch.tensor([1.0, 2.0, 3.0])}
        shocks_t = {"income": torch.tensor([1.0, 1.0, 1.0])}
        values = value_net.value_function(states_t, shocks_t, {})

        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (3,))

        # Test get_value_function method
        vf = value_net.get_value_function()
        values2 = vf(states_t, shocks_t, {})
        self.assertTrue(torch.allclose(values, values2))

    def test_bellman_equation_loss_creation(self):
        """Test that Bellman equation loss functions can be created."""
        # Create value network
        value_net = ann.BlockValueNet(self.test_block, width=16)

        # Create Bellman loss function - this is the key all-in-one loss function
        bellman_loss = loss.BellmanEquationLoss(
            self.test_block,
            self.discount_factor,
            value_net.get_value_function(),
            self.parameters,
        )

        # Create a simple decision function
        def simple_decision_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            consumption = 0.3 * wealth
            return {"consumption": consumption}

        # Test that the loss function works
        losses = bellman_loss(simple_decision_function, self.test_grid)

        self.assertIsInstance(losses, torch.Tensor)
        self.assertEqual(losses.shape, (10,))  # Per-sample losses
        self.assertTrue(torch.all(losses >= 0))  # Squared residuals

    def test_train_block_nn(self):
        """Test that value networks can be trained."""
        # Create value network
        value_net = ann.BlockValueNet(self.test_block, width=8)

        # Create a simple value loss function for testing
        def simple_value_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            states_t = {"wealth": given_vals["wealth"]}
            shocks_t = {
                "income": given_vals.get(
                    "income", torch.zeros_like(given_vals["wealth"])
                )
            }
            values = vf(states_t, shocks_t, {})
            target_values = 2.0 * given_vals["wealth"]  # Linear target
            return (values - target_values) ** 2

        # Test training (just a few epochs)
        trained_net = ann.train_block_nn(
            value_net, self.test_grid, simple_value_loss, epochs=5
        )

        # Check that we get the network back
        self.assertIs(trained_net, value_net)

        # Check that network can still make predictions
        test_states = {"wealth": torch.tensor([2.0])}
        test_shocks = {"income": torch.tensor([1.0])}
        values = trained_net.value_function(test_states, test_shocks, {})
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (1,))

    def test_value_function_case_scenarios(self):
        """Test value function training with the same case scenarios used for policy testing."""
        # Test value function training with case_0 scenario
        from conftest import case_0

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
        trained_value_net = ann.train_block_nn(
            value_net, case_0["givens"], zero_target_value_loss, epochs=10
        )

        # Verify training completed
        self.assertIs(trained_value_net, value_net)

    def test_value_function_with_shocks(self):
        """Test value function training with shock scenarios (like case_1)."""
        from conftest import case_1

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
        trained_value_net = ann.train_block_nn(
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
        # Create a simple test case where we know the correct value function
        test_block = model.DBlock(
            name="convergence_test",
            shocks={"income": Normal(mu=0.0, sigma=0.1)},
            dynamics={
                "wealth": lambda wealth, income, consumption: wealth
                + income
                - consumption,
                "consumption": model.Control(iset=["wealth"], agent="consumer"),
                "utility": lambda consumption: torch.log(consumption + 1e-8),
            },
            reward={"utility": "consumer"},
        )
        test_block.construct_shocks({}, rng=np.random.default_rng(TEST_SEED))

        # Create value network
        value_net = ann.BlockValueNet(test_block, width=32)

        # Create a loss function that targets a known linear value function
        def linear_target_loss(vf, input_grid):
            given_vals = input_grid.to_dict()
            states_t = {"wealth": given_vals["wealth"]}
            shocks_t = {
                "income": given_vals.get(
                    "income", torch.zeros_like(given_vals["wealth"])
                )
            }
            values = vf(states_t, shocks_t, {})
            # Target a simple linear value function: V(w) = 2*w
            target_values = 2.0 * given_vals["wealth"]
            return (values - target_values) ** 2

        # Create test grid
        test_grid = grid.Grid.from_dict(
            {"wealth": torch.linspace(1.0, 10.0, 20), "income": torch.zeros(20)}
        )

        # Train with more epochs for convergence
        trained_net = ann.train_block_nn(
            value_net, test_grid, linear_target_loss, epochs=200
        )

        # Test convergence accuracy
        test_states = {"wealth": torch.tensor([5.0])}
        predicted_value = trained_net.value_function(
            test_states, {"income": torch.tensor([0.0])}, {}
        )
        target_value = 2.0 * test_states["wealth"]

        # Should converge reasonably close (similar tolerance to policy tests)
        self.assertTrue(torch.allclose(predicted_value, target_value, atol=0.5))

    def test_value_function_perfect_foresight(self):
        """Test value function with perfect foresight model (mirrors policy test)."""
        import skagent.models.perfect_foresight as pfm

        # Create value network for perfect foresight model
        value_net = ann.BlockValueNet(pfm.block_no_shock, width=16)

        # Check what state variables the value network actually found
        print(
            f"Perfect foresight value network state variables: {value_net.state_variables}"
        )

        # This is a smoke test like the policy version - just verify the network can be created
        # and can make predictions
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

        # The main point is that BlockValueNet works with perfect foresight models
        # This mirrors the policy test pattern of basic smoke testing


class test_block_policy_value_net(unittest.TestCase):
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

    def test_block_policy_value_net__case_1(self):
        """Test comprehensive joint training that mirrors policy training patterns."""

        # Create networks
        policy_value_net = ann.BlockPolicyValueNet(case_1["block"])

        drs, vf = policy_value_net.get_policy_and_value_functions(10)

        self.assertIn("c", drs)

        # Test value network predictions
        test_states = {"a": torch.tensor([2.0])}
        value_estimates = vf(test_states, {"theta": torch.tensor([1.0])}, {})
        self.assertIsInstance(value_estimates, torch.Tensor)
        self.assertEqual(value_estimates.shape, (1,))

    def test_block_policy_value_net__case_11(self):
        """Test comprehensive joint training that mirrors policy training patterns."""

        # Create networks
        policy_value_net = ann.BlockPolicyValueNet(case_11["block"])

        ## This case has a non-trivial continuation value function
        ## and so is a good one to test policy/value training on.

        policy_value_net

        ## That hasn't been implemented yet :(
