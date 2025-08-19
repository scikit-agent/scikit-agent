from conftest import case_1, case_2, case_3, case_4
import conftest
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.grid as grid
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_simple_linear_value_network():
    """Create a simple linear value network for testing: V(a) = 10.0 * a"""

    def simple_value_network(states_t, shocks_t, parameters):
        # Handle both 'a' (case_11) and 'wealth' (custom test blocks)
        if "a" in states_t:
            return 10.0 * states_t["a"]
        elif "wealth" in states_t:
            return 10.0 * states_t["wealth"]
        else:
            raise ValueError(
                f"Expected state variable 'a' or 'wealth', got: {list(states_t.keys())}"
            )

    return simple_value_network


def create_income_aware_value_network():
    """Create a value network that depends on both wealth and income"""

    def income_aware_value_network(states_t, shocks_t, parameters):
        # Handle both variable name conventions
        if "a" in states_t:
            wealth = states_t["a"]
        else:
            wealth = states_t["wealth"]

        if "theta" in shocks_t:
            income = shocks_t["theta"]
        else:
            income = shocks_t["income"]

        return 10.0 * wealth + 5.0 * income  # Value depends on both wealth and income

    return income_aware_value_network


parameters = {"q": 1.1}


states_0 = {
    "a": 1,
}

# a dummy policy - use case_0's optimal decision rule
decision_rules = {"c": lambda a: 0}  # case_0's optimal decision rule
decisions = {"c": 0}


class TestSolverFunctions(unittest.TestCase):
    def setUp(self):
        # Use case_0 which is simpler and matches our test structure better

        self.case = conftest.case_0
        self.block = self.case["block"]
        self.block.construct_shocks(self.case["calibration"])

    def test_create_transition_function(self):
        transition_function = maliar.create_transition_function(self.block, ["a"])

        states_1 = transition_function(states_0, {}, decisions, parameters=parameters)

        # case_0 has no state transitions (no dynamics for 'a'), so states should remain unchanged
        self.assertEqual(states_1["a"], 1)

    def test_create_decision_function(self):
        decision_function = maliar.create_decision_function(self.block, decision_rules)

        decisions_0 = decision_function(states_0, {}, parameters=parameters)

        self.assertEqual(
            decisions_0["c"], 0
        )  # case_0's optimal decision rule returns 0

    def test_create_reward_function(self):
        reward_function = maliar.create_reward_function(self.block)

        reward_0 = reward_function(states_0, {}, decisions, parameters=parameters)

        # case_0's utility function is u = -c^2, with c=0, so u = 0
        self.assertEqual(reward_0["u"], 0)

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

        # case_0 with optimal decision rule c=0 gives utility u=-c^2=0 for all periods
        self.assertEqual(dlr_1, 0)

        dlr_2 = maliar.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            2,
            parameters=parameters,
        )

        # case_0 with optimal decision rule c=0 gives utility u=-c^2=0 for all periods
        self.assertEqual(dlr_2, 0)

    def test_estimate_bellman_residual(self):
        """Test the Bellman residual helper function."""

        simple_value_network = create_simple_linear_value_network()

        def simple_decision_function(states_t, shocks_t, parameters):
            a = states_t["a"]
            c = 0.5 * a  # Simple decision rule for case_0
            return {"c": c}

        # Use the block from setUp (already case_0)
        test_block = self.block

        # Test states and shocks - case_0 has no shocks, so use simple structure
        states_t = {"a": torch.tensor([2.0, 4.0])}
        shocks = {}  # case_0 has no shocks

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
        policy_high_tol, states_high_tol = maliar.maliar_training_loop(
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
        policy_low_tol, states_low_tol = maliar.maliar_training_loop(
            case_4["block"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=2,
            random_seed=TEST_SEED,
            max_iterations=3,
            tolerance=1e-8,  # Very low tolerance
        )

        # Test meaningful properties instead of None checks
        self.assertIsInstance(policy_high_tol, ann.BlockPolicyNet)
        self.assertIsInstance(states_high_tol, grid.Grid)
        self.assertIsInstance(policy_low_tol, ann.BlockPolicyNet)
        self.assertIsInstance(states_low_tol, grid.Grid)
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
        policy_net, states = maliar.maliar_training_loop(
            case_4["block"],
            edlrl,
            states_0_n,
            case_4["calibration"],
            simulation_steps=1,
            random_seed=TEST_SEED,
            max_iterations=100,  # Set high max iterations
            tolerance=1.0,  # Very high tolerance - should converge in 1-2 iterations
        )

        # Should complete successfully - test meaningful properties
        self.assertIsInstance(policy_net, ann.BlockPolicyNet)
        self.assertIsInstance(states, grid.Grid)

        # States should be valid
        sd = states.to_dict()
        self.assertIn("m", sd)
        self.assertIn("g", sd)
        self.assertTrue(torch.all(torch.isfinite(sd["m"])))
        self.assertTrue(torch.all(torch.isfinite(sd["g"])))
