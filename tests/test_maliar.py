from conftest import case_1, case_3, case_4
import logging
import numpy as np
import os
import skagent.algos.maliar as maliar
import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
import skagent.block as model
import torch
import unittest
from skagent.ann import BlockPolicyNet, BlockPolicyValueNet, train_block_nn
from skagent.distributions import Normal
from skagent.models.benchmarks import (
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
    get_reference_policy,
)
from skagent.simulation.monte_carlo import draw_shocks
from skagent.utils import reconcile
from test_benchmarks import assert_consumption_policy_diagnostics

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


# Shared comparison helpers: the Euler residual of a decision function and the
# pointwise error of a policy vs a reference, factored out so the same checks
# apply to both analytical and trained policies without duplicated plumbing.


def euler_residual_of(bp, decision_fn, states, shocks, parameters):
    """Return the (single-control) Euler residual tensor for a decision function.

    Centralizes the ``estimate_euler_residual`` -> single-control ->
    ``detach`` sequence shared by the analytical-policy and trained-policy
    tests, so the same residual computation is reused across solution methods.
    """
    residuals = bellman.estimate_euler_residual(
        bp, decision_fn, states, shocks, parameters
    )
    return next(iter(residuals.values())).detach()


def policy_rel_error_of(decision_fn, reference_fn, states, shocks, parameters):
    """Return the pointwise relative error of a policy against a reference.

    Both arguments are decision functions ``(states, shocks, parameters) ->
    {control: tensor}``, so the comparison is identical whether ``decision_fn``
    is an analytical solution or the output of a training algorithm. Callers
    take ``.mean()`` and/or ``.max()`` of the returned tensor.
    """
    actual = decision_fn(states, shocks, parameters)["c"].detach()
    reference = reference_fn(states, shocks, parameters)["c"].detach().to(actual.device)
    return torch.abs(actual - reference) / (torch.abs(reference) + 1e-8)


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
        def dummy_value_function(wealth):
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
            loss.BellmanEquationLoss(nrbp, dummy_value_function)
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
        def simple_value_function(states_t, shocks_t, parameters):
            wealth = states_t["wealth"]
            return 10.0 * wealth  # Linear value function

        loss_function = loss.BellmanEquationLoss(
            self.bp,
            simple_value_function,
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
        def simple_value_function(states_t, shocks_t, parameters):
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
            simple_value_function,
            self.decision_function,
            states_t,
            shocks_identical,
            parameters=None,
        )

        residual_independent = bellman.estimate_bellman_residual(
            self.bp,
            simple_value_function,
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

        def simple_value_function(states_t, shocks_t, parameters):
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
                simple_value_function,
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
    assert isinstance(losses, torch.Tensor), "losses must be a torch.Tensor"
    assert losses.shape[0] == n_points, "Expected one loss value per grid point"
    assert torch.all(losses >= 0), "Squared residuals must be non-negative"

    # For the analytical optimal policy, Euler residuals should be near zero
    # (within numerical tolerance for perfect foresight model)
    mean_loss = torch.mean(losses).item()
    assert mean_loss < 1e-8, (
        f"Analytical optimal policy should have near-zero Euler loss. Got mean loss: {mean_loss:.6e}"
    )


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
        bp = bellman.BellmanPeriod(d2_block, "DiscFac", d2_calibration)

        # Create test states
        n_samples = 100
        test_states = {"a": torch.linspace(0.1, 10.0, n_samples)}
        shocks = {}

        # Compute Euler residual with analytical optimal policy
        optimal_residual = euler_residual_of(
            bp, d2_policy, test_states, shocks, d2_calibration
        )

        # For the analytical optimal policy every Euler residual should be
        # at machine precision: the max over all samples must be tiny, not
        # just the mean. A mean test could mask a single bad sample.
        max_abs_residual = torch.max(torch.abs(optimal_residual)).item()
        mse_residual = torch.mean(optimal_residual**2).item()

        # Tolerance accounts for numerical precision in autograd
        self.assertLess(
            max_abs_residual,
            1e-5,
            f"Analytical optimal policy should have near-zero Euler residual. "
            f"Got max |residual| = {max_abs_residual:.6e}",
        )

        self.assertLess(
            mse_residual,
            1e-10,
            f"Analytical optimal policy should have near-zero Euler MSE. "
            f"Got MSE = {mse_residual:.6e}",
        )

    def test_euler_equation_training(self):
        """
        Test that a policy can be trained via Euler equation loss to achieve near-zero residual.

        This is the key validation test for the Maliar et al. (2021) methodology:
        train a neural network policy by minimizing squared Euler residuals,
        and verify the trained policy satisfies the first-order conditions.

        Uses scikit-agent's BlockPolicyNet and train_block_nn for proper integration.
        """

        # Get D-2 benchmark model and calibration
        d2_block = get_benchmark_model("D-2")
        d2_calibration = get_benchmark_calibration("D-2")

        # Construct shocks for the block
        d2_block.construct_shocks(d2_calibration)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(d2_block, "DiscFac", d2_calibration)

        # Create policy network using scikit-agent's BlockPolicyNet
        torch.manual_seed(TEST_SEED)
        policy_net = BlockPolicyNet(bp, width=32, init_seed=TEST_SEED)

        # Create Euler equation loss
        euler_loss_fn = loss.EulerEquationLoss(bp, parameters=d2_calibration)

        # Create training grid with states (D-2 is deterministic, no shocks needed)
        n_grid_points = 64
        train_grid = grid.Grid.from_dict(
            {
                "a": torch.rand(n_grid_points, device=device) * 9.9
                + 0.1,  # a in [0.1, 10]
            }
        )

        # Train using scikit-agent's train_block_nn
        trained_net, final_loss, _ = train_block_nn(
            policy_net, train_grid, euler_loss_fn, epochs=1000, verbose=False
        )

        # Verify training achieved small loss
        self.assertLess(
            final_loss,
            0.01,
            f"Training should achieve small Euler loss. Got final loss: {final_loss:.6e}",
        )

        # Evaluate on test grid using the trained policy
        test_states = {"a": torch.linspace(0.1, 10.0, 100, device=device)}
        shocks = {}

        # Get decision function from trained network
        decision_fn = trained_net.get_decision_function()

        # Compute final Euler residual
        final_residual = euler_residual_of(
            bp, decision_fn, test_states, shocks, d2_calibration
        )

        # Trained NN approximation, not analytical: a mean threshold fits here;
        # a max threshold suits only near-machine-precision tests.
        mean_residual = torch.mean(torch.abs(final_residual)).item()
        mse_residual = torch.mean(final_residual**2).item()

        # Trained policy should have mean |residual| < 0.07
        # (robust across seeds with lr=0.001 and 1000 epochs)
        self.assertLess(
            mean_residual,
            0.07,
            f"Trained policy should have small Euler residual. "
            f"Got mean |residual| = {mean_residual:.6e}",
        )

        self.assertLess(
            mse_residual,
            0.05,
            f"Trained policy should have small Euler residual MSE. "
            f"Got MSE = {mse_residual:.6e}",
        )

    # Euler-only training under-identifies the level (residual -> 0 but level is
    # platform-dependent), so it has no robust level test. Robust level recovery
    # uses the value head: see test_maliar_training_loop_u2_analytical.

    def test_maliar_training_loop_u2_analytical(self):
        """
        Test Bellman equation training with U-2 model against analytical PIH solution.

        U-2 uses NORMALIZED variables (all divided by permanent income P):
        - a = A/P (normalized assets, arrival state)
        - m = M/P = R*a/ψ + 1 (normalized cash-on-hand)
        - c = C/P (normalized consumption)

        Analytical solution: c = (1-β)(m + 1/r) where 1/r is normalized human wealth.

        Uses a shared-backbone BlockPolicyValueNet so that a single optimizer
        updates both the policy head and the value head simultaneously. The
        value head pins down the consumption LEVEL via the Bellman equation
        V(m) = u(c) + β E[V(m')], resolving the level-identification problem
        inherent in pure Euler equation training.
        """

        # Get U-2 benchmark model
        u2_block = get_benchmark_model("U-2")
        u2_calibration = get_benchmark_calibration("U-2")
        analytical_policy = get_analytical_policy("U-2")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u2_block.construct_shocks(u2_calibration, rng=rng)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(u2_block, "DiscFac", u2_calibration)

        # Create shared-backbone policy+value network
        torch.manual_seed(TEST_SEED)
        pvnet = BlockPolicyValueNet(bp, width=32)

        # Bellman equation loss using the value head of the same network.
        # foc_weight=1.0 adds the FOC term (Maliar et al. 2021, eq. 14)
        # for faster convergence.
        bellman_loss_fn = loss.BellmanEquationLoss(
            bp,
            pvnet.get_value_function(),
            parameters=u2_calibration,
            foc_weight=1.0,
        )

        # Training grid with two shock copies (AiO expectation operator)
        n_pts = 15
        train_grid = grid.Grid.from_dict(
            {
                "a": torch.linspace(0.5, 5.0, n_pts, device=device),
                "psi_0": torch.ones(n_pts, device=device),
                "psi_1": torch.ones(n_pts, device=device),
            }
        )

        # Train with single optimizer (both heads share the backbone)
        trained_net, _, _ = train_block_nn(
            pvnet, train_grid, bellman_loss_fn, epochs=2000, verbose=False
        )

        # Get decision function from trained network
        decision_fn = trained_net.get_decision_function()

        # Test on grid within training range (normalized assets)
        n_test = 25
        test_a = torch.linspace(0.5, 5.0, n_test, device=device)
        test_states = {"a": test_a}
        test_shocks = {"psi": torch.ones(n_test, device=device)}

        # Compare trained vs analytical through the shared helper, so the exact
        # same comparison can later be applied to any other solution method.
        rel_error = policy_rel_error_of(
            decision_fn, analytical_policy, test_states, test_shocks, u2_calibration
        )
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()

        # The value function anchors the consumption level (Euler-only training
        # leaves it indeterminate). The error is near-uniform across the grid, so
        # we assert a pointwise max bound, not just the mean.
        self.assertLess(
            mean_rel_error,
            0.05,
            f"Bellman-trained policy should closely match analytical PIH solution. "
            f"Mean relative error: {mean_rel_error:.4f}",
        )
        self.assertLess(
            max_rel_error,
            0.08,
            f"Every tested state should match the analytical PIH solution, not "
            f"only on average. Max relative error: {max_rel_error:.4f}",
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
        policy using two checks targeted at the training loop itself:

        1. The reusable :func:`assert_consumption_policy_diagnostics` helper
           (positive consumption, budget constraint, strict monotonicity,
           MPC in (0,1)) - the same checks that
           :class:`TestConsumptionPolicyDiagnostics` runs against the
           analytical benchmarks.
        2. Euler residual near zero in the unconstrained region (high
           wealth), which is what the training loop actually optimizes.
        """
        # Get U-3 buffer stock model (CRRA=2, with borrowing constraint)
        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u3_block.construct_shocks(u3_calibration, rng=rng)

        # Create BellmanPeriod
        bp = bellman.BellmanPeriod(u3_block, "DiscFac", u3_calibration)

        # Create initial states grid (normalized assets)
        # Use wider range to test both constrained and unconstrained regions
        states_0_n = grid.Grid.from_config(
            {
                "a": {"min": 0.5, "max": 10.0, "count": 25},
            }
        )

        # constrained=True: the borrowing constraint c <= m turns the Euler
        # equation into an inequality (u'(c) >= betaR E[u'(c')]); the one-sided
        # loss penalizes only negative residuals (overconsumption).
        euler_loss_fn = loss.EulerEquationLoss(
            bp,
            parameters=u3_calibration,
            constrained=True,  # Key fix: use one-sided loss for borrowing constraint
        )

        # Train the policy with enough iterations for convergence.
        # More iterations allow the Maliar simulation-based state updates
        # to explore the ergodic distribution, improving accuracy.
        trained_net, final_states = maliar.maliar_training_loop(
            bp,
            euler_loss_fn,
            states_0_n,
            u3_calibration,
            shock_copies=2,
            max_iterations=25,
            tolerance=1e-6,
            random_seed=TEST_SEED,
            simulation_steps=1,
        )

        # Get decision function and parameters
        decision_fn = trained_net.get_decision_function()
        R = u3_calibration["R"]
        beta = u3_calibration["DiscFac"]  # β ≡ DiscFac (standard economics notation)
        gamma = u3_calibration["CRRA"]

        # 1. Economic sanity via the reusable diagnostics helper: positive c,
        # budget c <= m, strict monotonicity in wealth, average MPC in (0, 1).
        n_test = 30
        test_a = torch.linspace(0.5, 10.0, n_test, device=device)
        test_states = {"a": test_a}
        test_shocks = {
            "psi": torch.ones(n_test, device=device),
            "theta": torch.ones(n_test, device=device),
        }

        def cash_on_hand(states, shocks, cal):
            return cal["R"] * states["a"] / shocks["psi"] + shocks["theta"]

        assert_consumption_policy_diagnostics(
            decision_fn,
            test_states=test_states,
            test_shocks=test_shocks,
            calibration=u3_calibration,
            cash_on_hand_func=cash_on_hand,
            monotone_tol=1e-2,
            label="U-3 trained",
        )

        # Reference values used in comments below.
        kappa_pf = (R - (beta * R) ** (1 / gamma)) / R
        self.assertGreater(
            kappa_pf, 0.0, "Perfect-foresight MPC must be positive for U-3."
        )

        # 2. Euler residual in the unconstrained (high-wealth) region: the
        # constraint is slack, so the Euler equation should hold closely.
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

        euler_residual = euler_residual_of(
            bp, decision_fn, high_wealth_states, high_wealth_shocks, u3_calibration
        )

        mean_euler_residual = torch.mean(torch.abs(euler_residual)).item()
        # Smoke bound calibrated across backends: seeded CPU trajectories
        # land < 0.1, CUDA lands ~0.21 at the 25-iteration cap (backends
        # diverge under the same seed; cf. 07d0191). 0.25 covers both.
        self.assertLess(
            mean_euler_residual,
            0.25,
            f"Euler residual should be small at high wealth (unconstrained). "
            f"Got mean |residual| = {mean_euler_residual:.4f}",
        )
        max_euler_residual = torch.max(torch.abs(euler_residual)).item()
        self.assertLess(
            max_euler_residual,
            0.6,
            "Max pointwise Euler residual exceeds tolerance",
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
        # Construct U-3's shocks here (seeded): the loss draws the all-in-one
        # operator's second next-period shock from the block, so it must be
        # constructed rather than relying on another test to do it first.
        torch.manual_seed(TEST_SEED)
        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")
        u3_block.construct_shocks(u3_calibration, rng=np.random.default_rng(TEST_SEED))

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


class TestU3OneSidedConstraintTraining(unittest.TestCase):
    """Train U-3 specifically on the constrained (low-wealth) region and
    check the one-sided Euler-loss training preserves the Karush-Kuhn-Tucker
    sign condition.
    """

    def test_u3_one_sided_euler_loss_preserves_kkt_sign(self):
        """At low wealth the borrowing constraint c <= m binds, so the
        Euler residual must be non-negative: u'(c) - betaR E[u'(c')] >= 0.
        The constrained=True one-sided loss only penalizes negative
        residuals, so a converged policy should produce residuals that
        are weakly non-negative in the constrained region.
        """
        torch.manual_seed(TEST_SEED)

        # Get U-3 model components
        u3_block = get_benchmark_model("U-3")
        u3_calibration = get_benchmark_calibration("U-3")

        # Construct shocks for the block
        rng = np.random.default_rng(TEST_SEED)
        u3_block.construct_shocks(u3_calibration, rng=rng)

        bp = bellman.BellmanPeriod(u3_block, "DiscFac", u3_calibration)

        # Create initial states grid focused on LOW wealth (constrained region)
        states_0_n = grid.Grid.from_config(
            {
                "a": {"min": 0.1, "max": 1.0, "count": 15},
            }
        )

        # Create Euler equation loss with constrained=True
        euler_loss_fn = loss.EulerEquationLoss(
            bp,
            parameters=u3_calibration,
            constrained=True,
        )

        # Train the policy
        trained_net, _ = maliar.maliar_training_loop(
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

        # Create decision function from trained network
        decision_fn = trained_net.get_decision_function()

        # Test at very low wealth points where constraint should bind
        low_wealth_a = torch.tensor([0.05, 0.1, 0.2], device=device)
        low_wealth_states = {"a": low_wealth_a}
        low_wealth_shocks = {
            "psi": torch.ones(3, device=device),
            "theta": torch.ones(3, device=device),
        }

        # Sanity: trained policy must respect the borrowing constraint c <= m
        # even at the constrained boundary.
        c = decision_fn(low_wealth_states, low_wealth_shocks, u3_calibration)["c"]
        R = u3_calibration["R"]
        m = R * low_wealth_a / low_wealth_shocks["psi"] + low_wealth_shocks["theta"]
        self.assertTrue(
            torch.all(c > 0),
            f"Consumption should be positive at low wealth. Got c = {c}",
        )
        self.assertTrue(
            torch.all(c <= m + 1e-6),
            f"Consumption should respect borrowing constraint c <= m at low "
            f"wealth. Got c = {c}, m = {m}",
        )

        # Core claim: where c <= m binds, the Euler residual
        # u'(c) - betaR E[u'(c')] is weakly non-negative; the one-sided loss
        # leaves positive residuals unpenalized, so trained residuals stay >~ 0.
        constraint_shocks = {
            "psi_0": torch.ones(3, device=device),
            "psi_1": torch.ones(3, device=device),
            "theta_0": torch.ones(3, device=device),
            "theta_1": torch.ones(3, device=device),
        }
        residual = euler_residual_of(
            bp, decision_fn, low_wealth_states, constraint_shocks, u3_calibration
        )
        # Allow a small slack for NN approximation; the principled
        # statement is residual >= 0, so any negative tail of meaningful
        # magnitude is a real failure of the one-sided loss.
        min_residual = residual.min().item()
        self.assertGreater(
            min_residual,
            -1e-2,
            f"One-sided Euler loss should keep residuals non-negative in "
            f"the constrained region. Got min residual = {min_residual:.4e}.",
        )


class TestD4ConstrainedEulerVFI(unittest.TestCase):
    """Euler + Fischer-Burmeister training recovers the VFI solution to <1% on D-4.

    D-4 is a deterministic CRRA model with a binding borrowing constraint
    (c <= m) and impatience (betaR = 0.9568 < 1). Pure Euler training identifies
    consumption GROWTH but not its LEVEL (a level shift changes the Euler
    residual only by a term proportional to 1 - betaR). The binding borrowing
    constraint supplies the boundary condition c = m that anchors the level, so
    minimizing the Fischer-Burmeister Euler/KKT residual (constrained=True)
    matches an independent value-function-iteration oracle to under 1%.

    This is the in-package demonstration that the Euler method of Maliar,
    Maliar, and Winant (2021, JME) reaches benchmark accuracy on a constrained
    problem, using only the Euler residual and Fischer-Burmeister (no value
    head). The contrast with the unconstrained interior models (U-2 stalls near
    8% under Euler-only training) isolates the binding constraint as the level
    anchor.
    """

    def test_d4_euler_fb_matches_vfi_within_one_percent(self):
        torch.manual_seed(TEST_SEED)

        d4_block = get_benchmark_model("D-4")
        d4_calibration = get_benchmark_calibration("D-4")
        d4_reference = get_reference_policy("D-4")

        # D-4 is deterministic; construct_shocks is a no-op but keeps the
        # block-construction contract uniform with the stochastic models.
        d4_block.construct_shocks(d4_calibration)
        bp = bellman.BellmanPeriod(d4_block, "DiscFac", d4_calibration)

        # Policy-only network (Euler method, no value head).
        policy_net = BlockPolicyNet(bp, width=64, init_seed=TEST_SEED)
        euler_loss_fn = loss.EulerEquationLoss(
            bp, parameters=d4_calibration, constrained=True
        )

        R = d4_calibration["R"]
        y = d4_calibration["y"]
        # Train on arrival assets a so cash-on-hand m = R*a + y spans [1, 9],
        # covering the binding (low m) and slack (high m) regions. Fresh uniform
        # resampling each step is Maliar all-domain training, not a fixed grid.
        a_lo, a_hi = (1.0 - y) / R, (9.0 - y) / R
        # 5000 steps is past the knee: the mean gap clears 1% by ~2000 steps and
        # the pointwise max at the constraint kink settles to ~0.7-1.2%. Measured
        # here: mean = 0.30%, max = 0.83% vs the VFI oracle.
        optimizer = None
        for _ in range(5000):
            train_grid = grid.Grid.from_dict(
                {"a": torch.empty(256, device=device).uniform_(a_lo, a_hi)}
            )
            policy_net, _, optimizer = train_block_nn(
                policy_net,
                train_grid,
                euler_loss_fn,
                epochs=1,
                lr=1e-2,
                optimizer=optimizer,
                verbose=False,
            )

        decision_fn = policy_net.get_decision_function()
        test_a = torch.linspace(a_lo, a_hi, 80, device=device)
        test_states = {"a": test_a}

        # Compare against the VFI oracle through the same helper used for the
        # analytical-solution tests, so the constrained Euler method is held to
        # the identical comparison as every other solution method.
        rel_error = policy_rel_error_of(
            decision_fn, d4_reference, test_states, {}, d4_calibration
        )
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()

        # The binding constraint anchors the level, so Euler + Fischer-Burmeister
        # matches VFI to well under 1% on average; the pointwise max sits at the
        # constraint kink and is held to a looser but still tight bound.
        self.assertLess(
            mean_rel_error,
            0.01,
            f"Euler+FB should match VFI within 1% on average for the constrained "
            f"D-4 model. Got mean relative error: {mean_rel_error:.4%}",
        )
        self.assertLess(
            max_rel_error,
            0.02,
            f"Euler+FB pointwise gap to VFI should stay tight even at the "
            f"constraint kink. Got max relative error: {max_rel_error:.4%}",
        )


class TestEulerLossAllInOneOperator(unittest.TestCase):
    """EulerEquationLoss forms the all-in-one *product* of two independent
    next-period residual draws, never the square of a single draw.

    On a stochastic model the single-draw square estimates
    ``E[f**2] = (E[f])**2 + Var(f)``, biased upward by the shock variance, which
    shifts the minimizer. The product of two independent draws is unbiased,
    ``E[f_a f_b] = (E[f])**2`` (Maliar, Maliar, and Winant 2021, JME, all-in-one
    operator). These tests fail if the loss reverts to squaring a single draw.

    The fixture is U-3 with a *widened transitory* shock. The transitory shock
    theta is not absorbed by the permanent-income normalization, so the
    next-period residual has genuine variance even at a fixed policy; the
    permanent-only models (e.g. U-2) divide their shock out and leave
    ``Var(f) = 0``, so they cannot exhibit the bias and would make this test
    vacuous.
    """

    def _stochastic_u3(self):
        # Widen the transitory dispersion so Var(f) dominates (E[f])**2 and the
        # bias is unmistakable; theta is transitory, so this does not just
        # rescale. get_benchmark_model returns a fresh copy, so the override holds.
        cal = dict(get_benchmark_calibration("U-3"))
        cal["sigma_theta"] = 0.3
        block = get_benchmark_model("U-3")
        block.construct_shocks(cal, rng=np.random.default_rng(TEST_SEED))
        bp = bellman.BellmanPeriod(block, "DiscFac", cal)
        return bp, block, cal

    @staticmethod
    def _half_cash_on_hand(states, shocks, parameters):
        # Fixed interior policy c = 0.5 m (0 < c < m). Not optimal, which is the
        # point: it leaves a nonzero, shock-varying Euler residual to estimate,
        # with no training required.
        a = states["a"]
        R = parameters["R"]
        psi = shocks.get("psi", torch.ones_like(a))
        theta = shocks.get("theta", torch.ones_like(a))
        m = R * a / torch.clamp(psi, min=1e-8) + theta
        return {"c": 0.5 * m}

    def test_two_evaluations_differ_independent_second_draw(self):
        """Two evaluations on the *same* grid differ, because the loss draws an
        independent second next-period residual internally. A single-draw square
        is deterministic given the grid, so identical outputs would betray it."""
        torch.manual_seed(TEST_SEED)
        bp, _, cal = self._stochastic_u3()
        n = 1024
        ones = torch.ones(n, dtype=torch.float64)
        g = grid.Grid.from_dict(
            {
                "a": torch.linspace(0.5, 5.0, n, dtype=torch.float64),
                "psi_0": ones,
                "theta_0": ones,
                "psi_1": ones,
                "theta_1": ones,
            }
        )
        loss_fn = loss.EulerEquationLoss(bp, parameters=cal)
        out1 = loss_fn(self._half_cash_on_hand, g)
        out2 = loss_fn(self._half_cash_on_hand, g)
        self.assertFalse(
            torch.allclose(out1, out2),
            "EulerEquationLoss must form the all-in-one product with an "
            "independently drawn second residual. Identical outputs across two "
            "calls on a stochastic model mean it is squaring a single draw, "
            "which is biased by Var(f).",
        )

    def test_all_in_one_loss_is_unbiased_below_single_draw_square(self):
        """Averaged over independent next-period draws, the all-in-one loss
        estimates the unbiased ``(E f)**2``, which sits strictly below the
        biased single-draw square ``E[f**2] = (E f)**2 + Var(f)``. Reverting the
        loss to a single-draw square would push it up to the biased value and
        break the bound."""
        torch.manual_seed(TEST_SEED)
        bp, block, cal = self._stochastic_u3()
        n = 1024
        ones = torch.ones(n, dtype=torch.float64)
        states = {"a": torch.linspace(0.5, 5.0, n, dtype=torch.float64)}
        loss_fn = loss.EulerEquationLoss(bp, parameters=cal)
        df = self._half_cash_on_hand

        n_draws = 200
        aio_total = 0.0
        biased_total = 0.0
        for _ in range(n_draws):
            drawn = draw_shocks(block.shocks, n=n)
            shocks = {
                "psi_0": ones,
                "theta_0": ones,
                "psi_1": reconcile(states["a"], drawn["psi"]).to(torch.float64),
                "theta_1": reconcile(states["a"], drawn["theta"]).to(torch.float64),
            }
            # All-in-one: the grid draw is residual A, the loss self-draws B.
            input_grid = grid.Grid.from_dict({"a": states["a"], **shocks})
            aio_total += loss_fn(df, input_grid).mean().item()
            # Biased comparison: square the single residual at the same draw.
            residual = bellman.estimate_euler_residual(bp, df, states, shocks, cal)
            biased_total += (next(iter(residual.values())).detach() ** 2).mean().item()
        aio_avg = aio_total / n_draws
        biased_avg = biased_total / n_draws

        self.assertGreater(
            biased_avg,
            1e-2,
            "Test would be vacuous without appreciable residual variance; "
            f"biased E[f**2] = {biased_avg:.4e} is too small to separate the "
            "two estimators.",
        )
        self.assertLess(
            aio_avg,
            0.7 * biased_avg,
            "The all-in-one product estimates the unbiased (E f)**2 and must "
            "sit well below the biased single-draw square E[f**2]. Got "
            f"AiO = {aio_avg:.4e}, biased = {biased_avg:.4e} "
            f"(ratio {aio_avg / biased_avg:.2f}); a ratio near 1 means the loss "
            "is squaring a single draw.",
        )


class TestConstrainedWarning(unittest.TestCase):
    """Test that EulerEquationLoss warns on misuse of constrained=True."""

    def test_constrained_warns_without_upper_bound(self):
        """constrained=True should warn when no Control has an upper_bound."""

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
        # case_4 Control(["g", "m"]): policy returns c given g, m
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


class TestMaliarTrainingLoopValidation(unittest.TestCase):
    """Test input validation in maliar_training_loop."""

    def setUp(self):
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)

        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        self.bp = case_4["bp"]
        self.states = grid.Grid.from_config(
            {
                "m": {"min": 0, "max": 5, "count": 3},
                "g": {"min": 0, "max": 5, "count": 3},
            }
        )
        self.loss_fn = loss.EstimatedDiscountedLifetimeRewardLoss(
            self.bp, 2, case_4["calibration"]
        )
        self.calibration = case_4["calibration"]

    def test_max_iterations_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                max_iterations=0,
                random_seed=TEST_SEED,
            )

    def test_max_iterations_negative_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                max_iterations=-5,
                random_seed=TEST_SEED,
            )

    def test_tolerance_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                tolerance=0,
                random_seed=TEST_SEED,
            )

    def test_tolerance_negative_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                tolerance=-1e-6,
                random_seed=TEST_SEED,
            )

    def test_lr_zero_raises(self):
        # lr=0 would silently produce a no-learning run; it must be rejected.
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                lr=0.0,
                random_seed=TEST_SEED,
            )

    def test_lr_negative_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                lr=-0.001,
                random_seed=TEST_SEED,
            )

    def test_shock_copies_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                shock_copies=0,
                random_seed=TEST_SEED,
            )

    def test_simulation_steps_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                simulation_steps=0,
                random_seed=TEST_SEED,
            )

    def test_network_width_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                network_width=0,
                random_seed=TEST_SEED,
            )

    def test_epochs_per_iteration_zero_raises(self):
        with self.assertRaises(ValueError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                self.calibration,
                epochs_per_iteration=0,
                random_seed=TEST_SEED,
            )

    def test_none_bellman_period_raises(self):
        with self.assertRaises(TypeError):
            maliar.maliar_training_loop(
                None,
                self.loss_fn,
                self.states,
                self.calibration,
                random_seed=TEST_SEED,
            )

    def test_non_callable_loss_raises(self):
        with self.assertRaises(TypeError):
            maliar.maliar_training_loop(
                self.bp,
                "not_a_function",
                self.states,
                self.calibration,
                random_seed=TEST_SEED,
            )

    def test_none_parameters_raises(self):
        with self.assertRaises(TypeError):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                self.states,
                None,
                random_seed=TEST_SEED,
            )

    def test_non_grid_states_raises(self):
        """Passing a dict instead of Grid for states_0_n should raise TypeError."""
        with self.assertRaises(TypeError, msg="Grid instance"):
            maliar.maliar_training_loop(
                self.bp,
                self.loss_fn,
                {"m": torch.tensor([1.0]), "g": torch.tensor([1.0])},
                self.calibration,
                random_seed=TEST_SEED,
            )


class TestMaliarHyperparameters(unittest.TestCase):
    """Test that network_width and epochs_per_iteration affect training."""

    def setUp(self):
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        rng = np.random.default_rng(TEST_SEED)
        case_4["block"].construct_shocks(case_4["calibration"], rng=rng)

        self.bp = case_4["bp"]
        self.states = grid.Grid.from_config(
            {
                "m": {"min": 0, "max": 5, "count": 3},
                "g": {"min": 0, "max": 5, "count": 3},
            }
        )
        self.loss_fn = loss.EstimatedDiscountedLifetimeRewardLoss(
            self.bp, 2, case_4["calibration"]
        )
        self.calibration = case_4["calibration"]

    def test_network_width_affects_parameter_count(self):
        net_narrow, _ = maliar.maliar_training_loop(
            self.bp,
            self.loss_fn,
            self.states,
            self.calibration,
            network_width=8,
            max_iterations=1,
            random_seed=TEST_SEED,
        )
        net_wide, _ = maliar.maliar_training_loop(
            self.bp,
            self.loss_fn,
            self.states,
            self.calibration,
            network_width=32,
            max_iterations=1,
            random_seed=TEST_SEED,
        )

        params_narrow = sum(p.numel() for p in net_narrow.parameters())
        params_wide = sum(p.numel() for p in net_wide.parameters())
        self.assertGreater(
            params_wide,
            params_narrow,
            "Wider network should have more parameters",
        )


class TestCheckConvergence(unittest.TestCase):
    """Test _check_convergence helper."""

    def test_param_convergence(self):
        """Convergence when parameter diff < tolerance."""
        params = torch.tensor([1.0, 2.0])
        converged, pdiff, ldiff, pc, lc = maliar._check_convergence(
            params,
            params,
            tolerance=1e-6,
            prev_loss=None,
            current_loss=0.1,
        )
        self.assertTrue(converged)
        self.assertTrue(pc)
        self.assertFalse(lc)
        self.assertAlmostEqual(pdiff, 0.0)
        self.assertIsNone(ldiff)

    def test_loss_convergence(self):
        """Convergence when loss diff < tolerance."""
        p1 = torch.tensor([1.0])
        p2 = torch.tensor([2.0])
        converged, pdiff, ldiff, pc, lc = maliar._check_convergence(
            p1,
            p2,
            tolerance=1e-6,
            prev_loss=0.5,
            current_loss=0.5,
        )
        self.assertTrue(converged)
        self.assertFalse(pc)
        self.assertTrue(lc)
        self.assertAlmostEqual(ldiff, 0.0)

    def test_no_convergence(self):
        """No convergence when both diffs exceed tolerance."""
        p1 = torch.tensor([1.0])
        p2 = torch.tensor([2.0])
        converged, pdiff, ldiff, pc, lc = maliar._check_convergence(
            p1,
            p2,
            tolerance=1e-6,
            prev_loss=1.0,
            current_loss=0.5,
        )
        self.assertFalse(converged)
        self.assertFalse(pc)
        self.assertFalse(lc)

    def test_tensor_loss_raises_type_error(self):
        """Passing an un-.item()-ed torch.Tensor as current_loss raises TypeError."""
        params = torch.tensor([1.0])
        tensor_loss = torch.tensor(0.5)  # forgot .item()
        with self.assertRaises(TypeError) as ctx:
            maliar._check_convergence(
                params,
                params,
                tolerance=1e-6,
                prev_loss=None,
                current_loss=tensor_loss,
            )
        self.assertIn("scalar", str(ctx.exception))


class TestLogIteration(unittest.TestCase):
    """Test _log_iteration helper."""

    def test_converged_by_params(self):
        """Convergence log should report only the triggering criterion."""

        with self.assertLogs(level=logging.INFO) as cm:
            maliar._log_iteration(
                converged=True,
                iteration=2,
                param_diff=1e-7,
                loss_diff=0.5,
                current_loss=0.01,
                param_converged=True,
                loss_converged=False,
            )
        self.assertIn("Converged after 3 iterations", cm.output[0])
        self.assertIn("parameters", cm.output[0])
        self.assertNotIn("loss", cm.output[0])

    def test_converged_by_loss(self):
        """Convergence by loss should report loss, not parameters."""

        with self.assertLogs(level=logging.INFO) as cm:
            maliar._log_iteration(
                converged=True,
                iteration=1,
                param_diff=0.5,
                loss_diff=1e-8,
                current_loss=0.01,
                param_converged=False,
                loss_converged=True,
            )
        self.assertIn("loss", cm.output[0])
        self.assertNotIn("parameters", cm.output[0])

    def test_log_shows_loss_when_provided(self):
        """Log should include loss when current_loss is supplied."""

        with self.assertLogs(level=logging.INFO) as cm:
            maliar._log_iteration(
                converged=False,
                iteration=0,
                param_diff=1e-3,
                loss_diff=None,
                current_loss=0.05,
            )
        self.assertIn("loss=", cm.output[0])

    def test_both_criteria_converged(self):
        """Log message includes both criteria when both trigger simultaneously."""

        with self.assertLogs(level=logging.INFO) as cm:
            maliar._log_iteration(
                converged=True,
                iteration=3,
                param_diff=1e-8,
                loss_diff=1e-9,
                current_loss=0.001,
                param_converged=True,
                loss_converged=True,
            )
        self.assertIn("parameters", cm.output[0])
        self.assertIn("loss", cm.output[0])


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


class TestValidationTypeChecks(unittest.TestCase):
    """Test integer type validation for training parameters (S3)."""

    def test_float_max_iterations_raises(self):
        """Float max_iterations should raise TypeError."""
        block = model.DBlock(
            name="test",
            dynamics={"c": model.Control(["a"]), "a": lambda a, c: a - c},
            reward={"a": "consumer"},
        )
        block.construct_shocks({})
        bp = bellman.BellmanPeriod(block, "beta", {"beta": 0.9})
        states = grid.Grid.from_config({"a": {"min": 1.0, "max": 2.0, "count": 5}})

        with self.assertRaises(TypeError, msg="must be an integer"):
            maliar.maliar_training_loop(
                bp,
                lambda df, g: torch.tensor(0.0),
                states,
                {"beta": 0.9},
                max_iterations=1.5,
            )


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

    def test_non_callable_value_function_raises(self):
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
        with self.assertRaises(
            TypeError, msg="value_function must be a callable or a dict"
        ):
            loss.BellmanEquationLoss(bp, value_function="not_callable")

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
                bp,
                value_function=lambda s, sh, p: s["a"],
                foc_weight=-0.5,
            )
