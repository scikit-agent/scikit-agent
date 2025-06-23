import unittest
import torch

from conftest import case_0, case_1
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.models.perfect_foresight as pfm


class TestEulerLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_euler_loss_function_creation(self):
        """Test that the Euler loss function can be created successfully."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        loss_fn = maliar.get_euler_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        self.assertTrue(callable(loss_fn))

    def test_euler_training_grid_generation(self):
        """Test generation of training grid with 2 shock copies for Euler method."""
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}

        # Need to construct shocks for the block first
        case_0["block"].construct_shocks(case_0["calibration"])

        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Should have state variable 'a' and any shock variables with _0 and _1 suffixes
        grid_dict = training_grid.to_dict()
        self.assertIn("a", grid_dict)

        # Check that we have 5 samples as specified in state_config
        self.assertEqual(len(grid_dict["a"]), 5)

    def test_euler_grid_alias_consistency(self):
        """Test that Euler grid generation is consistent with Bellman (both use 2 shocks)."""
        state_config = {"a": {"min": 0, "max": 2, "count": 3}}
        case_0["block"].construct_shocks(case_0["calibration"])

        # Generate grids using both methods
        euler_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=3, parameters=case_0["calibration"]
        )
        bellman_grid = maliar.generate_bellman_training_grid(
            state_config, case_0["block"], n_samples=3, parameters=case_0["calibration"]
        )

        # Both should have identical structure (same variable names and shapes)
        euler_dict = euler_grid.to_dict()
        bellman_dict = bellman_grid.to_dict()

        self.assertEqual(set(euler_dict.keys()), set(bellman_dict.keys()))

        for key in euler_dict:
            self.assertEqual(euler_dict[key].shape, bellman_dict[key].shape)

    def test_euler_loss_computation(self):
        """Test that Euler loss can be computed without errors."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create policy network
        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)

        # Create loss function
        loss_fn = maliar.get_euler_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}
        case_0["block"].construct_shocks(case_0["calibration"])
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Compute loss
        loss = loss_fn(policy_net.get_decision_function(), training_grid)

        # Should return a tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (5,))  # One loss per grid point
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_euler_training_loop(self):
        """Test that Euler training can run without errors."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create policy network
        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)

        # Create loss function
        loss_fn = maliar.get_euler_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}
        case_0["block"].construct_shocks(case_0["calibration"])
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Train for a few epochs (smoke test)
        trained_policy = ann.train_block_policy_nn(
            policy_net, training_grid, loss_fn, epochs=10
        )

        self.assertIsInstance(trained_policy, ann.BlockPolicyNet)

    def test_euler_loss_with_fischer_burmeister(self):
        """Test Euler loss with Fischer-Burmeister constraint handling."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create policy network
        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)

        # Create loss function with Fischer-Burmeister constraints
        loss_fn = maliar.get_euler_residual_loss(
            state_variables,
            case_0["block"],
            discount_factor,
            parameters,
            use_fischer_burmeister=True,
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 2, "count": 3}}
        case_0["block"].construct_shocks(case_0["calibration"])
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=3, parameters=case_0["calibration"]
        )

        # Compute loss
        loss = loss_fn(policy_net.get_decision_function(), training_grid)

        # Should return finite values
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_euler_loss_with_perfect_foresight_model(self):
        """Test Euler loss with a more realistic model."""
        # Use perfect foresight model
        block = pfm.block_no_shock  # No shocks version for simplicity
        state_variables = ["a", "p"]
        discount_factor = 0.96
        parameters = pfm.calibration

        # Create policy network
        policy_net = ann.BlockPolicyNet(block, width=8)

        # Create loss function
        loss_fn = maliar.get_euler_residual_loss(
            state_variables, block, discount_factor, parameters
        )

        # Create training grid
        state_config = {
            "a": {"min": 0.1, "max": 5.0, "count": 3},
            "p": {"min": 0.5, "max": 1.5, "count": 3},
        }

        training_grid = maliar.generate_euler_training_grid(
            state_config,
            block,
            n_samples=9,
            parameters=parameters,  # 3x3 grid
        )

        # Compute loss
        loss = loss_fn(policy_net.get_decision_function(), training_grid)

        # Should return finite values
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (9,))  # 3x3 = 9 grid points
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_case_1_with_shocks(self):
        """Test Euler loss with a model that has shocks."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_1["calibration"]

        # Construct shocks
        case_1["block"].construct_shocks(parameters)

        # Create policy network
        policy_net = ann.BlockPolicyNet(case_1["block"], width=8)

        # Create loss function
        loss_fn = maliar.get_euler_residual_loss(
            state_variables, case_1["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 1, "count": 3}}
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_1["block"], n_samples=3, parameters=parameters
        )

        # Check that we have shock variables in the grid
        grid_dict = training_grid.to_dict()
        self.assertIn("a", grid_dict)

        # Should have theta_0 and theta_1 for the two time periods
        shock_vars = [k for k in grid_dict.keys() if "theta" in k]
        self.assertTrue(any("_0" in var for var in shock_vars))
        self.assertTrue(any("_1" in var for var in shock_vars))

        # Compute loss
        loss = loss_fn(policy_net.get_decision_function(), training_grid)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_euler_vs_bellman_api_consistency(self):
        """Test that Euler and Bellman APIs are consistent."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Both should accept the same parameters
        euler_loss_fn = maliar.get_euler_residual_loss(
            state_variables,
            case_0["block"],
            discount_factor,
            parameters,
            use_fischer_burmeister=True,
        )

        bellman_loss_fn = maliar.get_bellman_residual_loss(
            state_variables,
            case_0["block"],
            discount_factor,
            parameters,
            use_fischer_burmeister=True,
        )

        # Both should be callable
        self.assertTrue(callable(euler_loss_fn))
        self.assertTrue(callable(bellman_loss_fn))

        # Test with same grid
        case_0["block"].construct_shocks(case_0["calibration"])
        state_config = {"a": {"min": 0, "max": 1, "count": 2}}
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=2, parameters=case_0["calibration"]
        )

        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)
        value_net = ann.BlockValueNet(case_0["block"], state_variables, width=8)

        # Euler should work with just policy function
        euler_loss = euler_loss_fn(policy_net.get_decision_function(), training_grid)

        # Bellman should work with both policy and value functions
        bellman_loss = bellman_loss_fn(
            policy_net.get_decision_function(),
            value_net.get_value_function(),
            training_grid,
        )

        # Both should return tensors of same shape
        self.assertEqual(euler_loss.shape, bellman_loss.shape)

    def test_marginal_utility_computation(self):
        """Test that marginal utility computation works correctly."""
        # This is a unit test for the core Euler computation
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create a simple policy network
        policy_net = ann.BlockPolicyNet(case_0["block"], width=4)

        # Create loss function
        loss_fn = maliar.get_euler_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        # Create minimal training grid
        case_0["block"].construct_shocks(case_0["calibration"])
        state_config = {"a": {"min": 1, "max": 1, "count": 1}}  # Single point
        training_grid = maliar.generate_euler_training_grid(
            state_config, case_0["block"], n_samples=1, parameters=case_0["calibration"]
        )

        # This should not raise errors and should produce finite gradients
        with torch.enable_grad():
            loss = loss_fn(policy_net.get_decision_function(), training_grid)
            self.assertTrue(torch.isfinite(loss).all())

            # Check that gradients can be computed
            loss_scalar = loss.mean()
            loss_scalar.backward()

            # Network should have gradients
            param_gradients = [
                p.grad for p in policy_net.parameters() if p.grad is not None
            ]
            self.assertTrue(len(param_gradients) > 0)

            # Gradients should be finite
            for grad in param_gradients:
                self.assertTrue(torch.isfinite(grad).all())


if __name__ == "__main__":
    unittest.main()
