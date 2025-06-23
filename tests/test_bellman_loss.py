import unittest
import torch

from conftest import case_0, case_1
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.models.perfect_foresight as pfm


class TestBellmanLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_bellman_loss_function_creation(self):
        """Test that the Bellman loss function can be created successfully."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        loss_fn = maliar.get_bellman_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        self.assertTrue(callable(loss_fn))

    def test_bellman_training_grid_generation(self):
        """Test generation of training grid with 2 shock copies."""
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}

        # Need to construct shocks for the block first
        case_0["block"].construct_shocks(case_0["calibration"])

        training_grid = maliar.generate_bellman_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Should have state variable 'a' and any shock variables with _0 and _1 suffixes
        grid_dict = training_grid.to_dict()
        self.assertIn("a", grid_dict)

        # Check that we have 5 samples as specified in state_config
        self.assertEqual(len(grid_dict["a"]), 5)

    def test_bellman_loss_computation(self):
        """Test that Bellman loss can be computed without errors."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create networks
        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)
        value_net = ann.BlockValueNet(case_0["block"], state_variables, width=8)

        # Create loss function
        loss_fn = maliar.get_bellman_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}
        case_0["block"].construct_shocks(case_0["calibration"])
        training_grid = maliar.generate_bellman_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Compute loss
        loss = loss_fn(
            policy_net.get_decision_function(),
            value_net.get_value_function(),
            training_grid,
        )

        # Should return a tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (5,))  # One loss per grid point
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_bellman_training_loop(self):
        """Test that Bellman training can run without errors."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_0["calibration"]

        # Create networks
        policy_net = ann.BlockPolicyNet(case_0["block"], width=8)
        value_net = ann.BlockValueNet(case_0["block"], state_variables, width=8)

        # Create loss function
        loss_fn = maliar.get_bellman_residual_loss(
            state_variables, case_0["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 2, "count": 5}}
        case_0["block"].construct_shocks(case_0["calibration"])
        training_grid = maliar.generate_bellman_training_grid(
            state_config, case_0["block"], n_samples=5, parameters=case_0["calibration"]
        )

        # Train for a few epochs (smoke test)
        trained_policy, trained_value = ann.train_bellman_nets(
            policy_net, value_net, training_grid, loss_fn, epochs=10
        )

        self.assertIsInstance(trained_policy, ann.BlockPolicyNet)
        self.assertIsInstance(trained_value, ann.BlockValueNet)

    def test_bellman_loss_with_perfect_foresight_model(self):
        """Test Bellman loss with a more realistic model."""
        # Use perfect foresight model
        block = pfm.block_no_shock  # No shocks version for simplicity
        state_variables = ["a", "p"]
        discount_factor = 0.96
        parameters = pfm.calibration

        # Create networks
        policy_net = ann.BlockPolicyNet(block, width=8)
        value_net = ann.BlockValueNet(block, state_variables, width=8)

        # Create loss function
        loss_fn = maliar.get_bellman_residual_loss(
            state_variables, block, discount_factor, parameters
        )

        # Create training grid
        state_config = {
            "a": {"min": 0.1, "max": 5.0, "count": 3},
            "p": {"min": 0.5, "max": 1.5, "count": 3},
        }

        training_grid = maliar.generate_bellman_training_grid(
            state_config,
            block,
            n_samples=9,
            parameters=parameters,  # 3x3 grid
        )

        # Compute loss
        loss = loss_fn(
            policy_net.get_decision_function(),
            value_net.get_value_function(),
            training_grid,
        )

        # Should return finite values
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (9,))  # 3x3 = 9 grid points
        self.assertTrue(torch.all(torch.isfinite(loss)))

    def test_case_1_with_shocks(self):
        """Test Bellman loss with a model that has shocks."""
        state_variables = ["a"]
        discount_factor = 0.9
        parameters = case_1["calibration"]

        # Construct shocks
        case_1["block"].construct_shocks(parameters)

        # Create networks
        policy_net = ann.BlockPolicyNet(case_1["block"], width=8)
        value_net = ann.BlockValueNet(case_1["block"], state_variables, width=8)

        # Create loss function
        loss_fn = maliar.get_bellman_residual_loss(
            state_variables, case_1["block"], discount_factor, parameters
        )

        # Create training grid
        state_config = {"a": {"min": 0, "max": 1, "count": 3}}
        training_grid = maliar.generate_bellman_training_grid(
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
        loss = loss_fn(
            policy_net.get_decision_function(),
            value_net.get_value_function(),
            training_grid,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.all(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()
