import torch
import pytest
from skagent.ann import Net

# Get device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestNet:
    def setup_method(self):
        self.n_inputs = 5
        self.n_outputs = 3
        self.width = 16
        self.n_layers = 3

        # Create test input data on the same device as the network
        self.test_input = torch.randn(10, self.n_inputs).to(device)

    @pytest.mark.parametrize("activation", ["silu", "relu", "tanh", "sigmoid"])
    def test_string_activation(self, activation):
        """Test Net with string activation."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=activation,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        assert len(net.activations) == self.n_layers
        # All layers should have the same activation
        for i in range(self.n_layers):
            assert net.activations[i] == net.activations[0]

    def test_list_activation(self):
        """Test Net with list of activations."""
        activation_list = ["relu", "tanh", "silu"]
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=activation_list,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        assert len(net.activations) == self.n_layers
        # Each layer should have different activation
        assert net.activations[0] != net.activations[1]
        assert net.activations[1] != net.activations[2]

    def test_activation_list_length_mismatch(self):
        """Test that wrong number of activations raises error."""
        with pytest.raises(ValueError):
            Net(
                self.n_inputs,
                self.n_outputs,
                width=self.width,
                n_layers=3,
                activation=["relu", "tanh"],  # Only 2 activations for 3 layers
            )

    @pytest.mark.parametrize("transform", ["sigmoid", "exp", "tanh", "relu"])
    def test_string_transform(self, transform):
        """Test Net with string transform."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            transform=transform,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)

        # Check that transformation is applied correctly
        if transform == "sigmoid":
            assert torch.all(output >= 0) and torch.all(output <= 1)
        elif transform == "exp":
            assert torch.all(output > 0)
        elif transform == "relu":
            assert torch.all(output >= 0)

    def test_list_transform(self):
        """Test Net with list of transforms."""
        transform_list = ["sigmoid", "exp", "tanh"]
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            transform=transform_list,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)

        # Check that different transforms are applied to different outputs
        # First output should be sigmoid (0 to 1)
        assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 1)
        # Second output should be exp (positive)
        assert torch.all(output[:, 1] > 0)
        # Third output should be tanh (-1 to 1)
        assert torch.all(output[:, 2] >= -1) and torch.all(output[:, 2] <= 1)

    def test_transform_list_length_mismatch(self):
        """Test that wrong number of transforms raises error."""
        with pytest.raises(ValueError):
            net = Net(
                self.n_inputs,
                self.n_outputs,
                width=self.width,
                n_layers=self.n_layers,
                transform=["sigmoid", "exp"],  # Only 2 transforms for 3 outputs
            )
            net(self.test_input)  # Error should occur on forward pass

    def test_no_transform(self):
        """Test Net with no transform."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            transform=None,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        # No constraints on output values since no transform applied

    def test_softmax_transform(self):
        """Test Net with softmax transform."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            transform="softmax",
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)

        # Check that softmax properties hold
        assert torch.all(output >= 0)  # All values non-negative
        assert torch.all(output <= 1)  # All values <= 1
        # Check that each row sums to 1 (within tolerance)
        row_sums = torch.sum(output, dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_combined_activation_and_transform(self):
        """Test Net with both list activation and list transform."""
        activation_list = ["relu", "tanh", "silu"]
        transform_list = ["sigmoid", "exp", "identity"]

        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=activation_list,
            transform=transform_list,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)

        # Check that transforms are applied correctly
        assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 1)  # sigmoid
        assert torch.all(output[:, 1] > 0)  # exp
        # Third output has identity transform, so no constraints

    def test_unsupported_activation(self):
        """Test that unsupported activation raises error."""
        with pytest.raises(ValueError):
            Net(self.n_inputs, self.n_outputs, activation="unsupported_activation")

    def test_unsupported_transform(self):
        """Test that unsupported transform raises error."""
        with pytest.raises(ValueError):
            net = Net(self.n_inputs, self.n_outputs, transform="unsupported_transform")
            net(self.test_input)  # Error should occur on forward pass

    def test_callable_activation(self):
        """Test Net with callable activation."""

        def custom_activation(x, **kwargs):
            return torch.sin(x)  # Custom sine activation

        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=custom_activation,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        # All layers should use the same custom activation
        for i in range(self.n_layers):
            assert net.activations[i] == custom_activation

    def test_none_activation(self):
        """Test Net with None activation (identity)."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=None,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        # All layers should use identity activation (None for optimization)
        for activation_fn in net.activations:
            assert activation_fn is None, (
                "Identity activations should be None for performance"
            )

    def test_identity_activation(self):
        """Test Net with 'identity' string activation."""
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation="identity",
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        # All layers should use identity activation (None for optimization)
        for activation_fn in net.activations:
            assert activation_fn is None, (
                "Identity activations should be None for performance"
            )

    def test_list_activation_with_callable_and_none(self):
        """Test Net with list of activations including callable and None."""

        def custom_activation(x, **kwargs):
            return torch.cos(x)

        activation_list = ["relu", custom_activation, None]
        net = Net(
            self.n_inputs,
            self.n_outputs,
            width=self.width,
            n_layers=self.n_layers,
            activation=activation_list,
        )
        output = net(self.test_input)
        assert output.shape == (10, self.n_outputs)
        assert len(net.activations) == self.n_layers

        # First should be ReLU
        assert net.activations[0] == torch.nn.functional.relu
        # Second should be custom function
        assert net.activations[1] == custom_activation
        # Third should be None (optimized identity)
        assert net.activations[2] is None, (
            "Identity activation should be None for performance"
        )

    def test_identity_activation_optimization(self):
        """Test that identity/None activations are optimized to skip function calls."""
        # Test identity activation
        net = Net(2, 1, width=4, n_layers=2, activation="identity")
        assert all(net.activation_is_identity), (
            "All layers should be marked as identity"
        )
        assert all(act is None for act in net.activations), (
            "Identity activations should be None"
        )

        # Test None activation
        net = Net(2, 1, width=4, n_layers=2, activation=None)
        assert all(net.activation_is_identity), (
            "All layers should be marked as identity"
        )
        assert all(act is None for act in net.activations), (
            "None activations should be None"
        )

        # Test mixed activations
        net = Net(2, 1, width=4, n_layers=3, activation=["relu", "identity", None])
        expected_identity = [False, True, True]
        assert net.activation_is_identity == expected_identity
        assert net.activations[0] is not None  # relu
        assert net.activations[1] is None  # identity
        assert net.activations[2] is None  # None

        # Verify forward pass still works - ensure tensor is on same device as network
        x = torch.randn(5, 2).to(net.device)
        output = net(x)
        assert output.shape == (5, 1)

    def test_softmax_as_regular_transform(self):
        """Test that softmax works as a regular string transform option."""
        # Test softmax as single transform
        net = Net(2, 3, transform="softmax")
        x = torch.randn(5, 2).to(net.device)
        output = net(x)

        # Verify softmax properties: outputs sum to 1 and are all positive
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(5).to(net.device), atol=1e-6
        )
        assert torch.all(output >= 0)
        assert output.shape == (5, 3)

        # Test softmax in list of transforms
        net = Net(2, 3, transform=["sigmoid", "softmax", "exp"])
        x = torch.randn(5, 2).to(net.device)
        output = net(x)

        # First output should be sigmoid (0,1)
        assert torch.all(output[..., 0] >= 0) and torch.all(output[..., 0] <= 1)

        # Second output should be softmax applied to that single value (still valid)
        assert torch.all(output[..., 1] >= 0) and torch.all(output[..., 1] <= 1)

        # Third output should be exp (positive)
        assert torch.all(output[..., 2] > 0)

        assert output.shape == (5, 3)

    def test_parameter_consistency_string_example(self):
        """Test that activation and transform parameters work consistently with strings."""
        net = Net(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation="relu",
            transform="sigmoid",
        )

        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        assert output.shape == (2, 3)
        # Sigmoid transform should ensure all outputs are in [0,1]
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_parameter_consistency_list_example(self):
        """Test that activation and transform parameters work consistently with lists."""
        net = Net(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation=["relu", "tanh", "silu"],
            transform=["sigmoid", "exp", "tanh"],
        )

        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        assert output.shape == (2, 3)
        # First output: sigmoid transform [0,1]
        assert torch.all(output[..., 0] >= 0) and torch.all(output[..., 0] <= 1)
        # Second output: exp transform (positive)
        assert torch.all(output[..., 1] > 0)
        # Third output: tanh transform [-1,1]
        assert torch.all(output[..., 2] >= -1) and torch.all(output[..., 2] <= 1)

    def test_parameter_consistency_callable_example(self):
        """Test that activation and transform parameters work consistently with callables."""

        def custom_activation(x):
            """Custom activation function: leaky ReLU with slope 0.1"""
            return torch.where(x > 0, x, 0.1 * x)

        def custom_transform(x):
            """Custom transform function: scale and shift"""
            return 2 * x + 1

        net = Net(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation=custom_activation,
            transform=custom_transform,
        )

        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        assert output.shape == (2, 3)
        # Verify custom transform applied: output should be 2*x + 1
        # So minimum should be at least 1 (when pre-transform was 0)
        # This is a basic sanity check rather than exact verification

    def test_parameter_consistency_none_identity_example(self):
        """Test that activation and transform parameters work consistently with None."""
        net = Net(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation=None,
            transform=None,
        )

        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        assert output.shape == (2, 3)
        # Verify identity optimizations are in place
        assert all(net.activation_is_identity)
        assert all(act is None for act in net.activations)

    def test_parameter_consistency_mixed_list_example(self):
        """Test that activation and transform parameters work consistently with mixed lists."""

        def custom_activation(x):
            """Custom activation function: leaky ReLU with slope 0.1"""
            return torch.where(x > 0, x, 0.1 * x)

        def custom_transform(x):
            """Custom transform function: scale and shift"""
            return 2 * x + 1

        net = Net(
            n_inputs=5,
            n_outputs=3,
            n_layers=3,
            activation=["relu", custom_activation, None],
            transform=["sigmoid", custom_transform, None],
        )

        test_input = torch.randn(2, 5).to(net.device)
        output = net(test_input)

        assert output.shape == (2, 3)

        # Verify activation handling
        assert net.activations[0] == torch.nn.functional.relu  # relu
        assert net.activations[1] == custom_activation  # custom
        assert net.activations[2] is None  # identity/None

        # Verify identity optimization flags
        expected_identity = [False, False, True]
        assert net.activation_is_identity == expected_identity

        # First output: sigmoid transform [0,1]
        assert torch.all(output[..., 0] >= 0) and torch.all(output[..., 0] <= 1)
        # Second output: custom transform applied
        # Third output: no transform applied

    def test_parameter_consistency_comprehensive(self):
        """Comprehensive test ensuring both parameters support identical capabilities."""
        # Test all supported types work for both parameters
        supported_types = [
            ("str", "relu", "sigmoid"),
            ("list", ["relu", "tanh"], ["sigmoid", "exp"]),
            (
                "callable",
                lambda x, **kwargs: torch.sin(x),
                lambda x, **kwargs: torch.abs(x),
            ),
            ("None", None, None),
        ]

        for type_name, activation, transform in supported_types:
            # Adjust outputs for list case
            n_outputs = 2 if isinstance(transform, list) else 3

            net = Net(
                n_inputs=4,
                n_outputs=n_outputs,
                n_layers=2,
                activation=activation,
                transform=transform,
            )

            test_input = torch.randn(3, 4).to(net.device)
            output = net(test_input)

            assert output.shape == (3, n_outputs), f"Failed for {type_name} case"
            assert torch.all(torch.isfinite(output)), (
                f"Non-finite outputs for {type_name} case"
            )
