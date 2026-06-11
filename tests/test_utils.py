import unittest

import skagent.utils as utils
import torch
from skagent.utils import compute_gradients_for_tensors


def test_utils_apply_fun_to_vals():
    def pow(x, y):
        return x**y

    vals = {"x": 2, "y": 3}
    a = utils.apply_fun_to_vals(pow, vals)
    assert a == 8


def test_extract_parameters():
    """Test parameter extraction from neural networks."""
    # Create a simple network
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
    )

    # Extract parameters
    params = utils.extract_parameters(net)

    # Should be a 1D tensor with all parameters flattened
    assert isinstance(params, torch.Tensor)
    assert params.dim() == 1

    # Expected number of parameters: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
    expected_params = (2 * 3 + 3) + (3 * 1 + 1)
    assert params.shape[0] == expected_params


def test_compute_parameter_difference():
    """Test parameter difference computation."""
    # Create two identical networks
    net1 = torch.nn.Linear(2, 1)
    net2 = torch.nn.Linear(2, 1)

    # Copy parameters from net1 to net2
    net2.load_state_dict(net1.state_dict())

    params1 = utils.extract_parameters(net1)
    params2 = utils.extract_parameters(net2)

    # Should have zero difference
    diff = utils.compute_parameter_difference(params1, params2)
    assert abs(diff) < 1e-6

    # Modify net2 slightly
    with torch.no_grad():
        net2.weight += 0.1

    params2_modified = utils.extract_parameters(net2)
    diff_modified = utils.compute_parameter_difference(params1, params2_modified)

    # Should have non-zero difference
    assert diff_modified > 0


class TestComputeGradientsForTensors(unittest.TestCase):
    """Direct tests for the compute_gradients_for_tensors utility."""

    def test_batched_gradient(self):
        """Batched input: target[i] = x[i]**2 should give gradient 2*x[i]."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = x**2
        grads = compute_gradients_for_tensors({"y": target}, {"x": x})
        expected = 2.0 * x
        self.assertTrue(torch.allclose(grads["y"]["x"], expected, atol=1e-6))

    def test_scalar_gradient(self):
        """Scalar input: target = x**2 should give gradient 2*x."""
        x = torch.tensor(3.0, requires_grad=True)
        target = x**2
        grads = compute_gradients_for_tensors({"y": target}, {"x": x})
        self.assertTrue(torch.allclose(grads["y"]["x"], 2.0 * x, atol=1e-6))

    def test_no_grad_returns_none(self):
        """Variable without requires_grad should produce None."""
        x = torch.tensor(3.0, requires_grad=False)
        y = torch.tensor(5.0, requires_grad=True)
        target = y**2
        grads = compute_gradients_for_tensors({"t": target}, {"x": x, "y": y})
        self.assertIsNone(grads["t"]["x"])
        self.assertIsNotNone(grads["t"]["y"])

    def test_unused_variable_returns_none(self):
        """Variable not used in computation should produce None."""
        x = torch.tensor(3.0, requires_grad=True)
        y = torch.tensor(5.0, requires_grad=True)
        target = x**2  # y is unused
        grads = compute_gradients_for_tensors({"t": target}, {"x": x, "y": y})
        self.assertIsNotNone(grads["t"]["x"])
        self.assertIsNone(grads["t"]["y"])

    def test_create_graph_enables_higher_order(self):
        """create_graph=True should allow second-order derivatives."""
        x = torch.tensor(3.0, requires_grad=True)
        target = x**3  # d/dx = 3x^2, d2/dx2 = 6x
        grads = compute_gradients_for_tensors(
            {"t": target}, {"x": x}, create_graph=True
        )
        first_deriv = grads["t"]["x"]
        # Verify first derivative is correct: 3 * 3^2 = 27
        self.assertTrue(torch.allclose(first_deriv, torch.tensor(27.0), atol=1e-5))
        # Verify we can take second derivative
        second_deriv = torch.autograd.grad(first_deriv, x)[0]
        # 6 * 3 = 18
        self.assertTrue(torch.allclose(second_deriv, torch.tensor(18.0), atol=1e-5))

    def test_empty_tensors_dict(self):
        """Empty tensors_dict should return empty dict."""
        x = torch.tensor(3.0, requires_grad=True)
        grads = compute_gradients_for_tensors({}, {"x": x})
        self.assertEqual(grads, {})


class TestFischerBurmeister(unittest.TestCase):
    """Test the Fischer-Burmeister complementarity function."""

    def test_both_zero(self):
        """FB(0, 0) = 0."""
        a = torch.tensor(0.0)
        h = torch.tensor(0.0)
        result = utils.fischer_burmeister(a, h)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_complementary_slackness(self):
        """FB(0, s) ≈ 0 for s > 0 and FB(f, 0) ≈ 0 for f > 0."""
        # When one is zero and the other is positive, FB should be ≈ 0
        s = torch.tensor(2.0)
        result = utils.fischer_burmeister(torch.tensor(0.0), s)
        self.assertAlmostEqual(result.item(), 0.0, places=4)

        f = torch.tensor(3.0)
        result = utils.fischer_burmeister(f, torch.tensor(0.0))
        self.assertAlmostEqual(result.item(), 0.0, places=4)

    def test_violation_nonzero(self):
        """FB(a, h) != 0 when both a > 0 and h > 0."""
        a = torch.tensor(1.0)
        h = torch.tensor(1.0)
        result = utils.fischer_burmeister(a, h)
        self.assertNotAlmostEqual(result.item(), 0.0, places=2)

    def test_differentiable(self):
        """FB is differentiable through autograd."""
        a = torch.tensor(1.0, requires_grad=True)
        h = torch.tensor(2.0, requires_grad=True)
        result = utils.fischer_burmeister(a, h)
        result.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(a.grad))
        self.assertTrue(torch.isfinite(h.grad))

    def test_differentiable_at_zero(self):
        """FB gradient is finite near zero due to epsilon safeguard."""
        a = torch.tensor(0.0, requires_grad=True)
        h = torch.tensor(0.0, requires_grad=True)
        result = utils.fischer_burmeister(a, h)
        result.backward()
        self.assertTrue(torch.isfinite(a.grad))
        self.assertTrue(torch.isfinite(h.grad))


class TestFischerBurmeisterEpsValidation(unittest.TestCase):
    """Test eps parameter validation."""

    def test_zero_eps_raises(self):
        a = torch.tensor(1.0)
        h = torch.tensor(1.0)
        with self.assertRaises(ValueError, msg="eps must be > 0"):
            utils.fischer_burmeister(a, h, eps=0.0)

    def test_negative_eps_raises(self):
        a = torch.tensor(1.0)
        h = torch.tensor(1.0)
        with self.assertRaises(ValueError, msg="eps must be > 0"):
            utils.fischer_burmeister(a, h, eps=-1e-12)
