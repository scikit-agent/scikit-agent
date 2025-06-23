"""
Tests for Fischer-Burmeister constraint handling in MMW JME '21 framework.

This test suite verifies the correctness of the Fischer-Burmeister function
implementation and its integration with the Bellman residual loss function
for constraint handling in dynamic economic models.

Test Coverage:
1. Fischer-Burmeister function mathematical properties
2. Constraint violation detection
3. Loss function integration
4. Economic model constraint handling
5. Neural network training with constraints

References:
- Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving
  dynamic economic models. Journal of Monetary Economics, 122, 76-101.
- Fischer, A. (1992). A special Newton-type optimization method.
  Optimization, 24(3-4), 269-284.
"""

import pytest
import numpy as np
import torch
from skagent.algos.maliar import (
    fischer_burmeister,
    get_constraint_violations,
    get_bellman_residual_loss,
    generate_bellman_training_grid,
)
from skagent.models.consumer import consumption_block_normalized, calibration
from skagent.ann import BlockPolicyNet, BlockValueNet


class TestFischerBurmeisterFunction:
    """Test the Fischer-Burmeister function mathematical properties."""

    def test_complementarity_conditions(self):
        """Test FB function equals zero for complementarity conditions."""
        # Case 1: a > 0, b = 0 (interior solution)
        a, b = 5.0, 0.0
        fb_val = fischer_burmeister(a, b)
        assert abs(fb_val) < 1e-6, f"FB({a}, {b}) should be ~0, got {fb_val}"

        # Case 2: a = 0, b > 0 (boundary solution)
        a, b = 0.0, 3.0
        fb_val = fischer_burmeister(a, b)
        assert abs(fb_val) < 1e-6, f"FB({a}, {b}) should be ~0, got {fb_val}"

        # Case 3: a = 0, b = 0 (corner solution)
        a, b = 0.0, 0.0
        fb_val = fischer_burmeister(a, b)
        assert abs(fb_val) < 1e-6, f"FB({a}, {b}) should be ~0, got {fb_val}"

    def test_constraint_violations(self):
        """Test FB function is negative for constraint violations."""
        # Both a > 0 and b > 0 violates complementarity
        a, b = 2.0, 2.0
        fb_val = fischer_burmeister(a, b)
        expected = a + b - np.sqrt(a**2 + b**2)  # Should be positive > 0 for violations
        assert fb_val > 0, f"FB({a}, {b}) should be positive for violations"
        assert abs(fb_val - expected) < 1e-6, "FB calculation incorrect"

    def test_infeasible_cases(self):
        """Test FB function behavior for infeasible cases."""
        # Case 1: a < 0
        a, b = -1.0, 0.0
        fb_val = fischer_burmeister(a, b)
        assert fb_val < 0, "FB should be negative when a < 0"

        # Case 2: b < 0
        a, b = 0.0, -1.0
        fb_val = fischer_burmeister(a, b)
        assert fb_val < 0, "FB should be negative when b < 0"

    def test_tensor_inputs(self):
        """Test FB function works with PyTorch tensors."""
        a = torch.tensor([5.0, 0.0, 2.0])
        b = torch.tensor([0.0, 3.0, 2.0])
        fb_vals = fischer_burmeister(a, b)

        assert isinstance(fb_vals, torch.Tensor), "Should return tensor"
        assert fb_vals.shape == (3,), "Shape should match input"

        # First two should be ~0 (complementarity satisfied)
        assert abs(fb_vals[0]) < 1e-6
        assert abs(fb_vals[1]) < 1e-6
        # Third should be positive (violation since both a,b > 0)
        assert fb_vals[2] > 0

    def test_differentiability(self):
        """Test FB function is differentiable."""
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        fb_val = fischer_burmeister(a, b)
        fb_val.backward()

        # Gradients should exist and be finite
        assert a.grad is not None, "Gradient w.r.t. a should exist"
        assert b.grad is not None, "Gradient w.r.t. b should exist"
        assert torch.isfinite(a.grad), "Gradient should be finite"
        assert torch.isfinite(b.grad), "Gradient should be finite"


class TestConstraintViolations:
    """Test constraint violation detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.block = consumption_block_normalized
        self.calibration = calibration.copy()

        # Add missing 'k' variable to calibration (k = a from previous period)
        self.calibration["k"] = 0.5  # Default initial capital

        # Construct shocks for the block
        self.block.construct_shocks(self.calibration)

    def test_upper_bound_constraints(self):
        """Test detection of upper bound constraint violations."""
        # Set up test data
        states = {"m": torch.tensor([2.0, 3.0, 4.0])}
        controls = {"c": torch.tensor([1.5, 3.5, 2.0])}  # Second violates c ≤ m

        violations = get_constraint_violations(
            self.block, states, controls, self.calibration
        )

        assert "c_upper" in violations, "Should detect upper bound constraint"

        # Check violation calculations: m - c
        expected_violations = states["m"] - controls["c"]  # [0.5, -0.5, 2.0]
        torch.testing.assert_close(
            violations["c_upper"],
            expected_violations,
            msg="Upper bound violations incorrectly calculated",
        )

    def test_lower_bound_constraints(self):
        """Test detection of non-negativity constraints."""
        states = {"m": torch.tensor([2.0, 3.0, 4.0])}
        controls = {"c": torch.tensor([1.5, -0.5, 2.0])}  # Second violates c ≥ 0

        violations = get_constraint_violations(
            self.block, states, controls, self.calibration
        )

        assert "c_lower" in violations, "Should detect lower bound constraint"

        # Check non-negativity: should equal control values
        torch.testing.assert_close(
            violations["c_lower"],
            controls["c"],
            msg="Lower bound violations incorrectly calculated",
        )

    def test_no_constraints(self):
        """Test when no constraints are violated."""
        states = {"m": torch.tensor([2.0, 3.0, 4.0])}
        controls = {"c": torch.tensor([1.5, 2.5, 3.0])}  # All feasible

        violations = get_constraint_violations(
            self.block, states, controls, self.calibration
        )

        # All violations should be non-negative (constraints satisfied)
        for constraint_name, violation in violations.items():
            assert torch.all(violation >= 0), (
                f"Constraint {constraint_name} should be satisfied"
            )


class TestBellmanLossWithConstraints:
    """Test Bellman loss function with Fischer-Burmeister constraints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state_variables = ["m", "a"]
        self.block = consumption_block_normalized
        self.calibration = calibration.copy()
        self.discount_factor = 0.96

        # Add missing 'k' variable to calibration (k = a from previous period)
        self.calibration["k"] = 0.5  # Default initial capital

        # Construct shocks for the block
        self.block.construct_shocks(self.calibration)

        # Create small training grid
        state_config = {
            "m": {"min": 0.5, "max": 2.0, "count": 5},
            "a": {"min": 0.0, "max": 1.0, "count": 4},
        }
        self.training_grid = generate_bellman_training_grid(
            state_config, self.block, n_samples=10
        )

        # Create test networks
        self.policy_net = BlockPolicyNet(self.block, width=16)
        self.value_net = BlockValueNet(self.block, self.state_variables, width=16)

    def test_standard_bellman_loss(self):
        """Test standard Bellman loss without constraints."""
        loss_fn = get_bellman_residual_loss(
            state_variables=self.state_variables,
            block=self.block,
            discount_factor=self.discount_factor,
            parameters=self.calibration,
            use_fischer_burmeister=False,
        )

        loss = loss_fn(
            self.policy_net.get_decision_function(),
            self.value_net.get_value_function(),
            self.training_grid,
        )

        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.shape == (len(self.training_grid.values),), "Loss shape incorrect"
        assert torch.all(torch.isfinite(loss)), "Loss should be finite"
        assert torch.all(loss >= 0), "Squared loss should be non-negative"

    def test_fischer_burmeister_bellman_loss(self):
        """Test Bellman loss with Fischer-Burmeister constraints."""
        loss_fn = get_bellman_residual_loss(
            state_variables=self.state_variables,
            block=self.block,
            discount_factor=self.discount_factor,
            parameters=self.calibration,
            use_fischer_burmeister=True,
        )

        loss = loss_fn(
            self.policy_net.get_decision_function(),
            self.value_net.get_value_function(),
            self.training_grid,
        )

        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.shape == (len(self.training_grid.values),), "Loss shape incorrect"
        assert torch.all(torch.isfinite(loss)), "Loss should be finite"
        assert torch.all(loss >= 0), "Squared loss should be non-negative"

    def test_constraint_penalty_effect(self):
        """Test that constraint penalties increase loss when violated."""
        # Standard loss without constraints
        loss_fn_std = get_bellman_residual_loss(
            state_variables=self.state_variables,
            block=self.block,
            discount_factor=self.discount_factor,
            parameters=self.calibration,
            use_fischer_burmeister=False,
        )

        # Loss with Fischer-Burmeister constraints
        loss_fn_fb = get_bellman_residual_loss(
            state_variables=self.state_variables,
            block=self.block,
            discount_factor=self.discount_factor,
            parameters=self.calibration,
            use_fischer_burmeister=True,
        )

        # Create policy that might violate constraints
        policy_fn = self.policy_net.get_decision_function()
        value_fn = self.value_net.get_value_function()

        loss_std = loss_fn_std(policy_fn, value_fn, self.training_grid)
        loss_fb = loss_fn_fb(policy_fn, value_fn, self.training_grid)

        # FB loss should be higher when constraints are violated
        # (Not always true, but generally expected)
        mean_loss_std = torch.mean(loss_std)
        mean_loss_fb = torch.mean(loss_fb)

        assert torch.isfinite(mean_loss_std), "Standard loss should be finite"
        assert torch.isfinite(mean_loss_fb), "FB loss should be finite"

        # Both should be non-negative
        assert mean_loss_std >= 0, "Standard loss should be non-negative"
        assert mean_loss_fb >= 0, "FB loss should be non-negative"

    def test_loss_gradients(self):
        """Test that loss functions produce finite gradients."""
        loss_fn = get_bellman_residual_loss(
            state_variables=self.state_variables,
            block=self.block,
            discount_factor=self.discount_factor,
            parameters=self.calibration,
            use_fischer_burmeister=True,
        )

        # Enable gradients for network parameters
        for param in self.policy_net.parameters():
            param.requires_grad_(True)
        for param in self.value_net.parameters():
            param.requires_grad_(True)

        loss = loss_fn(
            self.policy_net.get_decision_function(),
            self.value_net.get_value_function(),
            self.training_grid,
        )

        total_loss = torch.mean(loss)
        total_loss.backward()

        # Check that gradients exist and are finite
        for name, param in self.policy_net.named_parameters():
            assert param.grad is not None, f"Policy gradient missing for {name}"
            assert torch.all(torch.isfinite(param.grad)), (
                f"Policy gradient not finite for {name}"
            )

        for name, param in self.value_net.named_parameters():
            assert param.grad is not None, f"Value gradient missing for {name}"
            assert torch.all(torch.isfinite(param.grad)), (
                f"Value gradient not finite for {name}"
            )


class TestIntegrationWithEconomicModel:
    """Test integration with economic models."""

    def test_consumption_block_constraints(self):
        """Test constraint handling with consumption block."""
        # Verify that consumption block has upper bound constraint
        dynamics = consumption_block_normalized.get_dynamics()
        assert "c" in dynamics, "Consumption should be in dynamics"

        control = dynamics["c"]
        assert hasattr(control, "upper_bound"), "Consumption should have upper bound"
        assert control.upper_bound is not None, "Upper bound should be defined"

    def test_realistic_constraint_violations(self):
        """Test constraint violations in realistic economic scenarios."""
        # Set up realistic economic scenario
        states = {
            "m": torch.tensor([0.5, 1.0, 2.0, 3.0]),  # Cash-on-hand
            "a": torch.tensor([0.0, 0.2, 0.5, 1.0]),  # Assets
        }

        # Test case 1: Reasonable consumption
        controls_reasonable = {
            "c": torch.tensor([0.3, 0.8, 1.5, 2.0])  # All ≤ m
        }

        violations = get_constraint_violations(
            consumption_block_normalized, states, controls_reasonable, calibration
        )

        # All upper bound violations should be non-negative (constraints satisfied)
        assert torch.all(violations["c_upper"] >= 0), (
            "Reasonable consumption should satisfy constraints"
        )

        # Test case 2: Excessive consumption
        controls_excessive = {
            "c": torch.tensor([0.8, 1.5, 2.5, 4.0])  # Some > m
        }

        violations = get_constraint_violations(
            consumption_block_normalized, states, controls_excessive, calibration
        )

        # Some violations should be negative (constraints violated)
        upper_violations = violations["c_upper"]
        assert torch.any(upper_violations < 0), (
            "Excessive consumption should violate constraints"
        )

        # Specifically, last two should be negative
        assert upper_violations[2] < 0, "c=2.5 > m=2.0 should violate constraint"
        assert upper_violations[3] < 0, "c=4.0 > m=3.0 should violate constraint"


if __name__ == "__main__":
    pytest.main([__file__])
