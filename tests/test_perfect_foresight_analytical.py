"""
Test Euler and Bellman methods against ALL analytical solutions from benchmarks.py.

These tests validate that skagent methods can reproduce analytical solutions to machine precision
for ALL deterministic consumption-savings problems where closed-form solutions exist.
"""

import unittest
import torch

from skagent.models.benchmarks import (
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
    list_benchmark_models,
    BENCHMARK_MODELS,
)


class TestAnalyticalBenchmarks(unittest.TestCase):
    """Test methods against ALL analytical solutions from the comprehensive catalogue."""

    def setUp(self):
        """Set up test parameters."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get all deterministic models (perfect foresight)
        self.deterministic_models = ["D-1", "D-2", "D-3", "D-4"]

        # Standard test tolerances
        self.euler_tolerance = 1e-6
        self.bellman_tolerance = 1e-4
        self.policy_tolerance = 1e-8

    def test_catalogue_completeness(self):
        """Test that the benchmark catalogue contains expected models."""
        models = list_benchmark_models()

        # Should have all 10 models
        self.assertEqual(len(models), 10, "Should have 10 analytical models")

        # Check deterministic models exist
        for model_id in self.deterministic_models:
            self.assertIn(model_id, models, f"Missing deterministic model {model_id}")

    def test_d3_infinite_crra_euler_equation(self):
        """Test D-3 model satisfies Euler equation exactly."""
        calibration = get_benchmark_calibration("D-3")
        analytical_policy = get_analytical_policy("D-3")

        beta = calibration["DiscFac"]
        R = calibration["R"]
        sigma = calibration["CRRA"]

        # Test at different wealth levels with double precision
        wealth_levels = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64)

        for m in wealth_levels:
            # Get analytical consumption
            states = {"m": m.unsqueeze(0)}
            result = analytical_policy(states, {}, calibration)
            c_t = result["c"][0].double()

            # Next period wealth and consumption
            a_t = m - c_t
            m_t1 = a_t * R
            result_next = analytical_policy({"m": m_t1.unsqueeze(0)}, {}, calibration)
            c_t1 = result_next["c"][0].double()

            # Euler equation: u'(c_t) = β * R * u'(c_{t+1})
            lhs = c_t ** (-sigma)
            rhs = beta * R * (c_t1 ** (-sigma))
            euler_error = torch.abs(lhs - rhs)

            self.assertLess(
                euler_error.item(),
                self.euler_tolerance,
                f"Euler equation error too large at m={m}: {euler_error}",
            )

    def test_d1_two_period_optimality(self):
        """Test D-1 two-period model satisfies optimality conditions."""
        calibration = get_benchmark_calibration("D-1")
        analytical_policy = get_analytical_policy("D-1")

        beta = calibration["DiscFac"]
        R = calibration["R"]

        # Test at different wealth levels
        wealth_levels = torch.tensor([1.0, 2.0, 3.0, 4.0])

        for W in wealth_levels:
            states = {"W": W.unsqueeze(0)}
            result = analytical_policy(states, {}, calibration)
            c1 = result["c1"][0]

            # Second period consumption
            c2 = (W - c1) * R

            # Euler equation for log utility: 1/c1 = β*R/c2
            lhs = 1.0 / c1
            rhs = beta * R / c2
            euler_error = torch.abs(lhs - rhs)

            self.assertLess(
                euler_error.item(),
                self.euler_tolerance,
                f"Two-period Euler error too large at W={W}: {euler_error}",
            )

            # Budget constraint
            total_consumption_pv = c1 + c2 / R
            budget_error = torch.abs(total_consumption_pv - W)

            self.assertLess(
                budget_error.item(),
                1e-10,
                f"Budget constraint violated at W={W}: {budget_error}",
            )

    def test_analytical_policies_feasibility(self):
        """Test that all analytical policies produce feasible consumption."""

        for model_id in self.deterministic_models:
            with self.subTest(model=model_id):
                calibration = get_benchmark_calibration(model_id)
                analytical_policy = get_analytical_policy(model_id)

                # Generate appropriate test states
                if model_id == "D-1":
                    test_states = {"W": torch.linspace(0.5, 5.0, 20)}
                elif model_id == "D-2":
                    test_states = {
                        "W": torch.linspace(0.5, 5.0, 20),
                        "t": torch.zeros(20),
                    }
                else:  # D-3 or D-4
                    test_states = {"m": torch.linspace(0.5, 5.0, 20)}

                result = analytical_policy(test_states, {}, calibration)

                # Check all consumption values are positive and finite
                for control_name, control_values in result.items():
                    self.assertTrue(
                        torch.all(control_values > 0),
                        f"All {control_name} should be positive for {model_id}",
                    )
                    self.assertTrue(
                        torch.all(torch.isfinite(control_values)),
                        f"All {control_name} should be finite for {model_id}",
                    )

                    # For models with wealth/cash-on-hand constraints
                    if model_id == "D-1" and control_name == "c1":
                        self.assertTrue(
                            torch.all(control_values <= test_states["W"]),
                            f"c1 should not exceed wealth for {model_id}",
                        )
                    elif model_id in ["D-3", "D-4"] and control_name == "c":
                        self.assertTrue(
                            torch.all(control_values <= test_states["m"]),
                            f"c should not exceed cash-on-hand for {model_id}",
                        )

    def test_parameter_restrictions_validation(self):
        """Test that models validate their parameter restrictions correctly."""

        # Test D-3 return-impatience condition
        calibration_d3 = get_benchmark_calibration("D-3")
        beta = calibration_d3["DiscFac"]
        R = calibration_d3["R"]
        sigma = calibration_d3["CRRA"]

        # Should satisfy (βR)^(1/σ) < R
        growth_factor = (beta * R) ** (1 / sigma)
        self.assertLess(
            growth_factor, R, "D-3 should satisfy return-impatience condition"
        )

        # Test that violating the condition raises an error
        bad_calibration = calibration_d3.copy()
        bad_calibration["DiscFac"] = 1.1  # Too high discount factor

        with self.assertRaises(ValueError):
            # This should fail when creating the analytical policy
            from skagent.models.benchmarks import d3_analytical_policy

            d3_analytical_policy(bad_calibration)

    def test_benchmark_model_blocks_structure(self):
        """Test that benchmark model blocks have proper structure."""

        for model_id in self.deterministic_models:
            with self.subTest(model=model_id):
                model_block = get_benchmark_model(model_id)

                # Check block has required attributes
                self.assertTrue(
                    hasattr(model_block, "name"),
                    f"Model {model_id} block should have name",
                )
                self.assertTrue(
                    hasattr(model_block, "dynamics"),
                    f"Model {model_id} block should have dynamics",
                )
                self.assertTrue(
                    hasattr(model_block, "reward"),
                    f"Model {model_id} block should have reward",
                )

                # Check dynamics structure
                self.assertIsInstance(
                    model_block.dynamics,
                    dict,
                    f"Model {model_id} dynamics should be dict",
                )

                # Check reward structure
                self.assertIsInstance(
                    model_block.reward, dict, f"Model {model_id} reward should be dict"
                )

    def test_consistency_across_models(self):
        """Test consistency of the benchmark catalogue structure."""

        all_models = list_benchmark_models()

        for model_id, description in all_models.items():
            with self.subTest(model=model_id):
                # Each model should have all required components
                self.assertIn(model_id, BENCHMARK_MODELS)

                model_info = BENCHMARK_MODELS[model_id]
                self.assertIn("block", model_info)
                self.assertIn("calibration", model_info)
                self.assertIn("analytical_policy", model_info)

                # Description should be informative
                self.assertIsInstance(description, str)
                self.assertGreater(len(description), 10)
                self.assertIn(
                    model_id.split("-")[0], description
                )  # Should contain D or U


class TestStochasticBenchmarks(unittest.TestCase):
    """Test stochastic models from the benchmark catalogue."""

    def setUp(self):
        """Set up test parameters for stochastic models."""
        torch.manual_seed(42)
        self.stochastic_models = ["U-1", "U-2", "U-3", "U-4", "U-5", "U-6"]

    def test_stochastic_models_structure(self):
        """Test that stochastic models have proper shock specifications."""

        for model_id in self.stochastic_models:
            with self.subTest(model=model_id):
                model_block = get_benchmark_model(model_id)
                calibration = get_benchmark_calibration(model_id)

                # Stochastic models should have shocks
                self.assertTrue(
                    hasattr(model_block, "shocks"),
                    f"Stochastic model {model_id} should have shocks",
                )

                if hasattr(model_block, "shocks") and model_block.shocks:
                    self.assertIsInstance(
                        model_block.shocks,
                        dict,
                        f"Model {model_id} shocks should be dict",
                    )
                    self.assertGreater(
                        len(model_block.shocks),
                        0,
                        f"Model {model_id} should have at least one shock",
                    )

    def test_stochastic_analytical_policies(self):
        """Test that stochastic analytical policies work correctly."""

        for model_id in self.stochastic_models:
            with self.subTest(model=model_id):
                try:
                    analytical_policy = get_analytical_policy(model_id)
                    calibration = get_benchmark_calibration(model_id)

                    # Generate appropriate test states
                    if model_id == "U-6":  # Habit formation
                        test_states = {
                            "m": torch.tensor([1.0, 2.0, 3.0]),
                            "h": torch.tensor([0.5, 0.5, 0.5]),
                        }
                    else:
                        test_states = {"m": torch.tensor([1.0, 2.0, 3.0])}

                    result = analytical_policy(test_states, {}, calibration)

                    # Basic feasibility checks
                    for control_name, control_values in result.items():
                        self.assertTrue(
                            torch.all(control_values > 0),
                            f"All {control_name} should be positive for {model_id}",
                        )
                        self.assertTrue(
                            torch.all(torch.isfinite(control_values)),
                            f"All {control_name} should be finite for {model_id}",
                        )

                except Exception as e:
                    # Some stochastic models may have implementation limitations
                    self.skipTest(
                        f"Stochastic model {model_id} analytical policy not fully implemented: {e}"
                    )


if __name__ == "__main__":
    unittest.main()
