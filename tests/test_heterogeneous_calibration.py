"""
Test cases for ex ante agent heterogeneity in calibration.

Tests both vector calibration (arrays/lists of values) and distribution calibration
(distributions that get sampled for N agents) across various solvers and simulators.
"""

import unittest
import numpy as np
import torch

import skagent.model as model
from skagent.model import (
    Control,
    DBlock,
    expand_heterogeneous_calibration,
    calibration_agent_subset,
)
from skagent.distributions import Normal, Uniform, Lognormal, Bernoulli
from skagent.simulation.monte_carlo import (
    AgentTypeMonteCarloSimulator,
    MonteCarloSimulator,
)


# Test seed for reproducibility
TEST_SEED = 42


class TestHeterogeneousCalibrationCore(unittest.TestCase):
    """Test core functionality of heterogeneous calibration expansion."""

    def setUp(self):
        self.rng = np.random.default_rng(TEST_SEED)

    def test_scalar_calibration(self):
        """Test that scalar calibration is properly expanded to all agents."""
        calibration = {"DiscFac": 0.96, "CRRA": 2.0, "Rfree": 1.03}

        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=5, rng=self.rng
        )

        # Check that each parameter is expanded correctly
        for param_name, param_value in calibration.items():
            self.assertIn(param_name, expanded)
            np.testing.assert_array_equal(expanded[param_name], np.full(5, param_value))

    def test_vector_calibration(self):
        """Test vector calibration with arrays/lists."""
        calibration = {
            "DiscFac": [0.90, 0.95, 0.96, 0.97, 0.98],  # List
            "CRRA": np.array([1.5, 2.0, 2.5, 3.0, 3.5]),  # Array
            "Rfree": 1.03,  # Scalar mixed with vectors
        }

        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=5, rng=self.rng
        )

        # Check vector parameters
        np.testing.assert_array_equal(
            expanded["DiscFac"], [0.90, 0.95, 0.96, 0.97, 0.98]
        )
        np.testing.assert_array_equal(expanded["CRRA"], [1.5, 2.0, 2.5, 3.0, 3.5])

        # Check scalar parameter expansion
        np.testing.assert_array_equal(expanded["Rfree"], np.full(5, 1.03))

    def test_distribution_calibration(self):
        """Test distribution calibration with sampling."""
        calibration = {
            "DiscFac": Normal(mu=0.96, sigma=0.01),  # Normal distribution
            "CRRA": Uniform(low=1.5, high=3.5),  # Uniform distribution
            "Rfree": 1.03,  # Scalar mixed with distributions
        }

        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=10, rng=self.rng
        )

        # Check that distributions were sampled
        self.assertEqual(len(expanded["DiscFac"]), 10)
        self.assertEqual(len(expanded["CRRA"]), 10)

        # Check that sampled values are reasonable
        self.assertTrue(np.all(expanded["DiscFac"] > 0.90))
        self.assertTrue(np.all(expanded["DiscFac"] < 1.02))
        self.assertTrue(np.all(expanded["CRRA"] >= 1.5))
        self.assertTrue(np.all(expanded["CRRA"] <= 3.5))

        # Check scalar parameter expansion
        np.testing.assert_array_equal(expanded["Rfree"], np.full(10, 1.03))

    def test_mixed_calibration(self):
        """Test calibration with mix of scalars, vectors, and distributions."""
        calibration = {
            "DiscFac": [0.95, 0.96, 0.97],  # Vector
            "CRRA": Normal(mu=2.0, sigma=0.1),  # Distribution
            "Rfree": 1.03,  # Scalar
            "LivPrb": np.array([0.98, 0.99, 0.995]),  # Array
        }

        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=3, rng=self.rng
        )

        # Check vector parameters
        np.testing.assert_array_equal(expanded["DiscFac"], [0.95, 0.96, 0.97])
        np.testing.assert_array_equal(expanded["LivPrb"], [0.98, 0.99, 0.995])

        # Check distribution parameter
        self.assertEqual(len(expanded["CRRA"]), 3)
        self.assertTrue(np.all(expanded["CRRA"] > 1.5))
        self.assertTrue(np.all(expanded["CRRA"] < 2.5))

        # Check scalar parameter
        np.testing.assert_array_equal(expanded["Rfree"], np.full(3, 1.03))

    def test_vector_length_validation(self):
        """Test that vector length validation works correctly."""
        # Test correct length
        calibration = {"CRRA": [2.0, 2.5, 3.0]}
        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=3, rng=self.rng
        )
        np.testing.assert_array_equal(expanded["CRRA"], [2.0, 2.5, 3.0])

        # Test single element expansion
        calibration = {"CRRA": [2.0]}
        expanded = expand_heterogeneous_calibration(
            calibration, agent_count=3, rng=self.rng
        )
        np.testing.assert_array_equal(expanded["CRRA"], [2.0, 2.0, 2.0])

        # Test incorrect length should raise error
        calibration = {"CRRA": [2.0, 2.5]}
        with self.assertRaises(ValueError):
            expand_heterogeneous_calibration(calibration, agent_count=3, rng=self.rng)

    def test_calibration_agent_subset(self):
        """Test extracting calibration for agent subsets."""
        calibration = {
            "DiscFac": np.array([0.90, 0.95, 0.96, 0.97, 0.98]),
            "CRRA": np.array([1.5, 2.0, 2.5, 3.0, 3.5]),
            "Rfree": np.array(
                [1.03, 1.03, 1.03, 1.03, 1.03]
            ),  # Homogeneous but vectorized
        }

        # Extract subset for agents 1, 3, 4
        subset = calibration_agent_subset(calibration, [1, 3, 4])

        np.testing.assert_array_equal(subset["DiscFac"], [0.95, 0.97, 0.98])
        np.testing.assert_array_equal(subset["CRRA"], [2.0, 3.0, 3.5])
        np.testing.assert_array_equal(subset["Rfree"], [1.03, 1.03, 1.03])


class TestHeterogeneousCalibrationSimulation(unittest.TestCase):
    """Test heterogeneous calibration with simulators."""

    def setUp(self):
        self.rng = np.random.default_rng(TEST_SEED)

        # Create a simple consumption-savings block for testing
        self.test_block = DBlock(
            name="heterogeneous_consumption",
            shocks={
                "theta": (Lognormal, {"mean": 1.0, "std": 0.1}),
                "live": (
                    Bernoulli,
                    {"p": 0.98},
                ),  # Add live shock for AgentTypeMonteCarloSimulator
            },
            dynamics={
                "m": lambda a, Rfree, theta: a * Rfree + theta,
                "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
                "a": lambda m, c: m - c,
                "u": lambda c, CRRA: (
                    torch.log(c) if CRRA == 1 else c ** (1 - CRRA) / (1 - CRRA)
                ),
            },
            reward={"u": "consumer"},
        )

        # Simple decision rule
        self.dr = {"c": lambda m: 0.7 * m}

        # Initial conditions (both simulators expect distributions)
        self.initial_dist = {"a": Uniform(low=1.0, high=3.0)}

    def test_monte_carlo_simulator_vector_calibration(self):
        """Test MonteCarloSimulator with vector calibration."""
        # Vector calibration with different risk aversion for each agent
        calibration = {
            "CRRA": [1.5, 2.0, 2.5],  # Different risk aversion
            "Rfree": [1.02, 1.03, 1.04],  # Different interest rates
        }

        simulator = MonteCarloSimulator(
            calibration=calibration,
            block=self.test_block,
            dr=self.dr,
            initial=self.initial_dist,
            agent_count=3,
            seed=TEST_SEED,
            T_sim=5,
        )

        # Check that calibration was expanded correctly
        np.testing.assert_array_equal(simulator.calibration["CRRA"], [1.5, 2.0, 2.5])
        np.testing.assert_array_equal(
            simulator.calibration["Rfree"], [1.02, 1.03, 1.04]
        )

        # Run simulation
        simulator.initialize_sim()
        history = simulator.simulate()

        # Check that simulation produced reasonable results
        self.assertIn("m", history)
        self.assertIn("c", history)
        self.assertIn("a", history)

        # Check that different agents have different behavior due to heterogeneous parameters
        # (This is a basic check - more sophisticated tests could verify specific differences)
        final_assets = history["a"][-1]
        self.assertEqual(len(final_assets), 3)
        self.assertTrue(np.all(final_assets > 0))

    def test_monte_carlo_simulator_distribution_calibration(self):
        """Test MonteCarloSimulator with distribution calibration."""
        # Distribution calibration
        calibration = {
            "CRRA": Normal(mu=2.0, sigma=0.2),  # Normal distribution around 2.0
            "Rfree": Uniform(low=1.02, high=1.04),  # Uniform distribution
        }

        simulator = MonteCarloSimulator(
            calibration=calibration,
            block=self.test_block,
            dr=self.dr,
            initial=self.initial_dist,
            agent_count=5,
            seed=TEST_SEED,
            T_sim=3,
        )

        # Check that calibration was sampled correctly
        self.assertEqual(len(simulator.calibration["CRRA"]), 5)
        self.assertEqual(len(simulator.calibration["Rfree"]), 5)

        # Check that sampled values are in reasonable ranges
        self.assertTrue(np.all(simulator.calibration["CRRA"] > 1.0))
        self.assertTrue(np.all(simulator.calibration["CRRA"] < 3.0))
        self.assertTrue(np.all(simulator.calibration["Rfree"] >= 1.02))
        self.assertTrue(np.all(simulator.calibration["Rfree"] <= 1.04))

        # Run simulation
        simulator.initialize_sim()
        history = simulator.simulate()

        # Check basic simulation results
        self.assertIn("m", history)
        self.assertIn("c", history)

    def test_agent_type_monte_carlo_simulator_heterogeneous(self):
        """Test AgentTypeMonteCarloSimulator with heterogeneous calibration."""
        # Vector calibration
        calibration = {
            "CRRA": [1.8, 2.0, 2.2],
            "Rfree": 1.03,  # Scalar mixed with vector
        }

        # Note: AgentTypeMonteCarloSimulator expects initial to be a distribution
        simulator = AgentTypeMonteCarloSimulator(
            calibration=calibration,
            block=self.test_block,
            dr=self.dr,
            initial=self.initial_dist,
            agent_count=3,
            seed=TEST_SEED,
            T_sim=3,
        )

        # Check calibration expansion
        np.testing.assert_array_equal(simulator.calibration["CRRA"], [1.8, 2.0, 2.2])
        np.testing.assert_array_equal(
            simulator.calibration["Rfree"], [1.03, 1.03, 1.03]
        )

        # Run simulation
        simulator.initialize_sim()
        history = simulator.simulate()

        # Check results
        self.assertIn("m", history)
        self.assertIn("c", history)


class TestHeterogeneousCalibrationDynamics(unittest.TestCase):
    """Test that heterogeneous calibration works correctly in dynamics simulation."""

    def setUp(self):
        self.rng = np.random.default_rng(TEST_SEED)

    def test_simulate_dynamics_with_heterogeneous_parameters(self):
        """Test that simulate_dynamics correctly handles heterogeneous parameters."""
        # Create dynamics with heterogeneous parameters
        dynamics = {
            "new_wealth": lambda initial_wealth, income, CRRA: initial_wealth
            + income * (1 + CRRA / 10),
            "consumption": Control(["new_wealth"], agent="consumer"),
            "utility": lambda consumption, CRRA: consumption ** (1 - CRRA) / (1 - CRRA),
        }

        # Pre-state with heterogeneous parameters
        pre = {
            "initial_wealth": np.array([10.0, 15.0, 20.0]),
            "income": np.array([5.0, 5.0, 5.0]),
            "CRRA": np.array([1.5, 2.0, 2.5]),  # Heterogeneous risk aversion
        }

        # Decision rules
        dr = {"consumption": lambda new_wealth: 0.6 * new_wealth}

        # Simulate dynamics
        post = model.simulate_dynamics(dynamics, pre, dr)

        # Check that dynamics worked correctly with heterogeneous parameters
        self.assertEqual(len(post["new_wealth"]), 3)
        self.assertEqual(len(post["consumption"]), 3)
        self.assertEqual(len(post["utility"]), 3)

        # Check that different agents have different outcomes due to heterogeneous CRRA
        # Agent 0: new_wealth = 10 + 5*(1+1.5/10) = 10 + 5*1.15 = 15.75
        # Agent 1: new_wealth = 15 + 5*(1+2.0/10) = 15 + 5*1.2 = 21.0
        # Agent 2: new_wealth = 20 + 5*(1+2.5/10) = 20 + 5*1.25 = 26.25
        expected_wealth = np.array([15.75, 21.0, 26.25])
        np.testing.assert_array_almost_equal(post["new_wealth"], expected_wealth)

        # Check consumption (60% of new_wealth)
        expected_consumption = 0.6 * expected_wealth
        np.testing.assert_array_almost_equal(post["consumption"], expected_consumption)

    def test_construct_shocks_with_heterogeneous_calibration(self):
        """Test that construct_shocks works with heterogeneous calibration parameters."""
        # Shock specification that depends on calibration
        shock_data = {"income": (Normal, {"mu": "base_income", "sigma": "income_std"})}

        # Heterogeneous calibration (but shocks use first agent's parameters)
        scope = {
            "base_income": np.array([50.0, 55.0, 60.0]),
            "income_std": np.array([5.0, 6.0, 7.0]),
        }

        constructed_shocks = model.construct_shocks(shock_data, scope, rng=self.rng)

        # Check that shock was constructed (using first agent's parameters)
        self.assertIn("income", constructed_shocks)
        self.assertIsInstance(constructed_shocks["income"], Normal)

        # The shock should use the first agent's parameters
        # (this is the current behavior for homogeneous shocks with heterogeneous calibration)


class TestHeterogeneousCalibrationCompatibility(unittest.TestCase):
    """Test backward compatibility and integration with existing functionality."""

    def test_backward_compatibility_scalar_only(self):
        """Test that existing scalar-only calibration still works."""
        # This tests backward compatibility
        calibration = {"DiscFac": 0.96, "CRRA": 2.0, "Rfree": 1.03}

        # Should preserve scalars when no heterogeneity is present (backward compatibility)
        expanded = expand_heterogeneous_calibration(calibration, agent_count=3)

        for param_name, param_value in calibration.items():
            self.assertIn(param_name, expanded)
            # Should remain as scalars for backward compatibility
            self.assertEqual(expanded[param_name], param_value)

    def test_integration_with_calibration_by_age(self):
        """Test that heterogeneous calibration integrates with age-varying calibration."""
        from skagent.simulation.monte_carlo import calibration_by_age

        # Age-varying base calibration
        ages = np.array([0, 1, 2])  # Use valid indices instead of actual ages
        base_calibration = {
            "DiscFac": [0.95, 0.96, 0.97],  # Age-varying discount factor
            "CRRA": 2.0,  # Constant across ages
        }

        # Get age-specific calibration
        age_calibration = calibration_by_age(ages, base_calibration)

        # Then apply heterogeneous expansion
        het_calibration = {
            "CRRA": [1.8, 2.0, 2.2],  # Heterogeneous risk aversion
            "Rfree": Normal(mu=1.03, sigma=0.005),  # Heterogeneous interest rate
        }

        # Override with heterogeneous parameters
        combined_calibration = {**age_calibration, **het_calibration}

        expanded = expand_heterogeneous_calibration(combined_calibration, agent_count=3)

        # Check that both age-varying and heterogeneous parameters are present
        self.assertIn("DiscFac", expanded)
        self.assertIn("CRRA", expanded)
        self.assertIn("Rfree", expanded)

        # CRRA should be heterogeneous (overriding age effect)
        np.testing.assert_array_equal(expanded["CRRA"], [1.8, 2.0, 2.2])

        # Rfree should be sampled
        self.assertEqual(len(expanded["Rfree"]), 3)


if __name__ == "__main__":
    unittest.main()
