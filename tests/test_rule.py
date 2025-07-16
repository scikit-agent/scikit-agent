"""
Tests for rule_module.py using real scikit-agent objects and scenarios.

Based on actual consumption-savings model structure from scikit-agent.
"""

from importlib.util import find_spec
import pytest
import sys

sys.path.append("../src")
from skagent.rule import extract_dependencies

SKAGENT_AVAILABLE = find_spec("skagent") is not None
HARK_AVAILABLE = find_spec("HARK") is not None
HAS_DEPENDENCIES = SKAGENT_AVAILABLE and HARK_AVAILABLE

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="Optional dependencies (`scikit‑agent` and/or `HARK`) not installed.",
)

if HAS_DEPENDENCIES:
    from skagent.models.consumer import consumption_block, calibration


class TestExtractDependencies:
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_lambda_dynamics_rules(self):
        """Test dependency extraction from actual lambda dynamics rules."""
        # From the consumption model dynamics
        dynamics = consumption_block.dynamics

        assert extract_dependencies(dynamics["b"]) == ["k", "R"]
        assert extract_dependencies(dynamics["y"]) == ["p", "theta"]
        assert extract_dependencies(dynamics["m"]) == ["b", "y"]
        assert extract_dependencies(dynamics["p"]) == ["PermGroFac", "p"]
        assert extract_dependencies(dynamics["a"]) == ["m", "c"]
        assert extract_dependencies(dynamics["u"]) == ["c", "CRRA"]

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_control_rule(self):
        """Test dependency extraction from actual Control rule."""
        # From the consumption model: c = Control(["m"], upper_bound=lambda m: m, agent="consumer")
        dynamics = consumption_block.dynamics
        c_rule = dynamics["c"]

        assert extract_dependencies(c_rule) == ["m"]

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_shock_tuple_rules(self):
        """Test dependency extraction from actual shock tuple rules."""
        # From the consumption model shocks
        shocks = consumption_block.shocks
        live_shock = shocks["live"]
        theta_shock = shocks["theta"]

        live_deps = extract_dependencies(live_shock)
        theta_deps = extract_dependencies(theta_shock)

        assert "LivPrb" in live_deps
        assert "TranShkStd" in theta_deps

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_calibration_parameters(self):
        """Test that calibration parameters are correctly identified."""
        # Test that our dependency extraction can identify calibration parameters
        dynamics = consumption_block.dynamics

        # These should extract parameter names that are in calibration dict
        u_deps = extract_dependencies(dynamics["u"])
        b_deps = extract_dependencies(dynamics["b"])
        p_deps = extract_dependencies(dynamics["p"])

        # Check that calibration parameters are identified
        assert "CRRA" in u_deps
        assert "R" in b_deps
        assert "PermGroFac" in p_deps

        # Verify these are actually in the calibration dict
        assert "CRRA" in calibration
        assert "R" in calibration
        assert "PermGroFac" in calibration

    def test_string_rules(self):
        """Test dependency extraction from string expressions."""
        # Common patterns in economic models
        assert "income" in extract_dependencies("income - consumption")
        assert "consumption" in extract_dependencies("income - consumption")

        assert "beta" in extract_dependencies("beta * future_value")
        assert "future_value" in extract_dependencies("beta * future_value")

    def test_mixed_variable_types(self):
        """Test rules that mix state variables and parameters."""

        # Create test functions that mirror the pattern in real models
        def mixed_rule1(c, m, CRRA):
            return (c / m) ** CRRA

        def mixed_rule2(assets, R, DiscFac):
            return assets * R * DiscFac

        deps1 = extract_dependencies(mixed_rule1)
        deps2 = extract_dependencies(mixed_rule2)

        assert "c" in deps1 and "m" in deps1 and "CRRA" in deps1
        assert "assets" in deps2 and "R" in deps2 and "DiscFac" in deps2

    def test_edge_cases_from_model(self):
        """Test edge cases that might appear in real models."""
        # Constants and simple expressions
        assert extract_dependencies("1.0") == []
        assert extract_dependencies("0") == []

        # Single variable references
        def single_var(wealth):
            return wealth

        assert extract_dependencies(single_var) == ["wealth"]

    def test_realistic_economic_patterns(self):
        """Test common economic modeling patterns."""

        # Budget constraint
        def budget(income, consumption, savings):
            return income - consumption - savings

        assert set(extract_dependencies(budget)) == {"income", "consumption", "savings"}

        # Asset evolution
        def assets(assets_prev, savings, R):
            return assets_prev * R + savings

        assert set(extract_dependencies(assets)) == {"assets_prev", "savings", "R"}

        # Utility function
        def utility(c, gamma):
            return c ** (1 - gamma) / (1 - gamma)

        assert set(extract_dependencies(utility)) == {"c", "gamma"}
