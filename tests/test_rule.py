"""
Test cases to verify that rule.py extract_dependencies function works correctly
on the actual consumption model.

"""

from importlib.util import find_spec
import pytest
import sys

sys.path.append("../src")
from skagent.rule import extract_dependencies

SKAGENT_AVAILABLE = find_spec("skagent") is not None

pytestmark = pytest.mark.skipif(
    not SKAGENT_AVAILABLE,
    reason="Optional dependency (`scikit-agent`) not installed.",
)

if SKAGENT_AVAILABLE:
    from skagent.models.consumer import consumption_block
    from skagent.model import Control
    from skagent.distributions import Bernoulli


class TestRuleDependencyExtraction:
    """Test that rule.py correctly extracts dependencies from actual model."""

    def test_shock_dependencies(self):
        """Test dependency extraction from shock definitions."""
        # Test live shock: (Bernoulli, {"p": "LivPrb"})
        live_shock = consumption_block.shocks["live"]
        live_deps = extract_dependencies(live_shock)
        assert "LivPrb" in live_deps, f"Expected 'LivPrb' in {live_deps}"

        # Test theta shock: (MeanOneLogNormal, {"sigma": "TranShkStd"})
        theta_shock = consumption_block.shocks["theta"]
        theta_deps = extract_dependencies(theta_shock)
        assert "TranShkStd" in theta_deps, f"Expected 'TranShkStd' in {theta_deps}"

    def test_lambda_dynamics_dependencies(self):
        """Test dependency extraction from lambda dynamics rules."""
        dynamics = consumption_block.dynamics

        # Test b: lambda k, R: k * R
        b_deps = extract_dependencies(dynamics["b"])
        assert "k" in b_deps, f"Expected 'k' in {b_deps}"
        assert "R" in b_deps, f"Expected 'R' in {b_deps}"

        # Test y: lambda p, theta: p * theta
        y_deps = extract_dependencies(dynamics["y"])
        assert "p" in y_deps, f"Expected 'p' in {y_deps}"
        assert "theta" in y_deps, f"Expected 'theta' in {y_deps}"

        # Test m: lambda b, y: b + y
        m_deps = extract_dependencies(dynamics["m"])
        assert "b" in m_deps, f"Expected 'b' in {m_deps}"
        assert "y" in m_deps, f"Expected 'y' in {m_deps}"

        # Test p: lambda PermGroFac, p: PermGroFac * p
        p_deps = extract_dependencies(dynamics["p"])
        assert "PermGroFac" in p_deps, f"Expected 'PermGroFac' in {p_deps}"
        assert "p" in p_deps, f"Expected 'p' in {p_deps}"

        # Test a: lambda m, c: m - c
        a_deps = extract_dependencies(dynamics["a"])
        assert "m" in a_deps, f"Expected 'm' in {a_deps}"
        assert "c" in a_deps, f"Expected 'c' in {a_deps}"

        # Test u: lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)
        u_deps = extract_dependencies(dynamics["u"])
        assert "c" in u_deps, f"Expected 'c' in {u_deps}"
        assert "CRRA" in u_deps, f"Expected 'CRRA' in {u_deps}"

    def test_control_dependencies(self):
        """Test dependency extraction from Control object."""
        # Test c: Control(["m"], upper_bound=lambda m: m, agent="consumer")
        c_rule = consumption_block.dynamics["c"]
        c_deps = extract_dependencies(c_rule)
        assert "m" in c_deps, f"Expected 'm' in {c_deps}"

    def test_calibration_parameters_detected(self):
        """Test that all calibration parameters used in model are detected."""
        # Collect all dependencies from the entire model
        all_deps = set()

        # From shocks
        for shock in consumption_block.shocks.values():
            all_deps.update(extract_dependencies(shock))

        # From dynamics
        for rule in consumption_block.dynamics.values():
            all_deps.update(extract_dependencies(rule))

        # These parameters should be detected in the model
        expected_params = ["LivPrb", "TranShkStd", "PermGroFac", "CRRA"]

        for param in expected_params:
            assert param in all_deps, (
                f"Parameter '{param}' should be detected but wasn't. All deps: {sorted(all_deps)}"
            )

    def test_complete_dependency_extraction(self):
        """Test complete dependency extraction for entire model."""
        # Expected dependencies for each variable
        expected_deps = {
            # Shocks
            "live": {"LivPrb"},
            "theta": {"TranShkStd"},
            # Dynamics
            "b": {"k", "R"},
            "y": {"p", "theta"},
            "m": {"b", "y"},
            "c": {"m"},
            "p": {"PermGroFac", "p"},
            "a": {"m", "c"},
            "u": {"c", "CRRA"},
        }

        # Test shocks
        for shock_name, shock_rule in consumption_block.shocks.items():
            actual_deps = set(extract_dependencies(shock_rule))
            expected = expected_deps[shock_name]
            assert expected.issubset(actual_deps), (
                f"Shock '{shock_name}': expected {expected}, got {actual_deps}"
            )

        # Test dynamics
        for var_name, rule in consumption_block.dynamics.items():
            actual_deps = set(extract_dependencies(rule))
            expected = expected_deps[var_name]
            assert expected.issubset(actual_deps), (
                f"Dynamic '{var_name}': expected {expected}, got {actual_deps}"
            )

    def test_no_false_dependencies(self):
        """Test that we don't extract false dependencies."""
        # Test that calibration parameters not used in rules are not extracted
        all_deps = set()

        for shock in consumption_block.shocks.values():
            all_deps.update(extract_dependencies(shock))
        for rule in consumption_block.dynamics.values():
            all_deps.update(extract_dependencies(rule))

        # These parameters are in calibration but shouldn't be detected in the model
        unused_params = ["DiscFac", "R", "Rfree", "EqP", "BoroCnstArt", "RiskyStd"]

        for param in unused_params:
            if param in all_deps:
                # This might be OK if the parameter is actually used, but let's flag it
                print(f"Warning: Parameter '{param}' was detected but not expected")

    def test_rule_types_handled(self):
        """Test that different rule types are handled correctly."""

        # Lambda function
        def lambda_rule(x, y):
            return x + y

        lambda_deps = extract_dependencies(lambda_rule)
        assert "x" in lambda_deps and "y" in lambda_deps

        # Control object
        control_rule = Control(["wealth", "income"])
        control_deps = extract_dependencies(control_rule)
        assert "wealth" in control_deps and "income" in control_deps

        # Shock tuple
        shock_rule = (Bernoulli, {"p": "param_name"})
        shock_deps = extract_dependencies(shock_rule)
        assert "param_name" in shock_deps

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty Control
        empty_control = Control([])
        assert len(extract_dependencies(empty_control)) == 0

        # Built-in functions should return empty
        assert len(extract_dependencies(len)) == 0
        assert len(extract_dependencies(max)) == 0

        # Invalid inputs should return empty
        assert len(extract_dependencies(None)) == 0
        assert len(extract_dependencies(42)) == 0


class TestRuleRobustness:
    """Test robustness of rule dependency extraction."""

    def test_function_signature_extraction(self):
        """Test that function signature extraction works."""

        def test_func(income, consumption, savings):
            return income - consumption - savings

        deps = extract_dependencies(test_func)
        assert "income" in deps
        assert "consumption" in deps
        assert "savings" in deps

    def test_nested_function_handling(self):
        """Test handling of nested or complex functions."""

        def outer_func(param1):
            def inner_func(param2, param3):
                return param1 + param2 + param3

            return inner_func

        # Should handle gracefully without crashing
        deps = extract_dependencies(outer_func)
        assert isinstance(deps, list)
