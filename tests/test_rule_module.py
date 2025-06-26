"""
Tests for rule_module.py using real scikit-agent objects and scenarios.

Based on actual consumption-savings model structure.
"""

import pytest
from skagent.model import Control
from HARK.distributions import Bernoulli

from skagent.rule_module import (
    format_rule,
    extract_dependencies,
    validate_rule,
    get_rule_type,
)


# Real calibration and block similar to user's example
@pytest.fixture
def sample_calibration():
    """Real calibration parameters."""
    return {
        "DiscFac": 0.96,
        "CRRA": 2.0,
        "Rfree": 1.03,
        "LivPrb": 0.98,
        "PermGroFac": 1.01,
        "BoroCnstArt": None,
    }


@pytest.fixture
def sample_rules(sample_calibration):
    """Real rules from consumption-savings model."""
    cal = sample_calibration
    return {
        # Real lambda functions from dynamics
        "simple_lambda": lambda p: p,
        "multi_param_lambda": lambda Rfree, a, y: Rfree * a + y,
        "asset_rule": lambda m, c: m - c,
        "growth_rule": lambda PermGroFac, p: PermGroFac * p,
        # Real Control object
        "consumption_control": Control(
            ["m"], upper_bound=lambda m: m, agent="consumer"
        ),
        # Real reward function
        "utility": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        # Real shock distribution
        "survival_shock": Bernoulli(p=cal["LivPrb"]),
        # String expressions (common in models)
        "production": "alpha * K + beta * L",
        # Parameters
        "discount_factor": cal["DiscFac"],
        "risk_aversion": cal["CRRA"],
    }


class TestFormatRule:
    """Test format_rule with real model components."""

    def test_simple_lambda(self, sample_rules):
        """Test simple lambda: y = lambda p: p"""
        rule = sample_rules["simple_lambda"]
        result = format_rule("y", rule)

        assert "y = " in result
        assert "p" in result

    def test_multi_param_lambda(self, sample_rules):
        """Test multi-parameter lambda: m = lambda Rfree, a, y: Rfree * a + y"""
        rule = sample_rules["multi_param_lambda"]
        result = format_rule("m", rule)

        assert "m = " in result
        assert "Rfree * a + y" in result

    def test_real_control(self, sample_rules):
        """Test real Control object."""
        control = sample_rules["consumption_control"]
        result = format_rule("c", control)

        assert "c = Control(" in result
        assert "m" in result
        assert "upper_bound" in result

    def test_utility_function(self, sample_rules):
        """Test utility function lambda."""
        rule = sample_rules["utility"]
        result = format_rule("u", rule)

        assert "u = " in result
        # Should contain the CRRA utility formula
        assert "CRRA" in result

    def test_parameters(self, sample_rules):
        """Test parameter formatting."""
        result = format_rule("DiscFac", sample_rules["discount_factor"])
        assert result == "DiscFac = 0.96"

        result = format_rule("CRRA", sample_rules["risk_aversion"])
        assert result == "CRRA = 2.0"

    def test_string_expression(self, sample_rules):
        """Test string expression."""
        result = format_rule("Y", sample_rules["production"])
        assert result == "Y = alpha * K + beta * L"


class TestExtractDependencies:
    """Test extract_dependencies with real model components."""

    def test_simple_lambda_deps(self, sample_rules):
        """Test dependencies from simple lambda."""
        rule = sample_rules["simple_lambda"]
        deps = extract_dependencies(rule)
        assert deps == ["p"]

    def test_multi_param_lambda_deps(self, sample_rules):
        """Test dependencies from multi-parameter lambda."""
        rule = sample_rules["multi_param_lambda"]
        deps = extract_dependencies(rule)
        assert set(deps) == {"Rfree", "a", "y"}

    def test_asset_rule_deps(self, sample_rules):
        """Test asset accumulation rule dependencies."""
        rule = sample_rules["asset_rule"]
        deps = extract_dependencies(rule)
        assert set(deps) == {"m", "c"}

    def test_real_control_deps(self, sample_rules):
        """Test Control object dependencies."""
        control = sample_rules["consumption_control"]
        deps = extract_dependencies(control)
        assert deps == ["m"]

    def test_utility_deps(self, sample_rules):
        """Test utility function dependencies."""
        rule = sample_rules["utility"]
        deps = extract_dependencies(rule)
        assert set(deps) == {"c", "CRRA"}

    def test_string_deps(self, sample_rules):
        """Test string expression dependencies."""
        deps = extract_dependencies(sample_rules["production"])
        assert set(deps) == {"alpha", "K", "beta", "L"}

    def test_parameter_deps(self, sample_rules):
        """Test that parameters have no dependencies."""
        assert extract_dependencies(sample_rules["discount_factor"]) == []
        assert extract_dependencies(sample_rules["risk_aversion"]) == []


class TestValidateRule:
    """Test rule validation with real components."""

    def test_validate_lambdas(self, sample_rules):
        """Test that all lambda functions are valid."""
        lambda_rules = [
            "simple_lambda",
            "multi_param_lambda",
            "asset_rule",
            "growth_rule",
            "utility",
        ]

        for rule_name in lambda_rules:
            rule = sample_rules[rule_name]
            assert validate_rule(rule) == True

    def test_validate_control(self, sample_rules):
        """Test that Control object is valid."""
        control = sample_rules["consumption_control"]
        assert validate_rule(control) == True

    def test_validate_parameters(self, sample_rules):
        """Test that parameters are valid."""
        assert validate_rule(sample_rules["discount_factor"]) == True
        assert validate_rule(sample_rules["risk_aversion"]) == True

    def test_validate_string(self, sample_rules):
        """Test that string expressions are valid."""
        assert validate_rule(sample_rules["production"]) == True

    def test_invalid_string(self):
        """Test that empty strings are invalid."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_rule("")


class TestGetRuleType:
    """Test rule type classification with real components."""

    def test_lambda_types(self, sample_rules):
        """Test that lambdas are classified as 'callable'."""
        lambda_rules = [
            "simple_lambda",
            "multi_param_lambda",
            "asset_rule",
            "growth_rule",
            "utility",
        ]

        for rule_name in lambda_rules:
            rule = sample_rules[rule_name]
            assert get_rule_type(rule) == "callable"

    def test_control_type(self, sample_rules):
        """Test that Control is classified as 'control'."""
        control = sample_rules["consumption_control"]
        assert get_rule_type(control) == "control"

    def test_distribution_type(self, sample_rules):
        """Test that distribution is classified as 'distribution'."""
        shock = sample_rules["survival_shock"]
        assert get_rule_type(shock) == "distribution"

    def test_parameter_types(self, sample_rules):
        """Test that parameters are classified as 'constant'."""
        assert get_rule_type(sample_rules["discount_factor"]) == "constant"
        assert get_rule_type(sample_rules["risk_aversion"]) == "constant"

    def test_string_type(self, sample_rules):
        """Test that strings are classified as 'string'."""
        assert get_rule_type(sample_rules["production"]) == "string"


class TestConsumptionModelIntegration:
    """Integration test with a complete consumption-savings model."""

    def test_complete_consumption_block(self, sample_calibration):
        """Test all rules from a complete consumption block."""
        cal = sample_calibration

        # Create the actual block structure
        rules = {
            # Dynamics
            "y": lambda p: p,
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
            # Reward
            "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
            # Shock
            "live": Bernoulli(p=cal["LivPrb"]),
        }

        # Test that all rules can be processed
        for var_name, rule in rules.items():
            # Format should work
            formatted = format_rule(var_name, rule)
            assert f"{var_name} = " in formatted

            # Dependencies should be extractable
            deps = extract_dependencies(rule)
            assert isinstance(deps, list)

            # Type should be classifiable
            rule_type = get_rule_type(rule)
            assert rule_type in ["callable", "control", "distribution"]

            # Should be valid
            assert validate_rule(rule) == True

    def test_parameter_processing(self, sample_calibration):
        """Test processing of calibration parameters."""
        for param_name, param_value in sample_calibration.items():
            if param_value is not None:  # Skip None values
                formatted = format_rule(param_name, param_value)
                assert f"{param_name} = {param_value}" == formatted

                deps = extract_dependencies(param_value)
                assert deps == []  # Parameters have no dependencies

                assert get_rule_type(param_value) == "constant"
                assert validate_rule(param_value) == True


class TestEconomicSemantics:
    """Test that the module correctly handles economic model semantics."""

    def test_budget_constraint(self):
        """Test budget constraint: a = m - c"""
        budget_rule = lambda m, c: m - c

        formatted = format_rule("a", budget_rule)
        deps = extract_dependencies(budget_rule)

        assert "a = " in formatted
        assert "m - c" in formatted
        assert set(deps) == {"m", "c"}

    def test_cash_on_hand(self):
        """Test cash-on-hand: m = R * a + y"""
        coh_rule = lambda R, a, y: R * a + y

        formatted = format_rule("m", coh_rule)
        deps = extract_dependencies(coh_rule)

        assert "m = " in formatted
        assert set(deps) == {"R", "a", "y"}

    def test_crra_utility(self):
        """Test CRRA utility function."""
        utility_rule = lambda c, rho: c ** (1 - rho) / (1 - rho)

        formatted = format_rule("u", utility_rule)
        deps = extract_dependencies(utility_rule)

        assert "u = " in formatted
        assert set(deps) == {"c", "rho"}
