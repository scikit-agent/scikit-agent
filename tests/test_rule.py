"""
Tests for rule_module.py using real scikit-agent objects and scenarios.

Based on actual consumption-savings model structure.
"""

import pytest
import sys
sys.path.append('../src')
from skagent.rule import extract_dependencies

# Try to import scikit-agent components
try:
    from skagent.model import Control
    from HARK.distributions import Bernoulli, MeanOneLogNormal
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


class TestExtractDependencies:
    
    def test_lambda_dynamics_rules(self):
        """Test dependency extraction from actual lambda dynamics rules."""
        # From the consumption model dynamics
        b_rule = lambda k, R: k * R
        y_rule = lambda p, theta: p * theta
        m_rule = lambda b, y: b + y
        p_rule = lambda PermGroFac, p: PermGroFac * p
        a_rule = lambda m, c: m - c
        u_rule = lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)
        
        assert extract_dependencies(b_rule) == ['k', 'R']
        assert extract_dependencies(y_rule) == ['p', 'theta']
        assert extract_dependencies(m_rule) == ['b', 'y']
        assert extract_dependencies(p_rule) == ['PermGroFac', 'p']
        assert extract_dependencies(a_rule) == ['m', 'c']
        assert extract_dependencies(u_rule) == ['c', 'CRRA']

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_control_rule(self):
        """Test dependency extraction from actual Control rule."""
        # From the consumption model: c = Control(["m"], upper_bound=lambda m: m, agent="consumer")
        c_rule = Control(["m"], upper_bound=lambda m: m, agent="consumer")
        
        assert extract_dependencies(c_rule) == ['m']

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="scikit-agent not available")
    def test_shock_tuple_rules(self):
        """Test dependency extraction from actual shock tuple rules."""
        # From the consumption model shocks
        live_shock = (Bernoulli, {"p": "LivPrb"})
        theta_shock = (MeanOneLogNormal, {"sigma": "TranShkStd"})
        
        live_deps = extract_dependencies(live_shock)
        theta_deps = extract_dependencies(theta_shock)
        
        assert 'LivPrb' in live_deps
        assert 'TranShkStd' in theta_deps

    def test_string_rules(self):
        """Test dependency extraction from string expressions."""
        # Common patterns in economic models
        assert 'income' in extract_dependencies("income - consumption")
        assert 'consumption' in extract_dependencies("income - consumption")
        
        assert 'beta' in extract_dependencies("beta * future_value")
        assert 'future_value' in extract_dependencies("beta * future_value")

    def test_parameter_dependencies(self):
        """Test extraction of calibration parameters from rules."""
        # These should extract parameter names that would be in calibration dict
        crra_rule = lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)
        discount_rule = lambda u_next, DiscFac: DiscFac * u_next
        growth_rule = lambda p, PermGroFac: PermGroFac * p
        
        assert 'CRRA' in extract_dependencies(crra_rule)
        assert 'DiscFac' in extract_dependencies(discount_rule)
        assert 'PermGroFac' in extract_dependencies(growth_rule)

    def test_mixed_variable_types(self):
        """Test rules that mix state variables and parameters."""
        # Rules that depend on both model variables and calibration parameters
        mixed_rule1 = lambda c, m, CRRA: (c / m) ** CRRA
        mixed_rule2 = lambda assets, R, DiscFac: assets * R * DiscFac
        
        deps1 = extract_dependencies(mixed_rule1)
        deps2 = extract_dependencies(mixed_rule2)
        
        assert 'c' in deps1 and 'm' in deps1 and 'CRRA' in deps1
        assert 'assets' in deps2 and 'R' in deps2 and 'DiscFac' in deps2

    def test_edge_cases_from_model(self):
        """Test edge cases that might appear in real models."""
        # Constants and simple expressions
        assert extract_dependencies("1.0") == []
        assert extract_dependencies("0") == []
        
        # Single variable references
        single_var = lambda wealth: wealth
        assert extract_dependencies(single_var) == ['wealth']

    def test_realistic_economic_patterns(self):
        """Test common economic modeling patterns."""
        # Budget constraint
        budget = lambda income, consumption, savings: income - consumption - savings
        assert set(extract_dependencies(budget)) == {'income', 'consumption', 'savings'}
        
        # Asset evolution
        assets = lambda assets_prev, savings, R: assets_prev * R + savings
        assert set(extract_dependencies(assets)) == {'assets_prev', 'savings', 'R'}
        
        # Utility function
        utility = lambda c, gamma: c ** (1 - gamma) / (1 - gamma)
        assert set(extract_dependencies(utility)) == {'c', 'gamma'}
