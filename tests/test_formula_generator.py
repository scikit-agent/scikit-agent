"""
Test cases for FormulaGenerator using the unittest framework.
"""

import unittest
import sys

# --- Core Imports ---
sys.path.append("../src")
from skagent.models.consumer import consumption_block
from skagent.model import DBlock, Control
from skagent.distributions import Bernoulli
from skagent.formula_generator import FormulaGenerator


# Helper functions to provide test data
def get_calibration():
    """Provides standard calibration for consumption model."""
    return {
        "DiscFac": 0.96,
        "CRRA": 2.0,
        "R": 1.03,
        "Rfree": 1.03,
        "EqP": 0.02,
        "LivPrb": 0.98,
        "PermGroFac": 1.01,
        "BoroConstArt": None,
        "TranShkStd": 0.1,
        "RiskyStd": 0.1,
    }


def get_simple_block_with_control():
    """Provides a simple test block that includes a Control variable."""
    block = DBlock(
        name="simple",
        shocks={"eps": (Bernoulli, {"p": 0.5})},
        dynamics={"x": lambda eps: eps, "y": lambda x, param: x + param},
        reward={"y": "agent1"},
    )
    block.dynamics["ctrl"] = Control(["x"], agent="test_agent")
    return block


class TestFormulaGenerator(unittest.TestCase):
    """Test formula generation functionality."""

    def setUp(self):
        """Set up the test environment before each test."""
        self.calibration = get_calibration()
        self.simple_block = get_simple_block_with_control()

    def test_formula_generation(self):
        """Test that formulas are generated correctly for the consumption model."""
        generator = FormulaGenerator(consumption_block, self.calibration)
        formulas = generator.generate()

        self.assertGreater(len(formulas), 0)

        for var, formula_str in formulas.items():
            self.assertIn(" = ", formula_str)
            self.assertTrue(formula_str.startswith(var))

    def test_control_formula_format(self):
        """Test Control object formula formatting."""
        calibration = {"param": 1.0}
        generator = FormulaGenerator(self.simple_block, calibration)
        formulas = generator.generate()

        self.assertIn("ctrl", formulas)
        self.assertIn("Control", formulas["ctrl"])
        self.assertIn("x", formulas["ctrl"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
