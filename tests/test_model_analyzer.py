"""
Test cases for ModelAnalyzer using the unittest framework.
"""

import unittest
import json
import sys

# --- Core Imports ---
sys.path.append("../src")
from skagent.models.consumer import consumption_block
from skagent.model import DBlock
from skagent.distributions import Bernoulli
from skagent.model_analyzer import ModelAnalyzer


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


def get_simple_block():
    """Provides a simple test block."""
    return DBlock(
        name="simple",
        shocks={"eps": (Bernoulli, {"p": 0.5})},
        dynamics={"x": lambda eps: eps, "y": lambda x, param: x + param},
        reward={"y": "agent1"},
    )


class TestModelAnalyzerCore(unittest.TestCase):
    """Test core ModelAnalyzer functionality."""

    def setUp(self):
        """Set up the test environment before each test."""
        self.calibration = get_calibration()

    def test_node_classification_consumption_model(self):
        """Test node classification on actual consumption model."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        result = analyzer.analyze()
        kinds = {meta["kind"] for meta in result.node_meta.values()}
        expected_kinds = {"shock", "state", "control", "reward", "param"}
        self.assertTrue(expected_kinds.issubset(kinds))
        self.assertIn("c", result.node_meta)
        self.assertIn("u", result.node_meta)
        self.assertEqual(result.node_meta["c"]["kind"], "control")
        self.assertEqual(result.node_meta["u"]["kind"], "reward")

    def test_lag_detection_consumption_model(self):
        """Test lag dependency detection on consumption model."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        result = analyzer.analyze()
        lag_edges = result.edges["lag"]
        self.assertGreater(len(lag_edges), 0)
        lag_vars = [var for var in result.node_meta.keys() if var.endswith("*")]
        self.assertGreater(len(lag_vars), 0)

    def test_edge_classification_consumption_model(self):
        """Test edge classification on consumption model."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        result = analyzer.analyze()
        self.assertGreater(len(result.edges["param"]), 0)
        self.assertGreater(len(result.edges["shock"]), 0)
        self.assertGreater(len(result.edges["instant"]), 0)
        param_targets = {target for source, target in result.edges["param"]}
        self.assertIn("u", param_targets)


class TestModelAnalyzerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling. (RESTORED)"""

    def setUp(self):
        """Set up test data."""
        self.simple_block = get_simple_block()

    def test_simple_block_analysis(self):
        """Test with simple custom block."""
        calibration = {"param": 1.0}
        analyzer = ModelAnalyzer(self.simple_block, calibration)
        result = analyzer.analyze()
        self.assertIn("eps", result.node_meta)
        self.assertIn("x", result.node_meta)
        self.assertIn("y", result.node_meta)
        self.assertEqual(result.node_meta["eps"]["kind"], "shock")
        self.assertEqual(result.node_meta["y"]["kind"], "reward")

    def test_empty_calibration(self):
        """Test analyzer with empty calibration."""
        analyzer = ModelAnalyzer(consumption_block, {})
        result = analyzer.analyze()
        self.assertEqual(len(result.edges["param"]), 0)
        param_nodes = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "param"
        ]
        self.assertEqual(len(param_nodes), 0)

    def test_block_with_no_rewards(self):
        """Test block without reward specification."""
        block = DBlock(name="no_rewards", reward={}, dynamics={"x": 1})
        analyzer = ModelAnalyzer(block, {})
        result = analyzer.analyze()
        reward_nodes = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "reward"
        ]
        self.assertEqual(len(reward_nodes), 0)


class TestAnalyzerOutputAndStructure(unittest.TestCase):
    """Test the output format and structure of the analyzer."""

    def setUp(self):
        """Set up the test environment."""
        self.calibration = get_calibration()

    def test_output_json_serializable(self):
        """Test that output is JSON serializable."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        analysis = analyzer.analyze().to_dict()
        try:
            json_str = json.dumps(analysis)
            parsed = json.loads(json_str)
            self.assertIsInstance(parsed, dict)
        except TypeError:
            self.fail("to_dict() output is not JSON serializable")

    def test_output_structure(self):
        """Test output dictionary structure."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        analysis = analyzer.analyze().to_dict()
        required_keys = {"node_meta", "edges", "plates"}
        self.assertSetEqual(set(analysis.keys()), required_keys)
        edge_types = {"instant", "lag", "param", "shock"}
        self.assertSetEqual(set(analysis["edges"].keys()), edge_types)
        for edge_list in analysis["edges"].values():
            for edge in edge_list:
                self.assertIsInstance(edge, tuple)
                self.assertEqual(len(edge), 2)


class TestAgentAssignments(unittest.TestCase):
    """Test agent and plate assignments."""

    def setUp(self):
        """Set up test data."""
        self.simple_block = get_simple_block()
        self.calibration = get_calibration()

    def test_agent_detection(self):
        """Test that agents are correctly detected."""
        analyzer = ModelAnalyzer(consumption_block, self.calibration)
        result = analyzer.analyze()
        agents = {meta["agent"] for meta in result.node_meta.values()}
        self.assertIn("consumer", agents)
        self.assertIsInstance(result.plates, dict)
        self.assertIn("consumer", result.plates)

    def test_block_agent_override(self):
        """Test block-level agent assignment."""
        analyzer = ModelAnalyzer(self.simple_block, {}, block_agent="test_agent")
        result = analyzer.analyze()
        state_vars = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "state"
        ]
        for var in state_vars:
            meta = result.node_meta[var]
            if meta["agent"] != "global":
                self.assertEqual(meta["plate"], "test_agent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
