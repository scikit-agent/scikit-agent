"""
Test cases to verify that ModelAnalyzer works correctly
on actual scikit-agent models.
"""

from importlib.util import find_spec
import pytest
import sys

sys.path.append("../src")

SKAGENT_AVAILABLE = find_spec("skagent") is not None
HAS_DEPENDENCIES = SKAGENT_AVAILABLE

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="Optional dependencies (`scikit-agent`) not installed.",
)

if HAS_DEPENDENCIES:
    from skagent.models.consumer import consumption_block
    from skagent.model import Control, DBlock
    from skagent.distributions import Bernoulli
    from skagent.model_analyzer import ModelAnalyzer


@pytest.fixture
def calibration():
    """Standard calibration for consumption model."""
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


@pytest.fixture
def simple_block():
    """Simple test block for basic functionality."""
    return DBlock(
        name="simple",
        shocks={"eps": (Bernoulli, {"p": 0.5})},
        dynamics={
            "x": lambda eps: eps,
            "y": lambda x, param: x + param,
        },
        reward={"y": "agent1"},
    )


class TestModelAnalyzerCore:
    """Test core ModelAnalyzer functionality."""

    def test_initialization_with_consumption_block(self, calibration):
        """Test analyzer initialization with actual consumption block."""
        analyzer = ModelAnalyzer(consumption_block, calibration)

        assert analyzer.model == consumption_block
        assert analyzer.calibration == calibration
        assert len(analyzer._blocks) > 0

    def test_node_classification_consumption_model(self, calibration):
        """Test node classification on actual consumption model."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()

        # Check that we have all expected node types
        kinds = {meta["kind"] for meta in result.node_meta.values()}
        expected_kinds = {"shock", "state", "control", "reward", "param"}
        assert expected_kinds.issubset(kinds)

        # Check specific variables exist
        assert "c" in result.node_meta  # consumption control
        assert "u" in result.node_meta  # utility reward
        assert result.node_meta["c"]["kind"] == "control"
        assert result.node_meta["u"]["kind"] == "reward"

    def test_lag_detection_consumption_model(self, calibration):
        """Test lag dependency detection on consumption model."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()

        # Should detect lag dependencies
        lag_edges = result.edges["lag"]
        assert len(lag_edges) > 0

        # Check that lag variables are created
        lag_vars = [var for var in result.node_meta.keys() if var.endswith("*")]
        assert len(lag_vars) > 0

    def test_edge_classification_consumption_model(self, calibration):
        """Test edge classification on consumption model."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()

        # Should have all edge types
        assert len(result.edges["param"]) > 0  # Parameter dependencies
        assert len(result.edges["shock"]) > 0  # Shock dependencies
        assert len(result.edges["instant"]) > 0  # Instant dependencies

        # Check parameter edges target correct variables
        param_targets = {target for source, target in result.edges["param"]}
        assert "u" in param_targets  # utility depends on CRRA


class TestModelAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    def test_simple_block_analysis(self, simple_block):
        """Test with simple custom block."""
        calibration = {"param": 1.0}
        analyzer = ModelAnalyzer(simple_block, calibration)
        result = analyzer.analyze()

        assert "eps" in result.node_meta
        assert "x" in result.node_meta
        assert "y" in result.node_meta
        assert result.node_meta["eps"]["kind"] == "shock"
        assert result.node_meta["y"]["kind"] == "reward"

    def test_empty_calibration(self):
        """Test analyzer with empty calibration."""
        analyzer = ModelAnalyzer(consumption_block, {})
        result = analyzer.analyze()

        # Should still work, just no parameter nodes/edges
        assert len(result.edges["param"]) == 0
        param_nodes = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "param"
        ]
        assert len(param_nodes) == 0

    def test_block_with_no_rewards(self):
        """Test block without reward specification."""
        block = DBlock(
            name="no_rewards",
            shocks={"eps": (Bernoulli, {"p": 0.5})},
            dynamics={"x": lambda eps: eps},
            reward={},
        )

        analyzer = ModelAnalyzer(block, {})
        result = analyzer.analyze()

        reward_nodes = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "reward"
        ]
        assert len(reward_nodes) == 0


class TestFormulasAndOutput:
    """Test formula generation and output format."""

    def test_formula_generation(self, calibration):
        """Test that formulas are generated correctly."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()

        # Should have formulas for dynamics
        assert len(result.formulas) > 0

        # Check formula format
        for var, formula in result.formulas.items():
            assert " = " in formula
            assert formula.startswith(var)

    def test_control_formula_format(self, simple_block):
        """Test Control object formula formatting."""
        calibration = {"param": 1.0}
        analyzer = ModelAnalyzer(simple_block, calibration)

        # Add a control variable to test
        simple_block.dynamics["ctrl"] = Control(["x"], agent="test_agent")

        result = analyzer.analyze()

        # Control formula should mention Control
        if "ctrl" in result.formulas:
            assert "Control" in result.formulas["ctrl"]

    def test_output_json_serializable(self, calibration):
        """Test that output is JSON serializable."""
        import json

        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()
        analysis = result.to_dict()

        # Should not raise exception
        json_str = json.dumps(analysis)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_output_structure(self, calibration):
        """Test output dictionary structure."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()
        analysis = result.to_dict()

        required_keys = {"node_meta", "edges", "formulas", "plates"}
        assert set(analysis.keys()) == required_keys

        # Check edges structure
        edge_types = {"instant", "lag", "param", "shock"}
        assert set(analysis["edges"].keys()) == edge_types

        # All edges should be tuples of length 2
        for edge_list in analysis["edges"].values():
            for edge in edge_list:
                assert isinstance(edge, tuple)
                assert len(edge) == 2


class TestAgentAssignments:
    """Test agent and plate assignments."""

    def test_agent_detection(self, calibration):
        """Test that agents are correctly detected."""
        analyzer = ModelAnalyzer(consumption_block, calibration)
        result = analyzer.analyze()

        # Should have consumer agent
        agents = {meta["agent"] for meta in result.node_meta.values()}
        assert "consumer" in agents

        # Should have consumer plate
        assert isinstance(result.plates, dict)

    def test_block_agent_override(self, simple_block):
        """Test block-level agent assignment."""
        analyzer = ModelAnalyzer(simple_block, {}, block_agent="test_agent")
        result = analyzer.analyze()

        # Non-global variables should use block agent
        state_vars = [
            var for var, meta in result.node_meta.items() if meta["kind"] == "state"
        ]

        for var in state_vars:
            meta = result.node_meta[var]
            if meta["agent"] != "global":
                assert meta["plate"] == "test_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
