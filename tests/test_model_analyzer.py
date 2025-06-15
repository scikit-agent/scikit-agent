# test_model_analyzer.py

import pytest
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)

# Ensure skagent is importable as a package
try:
    # Use the exact same imports as ModelAnalyzer
    from skagent.model import Control, DBlock, RBlock
    from skagent.model_analyzer import ModelAnalyzer
except ImportError as e:
    print(f"Package import failed: {e}")
    # Fallback: direct module import from correct path
    skagent_path = os.path.join(src_dir, "skagent")
    sys.path.insert(0, skagent_path)

    # Import the modules directly and extract classes
    import importlib.util

    # Import model module
    model_spec = importlib.util.spec_from_file_location(
        "model", os.path.join(skagent_path, "model.py")
    )
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    # Import model_analyzer module
    analyzer_spec = importlib.util.spec_from_file_location(
        "model_analyzer", os.path.join(skagent_path, "model_analyzer.py")
    )
    analyzer_module = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(analyzer_module)

    # Extract classes - this ensures they're the same objects
    DBlock = model_module.DBlock
    Control = model_module.Control
    RBlock = getattr(model_module, "RBlock", None)
    ModelAnalyzer = analyzer_module.ModelAnalyzer

# Handle distributions
try:
    from HARK.distributions import Bernoulli, MeanOneLogNormal
except ImportError:
    try:
        from econ_ark.HARK.distributions import Bernoulli, MeanOneLogNormal
    except ImportError:
        # Mock distributions for testing - must be compatible with the analyzer
        class Bernoulli:
            def __init__(self, **params):
                self.params = params

        class MeanOneLogNormal:
            def __init__(self, **params):
                self.params = params


print(f"Successfully imported DBlock: {DBlock}")
print(f"Successfully imported ModelAnalyzer: {ModelAnalyzer}")


def make_dummy_block():
    """
    Creates a minimal DBlock with:
      - two shocks referencing different string parameters
      - two state transformations (one combining shocks, one combining params)
      - one Control that uses those states
      - one reward that depends on the control
    """
    shocks = {
        "shock_1": (Bernoulli, {"p": "param_1"}),
        "shock_2": (MeanOneLogNormal, {"sigma": "param_2"}),
    }

    # state_1 combines shock_1 and shock_2
    def state_1(shock_1, shock_2):
        return shock_1 + shock_2

    # state_2 combines param_1 and param_2
    def state_2(param_1, param_2):
        return param_1 * param_2

    dynamics = {
        "state_1": state_1,
        "state_2": state_2,
        "control_1": Control(
            iset=["state_1", "state_2"],
            lower_bound=lambda s1, s2: 0,
            upper_bound=lambda s1, s2: s1 + s2,
        ),
        # reward_var is included in dynamics to test dynamic parsing
        "reward_var": lambda control_1: control_1,
    }

    # reward mapping uses a non-global agent name
    reward = {"reward_var": "agent"}

    return DBlock(name="dummy", shocks=shocks, dynamics=dynamics, reward=reward)


def infer_calibration(block: DBlock):
    """
    Extracts all string-named parameters from the block's shocks for calibration.
    """
    cal = {}
    for _, spec in block.shocks.items():
        dist, params = spec
        if isinstance(params, dict):
            for v in params.values():
                if isinstance(v, str):
                    cal[v] = 1.0
    return cal


@pytest.fixture
def analysis():
    # build dummy block and calibration
    blk = make_dummy_block()
    cal = infer_calibration(blk)
    # pick a dynamic variable as observable
    obs = ["reward_var"]
    # run analysis
    return ModelAnalyzer(blk, cal, observables=obs).analyze()


def test_walk_and_collect_nodes(analysis):
    """
    Verify that every shock, dynamic (excluding reward), and parameter appears in node_meta,
    and that reward variables are correctly labeled.
    """
    meta = analysis.node_meta
    blk = analysis.model

    # shocks and dynamics (except reward) and calibration keys should be subsets of node_meta
    assert set(blk.shocks.keys()).issubset(meta)
    dyn_vars = set(blk.dynamics.keys()) - set(blk.reward.keys())
    assert dyn_vars.issubset(meta)
    assert set(analysis.calibration.keys()).issubset(meta)

    # each shock must be labeled 'shock'
    for s in blk.shocks:
        assert meta[s]["kind"] == "shock"

    # each dynamic not a Control and not reward is 'state'
    for var, rule in blk.dynamics.items():
        if var in blk.reward:
            continue
        expected_kind = "control" if isinstance(rule, Control) else "state"
        assert meta[var]["kind"] == expected_kind

    # reward var must be labeled 'reward'
    for rv in blk.reward:
        assert meta[rv]["kind"] == "reward"

    # parameters must be labeled 'param'
    for p in analysis.calibration:
        if p in meta:  # Only check if parameter is in meta
            assert meta[p]["kind"] == "param"


def test_collect_dependencies_and_param_deps(analysis):
    """
    Ensure raw and param dependencies capture at least the key relationships:
    - shock_1 ← param_1
    - shock_2 ← param_2
    - state_1 ← shock_1, shock_2
    - state_2 ← param_1, param_2
    - reward_var ← control_1
    """
    raw = analysis._raw_deps
    pd = analysis._param_deps

    # shock↔param
    assert "param_1" in pd and "shock_1" in pd["param_1"]
    assert "param_2" in pd and "shock_2" in pd["param_2"]

    # state_1 depends on both shocks
    deps_s1 = set(raw.get("state_1", []))
    assert {"shock_1", "shock_2"}.issubset(deps_s1)

    # state_2 depends on both params
    deps_s2 = set(raw.get("state_2", []))
    assert {"param_1", "param_2"}.issubset(deps_s2)

    # reward depends on control
    deps_r = raw.get("reward_var", [])
    assert "control_1" in deps_r


def test_identify_time_dependencies(analysis):
    """
    Confirm that _time_deps is a set of (tgt, src) tuples over valid names.
    """
    time_deps = analysis._time_deps
    # must be a set of 2-tuples of strings
    assert isinstance(time_deps, set)
    for pair in time_deps:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert all(isinstance(x, str) for x in pair)


def test_assemble_edges(analysis):
    """
    Check that assemble_edges classifies edges correctly:
    """
    edges = analysis.edges

    # param edges include key pairs
    assert ("param_1", "shock_1") in edges["param"]
    assert ("param_1", "state_2") in edges["param"]
    assert ("param_2", "shock_2") in edges["param"]
    assert ("param_2", "state_2") in edges["param"]

    # shock edges
    assert ("shock_1", "state_1") in edges["shock"]
    assert ("shock_2", "state_1") in edges["shock"]

    # instant edges: state->control, control->reward
    assert ("state_1", "control_1") in edges["instant"]
    assert ("state_2", "control_1") in edges[
        "instant"
    ]  # Control depends on both states
    assert ("control_1", "reward_var") in edges["instant"]

    # In this simple case, there should be no lag dependencies
    # since there are no forward references or self-dependencies
    # But we should check the structure is correct
    assert isinstance(edges["lag"], list)


def test_collect_formulas(analysis):
    """
    Verify formulas reflect the code structure:
    """
    f = analysis.formulas

    # Parameters should show their values
    assert f["param_1"] == "param_1 = 1.0"
    assert f["param_2"] == "param_2 = 1.0"

    # Functions should show [Function] since they can't be introspected easily
    assert f["state_1"] == "state_1 = [Function]"
    assert f["state_2"] == "state_2 = [Function]"

    # Control formula should show dependencies and bounds info
    assert "Control" in f["control_1"]
    assert "state_1" in f["control_1"]
    assert "state_2" in f["control_1"]
    assert "lower_bound" in f["control_1"]
    assert "upper_bound" in f["control_1"]

    # Lambda function - based on actual output format
    # The formula might be [Unknown] or show lambda content depending on implementation
    reward_formula = f["reward_var"]
    assert (
        "reward_var" in reward_formula
    )  # At minimum, should contain the variable name


def test_node_metadata_structure(analysis):
    """
    Test that node metadata has the correct structure for all node types.
    """
    meta = analysis.node_meta

    # Check that all nodes have required metadata fields
    for var, data in meta.items():
        assert "kind" in data
        assert "agent" in data
        assert "plate" in data
        assert "observed" in data

        # Check specific properties by kind
        if data["kind"] == "shock":
            assert data["agent"] == "global"
            assert data["plate"] is None
            assert data["observed"] is False

        elif data["kind"] == "param":
            assert data["agent"] == "global"
            assert data["plate"] is None
            assert data["observed"] is False

        elif data["kind"] == "control":
            assert data["observed"] is True  # Controls are observed

        elif data["kind"] == "reward":
            assert data["observed"] is True  # Rewards are observed


def test_plates_collection(analysis):
    """
    Test that plates are collected correctly from agent assignments.
    """
    # Since we don't specify block_agent, most variables should be global
    # But the reward has agent="agent" in the reward mapping
    plates = analysis.plates

    # The plates dict should be properly structured
    assert isinstance(plates, dict)

    # Each plate should have label and size
    for plate_name, plate_info in plates.items():
        assert "label" in plate_info
        assert "size" in plate_info


def test_to_dict_serialization(analysis):
    """
    Test that to_dict returns a properly structured dictionary.
    """
    result = analysis.to_dict()

    # Check required keys
    assert "node_meta" in result
    assert "edges" in result
    assert "formulas" in result
    assert "plates" in result

    # Check edges structure
    edges = result["edges"]
    assert "instant" in edges
    assert "lag" in edges
    assert "param" in edges
    assert "shock" in edges

    # All edge lists should be lists of tuples
    for edge_type, edge_list in edges.items():
        assert isinstance(edge_list, list)
        for edge in edge_list:
            assert isinstance(edge, tuple)
            assert len(edge) == 2


def test_empty_dependencies():
    """
    Test analyzer behavior with minimal dependencies.
    """
    # Create a block with no dependencies
    shocks = {}
    dynamics = {
        "simple_var": lambda: 1.0  # No dependencies
    }
    reward = {}

    block = DBlock(name="simple", shocks=shocks, dynamics=dynamics, reward=reward)
    analyzer = ModelAnalyzer(block, {})
    result = analyzer.analyze()

    # Should not crash and should have the variable
    assert "simple_var" in result.node_meta
    assert result.node_meta["simple_var"]["kind"] == "state"


def test_forward_reference_lag_detection():
    """
    Test that forward references are correctly identified as lag dependencies.
    """
    # Create dynamics with forward reference
    shocks = {}
    dynamics = {
        "var_a": lambda var_b: var_b + 1,  # Forward reference to var_b
        "var_b": lambda var_a: var_a * 2,  # Backward reference to var_a
    }
    reward = {}

    block = DBlock(name="test", shocks=shocks, dynamics=dynamics, reward=reward)
    analyzer = ModelAnalyzer(block, {})
    result = analyzer.analyze()

    # var_a depends on var_b (forward ref) -> should be lag
    assert ("var_a", "var_b") in result._time_deps

    # Check edges
    assert ("var_b", "var_a") in result.edges["lag"]  # var_b* -> var_a
    assert ("var_a", "var_b") in result.edges["instant"]  # var_a -> var_b

    # Check lag variable metadata is created
    assert "var_b*" in result.node_meta


def test_rblock_handling():
    """
    Test that RBlocks are properly handled and correctly flattened to DBlocks.
    """
    # Create a simple DBlock
    dblock1 = make_dummy_block()

    # Create another DBlock for testing multiple blocks
    dblock2 = DBlock(
        name="second_block",
        shocks={"shock_3": (Bernoulli, {"p": "param_3"})},
        dynamics={"state_3": lambda shock_3: shock_3 * 2},
        reward={},
    )

    # Check the actual RBlock class from our imports
    try:
        # Get the RBlock class that was imported at the top
        # We need to import it the same way ModelAnalyzer does
        from skagent.model import RBlock

        # Inspect RBlock to understand its constructor
        import inspect

        sig = inspect.signature(RBlock.__init__)
        print(f"RBlock constructor: {sig}")

        # Try different ways to construct RBlock based on common patterns
        try:
            # Try: RBlock(blocks=[...])
            rblock = RBlock(blocks=[dblock1, dblock2])
        except TypeError:
            try:
                # Try: RBlock(name="...", blocks=[...])
                rblock = RBlock(name="test_rblock", blocks=[dblock1, dblock2])
            except TypeError:
                try:
                    # Try: RBlock([...])
                    rblock = RBlock([dblock1, dblock2])
                except TypeError:
                    # Try: empty constructor then set blocks
                    rblock = RBlock()
                    rblock.blocks = [dblock1, dblock2]

        print(f"Successfully created RBlock: {type(rblock)}")
        print(f"RBlock has blocks: {hasattr(rblock, 'blocks')}")
        print(
            f"Number of blocks: {len(rblock.blocks) if hasattr(rblock, 'blocks') else 'N/A'}"
        )

        # Now test the analyzer
        cal = {**infer_calibration(dblock1), "param_3": 1.0}
        analyzer = ModelAnalyzer(rblock, cal)
        result = analyzer.analyze()

        # Should have flattened to 2 DBlocks
        assert len(result._blocks) == 2

        # Should contain variables from both blocks
        assert "state_1" in result.node_meta  # from dblock1
        assert "state_3" in result.node_meta  # from dblock2
        assert "control_1" in result.node_meta  # from dblock1

        print("✓ RBlock test passed!")

    except Exception as e:
        print(f"Error in RBlock test: {e}")
        import traceback

        traceback.print_exc()

        # Re-raise to fail the test so we can see what went wrong
        raise


def test_invalid_model_type():
    """
    Test that invalid model types raise ValueError.
    """
    with pytest.raises(ValueError, match="Model must be a DBlock or RBlock"):
        ModelAnalyzer(model="invalid", calibration={})
