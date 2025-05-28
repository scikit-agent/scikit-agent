# test_model_analyzer.py

import pytest

from skagent.model import DBlock, Control

try:
    from HARK.distributions import Bernoulli, MeanOneLogNormal
except ImportError:
    from econ_ark.HARK.distributions import Bernoulli, MeanOneLogNormal

from skagent.model_analyzer import ModelAnalyzer


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
    Confirm that _prev_deps is a set of (tgt, src) tuples over valid names.
    """
    prev = analysis._prev_deps
    # must be a set of 2-tuples of strings
    assert isinstance(prev, set)
    for pair in prev:
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

    # shock edges
    assert ("shock_1", "state_1") in edges["shock"]
    assert ("shock_2", "state_1") in edges["shock"]

    # instant edges: state->control, control->reward
    assert ("state_1", "control_1") in edges["instant"]
    assert ("control_1", "reward_var") in edges["instant"]

    # no cross-period lags beyond self-reference
    assert all(src == tgt for src, tgt in edges["lag"])


def test_collect_formulas(analysis):
    """
    Verify formulas reflect the code:
    """
    f = analysis.formulas
    # functions produce [Function]
    assert f["state_1"] == "state_1 = [Function]"
    assert f["state_2"] == "state_2 = [Function]"

    # Control formula
    expected = "control_1 = Control(state_1, state_2)"
    assert f["control_1"] == expected

    # reward body comes from lambda
    # reward_var formula is not introspectable -> [Function]
    assert f["reward_var"] == "reward_var = [Function]"
