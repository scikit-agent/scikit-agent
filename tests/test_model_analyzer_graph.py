"""
Tests for the ModelAnalyzer graph refactor (PR 2):

  1. Regression: to_dict() is byte-for-byte equivalent to a golden snapshot
     captured from the pre-refactor code, on the existing consumer benchmark
     models. This guarantees ModelVisualizer's contract is unchanged.
  2. influence_graph(): the SCIM view consumed by skagent.relevance.

Golden baseline summary is recorded in i251_design.md; the full golden dict
lives in tests/data/model_analyzer_golden.json.
"""

import copy
import json
from pathlib import Path

import pytest

from skagent.model_analyzer import ModelAnalyzer
from skagent.relevance import RelevanceGraph
from skagent.models.consumer import (
    consumption_block,
    cons_portfolio_problem,
    calibration,
)

# Snapshot pristine copies at import time (during collection, before any test's
# setUp runs). Other suites mutate these shared module-level blocks in place --
# e.g. test_model.py calls cons_portfolio_problem.construct_shocks(...) -- which
# would otherwise make this order-dependent golden regression flaky.
MODELS = {
    name: copy.deepcopy(model)
    for name, model in {
        "consumption_block": consumption_block,
        "cons_portfolio_problem": cons_portfolio_problem,
    }.items()
}
CALIBRATION = copy.deepcopy(calibration)

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "model_analyzer_golden.json").read_text()
)


def _analyze(name):
    return ModelAnalyzer(MODELS[name], CALIBRATION).analyze()


# ---------------------------------------------------------------------------
# 1. Regression: to_dict() unchanged (ModelVisualizer contract preserved).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
def test_to_dict_matches_golden(name):
    # Round-trip through JSON so tuple edges compare equal to the golden's lists.
    got = json.loads(json.dumps(_analyze(name).to_dict()))
    assert got == GOLDEN[name]


# ---------------------------------------------------------------------------
# 2. Graph is the source of truth.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
def test_edges_derived_from_graph(name):
    a = _analyze(name)
    # Regroup edges straight off G and compare to the public edge lists.
    regrouped = {"instant": [], "lag": [], "param": [], "shock": []}
    for s, t, data in a.G.edges(data=True):
        regrouped[data["kind"]].append((s, t))
    regrouped = {k: sorted(set(v)) for k, v in regrouped.items()}
    assert regrouped == a.edges


@pytest.mark.parametrize("name", list(MODELS))
def test_node_meta_matches_graph_attrs(name):
    a = _analyze(name)
    for node, meta in a.node_meta.items():
        if node.endswith("*"):  # lag-display variables are not graph nodes
            continue
        assert a.G.nodes[node] == meta


# ---------------------------------------------------------------------------
# 3. influence_graph(): the SCIM view.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
def test_influence_graph_kinds_and_no_params(name):
    scim = _analyze(name).influence_graph()
    kinds = {scim.graph.nodes[n]["kind"] for n in scim.graph.nodes}
    assert kinds <= {"chance", "decision", "utility"}
    # No param node survives, and no param-derived edge either.
    assert "param" not in kinds


@pytest.mark.parametrize("name", list(MODELS))
def test_influence_graph_only_causal_edges(name):
    a = _analyze(name)
    scim = a.influence_graph()
    # Every SCIM edge must be an instant or shock edge of G (never param/lag).
    causal = {
        (s, t) for s, t, d in a.G.edges(data=True) if d["kind"] in ("instant", "shock")
    }
    assert set(scim.graph.edges) <= causal


def test_influence_graph_consumption_block_details():
    scim = _analyze("consumption_block").influence_graph()
    assert scim.decisions == ["c"]
    assert scim.decision_agent == {"c": "consumer"}
    assert scim.agent_utilities == {"consumer": ["u"]}
    # Decision parents are the information set (Control.iset) minus params.
    assert scim.parents["c"] == ["m"]


def test_from_scim_integration():
    """The namedtuple unpacks straight into RelevanceGraph.from_scim."""
    scim = _analyze("cons_portfolio_problem").influence_graph()
    rg = RelevanceGraph.from_scim(*scim)
    assert set(rg.nodes()) == set(scim.decisions)
    assert rg.is_acyclic() is True
