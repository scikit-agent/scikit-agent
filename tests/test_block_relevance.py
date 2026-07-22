"""
Tests for the Block strategic-relevance API (PR 3):
``Block.relevance_graph`` and ``Block.relies_on``, exercised end-to-end
(Block -> ModelAnalyzer.influence_graph -> RelevanceGraph) on the Tree Killer
benchmark and a single-decision economic block.

Tree Killer (Koller & Milch 2001, Fig. 1) has the known relevance graph of
Fig. 4a; the expected values below were cross-checked against PyCID's
story_macids.tree_doctor (see the oracle test at the end).
"""

import copy

import pytest

from skagent.models.macid import tree_killer_block
from skagent.models.consumer import consumption_block

# Pristine copy: other suites mutate the shared consumption_block in place.
CONSUMPTION_BLOCK = copy.deepcopy(consumption_block)


# ---------------------------------------------------------------------------
# Tree Killer: the full assertion table.
# ---------------------------------------------------------------------------
def test_tree_killer_relevance_graph():
    rg = tree_killer_block.relevance_graph()
    assert set(rg.edges()) == {("PT", "BP"), ("PT", "TDoc"), ("BP", "TDoc")}
    assert set(rg.nodes()) == {"PT", "BP", "TDoc"}
    assert rg.is_acyclic() is True
    # Backward-induction solve order: leaves (relies on nothing) first.
    assert rg.condensation() == [{"TDoc"}, {"BP"}, {"PT"}]


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("PT", "BP", True),
        ("PT", "TDoc", True),
        ("BP", "TDoc", True),
        ("TDoc", "PT", False),
        ("TDoc", "BP", False),
        ("BP", "PT", False),
    ],
)
def test_tree_killer_relies_on(first, second, expected):
    assert tree_killer_block.relies_on(first, second) is expected


def test_relies_on_rejects_non_control():
    with pytest.raises(ValueError):
        tree_killer_block.relies_on("PT", "TS")  # TS is a chance node
    with pytest.raises(ValueError):
        tree_killer_block.relies_on("nope", "PT")


def test_calibration_none_defaults_to_empty():
    # Relevance is structural; None must not raise and must match the {} result.
    assert tree_killer_block.relies_on("PT", "TDoc", calibration=None) is True
    assert set(tree_killer_block.relevance_graph(calibration=None).edges()) == set(
        tree_killer_block.relevance_graph(calibration={}).edges()
    )


# ---------------------------------------------------------------------------
# A single-decision economic block: one node, no reliance edges.
# ---------------------------------------------------------------------------
def test_single_decision_block_has_no_edges():
    rg = CONSUMPTION_BLOCK.relevance_graph()
    assert rg.nodes() == ["c"]
    assert rg.edges() == []
    assert rg.is_acyclic() is True


# ---------------------------------------------------------------------------
# Oracle cross-check: PyCID must agree (skipped if PyCID is not importable).
# ---------------------------------------------------------------------------
def test_pycid_oracle_agreement():
    pytest.importorskip("pycid")
    from pycid.core.relevance_graph import RelevanceGraph as PyCIDRelevanceGraph
    from pycid.examples.story_macids import tree_doctor

    oracle_edges = set(PyCIDRelevanceGraph(tree_doctor()).edges())
    ours = set(tree_killer_block.relevance_graph().edges())
    # PyCID and our encoding share node names (PT, BP, TDoc).
    assert ours == oracle_edges
