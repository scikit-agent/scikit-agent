"""
Unit tests for skagent.relevance -- the s-reachability criterion and the
RelevanceGraph wrapper, exercised in isolation on hand-built influence-diagram
(SCIM) graphs. No Block / ModelAnalyzer coupling: each fixture is a plain
networkx.DiGraph so the graphical criterion is tested directly against ground
truth.

The five two-decision fixtures are Koller & Milch (2001) Fig. 3 (a)-(e). Their
expected relevance graphs are taken from that figure and were cross-checked
against PyCID's RelevanceGraph on the corresponding story_macids examples.

Convention: D is owned by agent "a", D' (spelled ``Dp``) by agent "b". Edge
D -> D' in the relevance graph means "D relies on D'".
"""

import networkx as nx
import pytest

from skagent.influence import DUMMY_PREFIX, SCIM
from skagent.relevance import RelevanceGraph, is_s_reachable


# ----------------------------------------------------------------------------
# Koller & Milch Fig. 3 fixtures: each returns a SCIM. A decision's information
# set is read off the graph, so the fixtures declare it as edges only.
# ----------------------------------------------------------------------------
def _fig3a():
    """Perfect information: chain D -> D' -> U, U owned by a; D' observes D."""
    G = nx.DiGraph([("D", "Dp"), ("Dp", "U")])
    agent_utilities = {"a": ["U"], "b": []}
    decision_agent = {"D": "a", "Dp": "b"}
    return SCIM(G, ["D", "Dp"], agent_utilities, decision_agent)


def _fig3b():
    """Imperfect but 'perfect enough': D -> C -> D'; C screens D from b's util."""
    G = nx.DiGraph(
        [
            ("D", "C"),
            ("C", "Dp"),
            ("C", "Ub"),
            ("Dp", "Ub"),
            ("D", "Ua"),
            ("Dp", "Ua"),
        ]
    )
    agent_utilities = {"a": ["Ua"], "b": ["Ub"]}
    decision_agent = {"D": "a", "Dp": "b"}
    return SCIM(G, ["D", "Dp"], agent_utilities, decision_agent)


def _fig3c():
    """Simultaneous; each agent's utility depends on both decisions."""
    G = nx.DiGraph([("D", "Ua"), ("Dp", "Ua"), ("D", "Ub"), ("Dp", "Ub")])
    agent_utilities = {"a": ["Ua"], "b": ["Ub"]}
    decision_agent = {"D": "a", "Dp": "b"}
    return SCIM(G, ["D", "Dp"], agent_utilities, decision_agent)


def _fig3d():
    """Simultaneous; a's utility does NOT depend on D'."""
    G = nx.DiGraph([("D", "Ua"), ("D", "Ub"), ("Dp", "Ub")])
    agent_utilities = {"a": ["Ua"], "b": ["Ub"]}
    decision_agent = {"D": "a", "Dp": "b"}
    return SCIM(G, ["D", "Dp"], agent_utilities, decision_agent)


def _fig3e():
    """Card game / signaling: a observes card (C -> D); b observes bet
    (D -> D'); both utilities depend on C, D, D'."""
    G = nx.DiGraph(
        [
            ("C", "D"),
            ("D", "Dp"),
            ("C", "Ua"),
            ("D", "Ua"),
            ("Dp", "Ua"),
            ("C", "Ub"),
            ("D", "Ub"),
            ("Dp", "Ub"),
        ]
    )
    agent_utilities = {"a": ["Ua"], "b": ["Ub"]}
    decision_agent = {"D": "a", "Dp": "b"}
    return SCIM(G, ["D", "Dp"], agent_utilities, decision_agent)


# (fixture builder, relies_on(D, Dp), relies_on(Dp, D), is_acyclic)
FIG3_CASES = {
    "a_perfect_info": (_fig3a, True, False, True),
    "b_perfect_enough": (_fig3b, True, False, True),
    "c_simultaneous_cyclic": (_fig3c, True, True, False),
    "d_simultaneous_acyclic": (_fig3d, False, True, True),
    "e_card_game_signaling": (_fig3e, True, True, False),
}


@pytest.mark.parametrize("case", list(FIG3_CASES))
def test_km_fig3_relevance(case):
    builder, d_relies_dp, dp_relies_d, acyclic = FIG3_CASES[case]
    rg = RelevanceGraph.from_scim(builder())

    assert rg.relies_on("D", "Dp") is d_relies_dp
    assert rg.relies_on("Dp", "D") is dp_relies_d
    assert rg.is_acyclic() is acyclic


def test_km_fig3c_single_scc():
    rg = RelevanceGraph.from_scim(_fig3c())
    assert rg.sccs() == [{"D", "Dp"}]


def test_is_s_reachable_matches_relies_on():
    """The primitive and the wrapper must agree."""
    scim = _fig3e()
    rg = RelevanceGraph.from_scim(scim)
    for d1 in scim.decisions:
        for d2 in scim.decisions:
            assert is_s_reachable(scim, d1, d2) is rg.relies_on(d1, d2)


# ----------------------------------------------------------------------------
# RelevanceGraph wrapper, tested directly on a known decision-only DiGraph.
# ----------------------------------------------------------------------------
def _tree_killer_relevance():
    """The Tree Killer relevance graph (KM Fig. 4a): PT relies on BP and TDoc;
    BP relies on TDoc; TDoc relies on nothing."""
    g = nx.DiGraph([("PT", "BP"), ("PT", "TDoc"), ("BP", "TDoc")])
    return RelevanceGraph(g)


def test_condensation_solve_order():
    rg = _tree_killer_relevance()
    assert rg.is_acyclic() is True
    # Solve leaves first: TDoc, then BP, then PT (each a singleton SCC).
    assert rg.condensation() == [{"TDoc"}, {"BP"}, {"PT"}]


def test_sccs_cyclic():
    rg = RelevanceGraph(nx.DiGraph([("D1", "D2"), ("D2", "D1")]))
    assert rg.is_acyclic() is False
    assert rg.sccs() == [{"D1", "D2"}]


def test_condensation_multi_node_scc():
    # {A,B} form a cycle that jointly relies on singleton C.
    g = nx.DiGraph([("A", "B"), ("B", "A"), ("A", "C")])
    rg = RelevanceGraph(g)
    order = rg.condensation()
    assert order == [{"C"}, {"A", "B"}]


# ----------------------------------------------------------------------------
# Edge cases.
# ----------------------------------------------------------------------------
def test_no_self_reliance():
    assert is_s_reachable(_fig3c(), "D", "D") is False


def test_empty_owned_utilities_not_reachable():
    """A decision whose agent owns no descendant utility relies on nothing."""
    # In Fig 3a, U is owned by a, so b (owner of Dp) owns no utility.
    assert is_s_reachable(_fig3a(), "Dp", "D") is False


def test_relies_on_rejects_non_decision():
    rg = _tree_killer_relevance()
    with pytest.raises(ValueError):
        rg.relies_on("PT", "not_a_decision")
    with pytest.raises(ValueError):
        rg.relies_on("nope", "TDoc")


def test_dummy_name_collision_is_avoided():
    """A real node literally named like the dummy must not corrupt the result."""
    scim = _fig3a()
    # Inject a decoy node colliding with the default dummy name for "Dp".
    scim.graph.add_edge(f"{DUMMY_PREFIX}Dp", "U")
    # D still relies on Dp (the collision must be sidestepped, result unchanged).
    assert is_s_reachable(scim, "D", "Dp") is True


def test_draw_returns_pydot_graph():
    pydot = pytest.importorskip("pydot")
    rg = _tree_killer_relevance()
    dot = rg.draw()
    assert isinstance(dot, pydot.Dot)
    assert {n.get_name() for n in dot.get_nodes()} == {"PT", "BP", "TDoc"}
