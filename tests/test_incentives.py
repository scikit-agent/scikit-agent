"""Unit tests for the single-decision incentive criteria in skagent.relevance:
``is_requisite``, ``minimal_reduction``, ``admits_voi``, ``admits_ri``,
``admits_voc`` and ``admits_ici``.

The fixtures are the four diagrams Everitt, Carey, Langlois, Ortega & Legg,
"Agent Incentives: A Causal Perspective" (AAAI-21) draw as its running examples:
grade prediction (Figs. 3a, 3b) and content recommendation (Figs. 4a, 4b). Each
example comes as a pair -- a diagram that admits an incentive and a redesign
that does not -- and in each pair the criteria come apart, which is what makes
the pairs a stronger test than any single figure: an implementation that has
accidentally computed plain reachability passes the figure and fails the pair.

As in test_relevance.py, each fixture is a plain networkx.DiGraph, so the
criteria are tested against the diagram directly, with no Block or
ModelAnalyzer in the loop.

Every expected value below is derived from the theorem named beside it. Where
the paper states the result in its own text this is noted; the rest follow from
the same criteria applied to the same figures.
"""

import itertools
import random

import networkx as nx
import pytest

from skagent.influence import SCIM
from skagent.relevance import (
    admits_ici,
    admits_ri,
    admits_voc,
    admits_voi,
    is_requisite,
    minimal_reduction,
)

AGENT = "agent"


def _cid(edges, decision, utility):
    """A single-decision, single-agent SCIM over *edges*."""
    graph = nx.DiGraph(edges)
    nx.set_node_attributes(graph, "chance", "kind")
    graph.nodes[decision]["kind"] = "decision"
    graph.nodes[utility]["kind"] = "utility"
    return SCIM(graph, [decision], {AGENT: [utility]}, {decision: AGENT})


# ----------------------------------------------------------------------------
# The four diagrams.
# ----------------------------------------------------------------------------
def _fig3a():
    """Grade prediction (Fig. 3a). A university predicts an applicant's grade
    (P) from their high school (HS) and gender (Ge); race (R) determines the
    high school, which determines education (E), which determines the grade
    (Gr); the payoff is prediction accuracy (Ac)."""
    return _cid(
        [
            ("R", "HS"),
            ("HS", "E"),
            ("HS", "P"),
            ("E", "Gr"),
            ("Gr", "Ac"),
            ("Ge", "P"),
            ("P", "Ac"),
        ],
        decision="P",
        utility="Ac",
    )


def _fig3b():
    """Grade prediction, redesigned (Fig. 3b): Fig. 3a minus the information
    link HS -> P, so the prediction no longer looks at the high school."""
    scim = _fig3a()
    return scim.without_edges([("HS", "P")])


def _fig4a():
    """Content recommendation (Fig. 4a). A recommender chooses posts (P) from a
    model (M) of the user's original opinions (O); the posts influence the
    user's opinions (I); clicks (C) depend on the posts and the influenced
    opinions, and are the payoff."""
    return _cid(
        [("O", "I"), ("O", "M"), ("M", "P"), ("P", "I"), ("I", "C"), ("P", "C")],
        decision="P",
        utility="C",
    )


def _fig4b():
    """Content recommendation, redesigned (Fig. 4b): the payoff is retargeted at
    clicks predicted from the model of the user's *original* opinions, so C is
    computed from M and P rather than from the influenced opinions I."""
    return _cid(
        [("O", "I"), ("O", "M"), ("M", "P"), ("P", "I"), ("P", "C"), ("M", "C")],
        decision="P",
        utility="C",
    )


# ----------------------------------------------------------------------------
# The assertion tables: node -> (VoI, RI, VoC, ICI). RAISES marks a query the
# criterion refuses as out of domain, which is not the same as a False.
# ----------------------------------------------------------------------------
RAISES = "raises"

CRITERIA = ("voi", "ri", "voc", "ici")
CRITERION_FUNCTIONS = {
    "voi": admits_voi,
    "ri": admits_ri,
    "voc": admits_voc,
    "ici": admits_ici,
}

FIG3A_TABLE = {
    # Race. RI by Thm. 12: R -> HS -> P is directed in the minimal reduction,
    # which keeps HS -> P. No VoI by Thm. 9: adding R -> P, every route from R
    # to Ac is intercepted -- R -> HS -> ... at the now-conditioned HS, and
    # R -> P -> Ac at the conditioned P. The paper states both (p. 11491).
    # VoC by Thm. 16: R -> HS -> E -> Gr -> Ac is directed. No ICI by Thm. 18:
    # R does not descend from the decision.
    "R": (False, True, True, False),
    # High school. Requisite, so VoI (Thm. 9) and, being a parent, RI (Thm. 12).
    "HS": (True, True, True, False),
    # Education and grade: on the directed route to the payoff but not to the
    # decision, so VoI and VoC without RI.
    "E": (True, False, True, False),
    "Gr": (True, False, True, False),
    # Gender. Nonrequisite (Def. 7): its only route to Ac runs through the
    # conditioned P, and the collider that opens at P is blocked at the
    # conditioned HS. So no VoI, and the reduction drops Ge -> P, killing the
    # RI. Nothing else leaves Ge, so no VoC either. The paper states that
    # gender is not requisite (p. 11491).
    "Ge": (False, False, False, False),
    # The decision: out of domain for three of the four, and trivially on the
    # path D -> Ac for the fourth.
    "P": (RAISES, RAISES, RAISES, True),
    # The payoff itself: controlling it is worth something (Thm. 16, path of
    # length zero) and the decision reaches it (Thm. 18).
    "Ac": (RAISES, False, True, True),
}

FIG3B_TABLE = {
    # Race, after deleting HS -> P. VoI by Thm. 9: with R -> P added, the route
    # R -> HS -> E -> Gr -> Ac is unblocked, since HS is no longer observed. No
    # RI by Thm. 12: nothing directed reaches P any more. The paper states both
    # (p. 11491) -- this row is the inverse of the Fig. 3a one.
    "R": (True, False, True, False),
    "HS": (True, False, True, False),
    "E": (True, False, True, False),
    "Gr": (True, False, True, False),
    # Gender is still nonrequisite: its one route to Ac runs through the
    # conditioned P, and P now has no other parent to open a collider with.
    "Ge": (False, False, False, False),
    "P": (RAISES, RAISES, RAISES, True),
    "Ac": (RAISES, False, True, True),
}

FIG4A_TABLE = {
    # Original user opinions: VoC by Thm. 16 (O -> I -> C is directed) but no
    # ICI by Thm. 18, since O does not descend from the decision -- the
    # recommender cannot control what the user already thought. This is the
    # separation the paper draws between the two control criteria.
    "O": (True, True, True, False),
    # The model of those opinions is the decision's one observation, and is
    # requisite (M <- O -> I -> C is active given P).
    "M": (True, True, True, False),
    # Influenced user opinions: ICI by Thm. 18, P -> I -> C. This is the
    # manipulation incentive the paper reads off Fig. 4a.
    "I": (RAISES, False, True, True),
    "P": (RAISES, RAISES, RAISES, True),
    "C": (RAISES, False, True, True),
}

FIG4B_TABLE = {
    # With the payoff retargeted, O loses VoI: with O -> P added, O -> M -> ...
    # is blocked at the observed M, and the collider at the now-childless I has
    # no observed descendant to open it. RI and VoC survive through M.
    "O": (False, True, True, False),
    "M": (True, True, True, False),
    # Influenced user opinions: no ICI by Thm. 18, because no directed path
    # leaves I at all -- which is the point of the redesign. That also costs it
    # the VoC it had in Fig. 4a.
    "I": (RAISES, False, False, False),
    "P": (RAISES, RAISES, RAISES, True),
    "C": (RAISES, False, True, True),
}

FIGURES = {
    "fig3a_grade_prediction": (_fig3a, FIG3A_TABLE),
    "fig3b_grade_prediction_redesigned": (_fig3b, FIG3B_TABLE),
    "fig4a_content_recommendation": (_fig4a, FIG4A_TABLE),
    "fig4b_content_recommendation_redesigned": (_fig4b, FIG4B_TABLE),
}


@pytest.mark.parametrize(
    "figure,node,criterion",
    [
        (figure, node, criterion)
        for figure, (_, table) in FIGURES.items()
        for node in table
        for criterion in CRITERIA
    ],
)
def test_incentive_table(figure, node, criterion):
    build, table = FIGURES[figure]
    scim = build()
    expected = table[node][CRITERIA.index(criterion)]
    criterion_function = CRITERION_FUNCTIONS[criterion]

    if expected is RAISES:
        with pytest.raises(ValueError):
            criterion_function(scim, "P", node)
    else:
        assert criterion_function(scim, "P", node) is expected


# ----------------------------------------------------------------------------
# The separations, asserted on their own so a regression names itself.
# ----------------------------------------------------------------------------
def test_race_swaps_ri_for_voi_across_one_edge():
    """Deleting HS -> P inverts both criteria on race, in opposite directions.

    Thms. 9 and 12 on Figs. 3a and 3b; the paper states this pair (p. 11491).
    An implementation that computed plain reachability from race would report
    the same thing on both diagrams.
    """
    before, after = _fig3a(), _fig3b()

    assert admits_ri(before, "P", "R") is True
    assert admits_voi(before, "P", "R") is False

    assert admits_ri(after, "P", "R") is False
    assert admits_voi(after, "P", "R") is True


def test_retargeting_the_payoff_removes_the_manipulation_incentive():
    """Influenced user opinions admit an ICI in Fig. 4a and none in Fig. 4b.

    Thm. 18: P -> I -> C is directed in Fig. 4a; in Fig. 4b nothing directed
    leaves I, because the payoff is computed from the model of the user's
    original opinions instead.
    """
    assert admits_ici(_fig4a(), "P", "I") is True
    assert admits_ici(_fig4b(), "P", "I") is False


def test_original_opinions_have_control_value_without_a_control_incentive():
    """In Fig. 4a, O admits positive VoC but no ICI.

    Thm. 16 gives the VoC (O -> I -> C is directed in the minimal reduction);
    Thm. 18 denies the ICI, since no directed path reaches O from the decision.
    The agent would gain from setting the user's original opinions, and has no
    way to.
    """
    scim = _fig4a()
    assert admits_voc(scim, "P", "O") is True
    assert admits_ici(scim, "P", "O") is False


# ----------------------------------------------------------------------------
# Requisiteness and the minimal reduction, which three of the four criteria
# rest on.
# ----------------------------------------------------------------------------
def test_requisiteness_of_the_grade_predictor_observations():
    scim = _fig3a()
    # HS reaches Ac through E and Gr, neither of which is observed.
    assert is_requisite(scim, "P", "HS") is True
    # Ge reaches Ac only through the observed P, and the collider that opens
    # there is blocked at the observed HS.
    assert is_requisite(scim, "P", "Ge") is False


def test_minimal_reduction_drops_exactly_the_nonrequisite_links():
    reduction = minimal_reduction(_fig3a())
    assert set(reduction.graph.edges) == {
        ("R", "HS"),
        ("HS", "E"),
        ("HS", "P"),
        ("E", "Gr"),
        ("Gr", "Ac"),
        ("P", "Ac"),
    }


def test_minimal_reduction_leaves_the_input_alone():
    scim = _fig3a()
    minimal_reduction(scim)
    assert ("Ge", "P") in scim.graph.edges


def test_minimal_reduction_is_a_noop_when_every_observation_is_requisite():
    scim = _fig4a()
    assert set(minimal_reduction(scim).graph.edges) == set(scim.graph.edges)


def _random_single_decision_cid(rng, n_nodes):
    """A random DAG with one decision that observes something, and one utility
    node downstream of it."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    for i, j in itertools.combinations(range(n_nodes), 2):
        if rng.random() < 0.35:
            graph.add_edge(i, j)

    decision = rng.randint(1, n_nodes - 1)
    if not list(graph.predecessors(decision)):
        graph.add_edge(rng.randint(0, decision - 1), decision)

    # A sink fed by the decision, so the agent always has an objective.
    utility = "U"
    graph.add_node(utility)
    graph.add_edge(decision, utility)
    for node in range(decision + 1, n_nodes):
        if rng.random() < 0.35:
            graph.add_edge(node, utility)

    return SCIM(graph, [decision], {AGENT: [utility]}, {decision: AGENT})


@pytest.mark.parametrize("seed", range(25))
def test_minimal_reduction_is_reached_in_one_pass(seed):
    """A second pass must find nothing left to remove.

    The reduction is computed by testing every information link once and
    dropping the failures together. That is only the minimal reduction if
    dropping a link cannot make another link nonrequisite, which this checks on
    random diagrams rather than assuming.
    """
    scim = _random_single_decision_cid(random.Random(seed), 8)
    once = minimal_reduction(scim)
    twice = minimal_reduction(once)
    assert set(twice.graph.edges) == set(once.graph.edges), f"seed={seed}"


# ----------------------------------------------------------------------------
# Refusals. Out-of-domain and multi-decision queries raise rather than answer,
# which is a deliberate departure from PyCID's quiet False.
# ----------------------------------------------------------------------------
def _two_decisions():
    graph = nx.DiGraph([("D", "Dp"), ("Dp", "U")])
    return SCIM(graph, ["D", "Dp"], {AGENT: ["U"]}, {"D": AGENT, "Dp": AGENT})


@pytest.mark.parametrize("criterion", CRITERIA)
def test_multi_decision_diagram_is_refused(criterion):
    """All four theorems are stated for a diagram with one decision."""
    with pytest.raises(ValueError, match="exactly one"):
        CRITERION_FUNCTIONS[criterion](_two_decisions(), "D", "U")


def test_minimal_reduction_refuses_a_multi_decision_diagram():
    with pytest.raises(ValueError, match="exactly one"):
        minimal_reduction(_two_decisions())


@pytest.mark.parametrize("criterion", CRITERIA)
def test_wrong_decision_is_refused(criterion):
    with pytest.raises(ValueError, match="not the decision"):
        CRITERION_FUNCTIONS[criterion](_fig3a(), "HS", "R")


@pytest.mark.parametrize("criterion", CRITERIA)
def test_unknown_node_is_refused(criterion):
    with pytest.raises(ValueError, match="not a node"):
        CRITERION_FUNCTIONS[criterion](_fig3a(), "P", "nope")


def test_voi_refuses_a_descendant_of_the_decision():
    """There is no diagram in which a descendant of the decision is observed:
    the information link would close a cycle."""
    with pytest.raises(ValueError, match="value of information"):
        admits_voi(_fig3a(), "P", "Ac")


def test_ri_and_voc_refuse_the_decision_itself():
    with pytest.raises(ValueError, match="response incentive"):
        admits_ri(_fig3a(), "P", "P")
    with pytest.raises(ValueError, match="value of control"):
        admits_voc(_fig3a(), "P", "P")


def test_requisiteness_refuses_a_node_that_is_not_an_observation():
    with pytest.raises(ValueError, match="not an observation"):
        is_requisite(_fig3a(), "P", "R")


@pytest.mark.parametrize("criterion", CRITERIA)
def test_unowned_payoff_is_refused(criterion):
    """A decision whose agent owns no downstream reward would answer False to
    everything, which is the direction that must not be silent."""
    graph = _fig3a().graph
    scim = SCIM(graph, ["P"], {"someone_else": ["Ac"]}, {"P": AGENT})
    with pytest.raises(ValueError, match="no objective nodes"):
        CRITERION_FUNCTIONS[criterion](scim, "P", "HS")


# ----------------------------------------------------------------------------
# Oracle cross-check: PyCID must agree (skipped if PyCID is not importable).
# PyCID is the reference implementation from the paper's own group, and ships
# three of the four diagrams. Fig. 3b is not among them; it is asserted by hand
# above.
# ----------------------------------------------------------------------------
def _or_none(criterion_function, scim, decision, node):
    """Our answer, or None where we refuse and PyCID returns a quiet False."""
    try:
        return criterion_function(scim, decision, node)
    except ValueError:
        return None


def test_pycid_oracle_agreement():
    pytest.importorskip("pycid")
    from pycid.analyze.instrumental_control_incentive import admits_ici as pycid_ici
    from pycid.analyze.response_incentive import admits_ri as pycid_ri
    from pycid.analyze.value_of_control import admits_voc as pycid_voc
    from pycid.analyze.value_of_information import admits_voi as pycid_voi
    from pycid.examples.story_cids import (
        get_content_recommender,
        get_grade_predictor,
        get_modified_content_recommender,
    )

    pairs = [
        (_fig3a(), get_grade_predictor()),
        (_fig4a(), get_content_recommender()),
        (_fig4b(), get_modified_content_recommender()),
    ]
    for ours, theirs in pairs:
        assert set(ours.graph.edges) == set(theirs.edges), "encodings must match"
        for node in ours.graph.nodes:
            assert _or_none(admits_voi, ours, "P", node) in (
                None,
                pycid_voi(theirs, "P", node),
            )
            assert _or_none(admits_ri, ours, "P", node) in (
                None,
                pycid_ri(theirs, "P", node),
            )
            assert _or_none(admits_voc, ours, "P", node) in (
                None,
                pycid_voc(theirs, node),
            )
            assert _or_none(admits_ici, ours, "P", node) in (
                None,
                pycid_ici(theirs, "P", node),
            )
