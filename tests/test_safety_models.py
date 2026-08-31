"""
Tests for skagent.models.safety.incentives -- the four diagrams of Everitt,
Carey, Langlois, Ortega & Legg (AAAI-21) encoded as blocks.

test_incentives.py already establishes what the criteria answer on these four
diagrams, drawn as graphs. What is unproven here is the encoding, so that is what
this file tests, in three claims:

1. Each block's influence-diagram view IS the paper's figure. This is where the
   encoding goes wrong quietly: a dependency read out of declaration order
   becomes a lag edge and is dropped from the diagram without complaint.
2. The noise nodes the structural-causal form adds change none of the answers.
   The convention says they cannot -- they are single-child exogenous roots --
   and this is the claim, not the assumption.
3. The mechanisms run. Nothing else here evaluates them, since the criteria read
   the diagram and never the rules.
"""

import numpy as np
import pytest

from skagent.model_analyzer import ModelAnalyzer
from skagent.models.safety.incentives import (
    content_recommender_block,
    content_recommender_redesign_block,
    draw_shocks,
    grade_predictor_block,
    grade_predictor_redesign_block,
)
from tests.test_incentives import (
    CRITERIA,
    CRITERION_FUNCTIONS,
    FIG3A_TABLE,
    FIG3B_TABLE,
    FIG4A_TABLE,
    FIG4B_TABLE,
    RAISES,
)

# The figures as drawn in the paper. The blocks carry one extra node per chance
# mechanism -- the exogenous noise the structural-causal encoding makes explicit,
# which the paper leaves implicit.
FIG3A_EDGES = {
    ("R", "HS"),
    ("HS", "E"),
    ("HS", "P"),
    ("E", "Gr"),
    ("Gr", "Ac"),
    ("Ge", "P"),
    ("P", "Ac"),
}
FIG3B_EDGES = FIG3A_EDGES - {("HS", "P")}
FIG4A_EDGES = {
    ("O", "I"),
    ("O", "M"),
    ("M", "P"),
    ("P", "I"),
    ("I", "C"),
    ("P", "C"),
}
FIG4B_EDGES = (FIG4A_EDGES - {("I", "C")}) | {("M", "C")}

# block, figure edges, noise nodes, table of expected answers
MODELS = {
    "fig3a": (grade_predictor_block, FIG3A_EDGES, {"u_HS", "u_E", "u_Gr"}, FIG3A_TABLE),
    "fig3b": (
        grade_predictor_redesign_block,
        FIG3B_EDGES,
        {"u_HS", "u_E", "u_Gr"},
        FIG3B_TABLE,
    ),
    "fig4a": (content_recommender_block, FIG4A_EDGES, {"u_M", "u_I"}, FIG4A_TABLE),
    "fig4b": (
        content_recommender_redesign_block,
        FIG4B_EDGES,
        {"u_M", "u_I"},
        FIG4B_TABLE,
    ),
}


def _scim(block):
    return ModelAnalyzer(block, {}).analyze().influence_graph()


@pytest.mark.parametrize("figure", list(MODELS))
def test_block_reproduces_the_figure(figure):
    block, edges, noise, _ = MODELS[figure]
    scim = _scim(block)

    assert set(scim.graph.nodes) == {n for edge in edges for n in edge} | noise
    assert {(s, t) for s, t in scim.graph.edges if s not in noise} == edges


@pytest.mark.parametrize("figure", list(MODELS))
def test_criteria_survive_the_encoding(figure):
    """Every criterion answers through the block what it answers on the figure.

    The block carries agent attribution and explicit noise the drawn diagram does
    not; neither may move an answer.
    """
    block, _, _, table = MODELS[figure]
    scim = _scim(block)

    answered = {}
    for node in table:
        row = []
        for criterion in CRITERIA:
            try:
                row.append(CRITERION_FUNCTIONS[criterion](scim, "P", node))
            except ValueError:
                row.append(RAISES)
        answered[node] = tuple(row)

    assert answered == table


@pytest.mark.parametrize("figure", list(MODELS))
def test_mechanisms_evaluate_on_the_numpy_path(figure):
    block = MODELS[figure][0]
    shocks = draw_shocks(block, n=1_000)

    vals = block.transition(shocks, {"P": lambda *observed: 0.5})
    (payoff,) = block.calc_reward(vals).values()

    assert np.shape(payoff) == (1_000,)
    assert np.all(np.isfinite(payoff))


def test_draw_shocks_leaves_the_block_it_was_given_alone():
    """The blocks here are module-level, so a draw must not construct in place.

    A draw resolves each declared ``(class, kwargs)`` pair against a
    calibration; were the result written back, the shared diagram would be left
    constructed for every later caller.
    """
    declared = grade_predictor_block.shocks["R"]
    shocks = draw_shocks(grade_predictor_block, n=8)

    assert grade_predictor_block.shocks["R"] is declared
    assert np.shape(shocks["R"]) == (8,)
