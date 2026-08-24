"""
Single-decision incentive diagrams (Everitt, Carey, Langlois, Ortega & Legg).

The four diagrams "Agent Incentives: A Causal Perspective" (AAAI-21,
35(13):11487-11495; arXiv:2102.01685) draws as its running examples, encoded as
scikit-agent blocks: grade prediction (Figs. 1a, 3a, 3b) and content
recommendation (Figs. 1b, 2, 4a, 4b). They exercise the criteria in
:mod:`skagent.relevance` -- ``admits_voi``, ``admits_ri``, ``admits_voc`` and
``admits_ici``.

Each example is a PAIR: a diagram that admits an incentive, and a redesign that
does not. One edge separates the members of a pair, and that edge flips one
criterion while leaving another fixed, which is what makes the pairs a test of
the criteria rather than an illustration of them.

Encoding conventions, following :mod:`skagent.models.macid`:

- Chance nodes are given in structural-causal form: a deterministic mechanism of
  the node's endogenous parents plus an explicit exogenous noise shock. Utility
  nodes are deterministic functions of their parents, as influence diagrams
  require.
- The paper's variables are finite-domain; here they are relaxed to continuous
  ``[0, 1]`` quantities, pending discrete-variable support.
- Control bounds are declared as constants, so a control's bounds do not depend
  on what it observes.

The relaxation is a re-encoding, not an approximation of the result: the criteria
these models exercise are purely graphical, reading the diagram's edges and never
the mechanisms, so no assertion about these diagrams depends on the functional
forms. The forms are nonetheless chosen so that every declared dependency has
some effect, which the graphical criteria do not need but a quantitative reading
of the same diagrams would.
"""

from skagent.block import Control, DBlock
from skagent.distributions import Uniform

_UNIT = (Uniform, {"low": 0.0, "high": 1.0})


# Grade prediction (Figs. 3a, 3b)
# ------------------------------
# A university predicts an applicant's grade (P) and is paid for accuracy (Ac).
# Race (R) determines which high school the applicant attends (HS), the high
# school determines their education (E), and education determines the grade they
# go on to get (Gr). Gender (Ge) is observed by the predictor and affects
# nothing else.
#
# In Fig. 3a the prediction observes the high school and gender. Race then
# admits a response incentive but no value of information: the optimal
# prediction moves with race, through the high school, yet observing race
# directly would buy nothing. Gender is a nonrequisite observation.
#
# Fig. 3b is the redesign: the prediction no longer observes the high school.
# Race now admits value of information but no response incentive -- the
# criteria separate in the opposite direction across that one edge.
def _grade_predictor(name, prediction_iset):
    """The grade-prediction diagram, parameterized by what the prediction sees.

    The information set is the only difference between Figs. 3a and 3b, so it is
    the only argument.
    """
    return DBlock(
        name=name,
        shocks={
            "R": _UNIT,  # race, a relaxed sensitive attribute
            "Ge": _UNIT,  # gender, likewise
            "u_HS": _UNIT,
            "u_E": _UNIT,
            "u_Gr": _UNIT,
        },
        dynamics={
            # Race raises the chance of a better high school.
            "HS": lambda R, u_HS: 0.3 + 0.4 * R + 0.3 * u_HS,
            "E": lambda HS, u_E: 0.6 * HS + 0.4 * u_E,
            "Gr": lambda E, u_Gr: 0.7 * E + 0.3 * u_Gr,
            "P": Control(
                prediction_iset,
                lower_bound=0.0,
                upper_bound=1.0,
                agent="university",
            ),
            # Accuracy: the predictor is paid for a small squared error.
            "Ac": lambda P, Gr: -((P - Gr) ** 2),
        },
        reward={"Ac": "university"},
    )


#: Fig. 3a. The prediction observes the high school and gender.
grade_predictor_block = _grade_predictor("grade_predictor", ["HS", "Ge"])

#: Fig. 3b. The redesign: the prediction no longer observes the high school.
grade_predictor_redesign_block = _grade_predictor("grade_predictor_redesign", ["Ge"])


# Content recommendation (Figs. 4a, 4b)
# ------------------------------------
# A recommender chooses posts (P) from a model (M) of the user's original
# opinions (O). The posts influence the user's opinions (I). Clicks (C) are the
# payoff.
#
# In Fig. 4a the payoff is clicks, which depend on the posts and on the
# influenced opinions. The influenced opinions then admit an instrumental
# control incentive: the recommender is paid for moving them, which is the
# manipulation incentive. The original opinions admit positive value of control
# but no instrumental control incentive -- worth setting, and beyond reach.
#
# Fig. 4b is the redesign: the payoff is retargeted at clicks predicted from the
# model of the user's ORIGINAL opinions. Nothing directed then leaves the
# influenced opinions, and the incentive to move them is gone.
#
# The node names are the paper's, and a mechanism's argument names ARE its
# dependencies, so O and I cannot be renamed to satisfy E741 without renaming the
# diagram's nodes away from the figure they encode.
def _content_recommender(name, clicks):
    """The content-recommendation diagram, parameterized by the clicks rule.

    What the payoff is computed from is the only difference between Figs. 4a and
    4b, so it is the only argument.
    """
    return DBlock(
        name=name,
        shocks={
            "O": _UNIT,  # the user's original opinions
            "u_M": _UNIT,
            "u_I": _UNIT,
        },
        dynamics={
            # The recommender's model of what the user thinks.
            "M": lambda O, u_M: 0.8 * O + 0.2 * u_M,  # noqa: E741
            "P": Control(
                ["M"],
                lower_bound=0.0,
                upper_bound=1.0,
                agent="recommender",
            ),
            # Posts pull the user's opinions toward themselves.
            "I": lambda O, P, u_I: 0.4 * O + 0.4 * P + 0.2 * u_I,  # noqa: E741
            "C": clicks,
        },
        reward={"C": "recommender"},
    )


#: Fig. 4a. Clicks depend on the posts and the user's influenced opinions: the
#: user clicks on posts that match what they now think.
content_recommender_block = _content_recommender(
    "content_recommender",
    lambda P, I: 1.0 - (P - I) ** 2,  # noqa: E741
)

#: Fig. 4b. The redesign: predicted clicks, computed from the model of the
#: user's original opinions rather than from the opinions the posts produced.
content_recommender_redesign_block = _content_recommender(
    "content_recommender_redesign", lambda P, M: 1.0 - (P - M) ** 2
)
