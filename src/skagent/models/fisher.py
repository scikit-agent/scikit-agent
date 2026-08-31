r"""
Fisher (1930) two-period intertemporal consumption.

The simplest dynamic programming problem with a closed-form solution. An
agent receives income :math:`y` each period, borrows or saves at gross rate
:math:`R`, and chooses consumption :math:`c_0, c_1` to maximize

.. math::
    u(c_0) + \beta \, u(c_1)

subject to the lifetime budget constraint
:math:`c_0 + c_1/R = m_0 + y/R`, with :math:`m_0 = R\, a_{-1} + y`. With
CRRA utility :math:`u(c) = c^{1-\sigma}/(1-\sigma)`, the Euler equation
:math:`u'(c_0) = \beta R \, u'(c_1)` together with the budget constraint
gives the closed form

.. math::
    c_0 \;=\; \frac{m_0 + y/R}{\,1 + (\beta R)^{1/\sigma}/R\,},
    \qquad
    c_1 \;=\; (\beta R)^{1/\sigma} \, c_0.

The two-period horizon makes the model an exact analogue of the
intertemporal-choice diagram in introductory macroeconomics, while the
recursive form is the simplest non-trivial test case for value-function
iteration and Euler-equation solvers in :mod:`skagent`.

The horizon is part of the model, not of the solver call: the block above is
one period, and the two-period problem is that period iterated exactly twice
against a terminal continuation. A solver that iterates it to a fixed point is
answering a different question -- the infinite-horizon one -- and the closed
form above is not its answer.

Notes
-----
The math above uses :math:`R` for the gross return; the block parameter
key is ``Rfree``.

References
----------
Fisher, I. (1930). *The Theory of Interest*. New York: Macmillan.
"""

from skagent.block import Control, DBlock

#: Number of periods. The closed form below is the period-0 rule under this
#: horizon; the terminal rule is :math:`c_1 = m_1`.
T = 2

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "Rfree": 1.03,
    "y": 1.0,
}

block = DBlock(
    **{
        "name": "fisher",
        "shocks": {},
        "dynamics": {
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(
                ["m"],
                lower_bound=1e-6,
                upper_bound=lambda m: m,
                agent="consumer",
            ),
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)


def analytical_policy(states, shocks, parameters):
    r"""
    Optimal period-0 consumption for the two-period problem.

    Parameters
    ----------
    states : dict
        Must contain ``"m"`` (cash-on-hand at the start of period 0).
    shocks : dict
        Unused; the model is deterministic.
    parameters : dict
        Must contain ``"DiscFac"``, ``"CRRA"``, ``"Rfree"`` and ``"y"``.

    Returns
    -------
    dict
        ``{"c": c_0}`` whose dtype follows the input cash-on-hand.

    Raises
    ------
    ValueError
        If ``CRRA <= 0`` or ``Rfree <= 0``, for which the closed form is
        undefined.
    """
    beta = parameters["DiscFac"]
    sigma = parameters["CRRA"]
    R = parameters["Rfree"]
    y = parameters["y"]

    if sigma <= 0:
        raise ValueError(f"CRRA must be positive, got {sigma}")
    if R <= 0:
        raise ValueError(f"Rfree must be positive, got {R}")

    m = states["m"]
    growth = (beta * R) ** (1 / sigma)
    return {"c": (m + y / R) / (1 + growth / R)}
