r"""
Fisher (1930) two-period intertemporal consumption.

The simplest dynamic programming problem with a closed-form solution. An
agent receives income :math:`y_0` in period 0 and :math:`y_1` in period 1,
borrows or saves at gross rate :math:`R`, and chooses consumption
:math:`c_0, c_1` to maximize

.. math::
    u(c_0) + \beta \, u(c_1)

subject to the lifetime budget constraint
:math:`c_0 + c_1/R = m_0 + y_1/R`, with :math:`m_0 = R\, a_{-1} + y_0`. With
CRRA utility :math:`u(c) = c^{1-\sigma}/(1-\sigma)`, the Euler equation
:math:`u'(c_0) = \beta R \, u'(c_1)` together with the budget constraint
gives the closed form

.. math::
    c_0 \;=\; \frac{m_0 + y_1/R}{\,1 + (\beta R)^{1/\sigma}/R\,},
    \qquad
    c_1 \;=\; (\beta R)^{1/\sigma} \, c_0.

The two-period horizon makes the model an exact analogue of the
intertemporal-choice diagram in introductory macroeconomics, while the
recursive form is the simplest non-trivial test case for value-function
iteration and Euler-equation solvers in :mod:`skagent`.

Notes
-----
The math above uses :math:`R` for the gross return; the block parameter
key is ``Rfree``.

References
----------
Fisher, I. (1930). *The Theory of Interest*. New York: Macmillan.
"""

from skagent.block import Control, DBlock


calibration = {
    "DiscFac": 0.96,
    "CRRA": (2.0,),
    "Rfree": 1.03,
    "y": [1.0, 1.0],
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "shocks": {},
        "dynamics": {
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"]),
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        },
        "reward": {"u": "consumer"},
    }
)
