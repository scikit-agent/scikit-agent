r"""
Cournot (1838) quantity competition among firms.

The smallest model that exercises an entity class, an agent role and an
aggregation. Several firms each draw a production cost, each choose a quantity,
and the market price falls with the *average* quantity supplied:

.. math::
    Q = \frac{1}{N}\sum_i q_i, \qquad P = A - b\,Q, \qquad
    u_i = (P - c_i)\, q_i

The model is static: there are no arrival states, so simulating it for
:math:`T` periods is :math:`T` independent plays of the same one-shot game.

Because the price is a function of the mean rather than the sum, the slope on
*total* quantity is :math:`b/N`. That keeps the two readings of :math:`N` below
in one model without changing an equation.

Two calibrations, and the pair is the point
-------------------------------------------

**Heterogeneous costs** (:func:`heterogeneous_calibration`). Under the supplied
rule :math:`q_i = (A - c_i)/2b`, with :math:`m = E[c]` and :math:`v = Var(c)`:

.. math::
    Q = \frac{A - m}{2b}, \qquad P = \frac{A + m}{2}, \qquad
    E[u] = \frac{(A - m)^2/2 + v}{2b}

The variance term is what makes this a real test. An aggregate-only claim on
:math:`Q` or :math:`P` is satisfied by a population that has silently lost its
cross-section, because a mean survives a collapse; :math:`E[u]` carries the
second moment and does not. Here :math:`N` is a numerical knob: the answers
converge as it grows.

**Three oligopolists** (:func:`collusion_calibration`). With cost degenerate and
:math:`N = 3` this is textbook Cournot, and three profiles are hand-derivable
(:data:`PROFILES`). Here :math:`N` is part of the model: it enters the effective
demand slope, so changing it asks a different question rather than refining the
answer.

What the library does not do with it
------------------------------------

It does not *find* the profiles. They are supplied as decision rules and
simulated forward. The model has a crossing, so each firm's optimal quantity
depends on what the others chose, which is an equilibrium problem rather than a
dynamic programming one.

The deviation profile is an asymmetric *assignment* of rules over a symmetric
model: three firms are handed three rules, two of which are equal. No rule reads
a firm's position, and the model is the same in all three profiles.
"""

import numpy as np

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Uniform

offer_block = DBlock(
    name="offer",
    shocks={"c": (Uniform, {"low": "cl", "high": "ch"})},
    dynamics={"q": Control(["c"], agent="firm")},
)

market_block = DBlock(
    name="market",
    dynamics={
        # Reads out of the firm class, so this is the model's one crossing.
        "Q": lambda q: q.mean(),
        "P": lambda A, b, Q: A - b * Q,
    },
)

payoff_block = DBlock(
    name="payoff",
    dynamics={"u": lambda P, c, q: (P - c) * q},
    reward={"u": "firm"},
)

# The firm class is declared twice because the population acts before the market
# clears and is paid after. Declaration order is what fixes the dynamics order,
# so there is no timing annotation anywhere in the model.
cournot_block = RBlock(
    name="cournot",
    blocks=[
        RBlock(name="firms", entity=Entity("firm"), blocks=[offer_block]),
        market_block,
        RBlock(name="payoffs", entity=Entity("firm"), blocks=[payoff_block]),
    ],
)

A = 10.0
"""Demand intercept, shared by both calibrations."""

B = 1.0
"""Demand slope on the average quantity, shared by both calibrations."""


def heterogeneous_calibration(size=200, low=2.0, high=6.0):
    """A market of *size* firms whose costs are uniform on ``[low, high]``.

    Parameters
    ----------
    size : int, optional
        How many firms. A numerical knob in this calibration.
    low, high : float, optional
        Support of the cost distribution.

    Returns
    -------
    dict
    """
    return {"A": A, "b": B, "cl": low, "ch": high, "firm": size}


def collusion_calibration(size=3, cost=4.0):
    """A market of *size* firms that all have cost *cost*.

    Parameters
    ----------
    size : int, optional
        How many firms. Part of the model in this calibration: it enters the
        effective slope on total quantity.
    cost : float, optional
        The common marginal cost.

    Returns
    -------
    dict
    """
    return {"A": A, "b": B, "cl": cost, "ch": cost, "firm": size}


def analytic_moments(low=2.0, high=6.0):
    """The heterogeneous-cost market's analytic aggregate and expected profit.

    Holds under the rule :func:`competitive_rule`, in the limit of many firms.

    Returns
    -------
    dict
        ``Q``, ``P`` and ``E[u]``.
    """
    mean = (low + high) / 2
    variance = (high - low) ** 2 / 12
    return {
        "Q": (A - mean) / (2 * B),
        "P": (A + mean) / 2,
        "E[u]": ((A - mean) ** 2 / 2 + variance) / (2 * B),
    }


def competitive_rule(c):
    """The supplied rule the heterogeneous-cost claims are stated under."""
    return (A - c) / (2 * B)


def nash_quantity(size=3, cost=4.0):
    """The symmetric Cournot-Nash quantity per firm.

    From the first-order condition ``A - bq - c - qb/N = 0``, which gives
    ``q = N(A - c) / (b(N + 1))``.
    """
    return size * (A - cost) / (B * (size + 1))


def monopoly_quantity(cost=4.0):
    """The quantity per firm that maximizes the industry's total profit."""
    return (A - cost) / (2 * B)


PROFILES = {
    "cournot-nash": [4.5, 4.5, 4.5],
    "joint-monopoly": [3.0, 3.0, 3.0],
    "one-defects": [6.0, 3.0, 3.0],
}
"""Three hand-derivable profiles at ``collusion_calibration()``.

Supplied, not found. Colluding pays jointly (27.0 against 20.25 in total
profit), the cartel is unstable because deviating pays the deviator (12 against
9) at its partners' expense (6 against 9), and Nash is stable.
"""


def profile_rules(quantities):
    """One fixed decision rule per firm, so a profile may be asymmetric.

    The asymmetry is in the ASSIGNMENT of rules to instances, not in the model:
    each rule is a constant function, and none reads a firm's position.

    Parameters
    ----------
    quantities : sequence of float
        One quantity per firm, in instance order.

    Returns
    -------
    numpy.ndarray
        Callables, one per firm, suitable as a ``Simulator`` decision rule.
    """
    return np.array([(lambda value: (lambda c: value))(q) for q in quantities])
