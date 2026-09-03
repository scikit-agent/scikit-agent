r"""
Akerlof (1970) adverse selection: the market for lemons.

Sellers know the quality of what they hold; buyers do not, and can price only
the average quality of what is offered. Each seller draws a quality
:math:`\theta`, values it at :math:`\theta`, and sells iff the going price
covers that. Buyers value the same item at a premium, and competition drives the
price to the expected buyer valuation of what actually trades:

.. math::
    \theta_i \sim U[0, 1], \qquad S_i = 1 \iff p \geq \theta_i, \qquad
    u_i = S_i (p - \theta_i), \qquad p' = 1.5\, E[\theta_i \mid S_i = 1]

The market is the model's one crossing: the price reads out of the seller class,
and every seller conditions on it.

**The price is a structural equation, not a decision.** Buyers are competitive
and have no decision node; what would be their inference is compressed into the
clearing equation. So the model has ONE decision, and its relevance graph is a
single node with no edges -- trivially acyclic. That is not the same as being
solvable one decision at a time. The price a seller conditions on is produced by
the seller's own rule, and a graph over decisions cannot see a feedback that
runs through a structural equation. Solving :math:`S` against a fixed price
returns a rule inconsistent with the price that rule induces, with no error
anywhere; reaching the equilibrium means iterating the price to a fixed point.

**The feedback is carried by a lag, so the iteration is a simulation.** ``p`` is
read by the sellers before the market writes it, which makes it an arrival
state: period :math:`t`'s sellers face period :math:`t-1`'s clearing price.
Simulating :math:`T` periods therefore runs :math:`T` rounds of the clearing
map, and the price column is the iterates.

**The equilibrium is collapse.** Conditional on sale :math:`\theta \sim U[0, p]`,
so :math:`E[\theta \mid \text{sale}] = p/2` and the map is
:math:`p' = 0.75\,p`: a contraction with the unique fixed point :math:`p = 0`.
No trade, though every item has a buyer who values it above its holder. The map
converges undamped from any starting price, which makes this the cheapest fixed
point in the library.

**The empty mask is the equilibrium, and the value there belongs to the model.**
At :math:`p = 0` nobody sells, so the clearing price is a mean over no sellers.
This model answers 0, which is what makes :math:`p = 0` a fixed point rather
than a NaN; a different market answers differently, so the choice is stated at
the reduction and not inferred.

Binary decisions are relaxed to continuous ``[0, 1]`` controls, pending
discrete-action support, following the convention of
:mod:`skagent.models.macid`. :func:`seller_rule` returns exact 0 and 1, so the
relaxation costs the supplied equilibrium nothing.
"""

import numpy as np

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Uniform

PREMIUM = 1.5
"""How much more a buyer values an item than the seller who holds it."""

CONTRACTION_SLOPE = PREMIUM / 2
"""Slope of the clearing map on the unit-support market: ``p' = 0.75 p``."""

EQUILIBRIUM_PRICE = 0.0
"""The unique fixed point of the clearing map: no trade."""


def clearing_price(theta, S):
    """The competitive price: mean buyer valuation over the items that sold.

    Parameters
    ----------
    theta : numpy.ndarray
        Quality, one per seller.
    S : numpy.ndarray
        The sell decision, one per seller; treated as sold above ``0.5``.

    Returns
    -------
    float
        ``PREMIUM`` times the mean quality of what sold, and 0 when nothing did.
    """
    sold = np.asarray(S) > 0.5
    if not sold.any():
        return 0.0
    return PREMIUM * np.asarray(theta)[sold].mean()


offer_block = DBlock(
    name="offer",
    shocks={"theta": (Uniform, {"low": "ql", "high": "qh"})},
    dynamics={
        "S": Control(["theta", "p"], lower_bound=0.0, upper_bound=1.0, agent="seller"),
        "u": lambda S, p, theta: S * (p - theta),
    },
    reward={"u": "seller"},
)

market_block = DBlock(name="market", dynamics={"p": clearing_price})

# The sellers act on the price they arrived at and are paid at that same price;
# the market then clears at what their decisions imply. Declaration order is
# what makes ``p`` a lag, so there is no timing annotation anywhere in the model.
lemons_block = RBlock(
    name="lemons",
    blocks=[
        RBlock(name="sellers", entity=Entity("seller"), blocks=[offer_block]),
        market_block,
    ],
)


def lemons_calibration(size=10000, low=0.0, high=1.0):
    """A market of *size* sellers whose quality is uniform on ``[low, high]``.

    Parameters
    ----------
    size : int, optional
        How many sellers. The clearing price is a mean over those who sold, so
        this is the accuracy of the fixed point rather than part of the model.
    low, high : float, optional
        Support of the quality distribution. The analytic results here assume
        ``low = 0``.

    Returns
    -------
    dict
    """
    return {"ql": low, "qh": high, "seller": size}


def seller_rule(theta, p):
    """Sell iff the price covers the seller's own valuation.

    Parameters
    ----------
    theta : numpy.ndarray
        Quality, one per seller.
    p : float
        The price sellers arrived at.

    Returns
    -------
    numpy.ndarray
        1.0 where the item sells, 0.0 where it does not.
    """
    return (theta <= p) * 1.0


def analytic_price(p0, t):
    """The clearing map iterated *t* times from *p0*, on the unit-support market.

    Parameters
    ----------
    p0 : float
        The starting price.
    t : int
        How many rounds.

    Returns
    -------
    float
    """
    return p0 * CONTRACTION_SLOPE**t
