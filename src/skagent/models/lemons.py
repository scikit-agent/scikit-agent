r"""
Akerlof (1970) adverse selection: the market for lemons.

Sellers know the quality of what they hold; buyers do not, and can price only
the average quality of what is offered. Each seller draws a quality
:math:`\theta`, values it at :math:`\theta`, and parts with it only if the price
covers that. A buyer values the same item at a premium :math:`m`, so the price
that clears a competitive market is the buyer valuation of what actually trades:

.. math::
    \theta_i \sim U[\ell, h], \qquad S_i = 1 \iff p \geq \theta_i, \qquad
    u_i = S_i (p - \theta_i), \qquad p = m \, E[\theta_i \mid S_i = 1]

Every version below shares those sellers and that clearing price. What differs
is WHEN the price is determined relative to the decision it feeds, and that
alone decides what it takes to solve the model.

**Sellers anticipate the price their own supply induces** --
:data:`lemons_block`, and the one an economist means by the Akerlof equilibrium.
Sellers act on their own quality alone, the market clears on what they offered,
and they are paid at that price. Nothing is lagged. A seller's rule is a best
response to the price that rule induces once every seller follows it, so the
equilibrium is a FIXED POINT IN RULES and a solver has to find it.

**Sellers respond to a price already posted** -- :data:`naive_lemons_block`. The
price is read before the market writes it, which makes it an arrival state:
period :math:`t`'s sellers face period :math:`t-1`'s clearing price. Simulating
:math:`T` periods therefore runs :math:`T` rounds of the clearing map, and the
price column is the iterates. This version needs no solver, which makes it the
cheapest way to watch the same equilibrium arrive.

**A buyer commits to a price before supply** -- :data:`monopsony_block`. The
price becomes a decision, owned by a buyer who names it to maximize its own
surplus and cannot see quality:

.. math::
    p \in [\ell, h] \text{ chosen by the buyer}, \qquad
    w = E[S_i (m \theta_i - p)]

The buyer's payoff is written per seller rather than as a total, so the closed
forms below do not depend on how many sellers there are.

**A buyer who priced AFTER supply was committed would pay the floor**, which is
why the buyer commits first and why the competitive price is a market condition
rather than anyone's choice. Once the pool is fixed, the buyer's surplus is
:math:`v(m\bar\theta - p)` in a volume and a grade it can no longer change, so
it falls in the price at every price: the best response is the lower bound, the
sellers anticipate that, and the market shuts for a reason that has nothing to
do with adverse selection. Competition among buyers is what pins the price up to
the value of what trades.

**None of the three has a cycle its relevance graph can see, and they need
three different treatments.** In :data:`lemons_block` a seller's payoff runs
through the price to every other seller's decision, so the model is a genuine
strategic fixed point among instances of one class -- and
``relies_on("S", "S")`` is nonetheless ``False``, because that reliance is not
derivable without expanding the class. In :data:`naive_lemons_block` the same
call is ``False`` and CORRECT: this period's payoff turns on last period's
price, so within a period there is no reliance to find. In
:data:`monopsony_block` the graph reports two nodes and the one edge from the
price to the sell decision, and its topological order is genuine backward
induction. So a cyclicity test cannot tell the first two apart, and only one of
them can be solved a decision at a time.

**Two knobs move the market between four regimes**, and both are calibration
parameters. The premium :math:`m` decides whether the clearing map contracts,
since its slope is :math:`m/2`; the quality floor :math:`\ell` decides whether
the unravelling has anywhere to stop. See :data:`MARKETS` for the four named
configurations and :func:`clearing_fixed_points` for the general answer.

With no floor and a premium below :math:`2`, conditional on sale
:math:`\theta \sim U[0, p]`, so :math:`E[\theta \mid \text{sale}] = p/2` and the
map is :math:`p' = (m/2)\,p`: a contraction whose only fixed point is
:math:`p = 0`. No trade, though every item has a buyer who values it above its
holder. With a floor the same map has a second, higher fixed point at
:math:`m\ell / (2 - m)`, and the market unravels down to it rather than away:
the top of the market is destroyed and the bottom keeps trading. Above the
threshold :math:`m = 2` the collapse is unstable instead, and the price runs up
to :math:`m(\ell + h)/2`, where every item trades.

:func:`clearing_map` is the same function under both readings. Iterated, it is
the price path :data:`naive_lemons_block` simulates; read once, it is the price
an anticipated price induces in :data:`lemons_block`, whose equilibrium is
therefore exactly where the map has a fixed point.

**The empty pool is the collapse, and the value there belongs to the model.**
Where the price is zero nobody sells, so the clearing price averages over
nothing. This model answers 0, which is what makes no trade a fixed point rather
than a NaN; a different market answers differently, so the choice is stated at
the reduction and not inferred.

Binary decisions are relaxed to continuous ``[0, 1]`` controls, pending
discrete-action support, following the convention of
:mod:`skagent.models.macid`. :func:`seller_rule` and :func:`supply_rule` return
exact 0 and 1, so the relaxation costs the supplied equilibria nothing.
"""

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Uniform

PREMIUM = 1.5
"""Default for how much more a buyer values an item than the seller holding it."""

PREMIUM_THRESHOLD = 2.0
"""The premium at which the clearing map's slope is exactly 1.

Below it the map contracts and the market collapses; above it no trade is an
unstable fixed point and the price runs up until every item trades. The
threshold does not depend on the quality support.
"""

_EMPTY_POOL = 1e-12
"""Guards the division when nothing is offered, so an empty pool prices at 0."""


def clearing_price(theta, S, premium):
    """The competitive price: buyer valuation of the average item that sold.

    A weighted mean rather than a mean over a selection, which is what keeps it
    usable on every path the library solves on: there is no boolean index, no
    branch on the data and no dynamic shape, so it differentiates and batches
    under ``torch`` as readily as it evaluates under ``numpy``. Where the sell
    decision is 0 or 1 the two forms agree exactly; where it is relaxed to a
    probability of selling, the weighted form is the expected quality of what
    trades, which is the mixed extension rather than an approximation of it.

    Parameters
    ----------
    theta : array
        Quality, one per seller.
    S : array
        The sell decision, one per seller, as a weight in ``[0, 1]``.
    premium : float
        How much more a buyer values an item than the seller holding it.

    Returns
    -------
    float
        *premium* times the average quality offered, and 0 when nothing is.
    """
    weight = S * 1.0
    return premium * (weight * theta).sum() / (weight.sum() + _EMPTY_POOL)


supply_block = DBlock(
    name="supply",
    shocks={"theta": (Uniform, {"low": "ql", "high": "qh"})},
    # No price in the information set: a seller acts on its own quality, and
    # what it expects the market to pay is carried by the rule rather than read.
    dynamics={
        "S": Control(["theta"], lower_bound=0.0, upper_bound=1.0, agent="seller")
    },
)

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

seller_payoff_block = DBlock(
    name="seller_payoff",
    dynamics={"u": lambda S, p, theta: S * (p - theta)},
    reward={"u": "seller"},
)

bid_block = DBlock(
    name="bid",
    dynamics={
        "p": Control(
            [],
            lower_bound=lambda ql: ql,
            upper_bound=lambda qh: qh,
            agent="buyer",
        )
    },
)

surplus_block = DBlock(
    name="surplus",
    # Reads out of the seller class, so this is the monopsony model's crossing.
    dynamics={"w": lambda S, theta, p, premium: (S * (premium * theta - p)).mean()},
    reward={"w": "buyer"},
)

# The sellers supply, the market clears on what they supplied, and they are paid
# at the price that clearing produced. The seller class is declared twice
# because it acts before the market and is paid after; declaration order is what
# fixes the dynamics order, so there is no timing annotation in the model.
lemons_block = RBlock(
    name="lemons",
    blocks=[
        RBlock(name="sellers", entity=Entity("seller"), blocks=[supply_block]),
        market_block,
        RBlock(name="payoffs", entity=Entity("seller"), blocks=[seller_payoff_block]),
    ],
)

# The same market with the price moved in front of the sellers, which is what
# makes it a lag: they respond to what the last round cleared at.
naive_lemons_block = RBlock(
    name="naive_lemons",
    blocks=[
        RBlock(name="sellers", entity=Entity("seller"), blocks=[offer_block]),
        market_block,
    ],
)

# The same sellers again, with the price named by a buyer who moves first rather
# than left over from the last round.
monopsony_block = RBlock(
    name="monopsony",
    blocks=[
        bid_block,
        RBlock(name="sellers", entity=Entity("seller"), blocks=[offer_block]),
        surplus_block,
    ],
)


def lemons_calibration(size=10000, low=0.0, high=1.0, premium=PREMIUM):
    """A market of *size* sellers whose quality is uniform on ``[low, high]``.

    Parameters
    ----------
    size : int, optional
        How many sellers. The clearing price is an average over those who sold,
        so this is the accuracy of the fixed point rather than part of the model.
    low, high : float, optional
        Support of the quality distribution. ``low`` is the quality floor.
    premium : float, optional
        How much more a buyer values an item than the seller holding it.

    Returns
    -------
    dict
    """
    return {"ql": low, "qh": high, "seller": size, "premium": premium}


MARKETS = {
    "collapse": {"premium": 1.5, "low": 0.0},
    "partial-collapse": {"premium": 1.5, "low": 0.2},
    "knife-edge": {"premium": 2.0, "low": 0.0},
    "no-collapse": {"premium": 2.5, "low": 0.0},
}
"""Four configurations of :func:`lemons_calibration`, on quality up to 1.

Each is a pair of arguments rather than a whole calibration, so the number of
sellers stays the caller's: ``lemons_calibration(size=1000, **MARKETS["collapse"])``.

    name               competitive prices     monopsony price and payoff
    collapse           0                      0,   and 0
    partial-collapse   0 and 0.6              0.4, and 0.0125
    knife-edge         every price up to 1    any, and 0
    no-collapse        0 (unstable) and 1.25  1,   and 0.25

Under ``partial-collapse`` the two competitive prices are both stable and the
quality floor divides their basins: a market that starts below the floor has
nobody willing to sell and stays collapsed, and one that starts above it
unravels down to 0.6 rather than to nothing.
"""


def seller_rule(theta, p):
    """Sell if and only if the posted price covers the seller's own valuation.

    Parameters
    ----------
    theta : array
        Quality, one per seller.
    p : float
        The price the sellers can see.

    Returns
    -------
    array
        1.0 where the item sells, 0.0 where it does not.
    """
    return (theta <= p) * 1.0


def supply_rule(price):
    """Supply as though the market will clear at *price*, seeing only quality.

    The equilibrium rule of :data:`lemons_block` is this one at a fixed point of
    :func:`clearing_map`.

    Parameters
    ----------
    price : float
        The price the sellers anticipate.

    Returns
    -------
    callable
        A rule of quality alone.
    """
    return lambda theta: (theta <= price) * 1.0


def bid_rule(price):
    """A buyer that always names *price*, as a decision rule for ``p``.

    Parameters
    ----------
    price : float

    Returns
    -------
    callable
        A rule taking no arguments, since the buyer's information set is empty.
    """
    return lambda: price


def clearing_map(p, premium=PREMIUM, low=0.0, high=1.0):
    """The price that sellers facing *p* bring about.

    Iterated, this is the path :data:`naive_lemons_block` simulates; read once,
    it is the price an anticipated *p* induces in :data:`lemons_block`.

    Parameters
    ----------
    p : float
        The price the sellers act on.
    premium, low, high : float, optional
        As in :func:`lemons_calibration`.

    Returns
    -------
    float
    """
    if p < low:
        return 0.0
    return premium * (low + min(p, high)) / 2


def clearing_path(p0, periods, premium=PREMIUM, low=0.0, high=1.0):
    """The clearing map iterated from *p0*, one price per round.

    Parameters
    ----------
    p0 : float
        The starting price.
    periods : int
        How many rounds.
    premium, low, high : float, optional
        As in :func:`lemons_calibration`.

    Returns
    -------
    list of float
        Length *periods*, the price after each round.
    """
    path = []
    p = p0
    for _ in range(periods):
        p = clearing_map(p, premium, low, high)
        path.append(p)
    return path


def clearing_fixed_points(premium=PREMIUM, low=0.0, high=1.0):
    """The prices the competitive market reproduces.

    These are the equilibria of :data:`lemons_block` and the rest points of
    :data:`naive_lemons_block`. No trade is always one of them. A second appears
    either where the unravelling stops at the quality floor, or, above
    :data:`PREMIUM_THRESHOLD`, where every item trades.

    Parameters
    ----------
    premium, low, high : float, optional
        As in :func:`lemons_calibration`.

    Returns
    -------
    tuple of float
        In increasing order. At the threshold with no quality floor the map is
        the identity below *high* and the two values returned are the ends of a
        continuum of fixed points rather than two isolated ones.
    """
    points = [0.0]
    if premium < PREMIUM_THRESHOLD:
        stalled = premium * low / (PREMIUM_THRESHOLD - premium)
        if 0.0 < stalled <= high:
            points.append(stalled)
    complete = premium * (low + high) / 2
    if complete >= high:
        points.append(complete)
    return tuple(points)


def buyer_payoff(p, premium=PREMIUM, low=0.0, high=1.0):
    """The monopsonist's surplus per seller at price *p*.

    Parameters
    ----------
    p : float
        The price the buyer names.
    premium, low, high : float, optional
        As in :func:`lemons_calibration`.

    Returns
    -------
    float
    """
    if p <= low:
        return 0.0
    share = min((p - low) / (high - low), 1.0)
    return share * (premium * (low + min(p, high)) / 2 - p)


def monopsony_price(premium=PREMIUM, low=0.0, high=1.0):
    """The price that maximizes :func:`buyer_payoff`.

    Parameters
    ----------
    premium, low, high : float, optional
        As in :func:`lemons_calibration`.

    Returns
    -------
    float
        At and above :data:`PREMIUM_THRESHOLD` the buyer takes the whole market
        at *high*. Below it the buyer bids the floor up by a factor of
        ``1 / (2 - premium)``, which with no floor is no trade at all.
    """
    if premium >= PREMIUM_THRESHOLD:
        return high
    return min(max(low / (PREMIUM_THRESHOLD - premium), low), high)
