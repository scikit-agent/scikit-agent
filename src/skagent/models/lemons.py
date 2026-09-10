r"""
Akerlof (1970) adverse selection: the market for lemons.

Sellers know the quality of what they hold; buyers do not, and can price only
the average quality of what is offered. Each seller holds one item of quality
:math:`\theta`, values it at :math:`\theta`, and parts with it only if the price
covers that. A buyer values the same item at a premium :math:`m`, so the price
that clears a competitive market is the buyer valuation of what actually trades:

.. math::
    S_i = 1 \iff p \geq \theta_i, \qquad u_i = S_i (p - \theta_i), \qquad
    p = m \, E[\theta_i \mid S_i = 1]

The defaults here are the paper's own numbers [Akerlof1970]_: quality uniform
on :math:`[0, 2]`, and buyers who value a car at :math:`3/2` of what its owner
does. So :func:`lemons_calibration` with no arguments is the model of that
paper's section II, and its answer is the paper's -- no trade at all, although
every car is worth more to a buyer than to the seller holding it.

The module builds its markets out of eight leaf blocks. Each states one thing
about the market, and every version below is a different composition of them, so
no equation is written twice:

    spread_quality_block    quality is uniform between the ends of the range
    two_type_quality_block  a car is a peach or a lemon, and nothing between
    supply_block            the seller decides, seeing only its own quality
    offer_block             the seller decides, seeing the price on the table
    seller_payoff_block     the seller's surplus, at whatever price is current
    market_block            the competitive price, given what was offered
    bid_block               a buyer names the price instead
    surplus_block           that buyer's surplus, per seller

The first two are alternatives to each other, and so are the next two: a market
takes one quality distribution and one seller decision. The last two are the
monopsony market's, and no other version composes them.

Where the payoff block sits determines which price the seller is paid at, and
where the market block sits determines whether the price is known when the
seller decides. Declaration order is therefore the whole of the timing, and
there is no timing annotation anywhere in the model. The three timings differ in
nothing else.

In :data:`lemons_block` and :data:`peaches_block` the sellers anticipate the
price their own supply induces. They decide, the market clears on what they
offered, and they are paid at that price. Nothing is lagged and no seller reads
a price, so a seller's rule is a best response to the price that rule induces
once every seller follows it. The equilibrium is a fixed point in rules, and a
solver has to find it.

In :data:`naive_lemons_block` and :data:`naive_peaches_block` the sellers
respond to a price already posted. The payoff block comes before the market
block, so sellers are paid at the price they responded to and the market then
clears at what their decisions imply. The price is read before it is written,
which makes it an arrival state: period :math:`t`'s sellers face period
:math:`t-1`'s clearing price. Simulating :math:`T` periods therefore runs
:math:`T` rounds of the clearing map, and no solver is needed to watch the same
equilibrium arrive.

In :data:`monopsony_block` a buyer commits to a price before supply. The price
becomes a decision, owned by a buyer who names it to maximize its own surplus
and cannot see quality:

.. math::
    p \in [\ell, h] \text{ chosen by the buyer}, \qquad
    w = E[S_i (m \theta_i - p)]

The buyer's payoff is written per seller rather than as a total, so the closed
forms below do not depend on how many sellers there are.

A buyer that named a price only after the cars had been offered would name the
lowest price it was allowed to. Once the pool is fixed, its surplus is
:math:`v(m\bar\theta - p)` in a volume and a grade it can no longer change, so
the surplus falls in the price at every price and the best response is the lower
bound. Sellers anticipate that and offer nothing, so the market shuts for a
reason that has nothing to do with adverse selection. That is why the buyer here
commits first, and why a competitive price is a condition on the market rather
than any one participant's choice: competition among buyers is what holds the
price up to the value of what trades.

The three timings need three different treatments, and the relevance graph
tells them apart. Where the price is anticipated, a seller's payoff runs
through it to every other seller's decision, so the model is a strategic fixed
point among instances of one class and ``relies_on("S", "S")`` is ``True``:
one decision, and it is its own predecessor, so no order solves it. Where the
price is posted, the same call is ``False`` and correct, since this period's
payoff turns on last period's price and within a period there is no reliance to
find. Where a buyer commits, the graph reports two nodes and one edge, from the
price to the sell decision, and its topological order is backward induction.

The market behaves differently under the two quality distributions. The
uniform of section II makes the clearing map exactly linear, since
:math:`E[\theta \mid \theta \leq p]` is :math:`(\ell + p)/2`, so the only prices
it can reproduce are no trade, a corner, or -- at :math:`m = 2` exactly -- every
price at once. The paper's automobiles are two types rather than a spread, and
that is where a second price becomes possible: only lemons trade at
:math:`m\ell`, and if peaches are common enough there is a second price,
:math:`m E[\theta]`, at which everything trades. Which one the market reaches
depends on where it starts. See :data:`MARKETS` and :data:`PEACH_MARKETS` for
the named configurations, and :func:`clearing_fixed_points` and
:func:`peaches_fixed_points` for the general answers.

Within the uniform family the quality floor is what decides whether the
unravelling has anywhere to stop. With no floor and a premium below 2 the map is
a contraction onto :math:`p = 0`. With a floor the same map has a second, higher
fixed point at :math:`m\ell/(2 - m)`, and the market unravels down to it rather
than away: the top of the market is destroyed and the bottom keeps trading.
Above the threshold :math:`m = 2` the collapse is unstable instead, and the
price runs up to :math:`m(\ell + h)/2`, where every item trades.

:func:`clearing_map` and :func:`peaches_clearing_map` are each the same function
under two readings. Iterated, they are the price path a posted-price market
simulates; read once, they give the price an anticipated price induces. An
anticipating market's equilibria are therefore exactly their fixed points.

Signalling is the paper's own answer to adverse selection, and this model is
shaped to reach it. Section IV of [Akerlof1970]_ names guarantees, brand names
and licensing as the institutions that counteract it, and all of them work the
same way: the seller takes a costly action whose cost is lower for higher quality, so
that taking it is credible. Two things have to change here. The first is
already in place, since a separating equilibrium separates types and
:data:`two_type_quality_block` supplies them. The second is that the price stops
being one number: the buyer commits to a pricing rule, taking the seller's
signal as its information set, and that rule is applied once per seller. Each
seller is then paid at the price its own signal earns. That adds a second
control to :data:`supply_block`, which is why the seller's information set is
its quality alone, so a signalling decision drops in beside the sell decision.

When the price is below what every
seller thinks their own car is worth, nothing is offered at all, and the
clearing price would be an average over an empty market. This model answers
zero. That is a choice rather than a definition, and it is what makes no trade
an equilibrium rather than an error: a market at a price of zero has nothing
offered to it, so it clears at zero again and stays there. A different market
could reasonably answer something else, so the value is written into the
clearing equation rather than supplied by the library.

Binary decisions are relaxed to continuous ``[0, 1]`` controls, pending
discrete-action support, following the convention of
:mod:`skagent.models.macid`. :func:`seller_rule` and :func:`supply_rule` return
exact 0 and 1, so the relaxation costs the supplied equilibria nothing.

References
----------
.. [Akerlof1970] Akerlof, G.A. (1970). "The Market for 'Lemons': Quality
       Uncertainty and the Market Mechanism." The Quarterly Journal of
       Economics, 84(3), 488-500. https://doi.org/10.2307/1879431
"""

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Bernoulli, Uniform

PREMIUM = 1.5
"""How much more a buyer values an item than the seller holding it.

Akerlof's own 3/2: his sellers value a car at its quality, his buyers at three
halves of it.
"""

QUALITY_LOW = 0.0
QUALITY_HIGH = 2.0
"""The ends of the quality range, and Akerlof's own: uniform on [0, 2]."""

PREMIUM_THRESHOLD = 2.0
"""The premium at which the uniform market's clearing map has slope exactly 1.

Below it the map contracts and the market collapses; above it no trade is an
unstable fixed point and the price runs up until every item trades. The
threshold does not depend on the quality range.
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


spread_quality_block = DBlock(
    name="quality",
    shocks={"theta": (Uniform, {"low": "ql", "high": "qh"})},
)

two_type_quality_block = DBlock(
    name="quality",
    shocks={"peach": (Bernoulli, {"p": "share"})},
    dynamics={"theta": lambda peach, ql, qh: ql + (qh - ql) * peach},
)

# No price in the information set: what a seller expects the market to pay is
# carried by its rule rather than read, which is what makes the equilibrium a
# fixed point in rules. A signalling decision would join this block and share
# the same information set.
supply_block = DBlock(
    name="supply",
    dynamics={
        "S": Control(["theta"], lower_bound=0.0, upper_bound=1.0, agent="seller")
    },
)

offer_block = DBlock(
    name="offer",
    dynamics={
        "S": Control(["theta", "p"], lower_bound=0.0, upper_bound=1.0, agent="seller")
    },
)

seller_payoff_block = DBlock(
    name="seller_payoff",
    dynamics={"u": lambda S, p, theta: S * (p - theta)},
    reward={"u": "seller"},
)

market_block = DBlock(name="market", dynamics={"p": clearing_price})

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


def _sellers(quality, decision):
    """The seller class: what its members hold, and what they decide."""
    return RBlock(name="sellers", entity=Entity("seller"), blocks=[quality, decision])


_payoffs = RBlock(name="payoffs", entity=Entity("seller"), blocks=[seller_payoff_block])

# Supply, then clearing, then payment: the price is not known when the sellers
# decide and is not left over from anywhere, so they can only anticipate it.
lemons_block = RBlock(
    name="lemons",
    blocks=[_sellers(spread_quality_block, supply_block), market_block, _payoffs],
)

peaches_block = RBlock(
    name="peaches",
    blocks=[_sellers(two_type_quality_block, supply_block), market_block, _payoffs],
)

# Payment before clearing: the sellers are paid at the price they responded to,
# and the market then clears at what their decisions imply. That is what makes
# the price a lag rather than an outcome of this round.
naive_lemons_block = RBlock(
    name="naive_lemons",
    blocks=[_sellers(spread_quality_block, offer_block), _payoffs, market_block],
)

naive_peaches_block = RBlock(
    name="naive_peaches",
    blocks=[_sellers(two_type_quality_block, offer_block), _payoffs, market_block],
)

# The price named first, by a buyer, rather than left over from the last round.
monopsony_block = RBlock(
    name="monopsony",
    blocks=[
        bid_block,
        _sellers(spread_quality_block, offer_block),
        _payoffs,
        surplus_block,
    ],
)


def lemons_calibration(size=10000, low=QUALITY_LOW, high=QUALITY_HIGH, premium=PREMIUM):
    """A market of *size* sellers whose quality is uniform on ``[low, high]``.

    The defaults are Akerlof's section II.

    Parameters
    ----------
    size : int, optional
        How many sellers. The clearing price is an average over those who sold,
        so this is the accuracy of the fixed point rather than part of the model.
    low, high : float, optional
        The ends of the quality range. ``low`` is the quality floor.
    premium : float, optional
        How much more a buyer values an item than the seller holding it.

    Returns
    -------
    dict
    """
    return {"ql": low, "qh": high, "seller": size, "premium": premium}


def peaches_calibration(
    size=10000, low=0.4, high=QUALITY_HIGH, premium=PREMIUM, share=0.5
):
    """A market of *size* sellers holding a peach or a lemon and nothing between.

    Parameters
    ----------
    size : int, optional
        How many sellers.
    low, high : float, optional
        The quality of a lemon and of a peach.
    premium : float, optional
        How much more a buyer values an item than the seller holding it.
    share : float, optional
        The fraction of cars that are peaches.

    Returns
    -------
    dict
    """
    return {
        "ql": low,
        "qh": high,
        "seller": size,
        "premium": premium,
        "share": share,
    }


MARKETS = {
    "akerlof": {"premium": 1.5, "low": 0.0},
    "partial-collapse": {"premium": 1.5, "low": 0.4},
    "knife-edge": {"premium": 2.0, "low": 0.0},
    "no-collapse": {"premium": 2.5, "low": 0.0},
}
"""Four configurations of :func:`lemons_calibration`, on quality up to 2.

Each is a pair of arguments rather than a whole calibration, so the number of
sellers stays the caller's: ``lemons_calibration(size=1000, **MARKETS["akerlof"])``.

    name               competitive prices     monopsony price and payoff
    akerlof            0                      0,   and 0
    partial-collapse   0 and 1.2              0.8, and 0.025
    knife-edge         every price up to 2    any, and 0
    no-collapse        0 (unstable) and 2.5   2,   and 0.5

``akerlof`` is the paper's own section II. Under ``partial-collapse`` the two
competitive prices are both stable and the quality floor divides their basins: a
market that starts below the floor has nobody willing to sell and stays
collapsed, and one that starts above it unravels down to 1.2 rather than to
nothing.
"""

PEACH_MARKETS = {
    "lemons-only": {"share": 0.3},
    "two-prices": {"share": 0.7},
}
"""Two configurations of :func:`peaches_calibration`, at its default qualities.

    name          competitive prices
    lemons-only   0 and 0.6
    two-prices    0 and 0.6 and 2.28

A lemon is worth 0.4 and a peach 2.0, so at a price of 0.6 the lemons all sell
and no peach does, whatever peaches are worth and however many there are. The
second price exists only where peaches are common enough to carry the average,
which is :func:`peach_share_for_trade` and is 0.583 here; above it a market that
starts high stays high and one that starts low still collapses to the lemons.
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

    The equilibrium rule of an anticipating market is this one at a fixed point
    of that market's clearing map.

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


def clearing_map(p, premium=PREMIUM, low=QUALITY_LOW, high=QUALITY_HIGH):
    """The price that sellers facing *p* bring about, on a uniform quality range.

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


def peaches_clearing_map(p, premium=PREMIUM, low=0.4, high=QUALITY_HIGH, share=0.5):
    """The price that sellers facing *p* bring about, in a two-type market.

    Below a lemon's worth nothing is offered; between the two qualities only
    lemons are, and their average quality does not depend on the price at all;
    at or above a peach's worth everything is.

    Parameters
    ----------
    p : float
        The price the sellers act on.
    premium, low, high, share : float, optional
        As in :func:`peaches_calibration`.

    Returns
    -------
    float
    """
    if p < low:
        return 0.0
    if p < high:
        return premium * low
    return premium * ((1 - share) * low + share * high)


def clearing_path(p0, periods, mapping=None, **market):
    """A clearing map iterated from *p0*, one price per round.

    Parameters
    ----------
    p0 : float
        The starting price.
    periods : int
        How many rounds.
    mapping : callable, optional
        The clearing map. Defaults to :func:`clearing_map`.
    **market
        Passed to *mapping*.

    Returns
    -------
    list of float
        Length *periods*, the price after each round.
    """
    step = clearing_map if mapping is None else mapping
    path = []
    p = p0
    for _ in range(periods):
        p = step(p, **market)
        path.append(p)
    return path


def clearing_fixed_points(premium=PREMIUM, low=QUALITY_LOW, high=QUALITY_HIGH):
    """The prices a uniform-quality competitive market reproduces.

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


def peaches_fixed_points(premium=PREMIUM, low=0.4, high=QUALITY_HIGH, share=0.5):
    """The prices a two-type competitive market reproduces.

    Up to three, and the middle one is what the uniform market cannot have: a
    price at which trade survives and every peach is withheld.

    Parameters
    ----------
    premium, low, high, share : float, optional
        As in :func:`peaches_calibration`.

    Returns
    -------
    tuple of float
        In increasing order.
    """
    points = [0.0]
    lemons_only = premium * low
    if low <= lemons_only < high:
        points.append(lemons_only)
    complete = premium * ((1 - share) * low + share * high)
    if complete >= high:
        points.append(complete)
    return tuple(points)


def peach_share_for_trade(premium=PREMIUM, low=0.4, high=QUALITY_HIGH):
    """The share of peaches above which a market in peaches exists at all.

    Below it, no price a buyer will pay for the average car is enough to bring a
    peach out, whatever the market does.

    Parameters
    ----------
    premium, low, high : float, optional
        As in :func:`peaches_calibration`.

    Returns
    -------
    float
        A share, which may exceed 1 where no share of peaches is enough.
    """
    return (high / premium - low) / (high - low)


def buyer_payoff(p, premium=PREMIUM, low=QUALITY_LOW, high=QUALITY_HIGH):
    """The monopsonist's surplus per seller at price *p*, on uniform quality.

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


def monopsony_price(premium=PREMIUM, low=QUALITY_LOW, high=QUALITY_HIGH):
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
