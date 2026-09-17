r"""
Aiyagari (1994) [Aiyagari1994]_: a cross-section that prices its own capital.

Many households each save out of labour income, and what they save between them
is the economy's capital stock. Capital sets the interest rate and the wage,
those set what each household has to spend, and what each household leaves over
is next period's capital. The households never interact directly; they meet only
through an average.

Symbols
--------

One value per household:

    theta   the labour endowment drawn this period, which has mean one
    a       assets carried in from last period, the model's arrival state
    z       cash on hand: labour income plus assets and the interest on them
    c       consumption, the household's one decision
    u       the household's payoff from consuming that much

One value for the whole economy, and capitalised for it:

    K       capital per head, the average of ``a`` over the households
    R       the net interest rate, so a household earns ``(1 + R)`` on assets
    W       the wage paid per unit of labour endowment

Calibration parameters:

    alpha        capital's share of output
    delta        the fraction of capital that wears out each period
    CRRA         relative risk aversion in the household's utility
    sigma_theta  how dispersed the labour endowment is
    household    how many households there are

The model
----------

.. math::
    K = \frac{1}{N}\sum_j a_j, \qquad R = \alpha K^{\alpha - 1} - \delta,
    \qquad W = (1 - \alpha) K^{\alpha}

.. math::
    z_i = \theta_i W + (1 + R) a_i, \qquad
    u_i = \frac{c_i^{1-\gamma}}{1-\gamma}, \qquad a_i' = z_i - c_i

The two prices are the marginal products of Cobb-Douglas production in its
intensive form, :math:`Y = K^{\alpha} L^{1-\alpha}`, with labour normalised to
one unit per head: the endowment has mean one and every household supplies it
whatever the wage, so :math:`L` drops out of both formulas. Factors are paid
their marginal products, so output is exactly exhausted before depreciation,
:math:`(R + \delta) K + W = K^{\alpha}`, and capital's share of it is
:math:`\alpha`.

A fraction :math:`\delta` of the capital stock wears out each period, so
:math:`R` is the marginal product of capital net of that, and a household's
assets earn :math:`(1 + R)`. This is the rate [Aiyagari1994]_ works with.
Depreciation is also what keeps the model's scale sensible: a household's
resources :math:`W + (1+R)K` come to output plus the capital that survived,
:math:`K^{\alpha} + (1-\delta)K`, so the stationary capital-output ratio is
exactly :math:`s/(1 - s(1-\delta))`. With no depreciation that is
:math:`s/(1-s)`, which reaches 9 at plausible savings rates, about three times
what an economy shows, because nothing ever wears out.

The savings rate is a fraction of cash on hand rather than of income, so it is
not the textbook savings rate and picking one by eye is misleading.
:func:`savings_rate_for` inverts the relationship instead, returning the rate at
which the economy settles at a given interest rate.

The market block is declared before the households, so :math:`K` is computed
from the assets the households arrived with rather than the ones they are about
to choose. That makes :math:`a` an arrival state, and simulating :math:`T`
periods runs :math:`T` rounds of the aggregate's own law of motion.

Two symbols differ from the design note this model was written from. It writes
the labour endowment :math:`l`, which is ``theta`` here, both because a bare
``l`` is not a legal identifier under the project's linter and because ``theta``
is what the rest of the library calls a mean-one transitory shock. And it writes
capital per head :math:`k`, which is ``K`` here, since the library capitalises a
variable that stands for the whole economy rather than for one member of it.

What the model is for
----------------------

Under a fixed savings rate the aggregate has a closed form, and that is what
makes this model a test rather than only a demonstration. If every household
consumes :math:`(1-s)z`, then :math:`a' = s z`, and averaging over the class
with :math:`E[\theta] = 1`:

.. math::
    K' = s\,(W + (1 + R) K) = s\,(K^{\alpha} + (1 - \delta) K)

so the stationary capital solves :math:`K^{1-\alpha} = s / (1 - s(1-\delta))`,
giving :math:`R^{*} = \alpha (1 - s(1-\delta))/s - \delta` exactly. The map's
slope there is :math:`\alpha + s(1-\delta)(1-\alpha)`, which is below 1 whenever
:math:`s(1-\delta) < 1`, so the aggregate converges from any starting capital
and the convergence needs no damping.

That claim is cross-sectional rather than optimal. The savings rate is a rule of
thumb, no household is solving anything, and the aggregate still arrives where
the arithmetic says it should -- which is what makes it a check on the
simulator's treatment of a class and its average rather than on a solver.

References
----------
.. [Aiyagari1994] Aiyagari, S.R. (1994). "Uninsured Idiosyncratic Risk and
       Aggregate Saving." *The Quarterly Journal of Economics*, 109(3),
       659-684. https://doi.org/10.2307/2118417
"""

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import MeanOneLogNormal

CAPITAL_SHARE = 0.36
"""Capital's share of output, the exponent on capital in the production function."""

DEPRECIATION = 0.08
"""The fraction of the capital stock that wears out each period.

A conventional value for an annual calibration. Setting it to zero gives a model
in which capital lasts forever, whose capital-output ratio is about three times
an economy's.
"""

market_block = DBlock(
    name="market",
    dynamics={
        # Reads out of the household class, so this is the model's one crossing.
        "K": lambda a: a.mean(),
        "R": lambda K, alpha, delta: alpha * K ** (alpha - 1) - delta,
        "W": lambda K, alpha: (1 - alpha) * K**alpha,
    },
)

household_block = DBlock(
    name="household",
    shocks={"theta": (MeanOneLogNormal, {"sigma": "sigma_theta"})},
    dynamics={
        "z": lambda theta, W, R, a: theta * W + (1 + R) * a,
        "c": Control(
            ["z"], lower_bound=0.0, upper_bound=lambda z: z, agent="household"
        ),
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
        # End-of-period assets, and next period's arrival value for the same
        # symbol. A tick block would only rename it, so there is none.
        "a": lambda z, c: z - c,
    },
    reward={"u": "household"},
)

# The market clears on the assets the households arrived with, and the
# households then decide. Declaration order is what makes ``a`` an arrival
# state, so there is no timing annotation anywhere in the model.
aiyagari_block = RBlock(
    name="aiyagari",
    blocks=[
        market_block,
        RBlock(
            name="households",
            entity=Entity("household"),
            blocks=[household_block],
        ),
    ],
)


def aiyagari_calibration(
    size=1000, alpha=CAPITAL_SHARE, delta=DEPRECIATION, crra=2.0, sigma=1.0
):
    """An economy of *size* households.

    Parameters
    ----------
    size : int, optional
        How many households. The capital stock is an average over them, so this
        is the accuracy of the aggregate rather than part of the model.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.
    crra : float, optional
        Relative risk aversion in the household's utility. It does not enter the
        aggregate under a fixed savings rate.
    sigma : float, optional
        Dispersion of the labour endowment, which has mean one whatever this is.

    Returns
    -------
    dict
    """
    return {
        "household": size,
        "alpha": alpha,
        "delta": delta,
        "CRRA": crra,
        "sigma_theta": sigma,
    }


def savings_rule(rate):
    """Consume all but *rate* of what is on hand, as a decision rule for ``c``.

    Parameters
    ----------
    rate : float
        The fraction saved, in ``(0, 1)``.

    Returns
    -------
    callable
        A rule of cash on hand alone.
    """
    return lambda z: (1 - rate) * z


def capital_map(capital, rate, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The capital an economy holding *capital* per head leaves for next period.

    Iterated, this is the path :data:`aiyagari_block` simulates under
    :func:`savings_rule`; its fixed point is :func:`stationary_capital`.

    Parameters
    ----------
    capital : float
        Capital per head this period.
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.

    Returns
    -------
    float
    """
    return rate * (capital**alpha + (1 - delta) * capital)


def stationary_capital(rate, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The capital per head that reproduces itself under *rate*.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.

    Returns
    -------
    float
    """
    return (rate / (1 - rate * (1 - delta))) ** (1 / (1 - alpha))


def stationary_prices(rate, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The interest rate and wage at :func:`stationary_capital`.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.

    Returns
    -------
    dict
        ``R`` and ``W``.
    """
    capital = stationary_capital(rate, alpha, delta)
    return {
        "R": alpha * (1 - rate * (1 - delta)) / rate - delta,
        "W": (1 - alpha) * capital**alpha,
    }


def convergence_rate(rate, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The slope of :func:`capital_map` at its fixed point.

    Below 1 whenever ``rate * (1 - delta)`` is, which is why the aggregate
    converges from any starting capital without damping.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.

    Returns
    -------
    float
    """
    return alpha + rate * (1 - delta) * (1 - alpha)


def savings_rate_for(interest, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The savings rate at which the economy settles at interest rate *interest*.

    The inverse of :func:`stationary_prices`' first entry. Cash on hand includes
    a household's whole asset position, so a savings rate here is not the
    textbook fraction of income and is easier to choose by the interest rate it
    implies than by eye.

    Parameters
    ----------
    interest : float
        The net interest rate the economy should settle at.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.

    Returns
    -------
    float
    """
    return alpha / (interest + delta + alpha * (1 - delta))
