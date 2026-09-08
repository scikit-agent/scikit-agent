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
    CRRA         relative risk aversion in the household's utility
    sigma_theta  how dispersed the labour endowment is
    household    how many households there are

The model
----------

.. math::
    K = \frac{1}{N}\sum_j a_j, \qquad R = \alpha K^{\alpha - 1}, \qquad
    W = (1 - \alpha) K^{\alpha}

.. math::
    z_i = \theta_i W + (1 + R) a_i, \qquad
    u_i = \frac{c_i^{1-\gamma}}{1-\gamma}, \qquad a_i' = z_i - c_i

The two prices are the marginal products of Cobb-Douglas production in its
intensive form, :math:`Y = K^{\alpha} L^{1-\alpha}`, with labour normalised to
one unit per head: the endowment has mean one and every household supplies it
whatever the wage, so :math:`L` drops out of both formulas. Factors are paid
their marginal products, so output is exactly exhausted,
:math:`RK + W = K^{\alpha}`, and capital's share of it is :math:`\alpha`.

There is no depreciation here, and that is a real difference from
[Aiyagari1994]_. A household earns :math:`(1+R)` on its assets where :math:`R`
is the gross marginal product of capital, whereas Aiyagari's own interest rate
is net of a depreciation rate. So :math:`W + (1+R)K` comes to output plus the whole
existing capital stock, which is where the :math:`+ K` in the law of motion
below comes from, and the savings rate is a fraction of cash on hand rather
than of income: at :math:`s = 0.9` a household consumes a tenth of its wealth
each period, not a tenth of its earnings. The stationary capital-output ratio
is then exactly :math:`s/(1-s)`, which is 9 at that rate.

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
    K' = s\,(W + (1 + R) K) = s\,(K^{\alpha} + K)

so the stationary capital solves :math:`K^{1-\alpha} = s / (1-s)`, giving
:math:`R^{*} = \alpha (1-s)/s` exactly. The map's slope there is
:math:`s(1-\alpha) + \alpha`, which is below 1 for every :math:`s < 1` and
:math:`\alpha < 1`, so the aggregate converges from any starting capital and the
convergence needs no damping.

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

market_block = DBlock(
    name="market",
    dynamics={
        # Reads out of the household class, so this is the model's one crossing.
        "K": lambda a: a.mean(),
        "R": lambda K, alpha: alpha * K ** (alpha - 1),
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


def aiyagari_calibration(size=1000, alpha=CAPITAL_SHARE, crra=2.0, sigma=1.0):
    """An economy of *size* households.

    Parameters
    ----------
    size : int, optional
        How many households. The capital stock is an average over them, so this
        is the accuracy of the aggregate rather than part of the model.
    alpha : float, optional
        Capital's share of output.
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


def capital_map(capital, rate, alpha=CAPITAL_SHARE):
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

    Returns
    -------
    float
    """
    return rate * (capital**alpha + capital)


def stationary_capital(rate, alpha=CAPITAL_SHARE):
    """The capital per head that reproduces itself under *rate*.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.

    Returns
    -------
    float
    """
    return (rate / (1 - rate)) ** (1 / (1 - alpha))


def stationary_prices(rate, alpha=CAPITAL_SHARE):
    """The interest rate and wage at :func:`stationary_capital`.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.

    Returns
    -------
    dict
        ``R`` and ``W``.
    """
    capital = stationary_capital(rate, alpha)
    return {
        "R": alpha * (1 - rate) / rate,
        "W": (1 - alpha) * capital**alpha,
    }


def convergence_rate(rate, alpha=CAPITAL_SHARE):
    """The slope of :func:`capital_map` at its fixed point.

    Below 1 for every savings rate below 1, which is why the aggregate converges
    from any starting capital without damping.

    Parameters
    ----------
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.

    Returns
    -------
    float
    """
    return rate * (1 - alpha) + alpha
