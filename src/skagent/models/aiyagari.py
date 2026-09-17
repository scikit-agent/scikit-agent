r"""
Aiyagari (1994) heterogeneous agents: a cross-section that prices its own capital.

Many households each save out of labour income, and what they save between them
is the economy's capital stock. Capital sets the interest rate and the wage,
those set what each household has to spend, and what each household leaves over
is next period's capital. The households never interact directly; they meet only
through an average.

Notation follows the module's own symbols. A household draws a labour endowment
:math:`\theta`, mean one, and arrives holding assets :math:`a`. The capital per
head is the average of those assets, and it is the only quantity that leaves the
household class:

.. math::
    k = \frac{1}{N}\sum_j a_j, \qquad r = \alpha k^{\alpha - 1}, \qquad
    w = (1 - \alpha) k^{\alpha}

.. math::
    z_i = \theta_i w + (1 + r) a_i, \qquad
    u_i = \frac{c_i^{1-\gamma}}{1-\gamma}, \qquad a_i' = z_i - c_i

The market block is declared before the households, so :math:`k` is computed
from the assets the households arrived with rather than the ones they are about
to choose. That makes :math:`a` an arrival state, and simulating :math:`T`
periods runs :math:`T` rounds of the aggregate's own law of motion.

The design note for this model writes the labour endowment :math:`l`. It is
``theta`` here, which is what the rest of the library calls a mean-one
transitory shock, and a bare ``l`` is not a legal identifier under the project's
linter.

What the model is for
----------------------

Under a fixed savings rate the aggregate has a closed form, and that is what
makes this model a test rather than only a demonstration. If every household
consumes :math:`(1-s)z`, then :math:`a' = s z`, and averaging over the class
with :math:`E[\theta] = 1`:

.. math::
    k' = s\,(w + (1 + r) k) = s\,(k^{\alpha} + k)

so the stationary capital solves :math:`k^{1-\alpha} = s / (1-s)`, giving
:math:`r^{*} = \alpha (1-s)/s` exactly. The map's slope there is
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
        "k": lambda a: a.mean(),
        "r": lambda k, alpha: alpha * k ** (alpha - 1),
        "w": lambda k, alpha: (1 - alpha) * k**alpha,
    },
)

household_block = DBlock(
    name="household",
    shocks={"theta": (MeanOneLogNormal, {"sigma": "sigma_theta"})},
    dynamics={
        "z": lambda theta, w, r, a: theta * w + (1 + r) * a,
        "c": Control(
            ["z"], lower_bound=0.0, upper_bound=lambda z: z, agent="household"
        ),
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
    },
    reward={"u": "household"},
)

tick_block = DBlock(name="tick", dynamics={"a": lambda z, c: z - c})

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
            blocks=[household_block, tick_block],
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


def capital_map(k, rate, alpha=CAPITAL_SHARE):
    """The capital an economy holding *k* leaves for the next period.

    Iterated, this is the path :data:`aiyagari_block` simulates under
    :func:`savings_rule`; its fixed point is :func:`stationary_capital`.

    Parameters
    ----------
    k : float
        Capital per head this period.
    rate : float
        The savings rate the households follow.
    alpha : float, optional
        Capital's share of output.

    Returns
    -------
    float
    """
    return rate * (k**alpha + k)


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
        ``r`` and ``w``.
    """
    capital = stationary_capital(rate, alpha)
    return {
        "r": alpha * (1 - rate) / rate,
        "w": (1 - alpha) * capital**alpha,
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
