r"""
Krusell and Smith (1998) [KrusellSmith1998]_, in the formulation of Maliar,
Maliar and Winant (2021) [MMW2021]_: an Aiyagari economy whose productivity
moves.

Many households each save out of labour income, and the average of what they
save is the economy's capital. Capital and aggregate productivity together set
the interest rate and the wage. Productivity follows its own random path, so the
prices move even when the households' distribution does not, and a household
deciding how much to save has to forecast both.

Symbols and parameters
----------------------

One value per household:

======  ===================================================================
y       log labour productivity
eps_y   this period's innovation to it
theta   the labour endowment, ``exp(y)`` scaled so the households average one
a       assets carried in from last period
w       cash on hand: labour income plus assets and the interest on them
c       consumption, the household's one decision
u       the household's payoff from consuming that much
======  ===================================================================

One value for the whole economy, and capitalised for it:

======  ===================================================================
Z       log aggregate productivity
eps_Z   this period's innovation to it
K       capital per head, the average of ``a`` over the households
Theta   the average of ``exp(y)`` over the households, which scales it
R       the net interest rate, so a household earns ``(1 + R)`` on assets
W       the wage paid per unit of labour endowment
======  ===================================================================

Calibration parameters:

=========  ================================================================
alpha      capital's share of output
delta      the fraction of capital that wears out each period
CRRA       relative risk aversion in the household's utility
DiscFac    the household's discount factor
rho_Z      the persistence of log aggregate productivity
sigma_Z    the standard deviation of its innovation
rho_y      the persistence of log labour productivity
sigma_y    the standard deviation of its innovation
household  how many households there are
=========  ================================================================

The period's equations
----------------------

.. math::
    Z = \rho_Z Z_{-1} + \epsilon_Z, \qquad
    y_i = \rho_y y_{i,-1} + \epsilon_{y,i}

.. math::
    K = \frac{1}{N}\sum_j a_j, \qquad \Theta = \frac{1}{N}\sum_j e^{y_j},
    \qquad R = e^{Z} \alpha K^{\alpha - 1} - \delta,
    \qquad W = e^{Z} (1 - \alpha) K^{\alpha}

.. math::
    \theta_i = e^{y_i} / \Theta, \qquad
    w_i = \theta_i W + (1 + R) a_i, \qquad
    u_i = \frac{c_i^{1-\gamma}}{1-\gamma} \ (\log c_i \text{ at } \gamma = 1),
    \qquad a_i' = w_i - c_i

The prices are the marginal products of the production function
:math:`Y = e^{Z} K^{\alpha} L^{1-\alpha}`. Each period every household's
endowment is divided by the class's average, so labour supply per head is
exactly one unit and :math:`L` drops out of both prices.

Each period opens with the innovations: aggregate productivity and every
household's productivity move first, then the market prices the capital the
households arrived with, and then each household decides. So :math:`a`,
:math:`y` and :math:`Z` are the arrival states, and every random draw that
bears on this period's decision is made in this period. The household decides
on its own cash on hand and productivity, on aggregate productivity, and on
capital per head, the mean of the distribution standing in for the whole of it.

The model is built from four blocks. :data:`productivity_block` and
:data:`income_block` draw the period's innovations, :data:`market_block`
computes the crossings and the prices, and :data:`household_block` holds the
budget, the decision and the payoff. The income and household blocks are each
inside the household class, and the market sits between them, so the class
appears twice in :data:`krusell_smith_block`.

The symbols follow [MMW2021]_ where the library's conventions allow. Their
aggregate productivity :math:`z` is ``Z`` here, since the library capitalises a
variable that stands for the whole economy. Their gross return :math:`R` is
``1 + R`` here, matching :mod:`skagent.models.aiyagari`.

Under a fixed savings rate
--------------------------

Under a fixed savings rate the aggregate has a closed form, as it has in the
Aiyagari economy, and that makes the model a test before anything is solved. If
every household consumes :math:`(1-s)w`, then averaging :math:`a' = s w` over a
class whose endowments average exactly one gives

.. math::
    K' = s\,(e^{Z} K^{\alpha} + (1 - \delta) K)

in every period of a simulated economy, whatever the draws. :func:`capital_map`
is that map, given the period's productivity.

References
----------
.. [KrusellSmith1998] Krusell, P. and Smith, A.A. (1998). "Income and Wealth
       Heterogeneity in the Macroeconomy." *Journal of Political Economy*,
       106(5), 867-896. https://doi.org/10.1086/250034
.. [MMW2021] Maliar, L., Maliar, S. and Winant, P. (2021). "Deep learning for
       solving dynamic economic models." *Journal of Monetary Economics*, 122,
       76-101. https://doi.org/10.1016/j.jmoneco.2021.07.004
"""

import numpy as np

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Normal

CAPITAL_SHARE = 0.36
"""Capital's share of output, the exponent on capital in the production function."""

DEPRECIATION = 0.08
"""The fraction of the capital stock that wears out each period."""

productivity_block = DBlock(
    name="productivity",
    shocks={"eps_Z": (Normal, {"sigma": "sigma_Z"})},
    dynamics={"Z": lambda Z, eps_Z, rho_Z: rho_Z * Z + eps_Z},
)
"""Log aggregate productivity's AR(1) step, drawn once for the economy."""

income_block = DBlock(
    name="income",
    shocks={"eps_y": (Normal, {"sigma": "sigma_y"})},
    dynamics={"y": lambda y, eps_y, rho_y: rho_y * y + eps_y},
)
"""Log labour productivity's AR(1) step, drawn once for each household."""

market_block = DBlock(
    name="market",
    dynamics={
        # Both read out of the household class, so these are the model's crossings.
        "K": lambda a: a.mean(),
        "Theta": lambda y: np.exp(y).mean(),
        "R": lambda Z, K, alpha, delta: np.exp(Z) * alpha * K ** (alpha - 1) - delta,
        "W": lambda Z, K, alpha: np.exp(Z) * (1 - alpha) * K**alpha,
    },
)
"""Capital and the endowment's scale as averages over the households, and the prices."""

household_block = DBlock(
    name="household",
    dynamics={
        "theta": lambda y, Theta: np.exp(y) / Theta,
        "w": lambda theta, W, R, a: theta * W + (1 + R) * a,
        "c": Control(
            ["w", "y", "K", "Z"],
            lower_bound=0.0,
            upper_bound=lambda w: w,
            agent="household",
        ),
        "u": lambda c, CRRA: np.log(c) if CRRA == 1 else c ** (1 - CRRA) / (1 - CRRA),
        # End-of-period assets, and next period's arrival value for the same
        # symbol. A tick block would only rename it, so there is none.
        "a": lambda w, c: w - c,
    },
    reward={"u": "household"},
)
"""The household's budget, decision and payoff."""

_households = Entity("household")

# Productivity moves, the market prices what the households arrived with, and
# the households decide. Declaration order is what makes ``a``, ``y`` and ``Z``
# arrival states, so there is no timing annotation anywhere in the model.
krusell_smith_block = RBlock(
    name="krusell_smith",
    blocks=[
        productivity_block,
        RBlock(name="incomes", entity=_households, blocks=[income_block]),
        market_block,
        RBlock(name="households", entity=_households, blocks=[household_block]),
    ],
)
"""The whole economy: the four blocks in the period's order."""


def krusell_smith_calibration(
    size=1000,
    alpha=CAPITAL_SHARE,
    delta=DEPRECIATION,
    crra=1.0,
    beta=0.96,
    rho_z=0.95,
    sigma_z=0.01,
    rho_y=0.9,
    sigma_y=0.2 * np.sqrt(1 - 0.9**2),
):
    """An economy of *size* households.

    The defaults are the values stated in [MMW2021]_'s text where it states
    them, and those of their reference code for ``alpha`` and ``delta``, which
    the text does not state.

    Parameters
    ----------
    size : int, optional
        How many households. Capital per head is an average over them.
    alpha : float, optional
        Capital's share of output.
    delta : float, optional
        The fraction of the capital stock that wears out each period.
    crra : float, optional
        Relative risk aversion in the household's utility, with 1 meaning log
        utility.
    beta : float, optional
        The household's discount factor.
    rho_z : float, optional
        The persistence of log aggregate productivity.
    sigma_z : float, optional
        The standard deviation of its innovation.
    rho_y : float, optional
        The persistence of log labour productivity.
    sigma_y : float, optional
        The standard deviation of its innovation. The endowment averages one
        over the households whatever this is.

    Returns
    -------
    dict
    """
    return {
        "household": size,
        "alpha": alpha,
        "delta": delta,
        "CRRA": crra,
        "DiscFac": beta,
        "rho_Z": rho_z,
        "sigma_Z": sigma_z,
        "rho_y": rho_y,
        "sigma_y": sigma_y,
    }


def savings_rule(rate):
    """Consume all but *rate* of cash on hand, as a decision rule for ``c``.

    Parameters
    ----------
    rate : float
        The fraction saved, in ``(0, 1)``.

    Returns
    -------
    callable
        A rule over the decision's information set that reads cash on hand
        alone.
    """
    return lambda w, y, K, Z: (1 - rate) * w


def capital_map(capital, productivity, rate, alpha=CAPITAL_SHARE, delta=DEPRECIATION):
    """The capital an economy leaves for next period under a fixed savings rate.

    Parameters
    ----------
    capital : float
        Capital per head this period.
    productivity : float
        Log aggregate productivity this period, ``Z``.
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
    return rate * (np.exp(productivity) * capital**alpha + (1 - delta) * capital)
