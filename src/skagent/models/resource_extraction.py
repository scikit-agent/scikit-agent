r"""
Resource Extraction with Optimal Escapement (Reed 1979)
========================================================

Resource extraction models analyze the optimal management of renewable or depletable resources over time. These models appear in:

    Fisheries management: Determining sustainable harvest rates for fish populations
    Forestry: Optimal timber harvesting schedules
    Environmental economics: Managing renewable natural resources (water, wildlife)
    Energy: Optimal depletion of oil fields and mineral deposits
    Finance: Portfolio liquidation and asset drawdown strategies

The core problem involves balancing immediate extraction (profit now) against preserving the resource stock (profit later), accounting for natural growth dynamics and environmental uncertainty.

This implementation follows the model from:

    Reed, W.J. (1979). "Optimal escapement levels in stochastic and deterministic
    harvesting models." Journal of Environmental Economics and Management, 6(4), 350-363.

Reed showed that under multiplicative environmental shocks and stock-dependent harvesting
costs, the optimal policy has a simple "constant escapement" form: maintain a target stock
level S* and harvest any surplus above it. This optimal escapement level S* can be computed
analytically without solving the full dynamic programming problem, making it an excellent
benchmark for testing reinforcement learning algorithms.\

Mathematical Model
------------------

**State**: :math:`x_t` = resource stock at time :math:`t`

**Control**: :math:`u_t` = harvest, constrained by :math:`0 \leq u_t \leq x_t`

**Dynamics**:

.. math::
    x_{t+1} = r (x_t - u_t) \epsilon_t

where :math:`r > 1` is the growth rate, :math:`\epsilon_t` is a mean-one
log-normal shock, and :math:`(x_t - u_t)` is the escapement (remaining stock).

**Profit**:

.. math::
    \pi(u_t, x_t) = \left(p - \frac{c_0}{x_t}\right) u_t

where :math:`p` is price and :math:`c_0/x_t` is the stock-dependent unit cost.

**Objective**: Maximize :math:`\mathbb{E}\left[\sum_{t=0}^{\infty} \delta^t \pi(u_t, x_t)\right]`

Optimal Policy
--------------

Reed (1979) [Reed1979]_ proved the optimal policy has constant escapement form:

.. math::
    u_t^* = \max(0, x_t - S^*)

where the optimal escapement level is:

.. math::
    S^* = \frac{c_0 (1 - \delta)}{p (1 - \delta r)}

This requires the impatience condition :math:`\delta r < 1`.

This model provides an excellent benchmark for RL algorithms since :math:`S^*`
can be computed analytically without dynamic programming.

References
----------
.. [Reed1979] Reed, W.J. (1979). "Optimal escapement levels in stochastic and
       deterministic harvesting models." Journal of Environmental Economics
       and Management, 6(4), 350-363.
"""

import numpy as np

from skagent.distributions import MeanOneLogNormal
from skagent.block import Control, DBlock

calibration = {
    "r": 1.02,  # growth rate
    "p": 5.0,  # price per unit extracted
    "c_0": 10.0,  # cost parameter
    "DiscFac": 0.95,  # discount factor
    "sigma": 0.1,  # standard deviation of growth shock
}

resource_extraction_block = DBlock(
    **{
        "name": "resource_extraction_reed",
        "shocks": {
            # Log-normal multiplicative shock with E[epsilon] = 1
            # ln(epsilon) ~ N(-sigma^2/2, sigma^2)
            "epsilon": (MeanOneLogNormal, {"sigma": "sigma"}),
        },
        "dynamics": {
            "u": Control(
                ["x"], lower_bound=0.0, upper_bound=lambda x: x, agent="extractor"
            ),
            # Stock-dependent unit cost: harder to harvest when stock is low
            "unit_cost": lambda x, c_0: c_0 / x,
            # Net profit per unit = price - unit_cost
            "profit": lambda u, p, unit_cost: (p - unit_cost) * u,
            # Reed's dynamics: x_{t+1} = r * (x_t - u_t) * epsilon_t
            "escapement": lambda x, u: x - u,
            "x": lambda escapement, r, epsilon: r * escapement * epsilon,
        },
        "reward": {"profit": "extractor"},
    }
)


def make_optimal_decision_rule(parameters):
    r"""
    Compute the optimal constant-escapement policy from Reed (1979).

    Reed showed that for the model with multiplicative shocks and stock-dependent
    costs :math:`c(x) = c_0/x`, the optimal policy has the form:

    .. math::
        u^*(x) = \max(0, x - S^*)

    where :math:`S^*` is the optimal escapement level (target stock to maintain).
    This can be computed analytically from the first-order condition without
    solving the full dynamic programming problem.

    Parameters
    ----------
    parameters : dict
        Model parameters including:

        - ``r`` : float, growth rate
        - ``p`` : float, price per unit
        - ``c_0`` : float, cost parameter
        - ``DiscFac`` : float, discount factor

    Returns
    -------
    decision_rule : callable
        Maps stock x to optimal harvest :math:`u^*(x) = \max(0, x - S^*)`
    decision_function : callable
        Compatible with DBlock interface: ``decision_function(states, shocks, parameters)``

    Notes
    -----
    The optimal escapement :math:`S^*` satisfies the first-order condition:

    .. math::
        p - \frac{c_0}{S^*} = \delta r \left(p - \frac{c_0}{r S^*}\right)

    where :math:`\delta` is the discount factor. Rearranging gives:

    .. math::
        S^* = \frac{c_0 (1 - \delta)}{p (1 - \delta r)}

    This requires the "impatience condition" :math:`\delta r < 1`, which ensures
    the agent prefers extraction over indefinite accumulation.
    """
    r = parameters["r"]
    p = parameters["p"]
    c_0 = parameters["c_0"]
    DiscFac = parameters["DiscFac"]

    # Check impatience condition
    if DiscFac * r >= 1:
        raise ValueError(
            f"Impatience condition violated: DiscFac * r = {DiscFac * r:.4f} >= 1. "
            "The model requires DiscFac * r < 1 for an interior solution. "
            "When DiscFac * r >= 1, the optimal policy is to never extract."
        )

    # Compute optimal escapement from first-order condition
    # Derivation: p - c_0/S = DiscFac * r * (p - c_0/(r*S))
    # Solving for S gives:
    numerator = c_0 * (1 - DiscFac)
    denominator = p * (1 - DiscFac * r)
    S_star = numerator / denominator

    def decision_rule(x):
        r"""
        Optimal harvest under constant escapement policy.

        Parameters
        ----------
        x : float or array_like
            Current stock level(s)

        Returns
        -------
        u : float or ndarray
            Optimal harvest :math:`u = \max(0, x - S^*)`
        """
        return np.maximum(0, x - S_star)

    def decision_function(states, shocks, parameters):
        """
        Decision function compatible with DBlock interface.

        Parameters
        ----------
        states : dict
            Contains ``'x'`` (current stock)
        shocks : dict
            Random shocks (not used in optimal policy)
        parameters : dict
            Model parameters (not used after S* is computed)

        Returns
        -------
        u : float
            Optimal harvest
        """
        x = states["x"]
        return decision_rule(x)

    return decision_rule, decision_function


# Create the optimal decision rule with calibrated parameters
dr_u, df_u = make_optimal_decision_rule(calibration)
