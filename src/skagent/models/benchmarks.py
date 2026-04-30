#!/usr/bin/env python3
"""
Analytically Solvable Consumption-Savings Models

This module implements a collection of discrete-time consumption-savings
dynamic programming problems for which the literature has succeeded in writing down
true closed-form policies. These represent well-known benchmark problems from
the economic literature with established analytical solutions.

THEORETICAL FOUNDATION
======================

An entry qualifies for inclusion ONLY if:
(i) The problem is a bona-fide dynamic programming problem
(ii) The optimal c_t (and any other control) can be written in closed form
with no recursive objects left implicit

Standard Timing Convention (Adopted Throughout)
-----------------------------------------------
t ∈ {0,1,2,...}     : period index
A_{t-1}             : beginning-of-period assets (arrival state, before interest)
y_t                 : non-capital income (realized in period t)
R                   : gross return on assets (R = 1 + r > 1)
m_t = R*A_{t-1} + y_t : cash-on-hand (market resources available for consumption)
c_t                 : consumption (control variable)
A_t = m_t - c_t     : end-of-period assets (state for next period)
H_t = E_t[∑_{s=1}^∞ R^{-s} y_{t+s}] : human wealth (present value of future income)
W_t = m_t + H_t     : total wealth (cash-on-hand plus human wealth)
u(c)                : period utility function
β                   : discount factor
TVC                 : lim_{T→∞} E_0[β^T u'(c_T) A_T] = 0 (transversality condition)
"""

from skagent.distributions import Normal, MeanOneLogNormal, Bernoulli
from skagent.block import Control, DBlock
import logging
import torch
from torch import as_tensor
import numpy as np
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


# UTILITY FUNCTIONS
# =================


def crra_utility(c, gamma):
    """
    CRRA utility: u(c) = c^(1-gamma)/(1-gamma) for gamma != 1, log(c) for gamma == 1
    """
    c_tensor = as_tensor(c)
    if abs(gamma - 1.0) < 1e-10:
        return torch.log(c_tensor)
    return c_tensor ** (1 - gamma) / (1 - gamma)


def _human_wealth_rate(R):
    """Return r = R - 1, raising ValueError if R <= 1.

    Used by analytical policies that compute human wealth as a present-value
    annuity y/r; R > 1 keeps that annuity well defined.
    """
    r = R - 1
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
    return r


# NUMERICAL TOLERANCE CONSTANTS
# =============================
EPS_STATIC = 1e-10  # Static identity verification (deterministic)
EPS_EULER = 1e-8  # Euler equation residuals (stochastic)
EPS_BUDGET = 1e-12  # Budget evolution (should be exact)
EPS_VALIDATION = 1e-8  # General validation tolerance


# DETERMINISTIC (PERFECT-FORESIGHT) BENCHMARKS
# ============================================

# D-1: Finite Horizon Log Utility
# -------------------------------
# T-period finite horizon log utility using backward induction. The key insight is that
# with log utility, the consumption rule has a simple closed form even with finite
# horizon effects.
#
# Mathematical formulation:
#   V_T(W_T) = ln W_T  (terminal condition)
#   V_t(W_t) = max_c ln c + β V_{t+1}((W_t - c)R)  for t < T
#
# The value function takes the form V_t(W_t) = A_t + ln W_t, leading to:
#   c_t = (1-β)/(1-β^{T-t}) * W_t  (remaining horizon consumption rule)
#
# This shows how finite horizon creates time-varying consumption rates that
# approach the infinite-horizon limit as T → ∞.

d1_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "T": 5,  # Finite horizon
    "W0": 2.0,  # Initial wealth
    # Note: t=0 is the initial STATE, not a parameter, so it's passed in initial_states
    "description": "D-1: Finite horizon log utility",
}

d1_block = DBlock(
    **{
        "name": "d1_finite_log",
        "shocks": {},
        "dynamics": {
            "c": Control(["W", "t"], upper_bound=lambda W, t: W, agent="consumer"),
            "u": lambda c, t, T: (as_tensor(t) < as_tensor(T)).float()
            * crra_utility(c, 1.0),  # Utility cutoff at T (life ends)
            "W": lambda W, c, R: (W - c) * R,  # Next period wealth
            "t": lambda t: t + 1,  # Time counter
        },
        "reward": {"u": "consumer"},
    }
)


def d1_analytical_policy(states, shocks, parameters):
    r"""
    Optimal policy for D-1: finite-horizon log-utility consumption.

    With log utility and a deterministic :math:`T`-period horizon, the agent
    solves

    .. math::
        V_T(W_T) = \log W_T,
        \qquad
        V_t(W_t) = \max_{c_t \in (0,\, W_t]} \, \log c_t
        + \beta \, V_{t+1}\bigl((W_t - c_t)\, R\bigr),

    where :math:`W_t` is wealth at the start of period :math:`t`, :math:`R`
    is the gross return, and :math:`\beta < 1` is the discount factor. The
    value function takes the form :math:`V_t(W) = \alpha_t + \log W` for a
    time-varying additive constant :math:`\alpha_t`, and the first-order
    condition gives the remaining-horizon rule

    .. math::
        c_t \;=\; \frac{1 - \beta}{\,1 - \beta^{T - t}\,} \, W_t.

    In the terminal period (:math:`T - t = 1`) the formula simplifies to
    :math:`c_t = W_t` since :math:`(1-\beta)/(1-\beta) = 1`; the
    implementation handles this case directly to avoid the :math:`0/0`
    form that would arise once :math:`T - t = 0`. As :math:`T - t \to
    \infty`, the rule converges to the infinite-horizon constant-MPC
    policy :math:`c_t = (1 - \beta) W_t`, the :math:`\sigma = 1` special
    case of D-2.

    Parameters
    ----------
    states : dict
        Arrival states. Must contain ``"W"`` (wealth) and ``"t"`` (time
        index, defaults to ``0``).
    shocks : dict
        Unused; the model is deterministic.
    parameters : dict
        Must contain ``"DiscFac"`` (:math:`\beta`) and ``"T"`` (horizon).

    Returns
    -------
    dict
        ``{"c": c_optimal}`` whose dtype matches the input wealth.

    Raises
    ------
    ValueError
        If :math:`\beta \geq 1`, since the closed form is undefined.
    """
    beta = parameters["DiscFac"]
    if beta >= 1.0:
        raise ValueError(
            f"Discount factor must satisfy β < 1 for finite horizon model, got β={beta}"
        )
    T = parameters["T"]
    W = states["W"]
    t = states.get("t", 0)

    # Pick an output dtype: use float64 for scalars (Python's float64 is the
    # most precise we can produce), and otherwise preserve the input tensor's
    # dtype.
    dtype = torch.float64 if not isinstance(W, torch.Tensor) else W.dtype

    # Stay in tensor space so scalar Python ``t = T`` does not raise a
    # ZeroDivisionError before the terminal branch can be masked. Clamping
    # the horizon at 1 collapses the denominator at ``T - t <= 1`` to
    # ``1 - beta``; the formula then evaluates to exactly ``W`` at the
    # boundary, matching the consume-everything terminal rule without
    # introducing a 0/0 form (or a masked NaN that would propagate through
    # autograd).
    #
    # The intermediate computation runs at float64 to match the precision of
    # the original Python-float path; without an explicit dtype here,
    # ``as_tensor`` would round Python scalars to float32 and silently lose
    # several digits of accuracy (visible at ``places=10`` in the test
    # suite). Integer ``safe_horizon`` keeps ``beta ** safe_horizon`` on the
    # fast integer-exponentiation path.
    W_tensor = as_tensor(W, dtype=torch.float64)
    safe_horizon = torch.clamp(as_tensor(T - t), min=1)
    denominator = 1 - torch.tensor(beta, dtype=torch.float64) ** safe_horizon
    c_optimal = (1 - beta) / denominator * W_tensor

    return {"c": as_tensor(c_optimal, dtype=dtype)}


# D-2: Infinite Horizon CRRA (Perfect Foresight)
# ----------------------------------------------
# The canonical perfect-foresight consumption model. With CRRA utility and
# constant income, the consumption function is linear in cash-on-hand with
# constant marginal propensity to consume κ.
#
# Mathematical formulation:
#   max E₀ ∑_{t=0}^∞ β^t [c_t^{1-σ}/(1-σ)]
#   s.t. A_t = R*A_{t-1} + y_t - c_t   (equivalently m_t = R*A_{t-1} + y_t, A_t = m_t - c_t)
#       lim_{T→∞} E₀[β^T u'(c_T) A_T] = 0  (TVC)
#
# Key condition: Return-Impatience (βR)^{1/σ} < R
# This ensures finite value function and non-explosive consumption.
#
# Closed-form solution:
#   c_t = κ*W_t  where κ = (R - (βR)^{1/σ})/R
#   W_t = m_t + H_t  (total wealth = cash-on-hand + human wealth)
#   H_t = y/r  (human wealth for constant income y)
#   m_t = A_t*R + y_t  (cash-on-hand)
#
# The MPC κ is constant and depends only on deep parameters, not state variables.

d2_calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,
    "y": 1.0,
    "description": "D-2: Infinite horizon CRRA perfect foresight",
}

d2_block = DBlock(
    **{
        "name": "d2_infinite_crra",
        "shocks": {},
        "dynamics": {
            "m": lambda a, R, y: a * R + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: crra_utility(c, CRRA),
        },
        "reward": {"u": "consumer"},
    }
)


def d2_analytical_policy(states, shocks, parameters):
    r"""
    Optimal policy for D-2: infinite-horizon CRRA perfect foresight.

    The canonical perfect-foresight consumption problem solves

    .. math::
        \max_{\{c_t\}} \, \sum_{t=0}^{\infty} \beta^t \,
        \frac{c_t^{\,1-\sigma}}{1 - \sigma}
        \quad \text{s.t.} \quad A_t = R\, A_{t-1} + y - c_t,

    with constant labor income :math:`y > 0`, gross return :math:`R > 1`,
    and the transversality condition
    :math:`\lim_{T\to\infty} \beta^T \, u'(c_T)\, A_T = 0`. Equivalently,
    cash-on-hand :math:`m_t = R\, A_{t-1} + y` evolves with end-of-period
    assets :math:`A_t = m_t - c_t`. Under return-impatience
    :math:`(\beta R)^{1/\sigma} < R`, the closed-form policy is linear in
    total wealth,

    .. math::
        c_t \;=\; \kappa \, W_t,
        \qquad
        \kappa \;=\; \frac{R - (\beta R)^{1/\sigma}}{R},
        \qquad
        W_t \;=\; m_t + H,

    where :math:`m_t = R\, A_{t-1} + y` is cash-on-hand at time :math:`t`,
    :math:`H = y / r` is human wealth (the present value of the constant
    future income stream), and :math:`r = R - 1`. The marginal propensity
    to consume :math:`\kappa` depends only on deep parameters, not on the
    state.

    Parameters
    ----------
    states : dict
        Must contain ``"a"`` (arrival assets).
    shocks : dict
        Unused.
    parameters : dict
        Must contain ``"DiscFac"`` (:math:`\beta`), ``"R"``, ``"CRRA"``
        (:math:`\sigma`), and ``"y"``.

    Returns
    -------
    dict
        ``{"c": c_optimal}``.

    Raises
    ------
    ValueError
        If return-impatience is violated, or if :math:`R \leq 1`.

    See Also
    --------
    d3_analytical_policy : Same model with i.i.d. survival risk.
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]
    sigma = parameters["CRRA"]
    y = parameters["y"]

    # Extract arrival state (assets from previous period)
    a = states["a"]

    # Compute MPC
    growth_factor = (beta * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError(
            f"Return-impatience violated: (βR)^(1/σ) = {growth_factor:.6f} >= R = {R}"
        )

    kappa = (R - growth_factor) / R

    # Compute information set: market resources = assets * return + income
    m = a * R + y

    # Compute human wealth: present value of constant income stream
    r = _human_wealth_rate(R)
    human_wealth = y / r

    # Optimal consumption: c = κ * (market resources + human wealth)
    c_optimal = kappa * (m + human_wealth)
    return {"c": c_optimal}


# D-3: Blanchard Discrete-Time Mortality
# ---------------------------------------
# Extension of D-2 with i.i.d. survival risk. Each period, the agent survives
# with probability s ∈ (0,1). This scales down effective patience but preserves
# the linear consumption function structure.
#
# Mathematical formulation:
#   max E₀ ∑_{t=0}^∞ (sβ)^t [c_t^{1-σ}/(1-σ)]
#   s.t. same budget constraint as D-2
#
# Key insight: Mortality risk is equivalent to reducing the discount factor
# from β to sβ. The consumption rule remains linear in cash-on-hand.
#
# Condition: sβR < 1 (mortality-adjusted return-impatience)
#
# Closed-form solution:
#   c_t = κ_s*W_t  where κ_s = (R - (sβR)^{1/σ})/R
#   W_t = m_t + H_t  (total wealth = cash-on-hand + human wealth)
#   H_t = y/r  (human wealth for constant income y)
#
# Note: κ_s > κ (higher MPC) because mortality makes the agent less patient.

d3_calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,
    "y": 1.0,
    "SurvivalProb": 0.99,  # Survival probability s ∈ (0,1)
    # Note: liv=1.0 is the initial STATE, not a parameter, so it's passed in initial_states
    "description": "D-3: Blanchard discrete-time mortality",
}

d3_block = DBlock(
    **{
        "name": "d3_blanchard_mortality",
        "shocks": {
            "live": (Bernoulli, {"p": "SurvivalProb"}),  # Survival shock each period
        },
        "dynamics": {
            "m": lambda a, R, y: a * R + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": lambda m, c: m - c,
            "liv": lambda liv, live: liv * live,  # liv becomes 0 if agent dies (live=0)
            "u": lambda c, liv, CRRA: liv
            * crra_utility(c, CRRA),  # Utility with survival
        },
        "reward": {"u": "consumer"},
    }
)


def d3_analytical_policy(states, shocks, parameters):
    r"""
    Optimal policy for D-3: Blanchard (1985) discrete-time mortality.

    Extends D-2 by giving the agent i.i.d. survival probability
    :math:`s \in (0, 1)` per period. The objective becomes

    .. math::
        \max \, \sum_{t=0}^{\infty} (s\beta)^t \,
        \frac{c_t^{\,1-\sigma}}{1 - \sigma},

    with the same budget constraint as D-2. Mortality is observationally
    equivalent to scaling the discount factor from :math:`\beta` to
    :math:`s\beta`, so the linear consumption rule survives:

    .. math::
        c_t \;=\; \kappa_s \, (m_t + H),
        \qquad
        \kappa_s \;=\; \frac{R - (s\beta R)^{1/\sigma}}{R},

    with :math:`H = y / r` as before. The MPC :math:`\kappa_s > \kappa`
    because mortality erodes effective patience: the agent consumes a
    larger share of wealth each period.

    Parameters
    ----------
    states : dict
        Must contain ``"a"`` (arrival assets). The ``"liv"`` alive
        indicator is part of the DBlock simulator dynamics but is not read
        by the analytical policy.
    shocks : dict
        May contain ``"live"`` (Bernoulli survival shock); unused for the
        analytical policy itself.
    parameters : dict
        Must contain ``"DiscFac"``, ``"R"``, ``"CRRA"``, ``"y"``, and
        ``"SurvivalProb"`` (:math:`s`).

    Returns
    -------
    dict
        ``{"c": c_optimal}``.

    Raises
    ------
    ValueError
        If mortality-adjusted return-impatience is violated, or if
        :math:`R \leq 1`.

    References
    ----------
    Blanchard, O.J. (1985). "Debt, deficits, and finite horizons."
    *Journal of Political Economy*, 93(2), 223-247.

    See Also
    --------
    d2_analytical_policy : Underlying perfect-foresight model without
        mortality.
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]
    sigma = parameters["CRRA"]
    s = parameters["SurvivalProb"]
    y = parameters["y"]

    # Extract arrival state (assets from previous period)
    a = states["a"]

    # Effective discount factor with mortality
    beta_eff = s * beta

    # Compute MPC with mortality
    growth_factor = (beta_eff * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError(
            f"Return-impatience violated: (sβR)^(1/σ) = {growth_factor:.6f} >= R = {R}"
        )

    kappa_s = (R - growth_factor) / R

    # Compute information set: market resources = assets * return + income
    m = a * R + y

    # Compute human wealth: present value of constant income stream
    r = _human_wealth_rate(R)
    human_wealth = y / r

    # Optimal consumption: c = κ_s * (market resources + human wealth)
    c_optimal = kappa_s * (m + human_wealth)
    return {"c": c_optimal}


# Remarks: D-2 is the canonical perfect-foresight workhorse; D-3 shows that adding
# i.i.d. survival risk does NOT break tractability; mortality just scales down patience.

# STOCHASTIC MODELS WITH CLOSED FORM
# ==================================

# U-1: Hall (1978) Random Walk
# ----------------------------
# The seminal result in consumption theory: with quadratic utility and βR = 1,
# consumption follows a martingale regardless of income process complexity.

# Mathematical foundation:
#   u(c) = ac - (b/2)c²  ⟹  u'(c) = a - bc
#   Euler equation: u'(c_t) = βR E_t[u'(c_{t+1})]
#   With βR = 1: a - bc_t = E_t[a - bc_{t+1}] = a - b E_t[c_{t+1}]
#   Therefore: c_t = E_t[c_{t+1}]  (martingale property)

# This implies: c_{t+1} = c_t + ε_{t+1} where E_t[ε_{t+1}] = 0

# Key insight: Quadratic utility makes marginal utility affine, so when the
# stochastic discount factor is neutral (βR = 1), the Euler equation collapses
# to a simple martingale condition.

# Restriction: βR = 1 exactly (neutral SDF)
# Income process: Any y_t with E_t[y_{t+1}] finite

u1_calibration = {
    "DiscFac": 0.970873786,  # Exact β*R = 1
    "R": 1.03,
    "quad_a": 1.0,
    "quad_b": 0.5,
    "y_mean": 1.0,
    "income_std": 0.1,
    "c_init": 1.0,  # Initial consumption level
    "description": "U-1: Hall random walk consumption",
}

u1_block = DBlock(
    **{
        "name": "u1_hall_random_walk",
        "shocks": {
            "eta": (Normal, {"mean": 0.0, "std": "income_std"}),
        },
        "dynamics": {
            "y": lambda y_mean, eta: y_mean + eta,  # i.i.d. income
            "m": lambda A, R, y: A * R + y,  # Cash-on-hand
            "c": Control(
                ["m"],
                upper_bound=lambda m: m,
                agent="consumer",
            ),
            "A": lambda m, c: m - c,  # End-of-period assets
            "u": lambda c, quad_a, quad_b: quad_a * c
            - quad_b * c**2 / 2,  # Quadratic utility
        },
        "reward": {"u": "consumer"},
    }
)


def u1_analytical_policy(states, shocks, parameters):
    r"""
    Optimal policy for U-1: Hall (1978) random-walk consumption.

    With quadratic utility :math:`u(c) = ac - bc^2/2` and the neutral
    stochastic discount factor :math:`\beta R = 1`, the Euler equation
    collapses to the martingale property

    .. math::
        \mathbb{E}_t[c_{t+1}] = c_t,

    so consumption follows a random walk regardless of the income process.
    Hall's contribution was to derive this implication and confront it
    with consumption data.

    The decision rule consistent with this Euler equation, plus
    transversality, is the Permanent Income Hypothesis: consume the
    annuity value of total wealth,

    .. math::
        c_t \;=\; \frac{r}{R} \, (m_t + H),

    where :math:`m_t = R \, A_{t-1} + y_t` is cash-on-hand,
    :math:`H = \mathbb{E}_t y / r` is the present value of the expected
    future income stream, and :math:`r = R - 1`. The martingale property
    is a *consequence* of this PIH policy, not the policy itself.

    Parameters
    ----------
    states : dict
        Must contain ``"A"`` (arrival assets) and ``"y"`` (current income
        realization).
    shocks : dict
        May contain ``"eta"`` (mean-zero income innovation); unused once
        :math:`y_t` is known.
    parameters : dict
        Must contain ``"DiscFac"``, ``"R"``, and ``"y_mean"``.

    Returns
    -------
    dict
        ``{"c": c_optimal}``.

    Notes
    -----
    Logs a warning via the module logger if :math:`|\beta R - 1| > 10^{-6}`,
    since the PIH derivation hinges on :math:`\beta R = 1` exactly. With
    high income variance, transversality may also fail.

    References
    ----------
    Hall, R.E. (1978). "Stochastic implications of the life cycle-permanent
    income hypothesis: Theory and evidence." *Journal of Political
    Economy*, 86(6), 971-987.
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]
    y_mean = parameters["y_mean"]
    income_std = parameters.get("income_std", 0.0)

    if abs(beta * R - 1.0) > 1e-6:
        logger.warning("β*R = %.6f ≠ 1; PIH conditions may not hold exactly", beta * R)

    if income_std > 0.5:
        logger.warning("high income variance (%g); verify TVC is satisfied", income_std)

    # Extract arrival states
    A = states["A"]  # Financial assets (arrival state)
    y_current = states["y"]  # Current income realization (from y_mean + eta)

    # Construct information set from arrival states
    m = A * R + y_current  # Information set: cash-on-hand/market resources

    # PIH solution with βR=1: consume annuity value of TOTAL wealth
    r = _human_wealth_rate(R)
    human_wealth = y_mean / r
    total_wealth = m + human_wealth

    # PIH MPC out of total wealth is r/R
    mpc = r / R
    c_optimal = mpc * total_wealth

    return {"c": c_optimal}


# U-2: Log Utility with Permanent Income Shocks (Normalized)
# ------------------------------------------------------------
# Uses NORMALIZED variables for proper handling of permanent income.
# All variables are expressed as ratios to permanent income P:
#   m = M/P  (normalized cash-on-hand)
#   c = C/P  (normalized consumption)
#   a = A/P  (normalized assets)
#
# This normalization is essential because:
# 1. The PIH solution depends only on normalized m, not on m and P separately
# 2. The network learns a 1D function c(m) instead of 2D c(m,P)
# 3. This is the standard formulation in the buffer stock literature (Carroll, HARK)
#
# U-2 is UNCONSTRAINED: agent can borrow against human wealth h = 1/r.
# At m = 0, the analytical solution is c = (1-β)/r ≈ 1.33 (borrowing against h).
# The upper bound is tightened to prevent Ponzi scheme solutions that satisfy
# the Euler equation but violate transversality. The analytical solution
# c = (1-β)(m+h) ≈ 0.04*m + 1.33 is well within the bound 0.1*m + 2.
# For the TRUE buffer stock model with BINDING borrowing constraint c ≤ m, see U-3.
#
# When σ_ψ = 0, the permanent shock ψ ≡ 1 and the model becomes deterministic.
# This is the default calibration, making the PIH analytical solution exact.

# Mathematical foundation (NORMALIZED):
#   u(c) = ln c  (log utility of normalized consumption)
#   Transition: a = m - c, then m' = R*a/ψ + θ
#   - ψ is the permanent income shock (= 1 when σ_ψ = 0)
#   - θ is normalized transitory income (here θ ≡ 1, i.e., E[θ] = 1)
#   Human wealth (normalized): h = 1/r  (constant when income is deterministic)

# Closed-form solution (NORMALIZED):
#   c = (1-β)(m + 1/r)

u2_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "sigma_psi": 0.0,  # No permanent shocks - PIH solution is exact when σ_ψ=0
    "description": "U-2: Log utility (normalized), no borrowing constraint, σ_ψ=0",
}

u2_block = DBlock(
    **{
        "name": "u2_log_permanent_normalized",
        "shocks": {
            "psi": (MeanOneLogNormal, {"sigma": "sigma_psi"}),  # Permanent income shock
        },
        "dynamics": {
            # Normalized cash-on-hand: m = R*a/ψ + θ where θ ≡ 1 (normalized income)
            # The "+1" represents E[θ] = 1, the mean of normalized transitory income.
            # In full buffer stock models (U-3), θ would be a shock; here it's deterministic.
            # Note: psi is strictly positive from MeanOneLogNormal, but we clamp for safety.
            "m": lambda a, R, psi: R * a / torch.clamp(psi, min=1e-8) + 1,
            # Normalized consumption - network sees only m
            # U-2 is UNCONSTRAINED PIH: agent can borrow against human wealth h = 1/r.
            # At m = 0, the analytical solution is c = (1-β)/r ≈ 1.33.
            # The analytical c ≈ (1-β)(m+h) ≈ 0.04*m + 1.33, which is always < 0.1*m + 2
            # for m in the training range. The bound is loose enough for the analytical
            # solution but prevents Ponzi schemes (over-borrowing that violates transversality).
            "c": Control(
                ["m"],  # Control depends ONLY on m (network input)
                lower_bound=lambda m: 0.01,  # Ensure c > 0 for log utility
                upper_bound=lambda m: 0.1 * m
                + 2,  # Loose bound; analytical c ≈ 0.04*m + 1.33
                agent="consumer",
            ),
            # Normalized assets (for transition to next period)
            "a": lambda m, c: m - c,
            # Log utility of normalized consumption
            "u": lambda c: crra_utility(c, 1.0),
        },
        "reward": {"u": "consumer"},
    }
)


def u2_analytical_policy(states, shocks, parameters):
    r"""
    Optimal policy for U-2: log utility with permanent income shocks
    (normalized).

    The buffer-stock problem with log utility, geometric random-walk
    permanent income, and no borrowing constraint admits a closed-form
    policy in *normalized* variables. Dividing every level variable by
    permanent income :math:`P_t` yields the lowercase ratios
    :math:`m = M/P`, :math:`c = C/P`, :math:`a = A/P`, with normalized
    transition

    .. math::
        m_{t+1} = \frac{R}{\psi_{t+1}} \, a_t + 1,

    where :math:`\psi_{t+1}` is the mean-one permanent income shock. The
    constant ``+1`` represents normalized transitory income, which is
    identically one in U-2; the more general two-shock case (with a
    stochastic transitory component) is U-3. Under log utility the closed
    form is

    .. math::
        c_t \;=\; (1 - \beta) \, (m_t + h),
        \qquad
        h \;=\; 1/r,

    independent of the realized shock path. The MPC :math:`(1 - \beta)`
    is the limiting MPC for any unconstrained CRRA agent; log utility (the
    :math:`\sigma = 1` case) makes it exact rather than asymptotic.

    Parameters
    ----------
    states : dict
        Must contain ``"a"`` (normalized arrival assets).
    shocks : dict
        May contain ``"psi"`` (permanent income shock, defaults to ones).
    parameters : dict
        Must contain ``"DiscFac"`` and ``"R"``.

    Returns
    -------
    dict
        ``{"c": c_optimal}`` (normalized consumption).

    Notes
    -----
    Setting ``"sigma_psi": 0`` makes :math:`\psi \equiv 1`, so the PIH
    analytical solution holds exactly. The ``Control`` upper bound
    :math:`0.1\, m + 2` is loose enough for the analytical policy
    :math:`c \approx 0.04\, m + 1.33` but rules out Ponzi-scheme solutions
    that satisfy the Euler equation while violating transversality.

    See Also
    --------
    u3_block : Same problem with a binding borrowing constraint and CRRA
        utility, which has no closed form.
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]

    # Extract arrival state (normalized assets from previous period)
    a = states["a"]

    # Get shock realization (default to ones tensor for type consistency)
    psi = shocks.get("psi", torch.ones_like(a))

    # Compute normalized cash-on-hand: m = R*a/ψ + 1
    clamped_psi = torch.clamp(psi, min=1e-8)
    if not torch.equal(psi, clamped_psi):
        logger.debug(
            "Clamped near-zero psi values in u2_analytical_policy: min(psi)=%g -> %g",
            psi.min().item(),
            clamped_psi.min().item(),
        )
    m = R * a / clamped_psi + 1

    # Human wealth (normalized): h = 1/r
    r = _human_wealth_rate(R)
    h = 1 / r

    # PIH solution (normalized): c = (1-β)(m + h)
    mpc = 1 - beta
    c_optimal = mpc * (m + h)

    return {"c": c_optimal}


# U-3: Buffer Stock Model - CRRA Utility with Borrowing Constraint
# -----------------------------------------------------------------
# Carroll's buffer stock consumption model (Carroll 1992, 1997) with:
#   - CRRA utility with γ > 1 (e.g., γ=2)
#   - Borrowing constraint: c ≤ m (cannot borrow against future income)
#   - Permanent income shocks (ψ ~ MeanOneLogNormal)
#   - Transitory income shocks (θ ~ MeanOneLogNormal)
#
# Uses NORMALIZED variables (lowercase = normalized by permanent income P):
#   m = M/P  (normalized cash-on-hand)
#   c = C/P  (normalized consumption)
#   a = A/P  (normalized assets)
#
# Normalized dynamics:
#   m = R*a/ψ + θ  (normalized cash-on-hand, where ψ is permanent shock, θ is transitory)
#   a' = m - c     (normalized end-of-period assets)
#
# This model does NOT have a closed-form analytical solution due to the
# borrowing constraint + income uncertainty interaction. However, it has
# well-known LIMITING MPC properties that can be tested:
#   - MPC is between 0 and 1
#   - MPC DECREASES with wealth (precautionary saving diminishes)
#   - As wealth → ∞: MPC → κ = (R - (βR)^(1/γ)) / R  (same as perfect foresight)

u3_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "CRRA": 2.0,  # γ = 2 (more risk averse than log utility)
    "sigma_psi": 0.1,  # Std of permanent shocks
    "sigma_theta": 0.1,  # Std of transitory shocks
    "description": "U-3: Buffer stock model (normalized) with CRRA=2, permanent + transitory shocks",
}

u3_block = DBlock(
    **{
        "name": "u3_buffer_stock_normalized",
        "shocks": {
            "psi": (MeanOneLogNormal, {"sigma": "sigma_psi"}),  # Permanent income shock
            "theta": (
                MeanOneLogNormal,
                {"sigma": "sigma_theta"},
            ),  # Transitory income shock
        },
        "dynamics": {
            # Normalized cash-on-hand: m = R*a/ψ + θ
            # Note: psi is strictly positive from MeanOneLogNormal, but we add epsilon for safety
            "m": lambda a, R, psi, theta: R * a / torch.clamp(psi, min=1e-8) + theta,
            # Normalized consumption - depends ONLY on m
            # Lower bound ensures c > 0 for CRRA utility
            # Upper bound is borrowing constraint: c ≤ m
            "c": Control(
                ["m"],  # Control depends ONLY on normalized m
                lower_bound=lambda m: 0.01,  # Ensure c > 0 for CRRA utility
                upper_bound=lambda m: m,  # BORROWING CONSTRAINT: c ≤ m
                agent="consumer",
            ),
            # Normalized assets (for transition)
            "a": lambda m, c: m - c,
            # Lambda maps calibration key CRRA to crra_utility's gamma parameter
            "u": lambda c, CRRA: crra_utility(c, CRRA),
        },
        "reward": {"u": "consumer"},
    }
)


def _generate_u3_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-3 buffer stock model: a (normalized arrival assets)"""
    a = torch.linspace(0.5, 5.0, test_points)  # Normalized assets
    return {"a": a}


# =============================================================================
# Model Registry
# =============================================================================


def _generate_d1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-1 model: W (wealth), t (time)"""
    return {
        "W": torch.linspace(1.0, 5.0, test_points),
        "t": torch.zeros(test_points, dtype=torch.int64),  # Time state (starts at 0)
    }


def _generate_d2_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-2 model: a (arrival assets)"""
    return {"a": torch.linspace(0.5, 4.0, test_points)}


def _generate_d3_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-3 model: a (arrival assets), liv (living state)"""
    return {
        "a": torch.linspace(0.5, 4.0, test_points),
        "liv": torch.ones(test_points),  # Living state (starts alive)
    }


def _generate_u1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-1 model: A (arrival assets), y (realized income)"""
    # Simple Hall random walk - no habit formation, no c_lag
    return {
        "A": torch.linspace(0.5, 3.0, test_points),
        "y": torch.ones(test_points),  # Default income realization for testing
    }


def _generate_u2_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-2 model: a (normalized arrival assets)"""
    # Arrival state is normalized assets a = A/P from previous period
    a = torch.linspace(0.5, 5.0, test_points)
    return {"a": a}


def _validate_d2_d3_solution(
    model_id: str,
    test_states: Dict[str, torch.Tensor],
    analytical_controls: Dict[str, torch.Tensor],
    parameters: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Validate D-2 and D-3 perfect foresight optimality conditions"""
    beta = parameters["DiscFac"]
    R = parameters["R"]
    sigma = parameters["CRRA"]
    y = parameters["y"]

    # For D-3, use effective discount factor
    if model_id == "D-3":
        s = parameters["SurvivalProb"]
        beta_eff = s * beta
    else:
        beta_eff = beta

    kappa = (R - (beta_eff * R) ** (1 / sigma)) / R

    # Compute m from arrival state a (proper decision function structure)
    a = test_states["a"]
    m = a * R + y  # market resources = assets * return + income
    c = analytical_controls["c"]

    # Compute human wealth: present value of constant income stream
    r = _human_wealth_rate(R)
    human_wealth = y / r

    # Check that c_t = κ*(m_t + H) where H is human wealth
    expected_c = kappa * (m + human_wealth)
    consumption_errors = torch.abs(c - expected_c)
    max_consumption_error = torch.max(consumption_errors).item()

    if max_consumption_error > tolerance:
        return {
            "success": False,
            "validation": "FAILED",
            "error": f"Consumption rule violated: max error = {max_consumption_error}",
        }

    # Check budget constraint feasibility: c_t < m_t + H
    # Note: With human wealth, agent can consume more than current cash-on-hand by borrowing against future income
    total_wealth = m + human_wealth
    if torch.any(c >= total_wealth):
        return {
            "success": False,
            "validation": "FAILED",
            "error": "Budget constraint violated: consumption >= total wealth (m + H)",
        }

    # If we get here, validation passed
    return {"success": True, "validation": "PASSED"}


BENCHMARK_MODELS = {
    "D-1": {
        "block": d1_block,
        "calibration": d1_calibration,
        "analytical_policy": d1_analytical_policy,
        "test_states": _generate_d1_test_states,
    },
    "D-2": {
        "block": d2_block,
        "calibration": d2_calibration,
        "analytical_policy": d2_analytical_policy,
        "test_states": _generate_d2_test_states,
        "custom_validation": _validate_d2_d3_solution,
    },
    "D-3": {
        "block": d3_block,
        "calibration": d3_calibration,
        "analytical_policy": d3_analytical_policy,
        "test_states": _generate_d3_test_states,
        "custom_validation": _validate_d2_d3_solution,
    },
    "U-1": {
        "block": u1_block,
        "calibration": u1_calibration,
        "analytical_policy": u1_analytical_policy,
        "test_states": _generate_u1_test_states,
    },
    "U-2": {
        "block": u2_block,
        "calibration": u2_calibration,
        "analytical_policy": u2_analytical_policy,
        "test_states": _generate_u2_test_states,
    },
    "U-3": {
        "block": u3_block,
        "calibration": u3_calibration,
        # NO analytical_policy - buffer stock requires numerical solution
        "test_states": _generate_u3_test_states,
    },
}


def get_benchmark_model(model_id: str) -> DBlock:
    """Get benchmark model by ID (D-1, D-2, D-3, U-1, U-2, U-3) - 6 models"""
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    return BENCHMARK_MODELS[model_id]["block"]


def get_benchmark_calibration(model_id: str) -> Dict[str, Any]:
    """Get benchmark calibration by model ID"""
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    return BENCHMARK_MODELS[model_id]["calibration"].copy()


def get_analytical_policy(model_id: str) -> Callable:
    """Get analytical policy function by model ID.

    Raises ValueError if the model does not have an analytical policy
    (e.g., U-3 buffer stock model requires numerical solution).
    """
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    if "analytical_policy" not in BENCHMARK_MODELS[model_id]:
        raise ValueError(
            f"Model '{model_id}' does not have an analytical policy. "
            "This model requires numerical solution via Euler equation training. "
            "Use EulerEquationLoss with maliar_training_loop (constrained=True for "
            "models with upper-bound constrained controls). See tests/test_maliar.py for examples."
        )

    return BENCHMARK_MODELS[model_id]["analytical_policy"]


def get_test_states(model_id: str, test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Get test states for model validation by model ID"""
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    test_states_func = BENCHMARK_MODELS[model_id]["test_states"]
    return test_states_func(test_points)


def get_custom_validation(model_id: str) -> Callable | None:
    """Get custom validation function for model (if it has one)"""
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    return BENCHMARK_MODELS[model_id].get("custom_validation", None)


def list_benchmark_models() -> Dict[str, str]:
    """List all discrete-time benchmark models.

    Note: Most models have analytical solutions, but some (e.g., U-3 buffer stock)
    require numerical solution due to borrowing constraints + income uncertainty.
    """
    return {
        model_id: model_info["calibration"]["description"]
        for model_id, model_info in BENCHMARK_MODELS.items()
    }


def validate_analytical_solution(
    model_id: str, test_points: int = 10, tolerance: float = EPS_VALIDATION
) -> Dict[str, Any]:
    """Validate analytical solution satisfies optimality conditions and budget constraints"""

    if model_id not in BENCHMARK_MODELS:
        return {
            "success": False,
            "validation": "FAILED",
            "model_id": model_id,
            "error": f"Unknown model: {model_id}",
        }

    try:
        parameters = get_benchmark_calibration(model_id)
        analytical_policy = get_analytical_policy(model_id)

        # Generate appropriate test states using the extensible approach
        test_states = get_test_states(model_id, test_points)

        # Test analytical policy
        analytical_controls = analytical_policy(test_states, {}, parameters)

        # Basic feasibility checks
        for control_name, control_values in analytical_controls.items():
            if torch.any(control_values <= 0):
                return {
                    "success": False,
                    "validation": "FAILED",
                    "model_id": model_id,
                    "test_points": test_points,
                    "error": f"Negative {control_name} in analytical solution",
                }
            if torch.any(~torch.isfinite(control_values)):
                return {
                    "success": False,
                    "validation": "FAILED",
                    "model_id": model_id,
                    "test_points": test_points,
                    "error": f"Non-finite {control_name} in analytical solution",
                }

        # Run custom validation if the model has one
        custom_validation = get_custom_validation(model_id)
        if custom_validation is not None:
            validation_result = custom_validation(
                model_id, test_states, analytical_controls, parameters, tolerance
            )
            if not validation_result["success"]:
                return {
                    "success": False,
                    "validation": "FAILED",
                    "model_id": model_id,
                    "test_points": test_points,
                    "error": validation_result["error"],
                }

        return {
            "success": True,
            "validation": "PASSED",
            "model_id": model_id,
            "test_points": test_points,
            "max_consumption": torch.max(
                analytical_controls[list(analytical_controls.keys())[0]]
            ).item(),
            "min_consumption": torch.min(
                analytical_controls[list(analytical_controls.keys())[0]]
            ).item(),
        }

    except (ValueError, KeyError, RuntimeError, TypeError, AttributeError) as e:
        return {
            "success": False,
            "validation": "FAILED",
            "model_id": model_id,
            "test_points": test_points,
            "error": f"Validation failed: {str(e)}",
        }


def euler_equation_test(model_id: str, test_points: int = 100) -> Dict[str, Any]:
    """Test Euler equation satisfaction for stochastic analytical solutions"""

    if model_id == "D-2":
        return {
            "success": False,
            "test": "SKIPPED",
            "model_id": model_id,
            "error": "D-2 is a perfect foresight model - Euler equation test not applicable",
        }

    # Add tests for stochastic models here when needed
    return {
        "success": False,
        "test": "NOT_IMPLEMENTED",
        "model_id": model_id,
        "error": f"Euler test not implemented for {model_id}",
    }


# ANALYTICAL LIFETIME REWARD FUNCTIONS
# ===================================


def d1_analytical_lifetime_reward(
    initial_wealth: float,
    discount_factor: float,
    interest_rate: float,
    time_horizon: int,
) -> float:
    """
    Analytical lifetime reward for D-1: Finite horizon log utility.

    Forward simulation that exactly matches the D-1 model implementation.
    """
    beta = discount_factor
    R = interest_rate
    W = initial_wealth
    T = time_horizon

    if W <= 0 or T <= 0:
        return -np.inf

    # Forward simulation matching D-1 model
    total_utility = 0.0
    current_wealth = W

    for t in range(T):
        # Remaining periods calculation: T - t
        remaining = T - t

        if remaining <= 1:
            # Terminal period: consume everything
            consumption = current_wealth
        else:
            # Use remaining horizon formula: c_t = (1-β)/(1-β^remaining) * W_t
            consumption_rate = (1 - beta) / (1 - beta**remaining)
            consumption = consumption_rate * current_wealth

        period_utility = np.log(consumption)
        total_utility += (beta**t) * period_utility

        if t < T - 1:
            current_wealth = (current_wealth - consumption) * R

    return total_utility


def d2_analytical_lifetime_reward(
    cash_on_hand: float,
    discount_factor: float,
    interest_rate: float,
    risk_aversion: float,
    income: float = 0.0,  # Income parameter to calculate human wealth
) -> float:
    """
    Analytical lifetime reward for D-2 using total wealth.

    With constant income y > 0, the value function is based on total wealth W = m + H
    where H = y/r is human wealth.

    Optimal policy: c = κ*(m + H) where κ = (R - (βR)^(1/σ))/R
    Value function: V(W) = κ^(1-σ)/(1-σ) * W^(1-σ) / (1 - β*(βR)^((1-σ)/σ))
    """
    beta = discount_factor
    R = interest_rate
    sigma = risk_aversion
    m = cash_on_hand
    y = income

    if m <= 0:
        return -np.inf

    growth_factor = (beta * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError("Return-impatience condition violated")

    kappa = (R - growth_factor) / R
    r = _human_wealth_rate(R)

    # Calculate total wealth W = m + H (including human wealth)
    if y > 0:
        human_wealth = y / r  # Human wealth for constant income
        total_wealth = m + human_wealth
    else:
        total_wealth = m  # No income case

    if total_wealth <= 0:
        return -np.inf

    if sigma == 1:  # Log utility case
        # Log utility value function:
        # V(W) = ln(W)/(1-β) + ln(1-β)/(1-β) + β*ln(R*β)/(1-β)²
        ln_wealth_term = np.log(total_wealth) / (1 - beta)
        constant_term1 = np.log(1 - beta) / (1 - beta)
        constant_term2 = beta * np.log(R * beta) / ((1 - beta) ** 2)
        return ln_wealth_term + constant_term1 + constant_term2
    else:  # General CRRA case
        denominator = 1 - beta * (beta * R) ** ((1 - sigma) / sigma)
        return (
            kappa ** (1 - sigma) * total_wealth ** (1 - sigma) / (1 - sigma)
        ) / denominator


def d3_analytical_lifetime_reward(
    cash_on_hand: float,
    discount_factor: float,
    interest_rate: float,
    risk_aversion: float,
    survival_prob: float,
    income: float = 0.0,
) -> float:
    """
    Analytical lifetime reward for D-3: Blanchard discrete-time mortality.

    Similar to D-2 but uses effective discount factor β_eff = s*β where s is survival probability.
    The mortality risk effectively increases the discount rate, making the agent more impatient.

    With income y > 0, uses total wealth W = m + H where H = y/r.
    Value function: V(W) = κ_eff^(1-σ)/(1-σ) * W^(1-σ) / (1 - β_eff*(β_eff*R)^((1-σ)/σ))
    where κ_eff uses β_eff = s*β in place of β.
    """
    beta = discount_factor
    R = interest_rate
    sigma = risk_aversion
    s = survival_prob
    m = cash_on_hand
    y = income

    if m <= 0:
        return -np.inf

    if not (0 < s <= 1):
        raise ValueError(f"Survival probability must be in (0,1], got s={s}")

    # Effective discount factor due to mortality risk
    beta_eff = s * beta

    growth_factor = (beta_eff * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError("Return-impatience condition violated with mortality")

    kappa_eff = (R - growth_factor) / R
    r = _human_wealth_rate(R)

    # Calculate total wealth W = m + H (same as D-2)
    if y > 0:
        human_wealth = y / r  # Human wealth for constant income
        total_wealth = m + human_wealth
    else:
        total_wealth = m  # No income case

    if total_wealth <= 0:
        return -np.inf

    if sigma == 1:  # Log utility case with mortality
        # Modified log utility value function with effective discount factor
        ln_wealth_term = np.log(total_wealth) / (1 - beta_eff)
        constant_term1 = np.log(1 - beta_eff) / (1 - beta_eff)
        constant_term2 = beta_eff * np.log(R * beta_eff) / ((1 - beta_eff) ** 2)
        return ln_wealth_term + constant_term1 + constant_term2
    else:  # General CRRA case with mortality
        denominator = 1 - beta_eff * (beta_eff * R) ** ((1 - sigma) / sigma)
        return (
            kappa_eff ** (1 - sigma) * total_wealth ** (1 - sigma) / (1 - sigma)
        ) / denominator


def get_analytical_lifetime_reward(model_id: str, *args, **kwargs) -> float:
    """
    Get analytical lifetime reward for a benchmark model.

    Parameters
    ----------
    model_id : str
        Model identifier (D-1, D-2, D-3, etc.)
    *args, **kwargs
        Arguments to pass to the specific analytical function

    Returns
    -------
    float
        Analytical lifetime reward value
    """
    analytical_functions = {
        "D-1": d1_analytical_lifetime_reward,
        "D-2": d2_analytical_lifetime_reward,
        "D-3": d3_analytical_lifetime_reward,  # ADDED: Missing D-3 lifetime reward function
    }

    if model_id not in analytical_functions:
        raise ValueError(f"No analytical lifetime reward function for {model_id}")

    return analytical_functions[model_id](*args, **kwargs)
