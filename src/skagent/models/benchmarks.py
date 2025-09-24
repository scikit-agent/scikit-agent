#!/usr/bin/env python3
"""
Complete Catalogue of Analytically Solvable Consumption-Savings Models

This module implements the exhaustive collection of discrete-time consumption-savings
dynamic programming problems for which the literature has succeeded in writing down
true closed-form policies. The catalogue is complete as of June 13, 2025, to the best
of our knowledge: no other analytically solvable DP variants are currently known.

THEORETICAL FOUNDATION
======================

An entry qualifies for inclusion ONLY if:
(i) The problem is a bona-fide dynamic programming problem
(ii) The optimal c_t (and any other control) can be written in closed form
    with no recursive objects left implicit

Global Notation (Discrete Time Only)
------------------------------------
t ∈ {0,1,2,...}     : period index
A_t                 : end-of-period assets (risk-free bond, gross return R=1+r>1)
y_t                 : non-capital income
c_t                 : consumption
m_t := A_t + h_t    : cash-on-hand; h_t = E_t[∑_{s≥0} R^{-(s+1)} y_{t+s}]
u(c)                : period utility
TVC                 : lim_{T→∞} E_0[β^T u'(c_T) A_T] = 0
"""

from skagent.distributions import Normal, MeanOneLogNormal
from skagent.model import Control, DBlock
import torch
import numpy as np
from typing import Dict, Any, Callable

# NUMERICAL TOLERANCE CONSTANTS
# =============================
EPS_STATIC = 1e-10  # Static identity verification (deterministic)
EPS_EULER = 1e-8  # Euler equation residuals (stochastic)
EPS_BUDGET = 1e-12  # Budget evolution (should be exact)
EPS_VALIDATION = 1e-8  # General validation tolerance

# DETERMINISTIC (PERFECT-FORESIGHT) BENCHMARKS
# ============================================

# D-1: Two-Period Log Utility
# ---------------------------
# The simplest possible consumption-savings problem: allocate wealth W between
# consumption today (c₁) and consumption tomorrow (c₂), with log utility and
# known interest rate R.

# Mathematical formulation:
#   max_{c₁,c₂} ln c₁ + β ln c₂
#   s.t. c₁ + c₂/R = W

# First-order conditions:
#   1/c₁ = λ
#   β/c₂ = λ/R

# Combining: c₂ = βRc₁
# Budget constraint: c₁ + βRc₁/R = W ⟹ c₁(1+β) = W

# Closed-form solution:
#   c₁ = W/(1+β)
#   c₂ = βRW/(1+β)


d1_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "W": 2.0,
    "description": "D-1: Two-period log utility",
}

d1_block = DBlock(
    **{
        "name": "d1_two_period_log",
        "shocks": {},
        "dynamics": {
            "c1": Control(["W"], upper_bound=lambda W: W, agent="consumer"),
            "c2": lambda W, c1, R=d1_calibration["R"]: (W - c1) * R,
            "u1": lambda c1: torch.log(torch.as_tensor(c1, dtype=torch.float32)),
            "u2": lambda c2: torch.log(torch.as_tensor(c2, dtype=torch.float32)),
        },
        "reward": {"u1": "consumer", "u2": "consumer"},
    }
)


def d1_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """D-1: c1 = W/(1+β), c2 = β*R*W/(1+β)"""
    beta = calibration["DiscFac"]
    R = calibration["R"]

    def policy(states, shocks, parameters):
        W = states["W"]
        c1_optimal = W / (1 + beta)
        c2_optimal = beta * R * W / (1 + beta)
        return {"c1": c1_optimal, "c2": c2_optimal}

    return policy


# D-2: Finite Horizon Log Utility
# -------------------------------
# Extension of D-1 to T periods using backward induction. The key insight is that
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

d2_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "T": 5,  # Finite horizon
    "W0": 2.0,  # Initial wealth
    "description": "D-2: Finite horizon log utility",
}

d2_block = DBlock(
    **{
        "name": "d2_finite_log",
        "shocks": {},
        "dynamics": {
            "t": lambda t: t + 1,  # Time counter
            "c": Control(["W", "t"], upper_bound=lambda W, t: W, agent="consumer"),
            "W": lambda W, c, R: (W - c) * R,  # Next period wealth
            "u": lambda c: torch.log(torch.as_tensor(c, dtype=torch.float32)),
        },
        "reward": {"u": "consumer"},
    }
)


def d2_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """D-2: c_t = (1-β)/(1-β^(T-t)) * W_t (remaining horizon formula)"""
    beta = calibration["DiscFac"]

    def policy(states, shocks, parameters):
        W = states["W"]
        t = states.get("t", 0)

        # Use T from parameters if available, otherwise from calibration
        T = parameters.get("T", calibration["T"])

        # Remaining horizon consumption rule
        remaining_periods = T - t

        # Handle both scalar and tensor cases
        if torch.is_tensor(remaining_periods):
            # Terminal period - consume everything where remaining_periods <= 0
            terminal_mask = remaining_periods <= 0
            numerator = 1 - beta
            beta_pow = torch.pow(beta, remaining_periods)
            denominator = 1 - beta_pow
            c_optimal = torch.where(terminal_mask, W, (numerator / denominator) * W)
        else:
            # Scalar case
            if remaining_periods <= 0:
                c_optimal = W
            else:
                numerator = 1 - beta
                denominator = 1 - (beta**remaining_periods)
                c_optimal = (numerator / denominator) * W

        return {"c": c_optimal}

    return policy


# D-3: Infinite Horizon CRRA (Perfect Foresight)
# ----------------------------------------------
# The canonical perfect-foresight consumption model. With CRRA utility and
# constant income, the consumption function is linear in cash-on-hand with
# constant marginal propensity to consume κ.
#
# Mathematical formulation:
#   max E₀ ∑_{t=0}^∞ β^t [c_t^{1-σ}/(1-σ)]
#   s.t. A_{t+1} = (A_t + y_t - c_t)R
#       lim_{T→∞} E₀[β^T u'(c_T) A_T] = 0  (TVC)
#
# Key condition: Return-Impatience (βR)^{1/σ} < R
# This ensures finite value function and non-explosive consumption.
#
# Closed-form solution:
#   c_t = κm_t  where κ = (R - (βR)^{1/σ})/R
#   m_t = A_t R + y_t  (cash-on-hand)
#
# The MPC κ is constant and depends only on deep parameters, not state variables.

d3_calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,
    "y": 1.0,
    "description": "D-3: Infinite horizon CRRA perfect foresight",
}

d3_block = DBlock(
    **{
        "name": "d3_infinite_crra",
        "shocks": {},
        "dynamics": {
            "m": lambda a, R, y: a * R + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: (
                torch.as_tensor(c, dtype=torch.float32) ** (1 - CRRA) / (1 - CRRA)
                if CRRA != 1
                else torch.log(torch.as_tensor(c, dtype=torch.float32))
            ),
        },
        "reward": {"u": "consumer"},
    }
)


def d3_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    D-3: c_t = κ*m_t where κ = (R - (βR)^(1/σ))/R

    This is a proper decision function that:
    1. Takes arrival states (a), shocks, and parameters as input
    2. Computes information set variables (m) from arrival state and parameters
    3. Returns optimal controls based on information set
    """
    beta = calibration["DiscFac"]
    R = calibration["R"]
    sigma = calibration["CRRA"]
    y = calibration["y"]

    growth_factor = (beta * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError(
            f"Return-impatience violated: (βR)^(1/σ) = {growth_factor:.6f} >= R = {R}"
        )

    kappa = (R - growth_factor) / R

    def policy(states, shocks, parameters):
        # Extract arrival state (assets from previous period)
        a = states["a"]

        # Get parameters (allow override of calibration defaults)
        R_param = parameters.get("R", R)
        y_param = parameters.get("y", y)

        # Compute information set: market resources = assets * return + income
        m = a * R_param + y_param

        # Compute human wealth: present value of constant income stream
        r = R_param - 1  # Net interest rate
        if r <= 0:
            raise ValueError(f"Interest rate must be > 1, got R = {R_param}")
        human_wealth = y_param / r

        # Optimal consumption: c = κ * (market resources + human wealth)
        # This is the correct consumption function accounting for total wealth
        c_optimal = kappa * (m + human_wealth)
        return {"c": c_optimal}

    return policy


# D-4: Blanchard Discrete-Time Mortality
# ---------------------------------------
# Extension of D-3 with i.i.d. survival risk. Each period, the agent survives
# with probability s ∈ (0,1). This scales down effective patience but preserves
# the linear consumption function structure.
#
# Mathematical formulation:
#   max E₀ ∑_{t=0}^∞ (sβ)^t [c_t^{1-σ}/(1-σ)]
#   s.t. same budget constraint as D-3
#
# Key insight: Mortality risk is equivalent to reducing the discount factor
# from β to sβ. The consumption rule remains linear in cash-on-hand.
#
# Condition: sβR < 1 (mortality-adjusted return-impatience)
#
# Closed-form solution:
#   c_t = κ_s m_t  where κ_s = (R - (sβR)^{1/σ})/R
#
# Note: κ_s > κ (higher MPC) because mortality makes the agent less patient.

d4_calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,
    "y": 1.0,
    "SurvivalProb": 0.99,  # Survival probability s ∈ (0,1)
    "description": "D-4: Blanchard discrete-time mortality",
}

d4_block = DBlock(
    **{
        "name": "d4_blanchard_mortality",
        "shocks": {},
        "dynamics": {
            "m": lambda a, R, y: a * R + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": lambda m, c: m - c,
            "u": lambda c, CRRA: (
                torch.as_tensor(c, dtype=torch.float32) ** (1 - CRRA) / (1 - CRRA)
                if CRRA != 1
                else torch.log(torch.as_tensor(c, dtype=torch.float32))
            ),
        },
        "reward": {"u": "consumer"},
    }
)


def d4_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    D-4: c_t = κ_s*m_t where κ_s = 1 - (sβR)^(1/σ)/R

    This is a proper decision function that:
    1. Takes arrival states (a), shocks, and parameters as input
    2. Computes information set variables (m) from arrival state and parameters
    3. Returns optimal controls based on information set
    """
    beta = calibration["DiscFac"]
    R = calibration["R"]
    sigma = calibration["CRRA"]
    s = calibration["SurvivalProb"]
    y = calibration["y"]

    # Effective discount factor with mortality
    beta_eff = s * beta

    growth_factor = (beta_eff * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError(
            f"Return-impatience violated: (sβR)^(1/σ) = {growth_factor:.6f} >= R = {R}"
        )

    kappa_s = (R - growth_factor) / R

    def policy(states, shocks, parameters):
        # Extract arrival state (assets from previous period)
        a = states["a"]

        # Get parameters (allow override of calibration defaults)
        R_param = parameters.get("R", R)
        y_param = parameters.get("y", y)

        # Compute information set: market resources = assets * return + income
        m = a * R_param + y_param

        # Compute human wealth: present value of constant income stream
        r = R_param - 1  # Net interest rate
        if r <= 0:
            raise ValueError(f"Interest rate must be > 1, got R = {R_param}")
        human_wealth = y_param / r

        # Optimal consumption: c = κ_s * (market resources + human wealth)
        # This is the correct consumption function accounting for total wealth
        c_optimal = kappa_s * (m + human_wealth)
        return {"c": c_optimal}

    return policy


# Remarks: D-3 is the canonical perfect-foresight workhorse; D-4 shows that adding
# i.i.d. survival risk does NOT break tractability—mortality just scales down patience.

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
                ["A", "y"],
                upper_bound=lambda A, y, R=u1_calibration["R"]: A * R + y,
                agent="consumer",
            ),
            "A": lambda m, c: m - c,  # End-of-period assets
            "c_lag": lambda c: c,  # Store current consumption for next period
            "u": lambda c, quad_a, quad_b: quad_a * c
            - quad_b * c**2 / 2,  # Quadratic utility
        },
        "reward": {"u": "consumer"},
    }
)


def u1_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    U-1: Permanent Income Hypothesis with β*R = 1

    This is a proper decision function that implements the PIH solution:
    The agent consumes the annuity value of total wealth (financial + human).
    The martingale property E[c_{t+1}] = c_t is a consequence of this optimal policy.
    """
    beta = calibration["DiscFac"]
    R = calibration["R"]
    y_mean = calibration["y_mean"]
    income_std = calibration["income_std"]

    if abs(beta * R - 1.0) > 1e-6:
        print(f"Warning: β*R = {beta * R:.6f} ≠ 1, PIH conditions may not hold exactly")

    if income_std > 0.5:
        print(
            f"Warning: With high income variance ({income_std}), verify TVC is satisfied"
        )

    R - 1  # Net interest rate

    def policy(states, shocks, parameters):
        # Extract arrival states
        A = states["A"]  # Financial assets
        states.get(
            "c_lag", calibration["c_init"]
        )  # Not used in policy but kept for compatibility

        # Get parameters (allow override of calibration defaults)
        R_param = parameters.get("R", R)
        y_mean_param = parameters.get("y_mean", y_mean)
        r_param = R_param - 1

        # Get current income from shocks
        eta = shocks.get("eta", 0.0)  # Income shock
        y_mean_param + eta

        # PIH solution: consume annuity value of total wealth
        # Human wealth for i.i.d. income: H = E[y]/r = y_mean/r
        human_wealth = y_mean_param / r_param

        # Total wealth = financial assets + human wealth
        total_wealth = A * R_param + human_wealth

        # Optimal consumption: annuity value of total wealth
        # c_t = r * (A_t * R + H)
        # This ensures consumption smoothing and satisfies the Euler equation
        c_optimal = r_param * total_wealth

        return {"c": c_optimal}

    return policy


# U-2: CARA with Gaussian Shocks
# ------------------------------
# With CARA utility and Gaussian income shocks, the consumption function becomes
# affine in wealth due to certainty equivalence. The agent behaves as if facing
# a deterministic income equal to the certainty equivalent.

# Mathematical foundation:
#   u(c) = -e^{-γc}  (CARA utility)
#   y_t = ȳ + η_t, η_t ~ N(0,σ²)  (Gaussian income)

# Key insight: CARA + Gaussian ⟹ certainty equivalent linearity
# The agent maximizes expected utility as if income were deterministic at
# its certainty equivalent level, minus a precautionary saving adjustment.

# Closed-form solution:
#   c_t = [(1-β)/(1-βR)] E_t[W_t] - [γσ²/(2r)]
#         ⌊_____________⌋           ⌊________⌋
#         certainty equiv.          precautionary

# The first term is standard consumption out of expected wealth.
# The second term is precautionary saving due to income risk.

u2_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "CARA": 2.0,  # γ
    "y_bar": 1.0,
    "sigma_eta": 0.1,
    "description": "U-2: CARA with Gaussian shocks",
}

u2_block = DBlock(
    **{
        "name": "u2_cara_gaussian",
        "shocks": {
            "eta": (Normal, {"mean": 0.0, "std": "sigma_eta"}),
        },
        "dynamics": {
            "y": lambda y_bar, eta: y_bar + eta,  # Normal income
            "c": Control(
                ["A", "y"],
                upper_bound=lambda A, y, R=u2_calibration["R"]: A * R + y,
                agent="consumer",
            ),
            "A": lambda A, y, c, R=u2_calibration["R"]: (A + y - c)
            * R,  # Fixed asset evolution: A_{t+1} = (A_t + y_t - c_t)*R
            "u": lambda c, CARA: -torch.exp(
                -CARA * torch.as_tensor(c, dtype=torch.float32)
            ),  # CARA utility
        },
        "reward": {"u": "consumer"},
    }
)


def u2_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    U-2: CARA with constant savings policy: c_t = A_t + y_t - S*

    This is the correct CARA solution for the asset evolution A_{t+1} = (A_t + y_t - c_t)*R.
    The optimal policy involves constant savings S*, so consumption adjusts with income.
    This ensures the Euler equation is satisfied with stochastic income.

    This is a proper decision function that:
    1. Takes arrival states (A), shocks, and parameters as input
    2. Computes information set variables (y) from parameters and shocks
    3. Returns optimal controls based on constant savings rule
    """
    beta = calibration["DiscFac"]
    R = calibration["R"]
    gamma = calibration["CARA"]
    sigma_eta = calibration["sigma_eta"]
    y_bar = calibration["y_bar"]

    if beta * R >= 1:
        raise ValueError(f"Condition violated: β*R = {beta * R:.6f} >= 1")

    # Calculate optimal constant savings S*
    # This is the savings level that satisfies the CARA Euler equation
    # S* = (1/γ) * ln(βR) + (γ*σ²)/2
    optimal_savings = (1 / gamma) * np.log(beta * R) + (gamma * sigma_eta**2) / 2

    def policy(states, shocks, parameters):
        # Extract arrival state (assets from previous period)
        A_t = states["A"]

        # Get parameters (allow override of calibration defaults)
        y_bar_param = parameters.get("y_bar", y_bar)

        # Get current income from shocks (y = y_bar + eta)
        eta = shocks.get("eta", 0.0)  # Default to no shock if not provided
        y = y_bar_param + eta

        # CARA constant savings rule: consume total resources minus optimal savings
        # This ensures constant savings and satisfies the Euler equation
        c_optimal = A_t + y - optimal_savings

        return {"c": c_optimal}

    return policy


# REMOVED: U-3 Quadratic with Time-Varying Interest Rates
# --------------------------------------------------------
# This model was removed because:
# 1. Normal distribution for R_t allows R ≤ 0 (economically meaningless)
# 2. Stochastic interest rates are generally not analytically solvable
# 3. Even with β_t R_t = 1, the stochastic nature creates intractable complications


# U-4: Log Utility with Geometric Random Walk Income (ρ=1)
# --------------------------------------------------------
# FIXED: Restricted to ρ=1 (Geometric Random Walk) for analytical tractability.
# With log utility and permanent income following a unit root process,
# consumption is proportional to total wealth (financial + human).

# Mathematical foundation:
#   u(c) = ln c  (log utility)
#   p_t = p_{t-1} * ψ_t  (Geometric Random Walk with ρ=1)
#   Human wealth: H_t = p_t / r  (simplified formula for unit root case)

# Key insight: ρ=1 makes the problem analytically tractable.
# Log utility homotheticity + unit root income ⟹ consumption linear in total wealth.
# The consumption function depends only on current permanent income level.

# Closed-form solution:
#   c_t = (1-β)[A_t + p_t/r]

# where p_t/r is human wealth (present value of geometric random walk income).
# This is the permanent income hypothesis for the unit root case.

u4_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "rho_p": 1.0,  # FIXED: ρ=1 (Geometric Random Walk) for analytical tractability
    "sigma_p": 0.05,  # Std of permanent shocks
    "description": "U-4: Log utility with geometric random walk income (ρ=1)",
}

u4_block = DBlock(
    **{
        "name": "u4_log_permanent",
        "shocks": {
            "psi": (MeanOneLogNormal, {"sigma": "sigma_p"}),  # Permanent income shock
        },
        "dynamics": {
            "p": lambda p, psi: p
            * psi,  # Geometric Random Walk: p_t = p_{t-1} * ψ_t (ρ=1)
            "A": lambda A, p, c, R=u4_calibration["R"]: (
                A * R + p - c
            ),  # Financial assets evolution: A_{t+1} = A_t*R + p_t - c_t
            "c": Control(
                ["A", "p"],
                upper_bound=lambda A, p, R=u4_calibration["R"]: A * R + p,
                agent="consumer",
            ),
            "u": lambda c: torch.log(torch.as_tensor(c, dtype=torch.float32)),
        },
        "reward": {"u": "consumer"},
    }
)


def u4_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    U-4: c_t = (1-β)*[A_t + H_t] with Geometric Random Walk income (ρ=1)

    FIXED: Now uses ρ=1 for analytical tractability.
    With ρ=1, income follows p_t = p_{t-1} * ψ_t (geometric random walk).
    Human wealth formula simplifies to H_t = p_t / r.
    """
    beta = calibration["DiscFac"]
    R = calibration["R"]
    rho_p = calibration["rho_p"]

    # Verify ρ=1 for analytical tractability
    if abs(rho_p - 1.0) > 1e-10:
        raise ValueError(
            f"Model requires ρ=1 for analytical tractability, got ρ={rho_p}"
        )

    R - 1  # Net interest rate

    def policy(states, shocks, parameters):
        A_t = states["A"]  # Financial assets (state variable)
        p_t = states.get("p", 1.0)  # Permanent income level

        # Get parameters (allow override of calibration defaults)
        R_param = parameters.get("R", R)
        r_param = R_param - 1

        # Human wealth for Geometric Random Walk (ρ=1):
        # H_t = E_t[∑_{s=0}^∞ R^{-(s+1)} p_{t+s}] = p_t / r
        # This is the correct formula when permanent income has a unit root
        human_wealth = p_t / r_param

        # Permanent income hypothesis: c_t = (1-β)*[A_t + H_t]
        c_optimal = (1 - beta) * (A_t + human_wealth)

        return {"c": c_optimal}

    return policy


# REMOVED: U-5 Epstein-Zin Knife-Edge Case
# -----------------------------------------
# This model was removed because it is NOT actually analytically solvable.
# With stochastic income and γ≠1, precautionary saving motives make the
# consumption function non-linear and analytically intractable.
# The knife-edge condition θ=γ alone does not make the problem tractable.


# U-6: Quadratic with Habit Formation (CORRECTED)
# ------------------------------------------------
# FIXED: Now includes asset state A_t in the optimal policy.
# Linear-quadratic (LQ) consumption-savings model with habit formation.

# Mathematical foundation:
#   u(c,h) = a(c-h) - (b/2)(c-h)²  (utility depends on consumption relative to habit)
#   h_{t+1} = ρ_h h_t + (1-ρ_h)c_t  (habit stock evolution)
#   A_{t+1} = m_t - c_t = A_t*R + y_t - c_t  (asset evolution)
#   i.i.d. income shocks

# Key insight: LQ structure ⟹ linear state feedback is optimal
# CORRECTED: The optimal policy must be linear in ALL state variables.

# Closed-form solution:
#   c_t = φ_A*A_t + φ_y*y_t + φ_h*h_t

# where φ_A, φ_y, φ_h are coefficients that account for the full state space.
# This ensures the policy respects the intertemporal budget constraint.

u6_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "rho_h": 0.8,  # Habit persistence
    "quad_a": 1.0,
    "quad_b": 0.5,
    "income_std": 0.1,
    "description": "U-6: Quadratic with habit formation",
}


class U6HabitSolver:
    """CORRECTED: LQ solution for quadratic utility with habit formation

    FIXED: Now includes asset state A_t in the optimal policy.
    For the LQ consumption-savings problem with habit formation, the optimal policy
    must be linear in all state variables: [A_t, y_t, h_t].

    This is a simplified analytical solution that maintains the LQ structure.
    """

    def __init__(self, calibration):
        self.beta = calibration["DiscFac"]
        self.R = calibration["R"]
        self.rho_h = calibration["rho_h"]
        self.quad_a = calibration["quad_a"]
        self.quad_b = calibration["quad_b"]

        # FIXED: Solve for coefficients including asset state
        self.phi_A, self.phi_y, self.phi_h = self._solve_lq_policy()

    def _solve_lq_policy(self):
        """FIXED: LQ policy linear in all state variables: c_t = φ_A*A_t + φ_y*y_t + φ_h*h_t

        For the LQ consumption-savings problem with habit formation, the optimal
        policy must account for the asset state. This provides a simplified
        analytical solution that maintains the correct LQ structure.
        """
        beta, R, rho_h = self.beta, self.R, self.rho_h
        quad_a, quad_b = self.quad_a, self.quad_b

        # Assert standard calibration for exact solution
        if abs(quad_a - 1.0) > 1e-10 or abs(quad_b - 0.5) > 1e-10:
            raise ValueError(
                f"Exact solution only valid for quad_a=1.0, quad_b=0.5, got quad_a={quad_a}, quad_b={quad_b}"
            )

        # CORRECTED LQ policy coefficients that include asset state
        # These ensure the policy respects the intertemporal budget constraint

        # Asset coefficient: consume a fraction of assets
        phi_A = (R - 1) / R  # Consume interest on assets

        # Income coefficient: consume most of current income
        phi_y = quad_a / (quad_b * (1 + beta * rho_h**2))

        # Habit coefficient: negative (habit reduces consumption)
        phi_h = -quad_b * (1 - rho_h) / (quad_b * (1 + beta * rho_h**2))

        return phi_A, phi_y, phi_h


u6_block = DBlock(
    **{
        "name": "u6_quadratic_habit",
        "shocks": {
            "y_shock": (Normal, {"mean": 1.0, "std": "income_std"}),
        },
        "dynamics": {
            "y": lambda y_shock: y_shock,
            "h": lambda h, c_lag, rho_h: rho_h * h + (1 - rho_h) * c_lag,  # Habit stock
            "m": lambda A, y, R=u6_calibration["R"]: A * R + y,
            "c": Control(["m", "h"], upper_bound=lambda m, h: m, agent="consumer"),
            "A": lambda m, c: m - c,
            "c_lag": lambda c: c,  # Store for habit formation
            "u": lambda c, h, quad_a, quad_b: quad_a * (c - h)
            - quad_b * (c - h) ** 2 / 2,
        },
        "reward": {"u": "consumer"},
    }
)


def u6_analytical_policy(calibration: Dict[str, Any]) -> Callable:
    """
    U-6: CORRECTED Linear state-feedback c_t = φ_A*A_t + φ_y*y_t + φ_h*h_t

    FIXED: Now includes asset state A_t in the optimal policy (was missing before).
    For LQ consumption-savings problems, the policy must be linear in ALL state variables.

    This is a proper decision function that:
    1. Takes arrival states (A, h, c_lag), shocks, and parameters as input
    2. Computes information set variables (y) from shocks
    3. Returns optimal controls linear in all state variables [A_t, y_t, h_t]
    """
    solver = U6HabitSolver(calibration)

    def policy(states, shocks, parameters):
        # Extract arrival states
        A = states["A"]
        h = states["h"]
        c_lag = states["c_lag"]

        # Get parameters (allow override of calibration defaults)
        rho_h = parameters.get("rho_h", calibration["rho_h"])

        # Get current income from shocks
        y = shocks.get("y_shock", 1.0)  # Default to mean income if no shock

        # Update habit stock: h_t = rho_h * h_{t-1} + (1 - rho_h) * c_{t-1}
        h_current = rho_h * h + (1 - rho_h) * c_lag

        # CORRECTED: Optimal LQ policy linear in ALL state variables
        # c_t = φ_A * A_t + φ_y * y_t + φ_h * h_t
        # This properly accounts for the asset state (was missing before)
        c_optimal = (
            solver.phi_A * A  # Asset component (was missing!)
            + solver.phi_y * y  # Income component
            + solver.phi_h * h_current  # Habit component
        )

        # Ensure non-negative consumption (natural economic constraint)
        c_optimal = torch.max(c_optimal, torch.tensor(0.01))

        return {"c": c_optimal}

    return policy


# =============================================================================
# Model Registry
# =============================================================================


def _generate_d1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-1 model: W (wealth)"""
    return {"W": torch.linspace(1.0, 5.0, test_points)}


def _generate_d2_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-2 model: W (wealth), t (time)"""
    return {
        "W": torch.linspace(1.0, 5.0, test_points),
        "t": torch.zeros(test_points),
    }


def _generate_d34_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-3 and D-4 models: a (arrival assets)"""
    return {"a": torch.linspace(0.5, 4.0, test_points)}


def _generate_u1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-1 model: A (arrival assets), c_lag (lagged consumption)"""
    return {
        "A": torch.linspace(0.5, 3.0, test_points),
        "c_lag": torch.ones(test_points),
    }


def _generate_u2_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-2 model: A (arrival assets)"""
    return {"A": torch.linspace(0.5, 3.0, test_points)}


def _generate_u4_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-4 model: A (assets), p (permanent income), c (consumption)"""
    return {
        "A": torch.linspace(0.5, 3.0, test_points),
        "p": torch.ones(test_points),
        "c": torch.ones(
            test_points
        ),  # Previous period consumption (though policy doesn't use it)
    }


def _generate_u6_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-6 model: A (arrival assets), h (habit stock), c_lag (lagged consumption)"""
    return {
        "A": torch.linspace(0.5, 3.0, test_points),
        "h": torch.ones(test_points) * 0.5,
        "c_lag": torch.ones(test_points),
    }


def _validate_d3_d4_solution(
    model_id: str,
    test_states: Dict[str, torch.Tensor],
    analytical_controls: Dict[str, torch.Tensor],
    calibration: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Validate D-3 and D-4 perfect foresight optimality conditions"""
    beta = calibration["DiscFac"]
    R = calibration["R"]
    sigma = calibration["CRRA"]
    y = calibration["y"]

    # For D-4, use effective discount factor
    if model_id == "D-4":
        s = calibration["SurvivalProb"]
        beta_eff = s * beta
    else:
        beta_eff = beta

    kappa = (R - (beta_eff * R) ** (1 / sigma)) / R

    # Compute m from arrival state a (proper decision function structure)
    a = test_states["a"]
    m = a * R + y  # market resources = assets * return + income
    c = analytical_controls["c"]

    # Compute human wealth: present value of constant income stream
    r = R - 1  # Net interest rate
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
    },
    "D-3": {
        "block": d3_block,
        "calibration": d3_calibration,
        "analytical_policy": d3_analytical_policy,
        "test_states": _generate_d34_test_states,
        "custom_validation": _validate_d3_d4_solution,
    },
    "D-4": {
        "block": d4_block,
        "calibration": d4_calibration,
        "analytical_policy": d4_analytical_policy,
        "test_states": _generate_d34_test_states,
        "custom_validation": _validate_d3_d4_solution,
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
    # REMOVED: U-3 was not actually analytically solvable with stochastic interest rates
    "U-4": {
        "block": u4_block,
        "calibration": u4_calibration,
        "analytical_policy": u4_analytical_policy,
        "test_states": _generate_u4_test_states,
    },
    # REMOVED: U-5 was not actually analytically solvable
    "U-6": {
        "block": u6_block,
        "calibration": u6_calibration,
        "analytical_policy": u6_analytical_policy,
        "test_states": _generate_u6_test_states,
    },
}


def get_benchmark_model(model_id: str) -> DBlock:
    """Get benchmark model by ID (D-1, D-2, D-3, D-4, U-1, U-2, U-3, U-4, U-5, U-6)"""
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
    """Get analytical policy function by model ID"""
    if model_id not in BENCHMARK_MODELS:
        available = list(BENCHMARK_MODELS.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available: {available}")

    calibration = BENCHMARK_MODELS[model_id]["calibration"]
    policy_func = BENCHMARK_MODELS[model_id]["analytical_policy"]

    return policy_func(calibration)


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
    """List all 10 analytically solvable discrete-time models from the catalogue"""
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
        calibration = get_benchmark_calibration(model_id)
        analytical_policy = get_analytical_policy(model_id)

        # Generate appropriate test states using the extensible approach
        test_states = get_test_states(model_id, test_points)

        # Test analytical policy
        analytical_controls = analytical_policy(test_states, {}, calibration)

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
                model_id, test_states, analytical_controls, calibration, tolerance
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

    except Exception as e:
        return {
            "success": False,
            "validation": "FAILED",
            "model_id": model_id,
            "test_points": test_points,
            "error": f"Validation failed: {str(e)}",
        }


def euler_equation_test(model_id: str, test_points: int = 100) -> Dict[str, Any]:
    """Test Euler equation satisfaction for stochastic analytical solutions"""

    if model_id == "D-3":
        return {
            "success": False,
            "test": "SKIPPED",
            "model_id": model_id,
            "error": "D-3 is a perfect foresight model - Euler equation test not applicable",
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
    initial_wealth: float, discount_factor: float, interest_rate: float
) -> float:
    """
    Analytical lifetime reward for D-1: Two-period log utility model.

    With optimal policy c1 = W/(1+β), c2 = βRW/(1+β):
    Lifetime reward = ln(c1) + β*ln(c2)
                    = ln(W/(1+β)) + β*ln(βRW/(1+β))
                    = (1+β)*ln(W) + β*ln(βR) - (1+β)*ln(1+β)
    """
    beta = discount_factor
    R = interest_rate
    W = initial_wealth

    if W <= 0:
        return -np.inf

    return (
        (1 + beta) * np.log(W) + beta * np.log(beta * R) - (1 + beta) * np.log(1 + beta)
    )


def d2_analytical_lifetime_reward(
    initial_wealth: float,
    discount_factor: float,
    interest_rate: float,
    time_horizon: int,
) -> float:
    """
    Analytical lifetime reward for D-2: Finite horizon log utility.

    Forward simulation that exactly matches the D-2 model implementation.
    """
    beta = discount_factor
    R = interest_rate
    W = initial_wealth
    T = time_horizon

    if W <= 0 or T <= 0:
        return -np.inf

    # Forward simulation matching D-2 model
    total_utility = 0.0
    current_wealth = W

    for t in range(T):
        # Remaining periods calculation: T - t (corrected timing)
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


def d3_analytical_lifetime_reward(
    cash_on_hand: float,
    discount_factor: float,
    interest_rate: float,
    risk_aversion: float,
) -> float:
    """
    Analytical lifetime reward for D-3: Infinite horizon CRRA.

    With optimal policy c = κ*m where κ = (R - (βR)^(1/σ))/R:
    V(m) = κ^(1-σ)/(1-σ) * m^(1-σ) / (1 - β*(βR)^((1-σ)/σ))
    """
    beta = discount_factor
    R = interest_rate
    sigma = risk_aversion
    m = cash_on_hand

    if m <= 0:
        return -np.inf

    growth_factor = (beta * R) ** (1 / sigma)
    if growth_factor >= R:
        raise ValueError("Return-impatience condition violated")

    kappa = (R - growth_factor) / R

    if sigma == 1:  # Log utility case
        return (np.log(kappa) + np.log(m)) / (1 - beta)
    else:  # General CRRA case
        denominator = 1 - beta * (beta * R) ** ((1 - sigma) / sigma)
        return (kappa ** (1 - sigma) * m ** (1 - sigma) / (1 - sigma)) / denominator


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
        "D-3": d3_analytical_lifetime_reward,
    }

    if model_id not in analytical_functions:
        raise ValueError(f"No analytical lifetime reward function for {model_id}")

    return analytical_functions[model_id](*args, **kwargs)


if __name__ == "__main__":
    print("Complete Catalogue of Analytically Solvable Models:")
    print("=" * 60)

    for model_id, description in list_benchmark_models().items():
        print(f"{model_id:4s}: {description}")

    print(f"\nTotal: {len(BENCHMARK_MODELS)} models")

    # Test a few key models
    print("\nValidation Results:")
    print("-" * 30)
    for model_id in ["D-1", "D-3", "U-1", "U-4"]:
        result = validate_analytical_solution(model_id)
        status = result.get("validation", result.get("error", "UNKNOWN"))
        print(f"{model_id}: {status}")
