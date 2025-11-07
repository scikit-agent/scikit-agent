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

from skagent.distributions import Normal, MeanOneLogNormal
from skagent.model import Control, DBlock
import torch
from torch import as_tensor
import numpy as np
from typing import Dict, Any, Callable


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
    "t": 0,  # Initial time period
    "description": "D-1: Finite horizon log utility",
}

d1_block = DBlock(
    **{
        "name": "d1_finite_log",
        "shocks": {},
        "dynamics": {
            "c": Control(["W", "t"], upper_bound=lambda W, t: W, agent="consumer"),
            "u": lambda c: crra_utility(c, 1.0),  # Log utility
            "W": lambda W, c, R: (W - c) * R,  # Next period wealth
            "t": lambda t: t + 1,  # Time counter
        },
        "reward": {"u": "consumer"},
    }
)


def d1_analytical_policy(states, shocks, parameters):
    """D-1: c_t = (1-β)/(1-β^(T-t)) * W_t (remaining horizon formula)"""
    beta = parameters["DiscFac"]
    if beta >= 1.0:
        raise ValueError(
            f"Discount factor must satisfy β < 1 for finite horizon model, got β={beta}"
        )
    T = parameters["T"]
    W = states["W"]
    t = states.get("t", 0)

    # Remaining horizon consumption rule
    remaining_periods = T - t

    # Terminal period: consume everything when remaining_periods <= 1
    # (remaining==1 means one period left, which is the terminal period)
    # Otherwise: c = (1-β)/(1-β^(T-t)) * W
    numerator = 1 - beta
    denominator = 1 - beta**remaining_periods

    # Infer dtype from W: use float64 for scalars, preserve tensor dtype
    W_tensor = as_tensor(W)
    dtype = torch.float64 if not isinstance(W, torch.Tensor) else W_tensor.dtype

    # Use torch.where to handle terminal period (works for both scalars and tensors)
    c_optimal = torch.where(
        as_tensor(remaining_periods <= 1),
        as_tensor(W, dtype=dtype),
        as_tensor((numerator / denominator) * W, dtype=dtype),
    )

    return {"c": c_optimal}


# D-2: Infinite Horizon CRRA (Perfect Foresight)
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
    """
    D-2: c_t = κ*W_t where κ = (R - (βR)^(1/σ))/R and W_t = m_t + H_t

    This is a proper decision function that:
    1. Takes arrival states (a), shocks, and parameters as input
    2. Computes information set variables (m) from arrival state and parameters
    3. Computes total wealth (W = m + H) including human wealth
    4. Returns optimal controls based on total wealth
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
    r = R - 1
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
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
    "liv": 1.0,  # Initial living state (agent starts alive)
    "description": "D-3: Blanchard discrete-time mortality",
}

d3_block = DBlock(
    **{
        "name": "d3_blanchard_mortality",
        "shocks": {},
        "dynamics": {
            "m": lambda a, R, y: a * R + y,
            "c": Control(["m"], upper_bound=lambda m: m, agent="consumer"),
            "a": lambda m, c: m - c,
            "liv": lambda liv, SurvivalProb: liv
            * SurvivalProb,  # Mortality probability
            "u": lambda c, liv, CRRA: liv
            * crra_utility(c, CRRA),  # Utility with survival
        },
        "reward": {"u": "consumer"},
    }
)


def d3_analytical_policy(states, shocks, parameters):
    """
    D-3: c_t = κ_s*(m_t + H) where κ_s = (R - (sβR)^(1/σ))/R

    This is a proper decision function that:
    1. Takes arrival states (a), shocks, and parameters as input
    2. Computes information set variables (m) from arrival state and parameters
    3. Returns optimal controls based on information set
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
    r = R - 1
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
    human_wealth = y / r

    # Optimal consumption: c = κ_s * (market resources + human wealth)
    c_optimal = kappa_s * (m + human_wealth)
    return {"c": c_optimal}


# Remarks: D-2 is the canonical perfect-foresight workhorse; D-3 shows that adding
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
    """
    U-1: Permanent Income Hypothesis with β*R = 1

    This is a proper decision function that implements the PIH solution:
    The agent consumes the annuity value of total wealth (financial + human).
    The martingale property E[c_{t+1}] = c_t is a consequence of this optimal policy.
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]
    y_mean = parameters["y_mean"]
    income_std = parameters.get("income_std", 0.0)

    if abs(beta * R - 1.0) > 1e-6:
        print(f"Warning: β*R = {beta * R:.6f} ≠ 1, PIH conditions may not hold exactly")

    if income_std > 0.5:
        print(
            f"Warning: With high income variance ({income_std}), verify TVC is satisfied"
        )

    # Extract arrival states
    A = states["A"]  # Financial assets (arrival state)
    y_current = states["y"]  # Current income realization (from y_mean + eta)

    # Construct information set from arrival states
    m = A * R + y_current  # Information set: cash-on-hand/market resources

    # PIH solution with βR=1: consume annuity value of TOTAL wealth
    r = R - 1
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
    human_wealth = y_mean / r
    total_wealth = m + human_wealth

    # PIH MPC out of total wealth is r/R
    mpc = r / R
    c_optimal = mpc * total_wealth

    return {"c": c_optimal}


# U-2: Log Utility with Geometric Random Walk Income (ρ=1)
# --------------------------------------------------------
# Uses standard timing convention for consistency across models.
# With log utility and permanent income following a unit root process,
# consumption is proportional to total wealth (financial + human).

# Mathematical foundation:
#   u(c) = ln c  (log utility)
#   p_t = p_{t-1} * ψ_t  (Geometric Random Walk with ρ=1)
#   Standard timing: m_t = R*A_{t-1} + p_t; A_t = m_t - c_t
#   Human wealth: H_t = p_t / r  (present value of geometric random walk)

# Key insight: ρ=1 makes the problem analytically tractable.
# Log utility homotheticity + standard timing ⟹ simple linear consumption.

# Closed-form solution (STANDARD TIMING):
#   c_t = (1-β)(m_t + H_t) = (1-β)(R*A_{t-1} + p_t + p_t/r)

# This is the standard PIH solution with geometric random walk income.

u2_calibration = {
    "DiscFac": 0.96,
    "R": 1.03,
    "rho_p": 1.0,  # ρ=1 (Geometric Random Walk) for analytical tractability
    "sigma_p": 0.05,  # Std of permanent shocks
    "description": "U-2: Log utility with geometric random walk income (ρ=1)",
}

u2_block = DBlock(
    **{
        "name": "u2_log_permanent",
        "shocks": {
            "psi": (MeanOneLogNormal, {"sigma": "sigma_p"}),  # Permanent income shock
        },
        "dynamics": {
            "p": lambda p, psi: p
            * psi,  # Geometric Random Walk: p_t = p_{t-1} * ψ_t (ρ=1)
            "m": lambda A, R, p: A * R
            + p,  # STANDARD TIMING: Cash-on-hand m_t = R*A_{t-1} + p_t
            "c": Control(
                ["m", "p"],  # Control depends on both m and p (policy uses both)
                upper_bound=lambda m, p: m,  # Budget constraint based on cash-on-hand
                agent="consumer",
            ),
            "A": lambda m, c: m
            - c,  # STANDARD TIMING: End-of-period assets A_t = m_t - c_t
            "u": lambda c: crra_utility(c, 1.0),  # Log utility (CRRA with gamma=1)
        },
        "reward": {"u": "consumer"},
    }
)


def u2_analytical_policy(states, shocks, parameters):
    """
    U-2: PIH with Geometric Random Walk Income using standard timing.

    Uses standard timing m_t = R*A_{t-1} + p_t for consistency.
    With ρ=1, income follows p_t = p_{t-1} * ψ_t (geometric random walk).
    Human wealth: H_t = p_t / r (present value of geometric random walk income).

    Standard timing analytical solution: c_t = (1-β)(m_t + H_t)
    """
    beta = parameters["DiscFac"]
    R = parameters["R"]
    rho_p = parameters["rho_p"]

    # Verify ρ=1 for analytical tractability
    if abs(rho_p - 1.0) > 1e-10:
        raise ValueError(
            f"Model requires ρ=1 for analytical tractability, got ρ={rho_p}"
        )

    # STANDARD TIMING: Use cash-on-hand from states (computed by DBlock)
    m_t = states["m"]  # Cash-on-hand m_t = R*A_{t-1} + p_t
    p_t = states["p"]  # Current permanent income level

    # Human wealth for Geometric Random Walk (ρ=1)
    r = R - 1
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
    human_wealth = p_t / r

    # Total wealth under standard timing: W_t = m_t + H_t
    total_wealth = m_t + human_wealth

    # Permanent income hypothesis: c_t = (1-β) * W_t
    mpc = 1 - beta
    c_optimal = mpc * total_wealth

    return {"c": c_optimal}


# =============================================================================
# Model Registry
# =============================================================================


def _generate_d1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-1 model: W (wealth), t (time)"""
    return {
        "W": torch.linspace(1.0, 5.0, test_points),
        "t": torch.zeros(test_points),
    }


def _generate_d23_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for D-2 and D-3 models: a (arrival assets)"""
    return {"a": torch.linspace(0.5, 4.0, test_points)}


def _generate_u1_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-1 model: A (arrival assets), y (realized income)"""
    # Simple Hall random walk - no habit formation, no c_lag
    return {
        "A": torch.linspace(0.5, 3.0, test_points),
        "y": torch.ones(test_points),  # Default income realization for testing
    }


def _generate_u2_test_states(test_points: int = 10) -> Dict[str, torch.Tensor]:
    """Generate test states for U-2 model with STANDARD TIMING: m (cash-on-hand), p (permanent income)"""
    # For standard timing, policy expects m (computed from A*R + p)
    A = torch.linspace(0.5, 3.0, test_points)
    p = torch.ones(test_points)
    # Use default R=1.03 for test state generation to avoid coupling to global calibration
    # Actual policy evaluation uses the calibration-specific R value
    R = 1.03

    return {
        "A": A,  # Keep for reference but policy uses m
        "p": p,  # Permanent income level
        "m": A * R + p,  # Cash-on-hand that policy expects under standard timing
    }


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
    r = R - 1  # Net interest rate
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )
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
        "test_states": _generate_d23_test_states,
        "custom_validation": _validate_d2_d3_solution,
    },
    "D-3": {
        "block": d3_block,
        "calibration": d3_calibration,
        "analytical_policy": d3_analytical_policy,
        "test_states": _generate_d23_test_states,
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
}


def get_benchmark_model(model_id: str) -> DBlock:
    """Get benchmark model by ID (D-1, D-2, D-3, U-1, U-2) - 5 models remain"""
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
    """List all analytically solvable discrete-time benchmark models"""
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
    r = R - 1  # Net interest rate
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )

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
    r = R - 1  # Net interest rate
    if r <= 0:
        raise ValueError(
            f"Interest rate must satisfy R > 1 for human wealth calculation, got R={R} (r={r})"
        )

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
