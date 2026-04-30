r"""
##############################
Benchmark Consumption Models
##############################

A guided tour of :py:mod:`skagent.models.benchmarks`. The closed-form
policies in this registry retrace dynamic consumption theory in roughly the
order the field discovered them, and each model teaches a fact that the
previous ones could not. Reading from top to bottom:

#. **D-1, D-2.** Finite horizons stop mattering once :math:`T - t` is far
   enough from the present.
#. **D-3.** Mortality acts like extra impatience: it scales the discount
   factor and pushes up the MPC.
#. **U-1.** Under :math:`\beta R = 1`, the *change* in consumption is the
   fundamental object, not its level. Income shocks of standard deviation
   :math:`\sigma_\eta` produce consumption changes of standard deviation
   :math:`(r/R)\,\sigma_\eta` only, a factor of roughly 30× smaller.
#. **U-2.** Dividing every level variable by permanent income collapses a
   2-D Bellman to a 1-D one. This trick is what makes neural-network
   solvers practical for richer models.
#. **U-3.** What happens when no normalization saves you, and how the
   registry still keeps the model around for limit-checking.

This example is the runnable companion to :doc:`/user_guide/benchmark_models`.
The code is intentionally verbose; production code should compose helpers,
but here every step is written out so the reader can follow the algebra.

References
==========

* Hall, R.E. (1978). Stochastic implications of the life cycle-permanent
  income hypothesis. *Journal of Political Economy* 86(6), 971-987.
* Blanchard, O.J. (1985). Debt, deficits, and finite horizons. *Journal of
  Political Economy* 93(2), 223-247.
* Carroll, C.D. (2024). *Solution Methods for Solving Microeconomic Dynamic
  Stochastic Optimization Problems*. https://llorracc.github.io/SolvingMicroDSOPs/

"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from skagent.models.benchmarks import (
    EPS_VALIDATION,
    get_analytical_policy,
    get_benchmark_calibration,
    list_benchmark_models,
    validate_analytical_solution,
)

# %%
# Step 1: What's in the Registry
# ------------------------------
#
# Five of the six entries carry an analytical policy that
# ``validate_analytical_solution`` checks against the standard test grid.
# U-3 is registered without one because the borrowing constraint plus
# uncertainty break the linearity that every other entry relies on; the
# helper still reports it as ``FAILED`` because no policy was found, not
# because anything is wrong with the model.


def _has_closed_form(model_id: str) -> bool:
    """Check whether the registry exposes an analytical policy for ``model_id``."""
    try:
        get_analytical_policy(model_id)
    except ValueError:
        return False
    return True


for model_id, description in list_benchmark_models().items():
    marker = "closed form" if _has_closed_form(model_id) else "numerical only"
    result = validate_analytical_solution(
        model_id, test_points=20, tolerance=EPS_VALIDATION
    )
    print(f"  {model_id} ({marker:14s}): {result['validation']:6s}  {description}")

# %%
# Step 2: Finite Horizons Fade Away (D-1 → D-2)
# ---------------------------------------------
#
# **Lesson:** A 30-year-old human has an *almost* infinite-horizon MPC.
# That is why infinite-horizon models survive as a baseline despite being
# obviously unrealistic.
#
# D-1's remaining-horizon MPC is :math:`(1-\beta) / (1 - \beta^{T-t})`. It
# is above the infinite-horizon constant :math:`(1-\beta)` for any finite
# horizon and decays geometrically as :math:`T - t \to \infty`. Holding
# wealth fixed at :math:`W = 5` and sweeping :math:`T`, the gap is already
# below 1% of the limit by :math:`T = 30`.

beta_d1 = get_benchmark_calibration("D-1")["DiscFac"]
W_fixed = 5.0
horizons = np.arange(1, 81)
# c_t = (1-β) / (1 - β^{T-t}) * W_t is the D-1 closed form (see
# ``d1_analytical_policy``). Sweeping T at t = 0 reduces to a single
# vectorized expression; the registry policy is exercised in Steps 1, 3,
# 4, and 6 below.
c_finite = (1 - beta_d1) / (1 - beta_d1**horizons) * W_fixed
c_infinite = (1 - beta_d1) * W_fixed

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(horizons, c_finite, label="D-1 (finite horizon)", linewidth=2)
ax.axhline(
    c_infinite,
    linestyle="--",
    color="C1",
    linewidth=2,
    label=rf"D-2 limit $(1-\beta)\, W = {c_infinite:.2f}$",
)
ax.set_xlabel("Horizon $T$", fontsize=11)
ax.set_ylabel("Optimal consumption at $W = 5$", fontsize=11)
ax.set_title(
    "Finite-Horizon Consumption Decays to the Infinite-Horizon Rule", fontsize=12
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

# %%
# Step 3: Mortality Erodes Patience (D-3)
# ---------------------------------------
#
# **Lesson:** I.i.d. mortality risk :math:`s` is observationally equivalent
# to scaling the discount factor from :math:`\beta` to :math:`s\beta`. The
# MPC :math:`\kappa_s = (R - (s\beta R)^{1/\sigma})/R` strictly exceeds the
# no-mortality MPC :math:`\kappa`, and the wedge widens as :math:`s` falls.
#
# Sweeping :math:`s \in \{1.0, 0.95, 0.9, 0.8, 0.7\}` (median lifetime
# falling from infinity to about 3 periods) makes the wedge visible. At
# :math:`s = 0.7` the agent consumes nearly six times more per dollar of
# total wealth than at :math:`s = 1`. Empirical annual mortality at age 30
# is around :math:`s = 0.999`, which is essentially indistinguishable from
# the no-mortality limit at this scale, but life-cycle models that
# aggregate over many decades pick up a measurable mortality wedge.

shared = get_benchmark_calibration("D-2")
a_grid = torch.linspace(0.0, 6.0, 121)
m_grid_np = (a_grid * shared["R"] + shared["y"]).numpy()


def _kappa(beta_eff: float) -> float:
    """MPC out of total wealth at effective discount factor ``beta_eff``."""
    return (shared["R"] - (beta_eff * shared["R"]) ** (1 / shared["CRRA"])) / shared[
        "R"
    ]


fig, ax = plt.subplots(figsize=(8, 5))
for s in [1.0, 0.95, 0.9, 0.8, 0.7]:
    if s == 1.0:
        c = get_analytical_policy("D-2")({"a": a_grid}, {}, shared)["c"]
        label = rf"D-2: $s = 1.00$, $\kappa\;\,= {_kappa(shared['DiscFac']):.4f}$"
    else:
        c = get_analytical_policy("D-3")(
            {"a": a_grid}, {}, {**shared, "SurvivalProb": s}
        )["c"]
        label = rf"D-3: $s = {s:.2f}$, $\kappa_s = {_kappa(s * shared['DiscFac']):.4f}$"
    ax.plot(m_grid_np, c.numpy(), label=label, linewidth=2)

ax.set_xlabel("Cash-on-hand $m_t$", fontsize=11)
ax.set_ylabel("Optimal consumption $c_t$", fontsize=11)
ax.set_title("Mortality Risk Visibly Increases the MPC", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

# %%
# Step 4: Hall's Martingale (U-1)
# -------------------------------
#
# **Lesson:** Hall's contribution wasn't the level of consumption. It was
# the prediction that, under :math:`\beta R = 1`, consumption changes are
# unforecastable from period-:math:`t` information, and that the standard
# deviation of those changes is *much smaller* than the standard deviation
# of income.
#
# We simulate U-1 forward with 1000 agents under the analytical policy. The
# left panel shows one agent's income (volatile, mean-reverting) against
# her consumption (smooth, slowly drifting): the textbook image of
# consumption smoothing. The right panel overlays the histograms of income
# innovations :math:`\eta_t` and consumption changes :math:`\Delta c_t`.
# Both are mean-zero, but :math:`\Delta c_t` is concentrated near zero
# while :math:`\eta_t` spreads out by a factor of about :math:`R/r \approx
# 34`. The closed-form prediction :math:`\sigma_{\Delta c} =
# (r/R)\,\sigma_\eta` drops out exactly because the agent has substituted
# saving for consumption volatility. Empirical PIH tests are precisely
# tests of whether this picture survives in real data.

torch.manual_seed(42)
n_agents = 1000
T_sim = 60
calib_u1 = get_benchmark_calibration("U-1")
beta_u1 = calib_u1["DiscFac"]
R_u1 = calib_u1["R"]
sigma_eta = calib_u1["income_std"]
y_mean = calib_u1["y_mean"]
r_u1 = R_u1 - 1
H_u1 = y_mean / r_u1

A_state = torch.zeros(n_agents)
c_paths = torch.zeros(T_sim, n_agents)
y_paths = torch.zeros(T_sim, n_agents)
for t in range(T_sim):
    eta = sigma_eta * torch.randn(n_agents)
    y = y_mean + eta
    m = R_u1 * A_state + y
    c = (r_u1 / R_u1) * (m + H_u1)
    A_state = m - c
    c_paths[t] = c
    y_paths[t] = y

dc = (c_paths[1:] - c_paths[:-1]).flatten()
eta_realized = (y_paths - y_mean).flatten()
theoretical_dc_std = (r_u1 / R_u1) * sigma_eta
empirical_dc_std = float(dc.std())
empirical_eta_std = float(eta_realized.std())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    y_paths[:, 0].numpy(),
    label=rf"income $y_t$ (std $\approx {empirical_eta_std:.3f}$)",
    color="C0",
    alpha=0.7,
    linewidth=1.5,
)
axes[0].plot(
    c_paths[:, 0].numpy(),
    label=rf"consumption $c_t$ (std $\approx {float(c_paths[:, 0].std()):.3f}$)",
    color="C3",
    linewidth=2,
)
axes[0].axhline(y_mean, linestyle=":", color="k", alpha=0.5, linewidth=1)
axes[0].set_xlabel("Period $t$", fontsize=11)
axes[0].set_ylabel("Level", fontsize=11)
axes[0].set_title("One Agent: Income Volatile, Consumption Smooth", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

bins = np.linspace(-0.4, 0.4, 60)
axes[1].hist(
    eta_realized.numpy(),
    bins=bins,
    density=True,
    color="C0",
    alpha=0.5,
    label=rf"income shock $\eta_t$, std$\,= {empirical_eta_std:.3f}$",
)
axes[1].hist(
    dc.numpy(),
    bins=bins,
    density=True,
    color="C3",
    alpha=0.8,
    label=rf"$\Delta c_t$, std$\,= {empirical_dc_std:.4f}$",
)
axes[1].axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
axes[1].set_xlabel("Innovation magnitude", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)
axes[1].set_title(
    rf"Theory: $\sigma_{{\Delta c}} = (r/R)\,\sigma_\eta = {theoretical_dc_std:.4f}$",
    fontsize=12,
)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()

# %%
# Step 5: Normalization Collapses the State Space (U-2)
# -----------------------------------------------------
#
# **Lesson:** A clever change of variables can turn an :math:`n`-dimensional
# state space into an :math:`(n-1)`-dimensional one. For U-2 the trick is
# dividing every level by permanent income :math:`P_t`, which removes
# :math:`P_t` from the state entirely. A neural-network solver that would
# have needed to learn a 2-D function :math:`C(M, P)` now only has to
# learn a 1-D function :math:`c(m)`, an enormous reduction in sample
# complexity.
#
# Left panel: the level rule :math:`C = (1-\beta)(M + P/r)` is a *family*
# of parallel lines, one per :math:`P`. Right panel: the same four
# policies in normalized variables :math:`(m, c) = (M/P, C/P)` collapse
# onto the single curve :math:`c = (1-\beta)(m + 1/r)`. The four
# line-styles all trace the same curve; the visual coincidence is the
# state-space collapse.

calib_u2 = get_benchmark_calibration("U-2")
beta_u2 = calib_u2["DiscFac"]
R_u2 = calib_u2["R"]
r_u2 = R_u2 - 1

P_values = [0.5, 1.0, 1.5, 2.0]
linestyles = ["-", "--", ":", "-."]
M_grid = torch.linspace(0.0, 4.0, 81)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for P, ls in zip(P_values, linestyles):
    C = (1 - beta_u2) * (M_grid + P / r_u2)
    axes[0].plot(
        M_grid.numpy(), C.numpy(), label=f"$P = {P}$", linewidth=2, linestyle=ls
    )
axes[0].set_xlabel("Level cash-on-hand $M$", fontsize=11)
axes[0].set_ylabel("Level consumption $C$", fontsize=11)
axes[0].set_title("Level Variables: 2-D Family", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

for P, ls in zip(P_values, linestyles):
    m = M_grid / P
    c = (1 - beta_u2) * (m + 1 / r_u2)
    axes[1].plot(m.numpy(), c.numpy(), label=f"$P = {P}$", linewidth=2, linestyle=ls)
axes[1].set_xlabel("Normalized cash-on-hand $m = M/P$", fontsize=11)
axes[1].set_ylabel("Normalized consumption $c = C/P$", fontsize=11)
axes[1].set_title("Normalized: All Four Lines Coincide", fontsize=12)
axes[1].legend(fontsize=10, title="all overlap")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()

# %%
# Step 6: When the Closed Form Breaks (U-3)
# -----------------------------------------
#
# **Lesson:** U-3 is U-2 plus a *binding* borrowing constraint
# :math:`c \leq m`. The U-2 closed form still satisfies the Euler equation
# everywhere — but it violates the constraint at low :math:`m`, because at
# :math:`m = 0` it prescribes :math:`c = (1-\beta)/r > 0` (the agent
# wants to borrow against future income). Below the intersection of the
# PIH line with the constraint :math:`c = m`, the U-2 policy is
# infeasible. Above the intersection, it is feasible but suboptimal,
# because the U-3 agent has *precautionary* saving motives that U-2 lacks.
# The actual U-3 policy bends below the PIH line at moderate :math:`m`
# and approaches D-2's :math:`\kappa` only as :math:`m \to \infty`.
# Neither bend nor approach has a closed form — that is exactly why U-3
# is in the registry as ``"numerical only"``.

m_u3 = torch.linspace(0.0, 4.0, 81)
c_pih = (1 - beta_u2) * (m_u3 + 1 / r_u2)
intersect_m = (1 - beta_u2) / (r_u2 * beta_u2)  # solves (1-β)(m+1/r) = m

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(
    m_u3.numpy(),
    c_pih.numpy(),
    label=r"U-2 PIH: $c = (1-\beta)(m + 1/r)$",
    linewidth=2,
)
ax.plot(
    m_u3.numpy(),
    m_u3.numpy(),
    label=r"Constraint: $c = m$",
    linewidth=2,
    linestyle="--",
    color="k",
)
ax.fill_between(
    m_u3.numpy(),
    c_pih.numpy(),
    m_u3.numpy(),
    where=(c_pih > m_u3).numpy(),
    color="red",
    alpha=0.18,
    label=r"PIH infeasible: $c > m$",
)
ax.axvline(
    intersect_m,
    color="grey",
    linestyle=":",
    linewidth=1.2,
    label=rf"intersection $m = (1-\beta)/(r\beta) \approx {intersect_m:.2f}$",
)
ax.set_xlabel("Normalized cash-on-hand $m$", fontsize=11)
ax.set_ylabel("Normalized consumption $c$", fontsize=11)
ax.set_title("U-3: PIH Violates the Borrowing Constraint at Low Wealth", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3.5)
fig.tight_layout()

try:
    get_analytical_policy("U-3")
except ValueError as exc:
    print("U-3 has no analytical policy by design:")
    print(f"  {exc}")

# %%
# Where to Read Next
# ------------------
#
# - :doc:`/user_guide/benchmark_models` for the per-model derivations,
#   notation conventions, and the standalone modules (Fisher, perfect
#   foresight, resource extraction).
# - :doc:`/user_guide/algorithms` covers the numerical solvers used for
#   models like U-3 that have no closed form.
# - The other examples in this gallery wire benchmark blocks into Monte
#   Carlo simulators and reinforcement learning loops.

plt.show()
