r"""
##################################################
Training a Policy Network Against a Known Solution
##################################################

This example demonstrates scikit-agent's neural-network training pipeline by
solving a model whose optimal policy is known in closed form, then comparing
the trained policy against that analytical solution. Validating a solver
against a known answer is the cleanest way to build trust in it before
applying it to models that have no closed form.

The model: normalized permanent-income consumption (U-2)
========================================================

The U-2 benchmark is a normalized version of the permanent-income hypothesis
(PIH) consumption-savings problem. Working in ratios to permanent income, the
state is normalized assets :math:`a`, the within-period resource is
cash-on-hand :math:`m = R a / \psi + 1`, and the control is normalized
consumption :math:`c`. The agent solves

.. math::

    V(m) = \max_{c} \; u(c) + \beta \, \mathbb{E}\!\left[ V(m') \right],

with constant-relative-risk-aversion utility :math:`u`. For this calibration
the optimal policy is linear in cash-on-hand,

.. math::

    c^*(m) = \kappa \, m,

where the marginal propensity to consume :math:`\kappa` has a closed form. This
gives us an exact target to check the trained policy against.

Why a value head: the level-identification problem
===================================================

A policy trained only on the Euler equation
:math:`u'(c_t) = \beta R \, \mathbb{E}[u'(c_{t+1})]` pins down the *slope* of
the consumption function but not its *level*: scaling the whole policy leaves
the Euler residual nearly unchanged. scikit-agent's
:class:`~skagent.ann.BlockPolicyValueNet` shares one hidden backbone between a
policy head and a value head, and trains them together under a single
optimizer with :class:`~skagent.loss.BellmanEquationLoss`. The value head
anchors the level through :math:`V(m) = u(c) + \beta \mathbb{E}[V(m')]`,
resolving the indeterminacy that Euler-only training suffers from.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
from skagent.ann import BlockPolicyValueNet, device, train_block_nn
from skagent.models.benchmarks import (
    get_analytical_policy,
    get_benchmark_calibration,
    get_benchmark_model,
)

SEED = 10077693

# %%
# Step 1: Load the U-2 benchmark and build a BellmanPeriod
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``get_benchmark_model`` returns the model block; ``get_analytical_policy``
# returns the closed-form solution we will validate against.

u2_block = get_benchmark_model("U-2")
u2_calibration = get_benchmark_calibration("U-2")
analytical_policy = get_analytical_policy("U-2")

rng = np.random.default_rng(SEED)
u2_block.construct_shocks(u2_calibration, rng=rng)

bp = bellman.BellmanPeriod(u2_block, "DiscFac", u2_calibration)

# %%
# Step 2: Build the shared-backbone policy/value network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A single :class:`~skagent.ann.BlockPolicyValueNet` carries both a (bounded)
# policy head and an unconstrained value head on top of one shared hidden
# stack. One optimizer updates all of its weights together.

torch.manual_seed(SEED)
pvnet = BlockPolicyValueNet(bp, width=32)

# %%
# Step 3: Define the Bellman-equation loss
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~skagent.loss.BellmanEquationLoss` evaluates the residual of
# :math:`V(m) = u(c) + \beta \mathbb{E}[V(m')]` using the network's own value
# head. ``foc_weight=1.0`` adds the first-order-condition term (Maliar et al.
# 2021, eq. 14), which speeds convergence.

bellman_loss_fn = loss.BellmanEquationLoss(
    bp,
    pvnet.get_value_function(),
    parameters=u2_calibration,
    foc_weight=1.0,
)

# %%
# Step 4: Train
# ^^^^^^^^^^^^^
#
# Training samples a grid of normalized assets together with two copies of the
# permanent-income shock (the "all-in-one" expectation operator from Maliar et
# al. evaluates the conditional expectation with independent shock draws).

n_pts = 15
train_grid = grid.Grid.from_dict(
    {
        "a": torch.linspace(0.5, 5.0, n_pts, device=device),
        "psi_0": torch.ones(n_pts, device=device),
        "psi_1": torch.ones(n_pts, device=device),
    }
)

trained_net, final_loss, _ = train_block_nn(
    pvnet, train_grid, bellman_loss_fn, epochs=2000, verbose=False
)
print(f"Final training loss: {final_loss:.3e}")

# %%
# Step 5: Compare the trained policy to the analytical solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We evaluate both policies on a fresh grid of normalized assets and report the
# pointwise relative error. With the value head anchoring the level, the mean
# relative error is typically a few percent.

decision_fn = trained_net.get_decision_function()

n_test = 50
test_a = torch.linspace(0.5, 5.0, n_test, device=device)
test_states = {"a": test_a}
test_shocks = {"psi": torch.ones(n_test, device=device)}

trained_c = decision_fn(test_states, test_shocks, u2_calibration)["c"].detach()
analytical_c = (
    analytical_policy(test_states, test_shocks, u2_calibration)["c"].detach().to(device)
)

rel_error = torch.abs(trained_c - analytical_c) / (analytical_c + 1e-8)
print(f"Mean relative error:  {rel_error.mean().item():.2%}")
print(f"Max  relative error:  {rel_error.max().item():.2%}")

# %%
# Step 6: Plot trained vs analytical policy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The trained consumption function tracks the closed-form linear PIH rule
# across the tested range of normalized assets. The lower panel shows the
# pointwise relative error.

a_np = test_a.cpu().numpy()
trained_np = trained_c.cpu().numpy()
analytical_np = analytical_c.cpu().numpy()
rel_error_np = rel_error.cpu().numpy()

fig, (ax_policy, ax_err) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, height_ratios=[3, 1]
)

ax_policy.plot(a_np, analytical_np, "k-", linewidth=2, label="Analytical $c^*(m)$")
ax_policy.plot(a_np, trained_np, "C1--", linewidth=2, label="Trained policy network")
ax_policy.set_ylabel("Normalized consumption $c$")
ax_policy.set_title("Trained policy vs. analytical PIH solution (U-2)")
ax_policy.legend()
ax_policy.grid(True, alpha=0.3)

ax_err.plot(a_np, rel_error_np * 100.0, "C3-", linewidth=1.5)
ax_err.set_xlabel("Normalized assets $a$")
ax_err.set_ylabel("Rel. error (%)")
ax_err.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
