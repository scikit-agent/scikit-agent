r"""
###############################################
Cournot: Solving for a Nash Equilibrium
###############################################

The Tree Killer example solves a game by *decomposing* it: its decisions fall
into an order, and each one can be settled once the decisions it relies on have
been settled. This example concerns a game where that approach fails.

Several firms each choose how much to produce. The market price falls with the
average quantity supplied, so a firm's best output depends on what the other
firms choose, and their best outputs depend in turn on its choice. There is no
first decision. Instead, the game has a **fixed point**, a quantity that is its
own best response; that quantity is the Cournot-Nash equilibrium.

This page does four things:

1. states the model and the three quantities worth knowing about it,
2. **projects** the population onto one firm and its rivals,
3. iterates best responses to the equilibrium, and
4. shows why the iteration has to be **damped**, by watching it fail without it.

The Model
==========

:math:`N` firms each draw a marginal cost :math:`c_i` and choose a quantity
:math:`q_i`. The market clears on the *average* quantity:

.. math::
    Q = \frac{1}{N}\sum_i q_i, \qquad P = A - b\,Q, \qquad
    u_i = (P - c_i)\, q_i

The model is static, since it has no arrival states, so solving it describes
one play of a one-shot game.
"""

# %%
import numpy as np

import skagent.models.cournot as cournot
from skagent.ground import GroundedBlock
from skagent.solver import ExactBestResponse, project, solve_symmetric_equilibrium

COST = 4.0
market = GroundedBlock(cournot.cournot_block, cournot.collusion_calibration(size=3))

# %%
# Three profiles, and why the equilibrium is not the good one
# =============================================================
#
# The model ships three hand-derived profiles. They are the prisoner's dilemma
# in disguise, which is what makes the equilibrium worth computing rather than
# guessing, because the outcome the firms reach is not the outcome they would
# prefer.

for label, quantities in cournot.PROFILES.items():
    print(f"{label:16s} {quantities}")

print()
print(f"joint monopoly, per firm : {cournot.monopoly_quantity()}  (pays 9.0 each)")
print(f"Cournot-Nash, per firm   : {cournot.nash_quantity()}  (pays 6.75 each)")
print("one firm deviating       : 6.0 against 3.0, and it pays 12.0, at the")
print("                           expense of the other two, which fall to 6.0")

# %%
# Colluding at 3.0 pays every firm more than the equilibrium does, but it is not
# stable, because any single firm gains by producing more. That is why 4.5 is
# where the market lands. The equilibrium is a prediction rather than a
# recommendation.
#
# Projecting the population
# ===========================
#
# A solver solves *one* decision. This model describes three firms at once, so
# something has to turn the question "what should the firms do?" into the
# question "what should *this* firm do, given what the others do?" That is the
# job of :func:`~skagent.solver.project`.

projected = project(market)

print("controls :", list(projected.block.get_controls()))
print("dynamics :", list(projected.block.get_dynamics()))

# %%
# The class has been split into the firm being solved (``_actor``) and the rest
# of the firms (``_other``), and one equation has been synthesized. That
# equation, ``q``, concatenates the two sides back into the population the
# market reads.
#
# That is the whole trick, and it is worth being precise about what it avoids.
# The projection did **not** rewrite ``Q = q.mean()`` into a formula about one
# firm and :math:`N-1` others. It reassembled the population and let the model's
# own equation run on it unchanged. So the solved firm's own share of the
# aggregate, which is its :math:`1/N` of the average, is there by construction
# rather than being computed, and a model that had written ``q.sum()`` or a
# maximum would project just as well without the library knowing which reduction
# was used.
#
# We can check the projection against the profiles above before solving
# anything: put the deviating firm at 6.0 with its rivals at 3.0, and it should
# earn the 12.0 the table promises.

values = projected.block.transition(
    {**projected.calibration, "c_actor": COST, "c_other": COST},
    {"q_actor": lambda c_actor: 6.0, "q_other": lambda c_other: 3.0},
)
payoff = projected.block.calc_reward(values, agent="firm_actor")["u_actor"]
print(f"deviator's payoff: {float(np.atleast_1d(payoff)[0])}")

# %%
# Iterating to the equilibrium
# ==============================
#
# We now come to the fixed point.
# :func:`~skagent.solver.solve_symmetric_equilibrium` solves the projected
# firm's decision against the rivals' current rule, substitutes the answer back
# in as the rivals' rule, and repeats until the rule stops moving.
#
# The method supplying each solve is an exact backup here. A policy network
# would serve just as well, and it reaches the same answer.


def cournot_method(size):
    ground = GroundedBlock(
        cournot.cournot_block, cournot.collusion_calibration(size=size)
    )
    projected = project(ground)
    return ExactBestResponse(
        projected,
        {"c_actor": np.array([COST])},
        scope={**projected.calibration, "c_other": COST},
    )


def quantity(rule):
    return float(np.atleast_1d(rule(np.array([COST]))).ravel()[0])


for size in (2, 3, 4):
    rule, info = solve_symmetric_equilibrium(
        cournot_method(size), damping=2.0 / (size + 1)
    )
    print(
        f"{size} firms: q* = {quantity(rule):.4f}   "
        f"analytic {cournot.nash_quantity(size=size)}   "
        f"({info['iterations']} iterations)"
    )

# %%
# Why the iteration is damped
# =============================
#
# ``damping`` moves the rule only part of the way toward the best response each
# round:
#
# .. math::
#     q_{k+1} = (1 - L)\, q_k + L \cdot \mathrm{BR}(q_k)
#
# Damping cannot change the answer, because a rule that equals its own damped
# update also equals its own best response, so it is not a tuning control for
# accuracy. It is what makes the iteration arrive at all.
#
# A Cournot best response *slopes down*: if the rivals produce more, this firm
# produces less, with slope :math:`-(N-1)/2`. Beyond two firms that response
# overshoots, and the undamped iteration does not settle. The runs below show it
# failing:

for size, cap in ((2, 30), (3, 30), (4, 8)):
    _, info = solve_symmetric_equilibrium(
        cournot_method(size), damping=1.0, max_iterations=cap
    )
    moves = [round(d, 2) for d in info["distances"][:6]]
    print(f"{size} firms undamped: converged={info['converged']}  first moves {moves}")

# %%
# There are three different failures here, and only the first one is benign:
#
# - **Two firms**: the slope is :math:`-0.5`, a contraction. It does converge,
#   halving its step each round, but it takes 14 iterations where the damped
#   iteration takes 2.
# - **Three firms**: the slope is exactly :math:`-1`. The iteration does not
#   diverge; it **cycles**, alternating between two quantities forever. Every
#   round moves by the same 9.0, and no iteration cap will change that.
# - **Four firms**: the slope is :math:`-1.5`. The moves grow, and the iterates
#   run off to whatever bounds the control declares.
#
# The middle case is the one to remember. A cycle returns a perfectly plausible
# quantity if it is stopped on a count of iterations, which is why the schedule
# tests whether the *rule* has stopped moving and reports ``converged: False``
# rather than handing back its last iterate.
#
# .. note::
#    :math:`L = 2/(N+1)` is the damping that zeroes the slope for this model,
#    which is why the runs above converge in a single step. That closed form
#    exists because Cournot's best response is linear; in general a damping
#    factor is chosen to make the iteration a contraction, not to solve it
#    instantly.
