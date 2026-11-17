r"""
========================================
Agentic Intention: A Counterfactual Test
========================================

When an outcome is produced by an action, it doesn’t necessarily mean that it was
intended. For example, a man plans to kill his uncle but, while driving to his uncle’s
house, accidentally runs over a pedestrian who turns out to be his uncle. If the man did
not intend to hit the pedestrian and did not know that the pedestrian was his uncle, can
we rightly say that he intended to kill his uncle through that collision?

This example motivates Ward et al.’s (2024) framework for attributing intention to
agents. [1]_ We test its conditions through short causal examples. The
:ref:`content recommendation case study <sphx_glr_auto_examples_models_plot_agentic_intention_2.py>`
then compares a larger set of policies and settings.

The companion page,
:ref:`sphx_glr_auto_examples_models_plot_agentic_intention_3.py`,
extends this framework to a learned agent and examines what an observer can
infer about intention from incomplete information.

References
==========

.. [1] Ward, F. R., MacDermott, M., Belardinelli, F., Toni, F., and Everitt, T. (2024).
   “The Reasons that Agents Act: Intention and Instrumental Goals.” In *Proceedings of
   the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS
   2024)*, Auckland, New Zealand, pp. 1901–1909. IFAAMAS. :arxiv:`2402.07221`.

.. [2] Ashton, H. (2023). “Definitions of intent suitable for algorithms.” *Artificial
   Intelligence and Law* 31, 515–546. :doi:`10.1007/s10506-022-09322-x`.

"""

# %%
# Setup
# =====

from collections.abc import Mapping
from itertools import product
from typing import Any

from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.ground import GroundedBlock
from skagent.intention import (
    compare_outcome_sets,
    compare_references,
    contextual_payoff,
)
from skagent.utils import plot_block_diagram

# sphinx_gallery_thumbnail_number = 1
# %%
# Intention tests
# ===============
#
# Garage fire: separating a reason from a side effect
# ---------------------------------------------------
#
# Bob burns his garage to collect insurance, knowing that Alice’s car will also be
# destroyed. The car’s destruction has no effect on the payout. Here, :math:`D=1` means
# proceeding with Bob’s plan and :math:`D=0` means doing nothing.
#
# We use :class:`~skagent.block.DBlock` to specify the causal dependencies
# and :class:`~skagent.block.Control` to represent Bob’s decision.

garage = DBlock(
    dynamics={
        "D": Control([], agent="bob"),
        "Insurance": lambda D: D,
        "Car": lambda D: D,
        "U": lambda Insurance: 3 * Insurance,
    },
    reward={"U": "bob"},
)

plot_block_diagram(garage, "Garage fire", figsize=(8, 3))

# %%
# Ward et al. give the first intention test as follows:
#
#    Definition 1 (Intention – Operationalisation I). An agent intended to cause an outcome
#    𝑜 with its action 𝑎, if guaranteeing that another action 𝑎′ also caused 𝑜 would make
#    𝑎′ just as good for the agent. [1]_
#
# :func:`~skagent.intention.contextual_payoff` tests the alternative while fixing
# an outcome to its original value. :class:`~skagent.ground.GroundedBlock` pairs
# the model with its calibration. This model has no shocks, so only one setting.


# The fixed key lets both shock-free examples use the same guarantee setting.
def one_setting(_shocks: Mapping[str, Any]) -> str:
    """Name the only setting in a model without shocks."""
    return "only"


burn = {"D": lambda: 1}
leave = {"D": lambda: 0}
garage_ground = GroundedBlock(garage, {})

for outcome in ("Insurance", "Car"):
    payoff = contextual_payoff(
        garage_ground, burn, leave, outcome, {"only"}, one_setting, agent="bob"
    )
    print(f"Guarantee {outcome}: {float(payoff):.1f}")

# %%
# Both insurance and the car’s destruction follow from burning, but only insurance
# contributes to Bob’s utility. By the above definition, it follows that the incidental
# destruction of Alice’s car was not intended, despite Bob being aware that it would be
# a consequence of burning the garage. Guaranteeing the payout without burning would
# remove Bob’s reason to choose burning over the alternative.
#
# This is the paper's distinction between direct intention, which requires a desired
# effect, and indirect intention, which can include a foreseeable side effect. [1]_
#
# Remodeling the garage: instrumental intention
# ---------------------------------------------
#
# Suppose Bob must first acquire gas to burn the garage. Here :math:`D=1` represents
# choosing the whole plan, including the later fire. Following the nodal structure
# in the original paper, the diagram treats acquiring gas and burning the garage as
# intermediate outcomes, rather than separate decisions. [1]_
# Gas is useful because it enables the payout, not because it pays Bob directly.

garage = DBlock(
    dynamics={
        "D": Control([], agent="bob"),
        "Gas": lambda D: D,
        "GarageBurned": lambda Gas: Gas,
        "Insurance": lambda GarageBurned: GarageBurned,
        "Car": lambda GarageBurned: GarageBurned,
        "U": lambda Insurance: 3 * Insurance,
    },
    reward={"U": "bob"},
)

plot_block_diagram(
    garage, "Garage fire: gas as an instrumental outcome", figsize=(8, 3)
)

garage_ground = GroundedBlock(garage, {})
for label, guaranteed in (("No gas", ()), ("Guarantee Gas", ("only",))):
    payoff = contextual_payoff(
        garage_ground, burn, leave, "Gas", guaranteed, one_setting, agent="bob"
    )
    print(f"{label}: {float(payoff):.1f}")

# %%
# Acquiring gas is instrumental to obtaining the payout: without it, Bob cannot carry
# out the fire that produces the insurance payment. The intermediate outcome can
# therefore be intended even though it has no value to Bob on its own.
#
# It is in this way that the authors invoke Ashton’s notion of *means-end consistency*
# as desideratum for intent [2]_. In their words:
#
#    “if an agent intends some final outcome, then any intermediary outcomes which are
#    instrumentally useful in achieving the final goal are also intended.” [1]_
#
# Adding subjective belief
# ------------------------
#
# Now we will circle back to the example in the introductory section to add *subjective
# belief* into the mix with Ward et al’s Example 4 [1]_
#
# Louis plans to kill his uncle. On the way, he accidentally runs over a pedestrian who
# turns out to be that uncle. In his understanding of events, running over the
# pedestrian does not accomplish his plan. Thus, in his subjective model he did not
# intend to kill his uncle by running over the pedestrian.
#
# Here :math:`D=1` means hitting the pedestrian. Louis believes this leaves his uncle
# alive; in the actual case, the pedestrian is his uncle. The diagrams show the two
# causal models without evaluating the intention test numerically.

louis_subjective = DBlock(
    dynamics={
        "D": Control([], agent="louis"),
        "PedestrianDead": lambda D: D,
        "UncleDead": lambda: 0,
        "U": lambda UncleDead: UncleDead,
    },
    reward={"U": "louis"},
)

louis_objective = DBlock(
    dynamics={
        "D": Control([], agent="louis"),
        "PedestrianDead": lambda D: D,
        "UncleDead": lambda PedestrianDead: PedestrianDead,
        "U": lambda UncleDead: UncleDead,
    },
    reward={"U": "louis"},
)

plot_block_diagram(louis_subjective, "Louis's Subjective Belief", figsize=(8, 3))
plot_block_diagram(louis_objective, "Objective Model", figsize=(8, 3))

# %%
# Leaning on Ashton’s definitions, the authors use this example to show that *subjective
# belief* is necessary for assigning instrumental agency:
#
#    “Ashton’s desiderata require the agent to foresee that its action can cause the
#    intended outcome. Therefore, the agent’s beliefs about the world are a determining
#    factor when assessing intent.”
#
# In other words, an outcome can achieve the utility without the agent intentionally
# bringing the outcome about. Conversely, if Louis were to have confused an unknown
# pedestrian for his uncle, and deliberately ran the pedestrian over as a result of that
# belief, then he would have intended to kill the pedestrian.
#
# Several reasons together
# ------------------------
#
# So far, a single outcome has been enough to test a proposed reason. The next example
# shows why that is insufficient without adding additional conditions.
#
# In Ward et al.’s Example 5 [1]_, Alice donates to charity because she values helping
# people, her reputation, and a tax deduction. These are her desired outcomes.
# Incidentally, she also receives a newsletter for donating.
#
# The model calls these outcomes Helping, Reputation, Taxes, and Newsletter,
# respectively; :math:`D=1` means donating.

donation = DBlock(
    dynamics={
        "D": Control([], agent="alice"),
        "Helping": lambda D: D,
        "Reputation": lambda D: D,
        "Taxes": lambda D: D,
        "Newsletter": lambda D: D,
        "Cost": lambda D: 1.5 * D,
        "U": lambda Helping, Reputation, Taxes, Cost: Helping
        + Reputation
        + Taxes
        - Cost,
    },
    reward={"U": "alice"},
)

plot_block_diagram(donation, "Charity donation", figsize=(7, 6))

# %%
# Alice believes the cost of donating is worth it if any two of her three desired
# outcomes happen.
#
#    Definition 2 (Intention – Operationalisation II). An agent intended to cause
#    outcomes 𝒐 with action 𝑎, if there exists a super-set of outcomes 𝒚 ⊇ 𝒐 such that
#    guaranteeing another action 𝑎′ also caused 𝒚 would make 𝑎′ just as good, and no
#    proper subset of 𝒚 makes 𝑎′ as good as 𝑎. [1]_
#
# :func:`~skagent.intention.compare_outcome_sets` finds the minimal sufficient
# outcome groups. We include the newsletter to test whether it belongs to any.
# Select the skip policy's result from the comparison keyed by reference name.
minimal_outcomes = compare_outcome_sets(
    GroundedBlock(donation, {}),
    {"D": lambda: 1},
    {"skip": {"D": lambda: 0}},
    ("Helping", "Reputation", "Taxes", "Newsletter"),
    {"only"},
    one_setting,
    agent="alice",
)["skip"]

print("Minimal outcome sets:", minimal_outcomes)

# %%
# Donating yields utility :math:`1.5` . Without donating, guaranteeing one benefit
# yields only :math:`1` , whereas guaranteeing any two yields :math:`2` . Each pair
# therefore makes the alternative at least as good, while neither benefit alone
# suffices.
#
# The three minimal sets are :math:`\{Helping, Reputation\}` ,
# :math:`\{Helping, Taxes\}` , and :math:`\{Reputation, Taxes\}` . Each desired outcome
# belongs to at least one minimal set, so each is individually intended under this test.
# In contrast, the newsletter appears in no minimal set. Adding the newsletter to any
# sufficient pair does not improve the alternative’s payoff, so it is excluded by
# minimality.
#
# The spy: intention in particular settings
# -----------------------------------------
#
# For our final refinement, we also need to distinguish the settings in which an outcome
# depends on the agent’s action from those in which random environmental factors might
# incidentally produce the outcome.
#
# In Ward et al.’s Example 6 [1]_, a double agent signals the direction of a minefield
# to an enemy submarine, in the hopes of guiding the submarine towards the minefield.
# The minefield may be East or West. If the submarine receives the signal, it follows
# it. Otherwise, it chooses a direction randomly.
#
# A setting fixes the relevant external conditions. In this case, the setting fixes
# where the minefield is, whether the signal is received, and which direction the
# submarine would choose without a signal.
# Here, ``shocks`` gives the exogenous variables their distributions, ``dynamics``
# defines the spy’s decision and the structural functions, and ``reward`` assigns
# utility to the spy.

spy = DBlock(
    shocks={
        name: (Bernoulli, {"p": 0.5})
        for name in ("MinefieldDirection", "Received", "Random")
    },
    dynamics={
        "D": Control(["MinefieldDirection"], agent="spy"),
        "SubmarineDirection": lambda D, Received, Random: D if Received else Random,
        "U": lambda SubmarineDirection, MinefieldDirection: int(
            SubmarineDirection == MinefieldDirection
        ),
    },
    reward={"U": "spy"},
)

plot_block_diagram(spy, "Spy signal", figsize=(8, 4))

# %%
# :func:`~skagent.intention.compare_references` finds the minimal sets of settings
# where guaranteeing the outcome makes another policy as good. Here a setting
# records the minefield, reception, and random fallback.
# Keep this order aligned with the eight settings generated by product below.

shock_names = ("MinefieldDirection", "Received", "Random")


def setting_of(shocks: Mapping[str, Any]) -> tuple[int, ...]:
    """Identify a minefield, reception, and fallback setting."""
    return tuple(int(shocks[name]) for name in shock_names)


signal_minefield = {"D": lambda MinefieldDirection: int(MinefieldDirection)}
signal_opposite = {"D": lambda MinefieldDirection: 1 - int(MinefieldDirection)}

# Because the first value is the opposite policy's payoff, only minimal settings matter here
_, minimal_settings = compare_references(
    GroundedBlock(spy, {}),
    signal_minefield,
    {"opposite": signal_opposite},
    "SubmarineDirection",
    product((0, 1), repeat=3),
    setting_of,
    agent="spy",
)["opposite"]

print("Minimal spy settings:", minimal_settings)

# %%
# A received signal and a lucky random choice can produce the same outcome. The
# matching signal earns :math:`0.75` on average and the opposite signal earns
# :math:`0.25` . Guaranteeing the submarine's direction in the four received-signal
# settings makes the opposite signal just as good.
#
# When the signal is not received, however, the submarine follows the same random
# fallback regardless of what the spy signals. Guaranteeing that outcome under an
# alternative signal changes nothing, so these settings are unnecessary and excluded by
# the minimality requirement.
#
# The spy therefore intentionally guides the submarine towards the minefield when the
# received signal causes it to go there, but does not intentionally cause the same
# outcome when it occurs through the random fallback.
#
# This leads us to definition 3:
#
#    Definition 3 (Intention – Operationalisation III). An agent intended to cause
#    outcomes 𝒐 in setting 𝒆 with action 𝑎, if
#
#    \(1) There is a super-set of outcomes 𝒚 ⊇ 𝒐 and a set of settings 𝒘 ⊇ 𝒆 in which 𝒚
#    is caused;
#
#    \(2) Guaranteeing that another action 𝑎′ also caused 𝒚 in 𝒘 would make 𝑎′ just as
#    good as 𝑎;
#
#    \(3) No proper subsets of 𝒚 and 𝒘 make 𝑎′ just as good as 𝑎. [1]_
#
# These examples add conditions to the intention test: outcomes may be valuable in
# combination, and the setting determines whether an action causes an outcome. The
# :ref:`content recommendation case study <sphx_glr_auto_examples_models_plot_agentic_intention_2.py>`
# uses these ideas to compare policies and calculate minimal guarantees.
