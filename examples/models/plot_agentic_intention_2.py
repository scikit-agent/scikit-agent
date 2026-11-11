r"""
=========================================
Agentic Intention: The Observer's Problem
=========================================

Suppose we see an agent repeatedly make decisions that produce a particular outcome.
We might infer that it intends that outcome, but how could we test our inference?
One approach is to guarantee the outcome and see whether the agent changes its
behavior. But how can we be sure we're guaranteeing the right outcome?

Ward et al. [1]_ give a behavioral definition of intention that tests how an agent's
policy responds when an outcome is guaranteed. We'll explore this through a bartender
who learns to allocate limited effort among customers from noisy feedback.

We'll examine what an outside observer can infer from the learned policy and use
those estimates to propose an outcome to guarantee.

In the
:ref:`companion page <sphx_glr_auto_examples_models_plot_agentic_intention_1.py>`,
we developed the intuition behind Ward et al.'s subjective intention test using fully
specified causal models. Here, we ask what happens when the observer lacks that
information.

References
==========

.. [1] Ward, F. R., MacDermott, M., Belardinelli, F., Toni, F., and Everitt, T.
   (2024). “The Reasons that Agents Act: Intention and Instrumental Goals.” In
   *Proceedings of the 23rd International Conference on Autonomous Agents and
   Multiagent Systems (AAMAS 2024)*, Auckland, New Zealand, pp. 1901–1909.
   IFAAMAS. :arxiv:`2402.07221` (arXiv version cited for numbering).

A bartender with an imperfect reading
=====================================

Imagine a bartender who can devote effort to making stronger
drinks, improving their quality, or talking with customers. Each customer values
one of these aspects of service, but the bartender doesn't know which one they
prefer (and it would be quite rude to ask). Instead, the bartender relies on
intuition, which we represent as a noisy reading of the customer's preference.

In a :class:`~skagent.block.DBlock`, we represent the customer's general
disposition and mood with two independent Bernoulli shocks, PreferenceA and
PreferenceB. Together, they determine the customer's true Preference, encoded as
0 (alcohol), 1 (quality), or 2 (conversation). A third shock, Misread, shifts the
bartender's observed Reading to the next category with probability 0.2. All three
inputs are modeled as
:class:`~skagent.distributions.Bernoulli` shocks.

The first diagram shows only what the bartender can infer before choosing a service.
"""

from itertools import product

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.utils import plot_block_diagram


reading_model = DBlock(
    shocks={
        "PreferenceA": (Bernoulli, {"p": 0.5}),
        "PreferenceB": (Bernoulli, {"p": 0.5}),
        "Misread": (Bernoulli, {"p": 0.2}),
    },
    dynamics={
        "Preference": lambda PreferenceA, PreferenceB: (
            int(PreferenceA + 2 * PreferenceB) % 3
        ),
        "Reading": lambda Preference, Misread: (Preference + int(Misread)) % 3,
    },
)

plot_block_diagram(reading_model, "Preference and reading", figsize=(8, 3))

# %%
# Reward and effort
# =================
#
# Based on this reading, the bartender chooses how much effort to devote to the
# suggested service. Greater effort incurs an increasing cost, even before we
# consider the other customers waiting to be served. In return, the bartender
# hopes to receive a tip.
#
# A bad tip might mean the bartender misread the customer's preference. It
# could also mean the customer simply isn't a good tipper.
#
# We add ``Time`` to record effort and a ``GoodTipper`` shock for the customer's
# unobserved tipping tendency. The reward is
#
# .. math::
#
#    U=1+2D\,\mathbb{1}\{P=R\}-D^2+\epsilon_T,
#
# where :math:`P` is the customer's true preference, :math:`R` is the bartender's
# reading, and :math:`D` is effort. Effort directed toward the preferred service
# increases the tip, while :math:`D^2` represents increasing opportunity cost.
#
# We take ``GoodTipper`` to be an independent
# :math:`\operatorname{Bernoulli}(0.5)` shock and set
# :math:`\epsilon_T=\mathrm{GoodTipper}-0.5`.


def base_tips(Preference, Intoxication, Quality, Rapport, Time, GoodTipper):
    """Compute the tip from service outcomes and the customer's tipping shock"""
    benefit = (
        Intoxication if Preference == 0 else Quality if Preference == 1 else Rapport
    )
    return 1 + 2 * benefit - Time**2 + (GoodTipper - 0.5)


# Keep the preference-reading mechanism while adding service outcomes and tips
bartender = reading_model.deep_replace(
    shocks={"GoodTipper": (Bernoulli, {"p": 0.5})},
    dynamics={
        "D": Control(["Reading"], agent="bartender"),
        "Intoxication": lambda D, Reading: D if Reading == 0 else 0.0,
        "Quality": lambda D, Reading: D if Reading == 1 else 0.0,
        "Rapport": lambda D, Reading: D if Reading == 2 else 0.0,
        "Time": lambda D: D,
        "Tips": base_tips,
    },
    reward={"Tips": "bartender"},
)

plot_block_diagram(bartender, "Bartender: effort and tips", figsize=(9, 4))

# %%
# Lastly, we enumerate the four Bernoulli shocks to calculate exact customer probabilities.

worlds = [
    dict(zip(("PreferenceA", "PreferenceB", "Misread", "GoodTipper"), values))
    for values in product((0, 1), repeat=4)
]

world_probabilities = np.array(
    [0.125 * (0.2 if world["Misread"] else 0.8) for world in worlds]
)


def reading_from(world):
    """Recover the bartender's noisy reading from a shock setting"""
    return (world["PreferenceA"] + 2 * world["PreferenceB"] + world["Misread"]) % 3


# %%
# Constraints and time context: a five-customer shift
# ===================================================
#
# So far, we've considered one customer in isolation. Intuitively, across a shift, effort
# spent now leaves less for later customers. Suppose five customers arrive, each
# with independent preference, misreading, and tipping shocks. Conserving effort
# may help earn future tips, much as acquiring gas helped achieve the garage-fire
# outcome in the previous notebook.
#
# The bartender observes the current ``Reading``, ``Budget``, and
# ``CustomersRemaining``. A :class:`~skagent.block.Control` bounds effort between
# zero and the lesser of one and the available budget. The quadratic effort cost
# still applies to each tip, and total reward is the sum of five tips.
# We carry budget and customer count between periods in a ``DBlock``. The budget
# follows the state transition:
#
# .. math::
#
#    B_{t+1}=B_t-D_t,\qquad B_0=2.
#
# Below, a one-period ``BellmanPeriod`` supplies transitions to
# :class:`~skagent.algos.sb3.PPOAgent`.
# Five transitions make a full shift. ``GymEnv`` marks its episode limit as a
# truncation, so the wrapper below marks the actual end of the shift as terminal for
# PPO's return calculation.

# Reuse the single-customer shocks and reward while adding a shift-wide budget
shift_model = bartender.deep_replace(
    dynamics={
        "D": Control(
            ["Reading", "Budget", "CustomersRemaining"],
            lower_bound=lambda Reading, Budget, CustomersRemaining: 0.0,
            upper_bound=lambda Reading, Budget, CustomersRemaining: min(
                1.0, max(0.0, Budget)
            ),
            agent="bartender",
        ),
        "Budget": lambda Budget, D: Budget - D,
        "CustomersRemaining": lambda CustomersRemaining: CustomersRemaining - 1,
    }
)


class CompleteShift(gym.Wrapper):
    """Treat the end of a five-customer shift as episode termination"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, False, info


shift_period = BellmanPeriod(shift_model, None, {})
shift_agent = PPOAgent(
    shift_period,
    initial={"Budget": 2.0, "CustomersRemaining": 5.0},
    max_episode_steps=5,
    seed=17,
    gym_kwargs={"bound_clearance": 0.0},
    ppo_kwargs={
        # Count later customers' tips fully because the objective is total shift reward
        "gamma": 1.0,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "learning_rate": 1e-4,
        # Narrow the initial action distribution to limit clipping at effort bounds
        "policy_kwargs": {"log_std_init": -1},
    },
)
shift_agent.model.set_env(CompleteShift(shift_agent.env))
shift_agent.learn(total_timesteps=60_000)

start_states = np.array([[r, 2.0, 5.0] for r in range(3)], dtype=np.float32)
start_efforts = shift_agent.predict_unscaled(start_states)
print(
    "PPO effort at shift start (alcohol, quality, conversation):",
    np.round(start_efforts, 3),
)

# %%
# At the start of a shift, PPO spends about 0.43–0.49 effort across readings,
# leaving most of its budget for later customers. To evaluate this learned policy,
# ``serve()`` uses :meth:`~skagent.block.Block.transition` to calculate one
# customer's tip and next budget. ``collect_shifts()`` repeats that step across
# five customers, retaining the causal state for further analysis in later sections.


def serve(state, world, effort):
    """Evaluate one customer interaction at a specified effort"""

    def rule(Reading, Budget, CustomersRemaining):
        return effort

    return shift_model.transition({**state, **world}, {"D": rule})


def collect_shifts(count, seed):
    """Generate PPO shifts while retaining the experimenter's causal state"""
    rng = np.random.default_rng(seed)
    shifts = []
    for shift_id in range(count):
        state = {"Budget": 2.0, "CustomersRemaining": 5.0}
        history = []
        for t in range(5):
            world = worlds[int(rng.choice(len(worlds), p=world_probabilities))]
            reading = reading_from(world)
            observation = np.array(
                [reading, state["Budget"], state["CustomersRemaining"]],
                dtype=np.float32,
            )
            effort = float(shift_agent.predict_unscaled(observation)[0])
            result = serve(state, world, effort)

            history.append(
                {
                    "shift": shift_id,
                    "t": t,
                    "world": world,
                    "reading": reading,
                    "budget": state["Budget"],
                    "remaining": state["CustomersRemaining"],
                    "effort": effort,
                    "tip": result["Tips"],
                    "next_budget": result["Budget"],
                }
            )
            state = {
                "Budget": result["Budget"],
                "CustomersRemaining": result["CustomersRemaining"],
            }
        shifts.append(history)
    return shifts


# %%
# To visualize how learned conversation effort changes with budget and customers
# remaining, we plot the heatmap below. Darker colors indicate lower effort;
# lighter colors indicate higher effort.

budgets = np.linspace(0, 2, 41)
remaining_counts = np.arange(5, 0, -1)
# Probe the learned policy across budgets and shift positions
probe_states = np.array(
    [[2, budget, remaining] for budget in budgets for remaining in remaining_counts],
    dtype=np.float32,
)
conversation_effort = shift_agent.predict_unscaled(probe_states).reshape(
    len(budgets), 5
)

fig, ax = plt.subplots()
image = ax.imshow(
    conversation_effort,
    origin="lower",
    aspect="auto",
    extent=(0.5, 5.5, 0, 2),
    vmin=0,
    vmax=1,
    cmap="viridis",
)
ax.set(
    xticks=np.arange(1, 6),
    xticklabels=remaining_counts,
    xlabel="Customers remaining",
    ylabel="Remaining budget",
)
fig.colorbar(image, ax=ax, label="PPO conversation effort")
plt.show()

# %%
# Testing intention in a trained agent
# ====================================
#
# The heatmap shows how our bartender adjusts its effort as its budget and
# remaining customers change. But observing these decisions alone doesn't
# tell us whether the bartender intends to preserve its budget.
#
# To address this and similar problems,
# Ward et al. [1]_ propose a behavioral definition of intention that infers
# intent from responses to interventions without knowing an agent's
# subjective causal model.
#
# Their formal definition in [1]_ is:
#
#    **Definition 13 (Behavioural Intention).** Let :math:`\Gamma` be a policy
#    oracle for a SCIM :math:`\mathcal M` which need not include utility
#    variables. Then :math:`\Gamma` behaviourally intends to cause
#    :math:`\mathbf O_\pi(\mathbf e)` with :math:`\pi:=\Gamma(\mathcal M)` if
#    there exists subset-minimal :math:`\mathbf Y\supseteq\mathbf O` and for
#    each :math:`Y\in\mathbf Y` subset-minimal
#    :math:`\mathbf w_Y\subseteq\operatorname{dom}(\mathbf E)`, with
#    :math:`\mathbf e\in\bigcap_{O\in\mathbf O}\mathbf w_O`, satisfying
#    :math:`\Gamma(\mathcal M)\ne\Gamma(\mathcal M_{\mathbf Y_\pi\mid\mathbf W})`.
#
# Ward et al. continue on to [1]_ connect this behavioral definition to
# subjective intention:
#
#    **Theorem 14 (Equivalence of subjective and behavioural intent).** Given a
#    SCIM :math:`\mathcal M` and policy-oracle :math:`\Gamma`, if 1)
#    :math:`\Gamma` is robustly optimal and 2) :math:`\Gamma` only maps to
#    :math:`\pi` instead of :math:`\hat\pi` if :math:`\pi` gets strictly higher
#    utility (i.e. :math:`\mathbb E_\pi[\sum_U U]>\mathbb E_{\hat\pi}[\sum_U U]`),
#    then :math:`\Gamma` behaviourally intends to cause
#    :math:`\mathbf O_\pi(\mathbf e)` (definition 13) if and only if it
#    subjectively intends to cause :math:`\mathbf O(\pi,\mathbf e)` (definition 6)
#    with :math:`\pi=\Gamma(\mathcal M)`.
#
#
# However, the authors argue that "it is reasonable to
# ascribe intent to GPT-4 using our behavioural definition."
# We'll investigate this difficulty
# using our PPO bartender in place of an LLM. Both learn policies through
# training, but the bartender operates in a causal model we can inspect.
#
# What can an observer see?
# -------------------------
#
# The above guarantee depends on the outcome produced by the original policy.
# However, an outside observer may not know that outcome.
# In the case of an LLM, the observer sees its inputs and outputs without knowing
# the causal process behind its responses. Accordingly, we give two observers
# different levels of access to the bartender's decisions.
#
# - **Partial observer:** Sees the current reading and the previous customer's
#   effort and tip, rounded to a tenth.
# - **Full-state observer:** Also sees the remaining budget and number of
#   customers left.
#
# Neither sees the current action, its tip, or future outcomes. The full-state
# view is an idealized bartender-state benchmark, not access to an LLM's weights
# or activations. We generate 800 shifts to train their predictions and 250
# held-out shifts to evaluate them.

training_shifts = collect_shifts(800, seed=1000)
heldout_shifts = collect_shifts(250, seed=5000)

# Construct each observer's view from the same shifts
for shifts in (training_shifts, heldout_shifts):
    for shift in shifts:
        for t, row in enumerate(shift):
            previous = shift[t - 1] if t else None
            # Mark a missing previous customer so first decisions remain distinguishable
            partial = [
                row["reading"],
                round(previous["effort"], 1) if previous else -1,
                round(previous["tip"], 1) if previous else -1,
            ]
            row["partial"] = partial
            row["full"] = partial + [row["budget"], row["remaining"]]

training_rows = [row for shift in training_shifts for row in shift]
heldout_rows = [row for shift in heldout_shifts for row in shift]

# %%
# Predicting the bartender's behavior
# -----------------------------------
#
# Now we examine if these different observational perspectives can be used to
# predict the bartender's decisions using linear least-squares
# models to estimate effort. The partial observer also estimates the hidden
# budget, which we'll need to propose a guarantee.


def observer_features(record):
    """Encode the information available to an observer for linear prediction"""
    reading, last_effort, last_tip = record[:3]
    columns = [
        1.0,
        *(float(reading == r) for r in range(3)),
        last_effort,
        last_tip,
        float(last_effort >= 0),
    ]
    return columns + list(record[3:])


def features(rows, view):
    return np.array([observer_features(row[view]) for row in rows])


# Both predict effort, but the partial observer must also estimate its hidden budget
partial_train = features(training_rows, "partial")
full_train = features(training_rows, "full")
partial_targets = np.array([[row["effort"], row["budget"]] for row in training_rows])
full_targets = np.array([row["effort"] for row in training_rows])

# Fit effort and hidden-budget estimates from partial observations
partial_weights = np.linalg.lstsq(partial_train, partial_targets, rcond=None)[0]
full_weights = np.linalg.lstsq(full_train, full_targets, rcond=None)[0]

# %%
# We evaluate both models on shifts they have not seen before. Linear regression
# can predict values outside the feasible range, so we clip effort to :math:`[0,1]`
# and budget to :math:`[0,2]` before comparing held-out mean absolute error (MAE).

partial_predictions = features(heldout_rows, "partial") @ partial_weights
full_predictions = features(heldout_rows, "full") @ full_weights

# Bound the observers' predictions to feasible effort and budget ranges
partial_predictions[:, 0] = np.clip(partial_predictions[:, 0], 0, 1)
partial_predictions[:, 1] = np.clip(partial_predictions[:, 1], 0, 2)
full_predictions = np.clip(full_predictions, 0, 1)
# Compare with held-out PPO actions
actual_efforts = np.array([row["effort"] for row in heldout_rows])
partial_mae = np.mean(abs(partial_predictions[:, 0] - actual_efforts))
full_mae = np.mean(abs(full_predictions - actual_efforts))

fig, ax = plt.subplots(layout="constrained")
bars = ax.barh(["Partial observer", "Full-state observer"], [partial_mae, full_mae])
ax.invert_yaxis()
ax.bar_label(bars, fmt="%.4f")
ax.set_xlabel("Held-out effort prediction MAE")
plt.show()

# %%
# From prediction to intervention
# -------------------------------
#
# The partial observer predicts effort less accurately than the full-state
# observer. To apply Definition 13, it must also identify the outcome produced
# by the original policy. We test whether it can estimate that outcome.
#
# Consider the third customer when the bartender's reading indicates conversation.
# We want to predict the budget PPO actually leaves after serving that customer,
# as Definition 6's contextual guarantee requires. Two customers remain, so the
# budget left after this decision can still affect later service.
#
# The partial observer cannot see the arrival budget or current effort, so it
# estimates the next budget from its predictions:
#
# .. math::
#
#    \widehat B_{t+1}=\widehat B_t-\widehat D_t.
#
# We can compare this estimate with the actual next budget because we know the
# underlying causal model.
#
focal_indices = [
    i for i, row in enumerate(heldout_rows) if row["t"] == 2 and row["reading"] == 2
]
tolerance = 0.05

# Errors in either estimated budget or effort shift the proposed guarantee
proposed_targets = np.clip(
    partial_predictions[focal_indices, 1] - partial_predictions[focal_indices, 0], 0, 2
)

actual_targets = np.array([heldout_rows[i]["next_budget"] for i in focal_indices])
target_errors = abs(proposed_targets - actual_targets)
target_mae = np.mean(target_errors)
within_tolerance = np.mean(target_errors <= tolerance)

print(f"Budget-target MAE: {target_mae:.4f}")
print(f"Within tolerance: {within_tolerance:.1%}")

fig, ax = plt.subplots()
ax.hist(target_errors, edgecolor="black", linewidth=0.5)
ax.axvline(
    tolerance, color="red", linestyle="--", label=f"{tolerance:.2f} budget units"
)
ax.set(
    xlabel="Absolute budget-target error",
    ylabel="Focal decisions",
    title=f"Target error across {len(focal_indices)} decisions",
)
ax.legend()
plt.show()

# %%
# Across 62 decisions, the observer's budget estimate was off by an average
# of 0.057 units. The dashed line marks an error of 0.05, and fewer than half
# of the estimates came that close.
#
# Even a small error matters here. Ward et al.'s guarantee requires knowing
# how much budget the bartender actually preserved.
#
# Conclusion
# ==========
#
# Our bartender shows why applying this test can be difficult. Even in a small
# causal model, the partial observer often proposed a different next budget from
# the one produced by PPO's original policy. We measured errors in specifying the
# guarantee, not a change in intention attribution. For a language model, an
# external observer sees inputs and outputs without knowing its subjective causal
# model or whether it responds optimally to interventions.
#
# **Ward et al.'s formal equivalence holds under specific assumptions, but how
# reliably can we infer intention from prompted responses when those assumptions
# cannot be verified?**
