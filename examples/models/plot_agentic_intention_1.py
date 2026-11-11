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
agents. [1]_ We introduce its conditions through illustrative examples and causal diagrams,
then apply the counterfactual test numerically to a toy content recommender. [3]_

The companion page,
:ref:`sphx_glr_auto_examples_models_plot_agentic_intention_2.py`,
extends this framework to a learned agent and examines what an observer can
infer about intention from incomplete information.

References
==========

.. [1] Ward, F. R., MacDermott, M., Belardinelli, F., Toni, F., and Everitt, T. (2024).
   “The Reasons that Agents Act: Intention and Instrumental Goals.” In *Proceedings of
   the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS
   2024)*, Auckland, New Zealand, pp. 1901–1909. IFAAMAS. :arxiv:`2402.07221`.

.. [2] Evans, C. and Kasirzadeh, A. (2021). *User Tampering in Reinforcement Learning
   Recommender Systems*. :arxiv:`2109.04083`. Ward et al. adapt the recommender thought
   experiment; this walkthrough follows their Example 7.

.. [3] Ward et al.’s Definitions 1–3 are successive operationalisations. Definition 4
   specifies the causal model. Definition 5 defines contextual interventions, and
   Definition 6 formalises subjective intention. The numerical payoffs and probabilities
   are illustrative choices. The recommender example is Ward et al.’s adaptation of
   Evans and Kasirzadeh [2]_. For brevity, we illustrate instrumental intention by
   extending the garage example rather than introducing the paper’s separate thought
   experiment. The paper also develops a behavioural account of intention, which we
   leave outside this walkthrough.

.. [4] Ashton, H. (2023). “Definitions of intent suitable for algorithms.” *Artificial
   Intelligence and Law* 31, 515–546. :doi:`10.1007/s10506-022-09322-x`.

"""

# %%
# Setup
# =====

from itertools import combinations, product

import matplotlib.pyplot as plt
from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.ground import Discretized, GroundedBlock
from skagent.utils import plot_block_diagram

# sphinx_gallery_thumbnail_number = 7
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
# Both insurance and the car’s destruction follow from burning, but only insurance
# contributes to Bob’s utility. By the above definition, it follows that the incidental
# destruction of Alice’s car was not intended, despite Bob being aware that it would be
# a consequence of burning the garage. Guaranteeing the payout without burning would
# remove Bob’s reason to choose burning over the alternative.
#
# This illustrates the paper’s notion of direct intention vs indirect intention, or, in
# the authors’ words:
#
#    “Definition 1 distinguishes desired effects from accidental side-effects. In legal
#    terms, we capture *direct intent*, which requires that intended outcomes are
#    desired [4]_. The alternative notion of *indirect intent* drops this requirement
#    and includes the ‘almost certain side-effects of directly intended outcomes’ [1]_.”
#
# Remodeling the garage: instrumental intention
# ---------------------------------------------
#
# Suppose Bob must first acquire gas to burn the garage. Acquiring gas contributes to
# his goal by enabling the insurance payout.

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

plot_block_diagram(garage, "Garage fire", figsize=(8, 3))

# %%
# Acquiring gas is instrumental to obtaining the payout: without it, Bob cannot carry
# out the fire that produces the insurance payment. The intermediate outcome can
# therefore be intended even though it has no value to Bob on its own.
#
# It is in this way that the authors invoke Ashton’s notion of *means-end consistency*
# as desideratum for intent [4]_. In their words:
#
#    “if an agent intends some final outcome, then any intermediary outcomes which are
#    instrumentally useful in achieving the final goal are also intended.” [1]_
#
# Louis: adding subjective belief
# -------------------------------
#
# Now we will circle back to the example in the introductory section to add *subjective
# belief* into the mix with Ward et al’s Example 4 [1]_
#
# Louis plans to kill his uncle. On the way, he accidentally runs over a pedestrian who
# turns out to be that uncle. In his understanding of events, running over the
# pedestrian does not accomplish his plan. Thus, in his subjective model he did not
# intend to kill his uncle by running over the pedestrian.
#
# These schematic diagrams illustrate the distinction between Louis’s understanding and
# the actual outcome. Their functions return ``None`` only to declare dependencies for
# drawing; we do not evaluate the intention test numerically for this example.

louis_subjective = DBlock(
    dynamics={
        "D": Control([], agent="louis"),
        "PedestrianDead": lambda D: None,
        "UncleDead": lambda D: None,
        "U": lambda UncleDead: None,
    },
    reward={"U": "louis"},
)

louis_objective = DBlock(
    dynamics={
        "D": Control([], agent="louis"),
        "UncleDead": lambda D: None,
        "U": lambda UncleDead: None,
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
# Alice: several reasons together
# -------------------------------
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
# In the spy example, a received signal and a lucky random choice can produce the same
# outcome. When the signal is received, guaranteeing that an alternative signal would
# also guide the submarine towards the minefield can make that alternative just as good.
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
# Content Recommendation: computing the intention test
# ----------------------------------------------------
#
# Ward et al.’s Example 7 [1]_ adapts a recommender thought experiment from Evans and
# Kasirzadeh [2]_ to build a SCIM that specifies how preference, sleep, and the
# recommendation determine watching and reward. In it, a user prefers comedy or drama. A
# recommender can show either genre or addictive content, and receives utility :math:`1`
# when the user watches. The user watches preference-matched or addictive content. If
# the user has fallen asleep with the app running, watching continues regardless of the
# recommendation.
#
# We assume the recommender’s beliefs match the causal model specified below and that
# its utility is watching. Under this assumption, we can apply Definition 6 to assess
# whether watching is intended in each setting.
#
#    Definition 4 (Structural Causal Influence Model). A SCIM is a tuple
#    :math:`\mathcal{M} = (G, \mathbf{F}, P)` where
#    :math:`G = (\mathbf{V} \cup \mathbf{E}, \mathcal{E})` is a directed acyclic graph
#    (DAG) with endogenous variables :math:`\mathbf{V}` and exogenous parents
#    :math:`\mathbf{E}` for each :math:`V \in \mathbf{V}` :
#    :math:`\mathbf{E} = \{E^V\}_{V \in \mathbf{V}}` . The endogenous variables
#    :math:`\mathbf{V}` are partitioned into chance (:math:`\mathbf{X}`), decision
#    (:math:`\mathbf{D}`), and utility (:math:`\mathbf{U}`) variables. The domains of
#    utility variables are real-valued. :math:`\mathcal{E}` is the set of edges in the
#    DAG. Edges into decision variables are called information links.
#    :math:`\mathbf{F} = \{f^V\}_{V \in \mathbf{V} \setminus \mathbf{D}}` is a set of
#    structural functions
#    :math:`f^V : \operatorname{dom}(\mathrm{Pa}^V) \to \operatorname{dom}(V)` which
#    specify how each non-decision endogenous variable depends on its parents. :math:`P`
#    is a probability distribution over :math:`\mathbf{E}` such that the exogenous
#    variables :math:`E^V` are mutually independent. [1]_
#
# The variable dependencies specify the graph, while ``dynamics`` supplies the
# structural functions. The ``shocks`` distributions give equal probability to each
# preference and assign sleep a probability of :math:`0.25` .

recommender = DBlock(
    shocks={"Preference": (Bernoulli, {"p": 0.5}), "Sleep": (Bernoulli, {"p": 0.25})},
    dynamics={
        "Recommendation": Control(["Preference"], agent="recommender"),
        # Addictive content (2) keeps the user watching even without a preference match
        "Watching": lambda Recommendation, Preference, Sleep: int(
            Sleep or Recommendation == Preference or Recommendation == 2
        ),
        # Only watch time rewards the recommender, so matching and addictive policies
        # can tie
        "Reward": lambda Watching: Watching,
    },
    reward={"Reward": "recommender"},
)

plot_block_diagram(recommender, "Content recommender", figsize=(8, 3))

# %%
# Compare policies
# ~~~~~~~~~~~~~~~~
#
# Recommendations are coded as comedy (``0``), drama (``1``), or addictive content
# (``2``). A policy specifies what to recommend for each preference. Both the original,
# always-addictive policy and the preference-matching policy earn an expected reward of
# :math:`1` . A policy that always recommends the wrong genre earns only :math:`0.25` :
# watching continues only when the user is asleep.
#
# Does guaranteeing watching under the wrong-genre policy eliminate this payoff gap?
#
# Guarantee watching in selected settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#    Definition 5 (Contextual Intervention). For an intervention :math:`\mathcal{I}^Y`
#    on :math:`Y` , and a set of settings
#    :math:`w^Y \subseteq \operatorname{dom}(\mathbf{E})` , the contextual intervention
#    :math:`\mathcal{I}^{Y}_{w^Y}: \operatorname{dom}(\mathrm{Pa}^Y \cup \mathbf{E}) \to \operatorname{dom}(Y)`
#    is
#
#    .. math::
#       :label: contextual-intervention
#
#       \mathcal{I}^{Y}_{w^Y}(\mathrm{pa}^Y, \mathbf{e}) =
#       \begin{cases}
#       \mathcal{I}^{Y}(\mathrm{pa}^Y) & \text{if } \mathbf{e} \in w^Y, \\
#       f^Y(\mathrm{pa}^Y) & \text{if } \mathbf{e} \notin w^Y.
#       \end{cases}
#
#    [1]_
#
# A **contextual intervention** replaces a variable’s mechanism only in selected
# settings. Here, we give the wrong-genre policy the original policy’s watching outcome
# in those settings, while leaving its recommendations unchanged. Outside those
# settings, the usual watching rule applies.
#
# Without a guarantee, :meth:`~skagent.ground.GroundedBlock.expected_payoff`
# computes the policy’s expected reward. To apply a contextual guarantee,
# the helper below uses :meth:`~skagent.block.Block.transition` with
# ``fix=["Watching"]`` in the selected settings. It then averages payoffs
# across settings using :meth:`~skagent.ground.Discretized.reduce`.

policy_codes = list(product(range(3), repeat=2))
settings = tuple(product(range(2), repeat=2))  # (preference, asleep)

# Four settings give 16 possible sets of contextual guarantees
subsets = [
    frozenset(choice) for size in range(5) for choice in combinations(settings, size)
]

original_codes = (2, 2)
asleep_settings = frozenset((preference, 1) for preference in range(2))


def policy(codes):
    """Map each preference to its coded recommendation"""
    return lambda Preference: codes[int(Preference)]


def expected_reward(ground, codes, guaranteed=frozenset()):
    """Average payoff, fixing Watching in the selected settings"""
    decision = policy(codes)
    if not guaranteed:
        return float(
            ground.expected_payoff(
                {"Recommendation": decision}, Discretized(), agent="recommender"
            )
        )

    def at_setting(shocks):
        """Apply the watching guarantee in one preference-and-sleep setting"""
        setting = (int(shocks["Preference"]), int(shocks["Sleep"]))
        fixed = setting in guaranteed
        pre = dict(shocks)
        if fixed:
            # The original policy's watching outcome is what the alternative is
            # guaranteed
            pre["Watching"] = ground.block.transition(
                shocks, {"Recommendation": policy(original_codes)}
            )["Watching"]

        # Fix Watching only in selected settings; elsewhere use the alternative's usual
        # mechanism
        outcome = ground.block.transition(
            pre, {"Recommendation": decision}, fix=["Watching"] if fixed else []
        )
        return ground.block.payoff(outcome, agent="recommender")

    return float(Discretized().reduce(ground, at_setting))


# %%
# With equally likely preferences and a sleep probability of :math:`0.25` , the
# comparison is:
#
# ======================== =============================
# Watching guaranteed in…  Alternative’s expected reward
# ======================== =============================
# No settings              :math:`0.25`
# Comedy preference, awake :math:`0.625`
# Both preferences, awake  :math:`1`
# Only asleep settings     :math:`0.25`
# ======================== =============================
#
# Both awake settings are necessary to match the original policy and removing either
# guarantee leaves the alternative worse off. In contrast, sleep settings add nothing
# because playback already continues regardless of the recommendation.
#
#    Definition 6 (Subjective Intention). Assume the agent follows policy :math:`\pi` in
#    SCIM :math:`\mathcal{M}` with utility variables :math:`\mathbf{U}` . Let
#    :math:`\mathbf{O} \subseteq \mathbf{V}` and let :math:`REF(\pi)` be a reference set
#    of policies which the agent could have followed instead. The agent intentionally
#    causes the outcomes of :math:`\mathbf{O}` in setting :math:`\mathbf{e}` with
#    :math:`\pi` if the following conditions are met.
#
#    \(1) There exists :math:`\mathbf{Y} \supseteq \mathbf{O}` such that, for each
#    :math:`Y \in \mathbf{Y}` , there is a set of settings
#    :math:`w^Y \subseteq \operatorname{dom}(\mathbf{E})` and
#    :math:`\mathbf{e} \in \bigcap_{O \in \mathbf{O}} w^O` ;
#
#    \(2) There is an alternate policy :math:`\hat{\pi} \in REF(\pi)` such that
#
#    .. math::
#       :label: subjective-intention
#
#       E_{\pi}\left[\sum_{U \in \mathbf{U}} U\right] \le E_{\hat{\pi}}\left[\sum_{U \in
#       \mathbf{U}} U_{\mathbf{Y}_{\pi}\mid\mathbf{W}}\right]
#
#    where :math:`\mathbf{W} = \{w^Y\}_{Y \in \mathbf{Y}}`,
#
#    \(3) No proper subsets of :math:`\mathbf{Y}` and any :math:`w^Y` satisfy Equation
#    (2), i.e., :math:`\mathbf{Y}` and every :math:`w^Y` are subset-minimal. [1]_
#
# We apply Definition 6 to the single outcome variable ``Watching`` , treating the model
# as the recommender’s beliefs and utility. For each reference policy, we search for a
# subset-minimal set of settings in which guaranteeing watching closes the payoff gap.
#
# Searching the policy space
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We evaluate all nine deterministic policies against all :math:`2^4=16` subsets of the
# four settings: comedy awake, comedy asleep, drama awake, and drama asleep. The results
# show the eight alternatives to the original policy; ``comedy / drama`` means
# recommending comedy to comedy-preferring users and drama to drama-preferring users.
#
# A :class:`~skagent.ground.GroundedBlock` pairs the recommender model with
# its parameter values; this model requires no additional parameters.


def compare_references(ground, sleep_probability):
    """Find each policy's payoff and minimal guarantees that tie the original"""
    baseline = expected_reward(ground, original_codes)
    assert abs(baseline - 1) < 1e-10
    comparison = {}
    for codes in policy_codes:
        payoffs = {subset: expected_reward(ground, codes, subset) for subset in subsets}
        sufficient = [
            subset for subset in subsets if payoffs[subset] >= baseline - 1e-10
        ]

        # Minimality removes settings that did not help close the payoff gap
        minimal = [
            subset
            for subset in sufficient
            if not any(other < subset for other in sufficient)
        ]

        # An awake miss neither matches the preference nor shows addictive content
        missed_awake = frozenset(
            (preference, 0)
            for preference in range(2)
            if codes[preference] not in (preference, 2)
        )
        assert minimal == [frozenset() if sleep_probability == 1 else missed_awake]

        # Watching already continues during sleep, so asleep guarantees add nothing
        assert all(
            abs(value - payoffs[subset - asleep_settings]) < 1e-10
            for subset, value in payoffs.items()
        )
        comparison[codes] = (payoffs[frozenset()], minimal[0])
    return comparison


names = ("comedy", "drama", "addictive")
ground = GroundedBlock(recommender, {})
nominal = compare_references(ground, 0.25)
references = [codes for codes in policy_codes if codes != original_codes]

print("Original policy (addictive / addictive): expected reward 1.000")
print("Policy (if comedy / if drama)           Reward   Gap   Minimal guarantee")
for codes in references:
    reward, needed = nominal[codes]
    label = f"{names[codes[0]]} / {names[codes[1]]}"
    guarantee = (
        ", ".join(f"{names[preference]} awake" for preference, _ in sorted(needed))
        or "No guarantee needed—already ties"
    )
    print(f"{label:<39} {reward:>5.3f}  {1 - reward:>5.3f}  {guarantee}")

# %%
# Policies that produce no watching in one awake setting require a guarantee there; the
# wrong-genre policy requires both awake settings. Policies already earning :math:`1`
# need no guarantee.
#
# Varying sleep probability
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we examine what happens when we vary the probability of sleep. Intuitively, as
# sleep becomes more common, more watching happens regardless of the recommendation, so
# the payoff gap should shrink. At each sleep probability, we assume the recommender’s
# beliefs match that probability.
#
# We use :meth:`~skagent.block.DBlock.deep_replace` to change the sleep
# distribution while retaining the model’s other specifications.

sleep_probabilities = sorted({step / 10 for step in range(11)} | {0.25})
results = {}
for sleep_probability in sleep_probabilities:
    # Change only sleep frequency to isolate watching that ignores the recommendation
    model = recommender.deep_replace(
        shocks={"Sleep": (Bernoulli, {"p": sleep_probability})}
    )
    results[sleep_probability] = compare_references(
        GroundedBlock(model, {}), sleep_probability
    )

fig, ax = plt.subplots(figsize=(8, 4))

# These policies miss zero, one, or both preferences while the user is awake
for misses, codes in ((0, (0, 1)), (1, (0, 0)), (2, (1, 0))):
    ax.plot(
        sleep_probabilities,
        [1 - results[p][codes][0] for p in sleep_probabilities],
        marker="o",
        markersize=3,
        label=f"{misses} awake misses",
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.set(
    xlabel="Probability user is asleep",
    ylabel="Original minus reference reward",
    title="Payoff gap across sleep rates",
)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %%
# At a sleep probability of :math:`1` , watching is independent of the recommendation.
# Every policy earns the same reward, so no guarantee is needed to make an alternative
# equally good. Watching still produces utility, but it no longer provides a reason to
# choose one recommendation over another.
#
# The example therefore separates what the recommender is rewarded for from what it
# intends to bring about. The reward function stays unchanged across the sleep sweep,
# but the intention assessment depends on how the recommender’s choices can affect that
# reward.
#
# Conclusion
# ==========
#
# An outcome is not intended simply because an agent causes it. Ward et al.’s test asks
# whether, under the agent’s subjective model, guaranteeing that outcome under an
# alternative choice would remove the original choice’s advantage, subject to
# minimality. Here, explicitly specified causal models made those comparisons
# computable. Applying the test in deployment raises a further question: what can we
# establish when the required outcomes cannot be guaranteed, or their consequences are
# uncertain?
