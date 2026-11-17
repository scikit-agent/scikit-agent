r"""
=======================================================
Agentic Intention: A Content Recommendation Case Study
=======================================================

Ward et al.'s subjective intention test asks whether guaranteeing an outcome under
an alternative policy would remove the original policy's advantage. The
:ref:`introduction <sphx_glr_auto_examples_models_plot_agentic_intention_1.py>`
develops the test through short causal examples. Here, we calculate it for a
recommender that can show a user's preferred genre or addictive content.

Whereas the
:ref:`content recommendation incentive example <sphx_glr_auto_examples_models_plot_incentives_2_content_recommendation.py>`
tests whether the agent has an incentive to influence users' opinions, this
page holds preferences fixed and asks which watching outcomes the agent intends
under particular policies and settings.

References
==========

.. [1] Ward, F. R., MacDermott, M., Belardinelli, F., Toni, F., and Everitt, T. (2024).
   “The Reasons that Agents Act: Intention and Instrumental Goals.” In *Proceedings of
   the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS
   2024)*, Auckland, New Zealand, pp. 1901–1909. IFAAMAS. :arxiv:`2402.07221`.

.. [2] Evans, C. and Kasirzadeh, A. (2021). *User Tampering in Reinforcement Learning
   Recommender Systems*. :arxiv:`2109.04083`.

"""

# %%
# Computing the intention test
# ============================
#
# Ward et al.’s Example 7 [1]_ adapts a recommender thought experiment from Evans and
# Kasirzadeh [2]_. Here, preference, sleep, and the recommendation determine watching
# and reward. A user prefers comedy or drama. A
# recommender can show either genre or addictive content, and receives utility :math:`1`
# when the user watches. The user watches preference-matched or addictive content. If
# the user has fallen asleep with the app running, watching continues regardless of the
# recommendation.
#
# We assume the recommender’s beliefs match the causal model specified below and that
# its utility is watching. Under this assumption, we can apply Definition 6 to assess
# whether watching is intended in each setting.
#
# The variable dependencies specify the graph, while ``dynamics`` supplies the
# structural functions. The ``shocks`` distributions give equal probability to each
# preference and assign sleep a probability of :math:`0.25` .

from itertools import combinations, product

import matplotlib.pyplot as plt
from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.ground import Discretized, GroundedBlock
from skagent.utils import plot_block_diagram

# sphinx_gallery_thumbnail_number = 1

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
