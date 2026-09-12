r"""
Tuning a differential-privacy parameter as a causal game.

An analyst wants to estimate a population mean and offers a differential-privacy
guarantee to the people it collects data from. Each data subject decides whether
to share. Sharing helps the estimate, which the subject values, and costs the
subject privacy, which it does not. How much noise the mechanism adds is neither
agent's choice: a designer sets it, knowing that the agents will then play a
best response to it. So the noise scale is a design parameter above an
equilibrium rather than a parameter of anyone's problem, and choosing it is the
mechanism being designed.

The model is Benthall and Cummings (2026) [BC2026]_, sections 4 to 6.

Two games, and they differ in one place
=======================================

The **local** model has each subject add its own noise before sharing, so what
reaches the analyst is already private. The **central** model has subjects share
their true data with a trusted analyst, which adds noise once to the estimate it
publishes. Every equation below is shared except where the noise enters:

.. math::
    b_i = a + \zeta_i, \qquad
    u_i = c_i \left(q_i - \frac{p_i}{\sigma}\right), \qquad
    g = -(f - a)^2

with :math:`a` the population mean, :math:`\zeta_i \sim U(-\Delta/2, \Delta/2)`
individual variation, :math:`p_i \sim N(0, \sigma_p^2)` how much subject
:math:`i` minds sharing, :math:`q_i` what it gains by sharing, and :math:`\sigma`
the noise scale. The report is :math:`d_i = c_i(b_i + \gamma_i)` locally, with
one :math:`\gamma_i \sim N(0, \sigma^2)` per subject, and :math:`d_i = c_i b_i`
centrally, where a single :math:`\gamma` is added to the estimate instead.

What each agent decides, and what is supplied
=============================================

Both decisions have closed forms in the paper, so this module supplies both as
rules and the blocks declare them as decisions:

- A subject shares iff sharing is worth more than it costs. The utility is
  linear in :math:`c_i`, so the optimum is at a vertex whatever the control's
  bounds: :math:`c_i = 1` iff :math:`p_i < q_i \sigma`, which is
  :func:`sharing_rule`. A solver on the relaxed :math:`[0, 1]` control returns
  that rule rather than an approximation of it.
- The analyst's estimate is the mean of the reports it received, which the paper
  proves is the minimum-variance unbiased estimator of :math:`a`, so maximizing
  its accuracy is computing that mean. It is :func:`analyst_rule`.

The analyst's information set is the whole subject class, which no policy network
in this library is shaped for -- a rule over a class is a permutation-invariant
architecture -- so this decision is represented and supplied rather than learned.
The subjects' is not: their information set is per-instance, and
:func:`sharing_rule` is what a solver should reproduce.

Two reductions written as weighted means
========================================

The analyst averages over the subjects who shared, and averages a *prior* when
nobody did. Both are written as weighted sums over the whole class rather than as
a mean over a selection and a branch on the count, so the estimate carries no
boolean index and no branch on the data: it differentiates and batches under
``torch`` as readily as it evaluates under ``numpy``. A non-sharer contributes
nothing to either the numerator or the denominator, which is what makes a null
report and a zero weight the same thing here, and the prior enters as one
pseudo-observation of negligible weight, which is what makes the empty database
return the prior exactly.

The designer's problem
======================

The designer maximizes an objective of its own over :math:`\sigma`, given that
the agents play their equilibrium at each :math:`\sigma`:

.. math::
    \max_\sigma h(\sigma), \qquad h = \mathbb{E}[g] + \mathbb{E}[u_i]

That is a sweep over equilibria rather than a solution concept or a method, so it
is a loop a caller writes; this module supplies what the loop evaluates.
:func:`designer_objective` is the paper's own example of an :math:`h`, and it
uses the bounded subject utility of the paper's Figure 5,
:math:`u_i = c_i(1 - e^{p_i}/(1 + \sigma))`, under which more noise always helps
a subject. The base utility above is unbounded as :math:`\sigma \to 0`: a subject
who positively enjoys sharing gains :math:`-p_i/\sigma`, without limit.

The closed forms here are exact, and the accuracy ones are the paper's equation
(3) and its central analogue. The number of subjects who share is
:math:`n' \sim \mathrm{Binomial}(n, F_p(q\sigma))`, and the error is a variance
divided by that count, so the count's distribution has to be summed over rather
than replaced by its mean.

What the estimator rests on
===========================

The empirical mean is unbiased because who shares is independent of what is
being measured: :math:`p_i` and :math:`b_i` are drawn independently, so
selecting on :math:`p_i` selects a random subset of the data. That is an
assumption about the population rather than a property of the mechanism, and
:func:`analyst_rule` is the minimum-variance unbiased estimator only while it
holds.

.. [BC2026] Benthall, S. and Cummings, R. (2026). "Principled Differential
       Privacy Parameter Tuning with Causal Games." FAccT '26.
       https://doi.org/10.1145/3805689.3806530
"""

import numpy as np
from scipy.stats import binom, norm

from skagent.block import Control, DBlock, Entity, RBlock
from skagent.distributions import Normal, Uniform

Q = 1.0
"""What a subject gains by sharing, and the paper's own value throughout."""

SIGMA_P = 1.0
"""The spread of privacy concern, and the paper's own: ``p ~ N(0, 1)``."""

DELTA = 1.0
"""The width of individual variation, so ``zeta ~ U(-1/2, 1/2)``.

Bounded rather than normal because it is what bounds the sensitivity of the
statistic being estimated, which is what the mechanism's noise is calibrated to.
"""

SUBJECTS = 1000
"""How many data subjects, and the size the paper's figures are computed at."""

_NOBODY = 1e-12
"""The weight the prior enters the analyst's estimate with.

One pseudo-observation, small enough to leave a non-empty database unmoved and
large enough to be the whole estimate when the database is empty. It is what
lets the paper's branch on ``n' = 0`` be written as a weighted mean.
"""


def sharing_rule(sigma, q=Q):
    """Share iff privacy concern is below what sharing is worth.

    The subject's utility ``c * (q - p / sigma)`` is linear in the decision, so
    the optimum is at a vertex: share when the bracket is positive.

    Parameters
    ----------
    sigma : float
        The mechanism's noise scale.
    q : float, optional
        What a subject gains by sharing.

    Returns
    -------
    callable
        A rule of ``(b, p)``, returning 1.0 where the subject shares. It ignores
        ``b``: the subject's own data does not enter its utility, which is what
        makes the sharing decision uninformative about the data.
    """
    return lambda b, p: (p < q * sigma) * 1.0


def bounded_sharing_rule(sigma):
    """Share iff a lognormal privacy cost is below the benefit of sharing.

    The rule for the bounded utility ``c * (1 - exp(p) / (1 + sigma))`` that
    :func:`designer_objective` is written against.

    Parameters
    ----------
    sigma : float
        The mechanism's noise scale.

    Returns
    -------
    callable
        A rule of ``(b, p)``, returning 1.0 where the subject shares.
    """
    return lambda b, p: (np.exp(p) < 1 + sigma) * 1.0


def estimate(c, d, a_mean):
    """The mean of the reports received, and the prior mean when there are none.

    The minimum-variance unbiased estimator of the population mean, given that
    who shares is independent of what is being measured.

    A weighted sum over the whole class rather than a mean over a selection: a
    subject that did not share carries weight 0 and contributes to neither the
    numerator nor the denominator, so a null report and a zero weight are the
    same thing. The prior enters as one pseudo-observation of weight
    :data:`_NOBODY`, which leaves a non-empty database unmoved and is the whole
    estimate when the database is empty -- the paper's ``n' = 0`` branch, without
    a branch.

    Parameters
    ----------
    c : array
        The sharing decision, one per subject, as a weight in ``[0, 1]``.
    d : array
        The report, one per subject, already weighted by the decision.
    a_mean : float
        The mean of the analyst's prior over the population mean.

    Returns
    -------
    float
    """
    weight = c * 1.0
    return (d.sum() + a_mean * _NOBODY) / (weight.sum() + _NOBODY)


def analyst_rule(a_mean):
    """:func:`estimate` against a fixed prior mean, as a decision rule for ``f``.

    Parameters
    ----------
    a_mean : float
        The mean of the analyst's prior over the population mean.

    Returns
    -------
    callable
        A rule of ``(c, d)``, the class's decisions and reports.
    """
    return lambda c, d: estimate(c, d, a_mean)


population_block = DBlock(
    name="population",
    shocks={"a": (Normal, {"mu": "a_mean", "sigma": "a_sd"})},
)

dp_noise_block = DBlock(
    name="dp_noise",
    # One draw for the whole population: the central model's trusted analyst
    # adds noise once, to the estimate, rather than once per subject.
    shocks={"gamma": (Normal, {"mu": 0.0, "sigma": "sigma"})},
)

data_block = DBlock(
    name="data",
    shocks={
        "zeta": (Uniform, {"low": "-delta/2", "high": "delta/2"}),
        "p": (Normal, {"mu": 0.0, "sigma": "sigma_p"}),
    },
    dynamics={"b": lambda a, zeta: a + zeta},
)

sharing_block = DBlock(
    name="sharing",
    # `b` is in the information set because the subject knows its own data, and
    # the analysis can then say that it does not need it: the utility below
    # reads the privacy concern and the noise scale, and nothing else.
    dynamics={
        "c": Control(["b", "p"], lower_bound=0.0, upper_bound=1.0, agent="subject")
    },
)

privacy_cost_block = DBlock(
    name="privacy_cost",
    dynamics={"u": lambda c, p, q, sigma: c * (q - p / sigma)},
    reward={"u": "subject"},
)

bounded_privacy_cost_block = DBlock(
    name="bounded_privacy_cost",
    # The paper's Figure 5 utility: a lognormal privacy cost that more noise
    # always reduces, so a subject's gain from sharing is bounded by what it is
    # worth rather than unbounded as the noise vanishes.
    dynamics={"u": lambda c, p, sigma: c * (1 - np.exp(p) / (1 + sigma))},
    reward={"u": "subject"},
)

local_report_block = DBlock(
    name="local_report",
    # The subject privatizes its own data before sharing it.
    dynamics={"d": lambda c, b, gamma: c * (b + gamma)},
)

central_report_block = DBlock(
    name="central_report",
    # The subject shares its data as it is, with an analyst it trusts.
    dynamics={"d": lambda c, b: c * b},
)

estimate_block = DBlock(
    name="estimate",
    # The whole class is in this information set, which is why the rule is
    # supplied rather than learned.
    dynamics={"f": Control(["c", "d"], agent="analyst")},
)

local_accuracy_block = DBlock(
    name="local_accuracy",
    # The estimate is already private, since every report it averages was.
    dynamics={"g": lambda f, a: -((f - a) ** 2)},
    reward={"g": "analyst"},
)

central_accuracy_block = DBlock(
    name="central_accuracy",
    # The estimate is private only once the analyst has added noise to it, so
    # accuracy is measured on what it publishes rather than on what it computed.
    dynamics={
        "f_dp": lambda f, gamma: f + gamma,
        "g": lambda f_dp, a: -((f_dp - a) ** 2),
    },
    reward={"g": "analyst"},
)


local_noise_block = DBlock(
    name="local_noise",
    # One draw per subject: the local model has each subject privatize its own
    # data, so the noise is an attribute of the subject rather than of the model.
    shocks={"gamma": (Normal, {"mu": 0.0, "sigma": "sigma"})},
)


def _subjects(data, cost, report):
    """The subject class: what its members hold, decide, are paid and report."""
    return RBlock(
        name="subjects",
        entity=Entity("subject"),
        blocks=[data, sharing_block, cost, *report],
    )


def _local(data, cost):
    return RBlock(
        name="local_dp",
        blocks=[
            population_block,
            _subjects(data, cost, [local_noise_block, local_report_block]),
            estimate_block,
            local_accuracy_block,
        ],
    )


local_block = _local(data_block, privacy_cost_block)
"""The local model: each subject privatizes its own report."""

bounded_local_block = _local(data_block, bounded_privacy_cost_block)
"""The local model under the bounded utility :func:`designer_objective` uses."""

central_block = RBlock(
    name="central_dp",
    blocks=[
        population_block,
        dp_noise_block,
        _subjects(data_block, privacy_cost_block, [central_report_block]),
        estimate_block,
        central_accuracy_block,
    ],
)
"""The central model: a trusted analyst adds noise once, to the estimate."""


def calibration(
    sigma,
    size=SUBJECTS,
    q=Q,
    sigma_p=SIGMA_P,
    delta=DELTA,
    a_mean=0.0,
    a_sd=1.0,
):
    """A calibration of either game at noise scale *sigma*.

    Parameters
    ----------
    sigma : float
        The mechanism's noise scale, and the designer's decision. Must be
        positive: the base utility divides by it.
    size : int, optional
        How many data subjects.
    q : float, optional
        What a subject gains by sharing.
    sigma_p : float, optional
        The spread of privacy concern.
    delta : float, optional
        The width of individual variation, which bounds the sensitivity of the
        statistic and so what the mechanism's noise is calibrated to.
    a_mean, a_sd : float, optional
        The analyst's prior over the population mean. ``a_sd`` is what an
        estimate falls back on when nobody shares, so it is part of the accuracy
        the designer weighs and not only a simulation detail.

    Returns
    -------
    dict
    """
    return {
        "sigma": sigma,
        "q": q,
        "sigma_p": sigma_p,
        "delta": delta,
        "a_mean": a_mean,
        "a_sd": a_sd,
        "subject": size,
    }


def share_probability(sigma, q=Q, sigma_p=SIGMA_P):
    """The fraction of subjects who share at noise scale *sigma*.

    ``F_p(q * sigma)``, the privacy-concern CDF at the benefit of sharing.

    Parameters
    ----------
    sigma, q, sigma_p : float
        As in :func:`calibration`.

    Returns
    -------
    float
    """
    return float(norm.cdf(q * sigma / sigma_p))


def bounded_share_probability(sigma, sigma_p=SIGMA_P):
    """The fraction who share under :func:`bounded_sharing_rule`.

    Parameters
    ----------
    sigma, sigma_p : float
        As in :func:`calibration`.

    Returns
    -------
    float
    """
    return float(norm.cdf(np.log1p(sigma) / sigma_p))


def _reciprocal_count(size, share):
    """``E[1/n' | n' > 0] Pr[n' > 0]`` for ``n' ~ Binomial(size, share)``.

    The error of a mean is a variance over a count, and the count is random, so
    this is summed over its distribution rather than evaluated at its mean.
    """
    counts = np.arange(1, size + 1)
    return float(np.sum(binom.pmf(counts, size, share) / counts))


def _averaged_variance(share, per_report, size, a_sd):
    """A per-report variance averaged over a random number of reports.

    The number of reports is binomial, and the error of a mean is a variance
    over a count, so the count is summed over rather than replaced by its mean.
    An empty database is what the prior variance is paid for.
    """
    return per_report * _reciprocal_count(size, share) + a_sd**2 * (1 - share) ** size


def local_error(sigma, size=SUBJECTS, q=Q, sigma_p=SIGMA_P, delta=DELTA, a_sd=1.0):
    """The analyst's expected squared error in the local model.

    Every report carries the mechanism's noise as well as individual variation,
    so the variance being averaged down is ``sigma^2 + delta^2 / 12``. More noise
    therefore costs accuracy directly and buys it back by drawing more subjects
    in, which is the trade the designer is making.

    Parameters
    ----------
    sigma, size, q, sigma_p, delta, a_sd : float
        As in :func:`calibration`.

    Returns
    -------
    float
        ``E[-g]``, so smaller is better.
    """
    share = share_probability(sigma, q, sigma_p)
    return _averaged_variance(share, sigma**2 + delta**2 / 12, size, a_sd)


def central_error(sigma, size=SUBJECTS, q=Q, sigma_p=SIGMA_P, delta=DELTA, a_sd=1.0):
    """The analyst's expected squared error in the central model.

    The noise is added once rather than per report, so it is not averaged down:
    it enters as ``sigma^2`` whatever the number of subjects, while what is
    averaged down is individual variation alone. That is why the two models make
    different designs optimal at the same privacy guarantee.

    Parameters
    ----------
    sigma, size, q, sigma_p, delta, a_sd : float
        As in :func:`calibration`.

    Returns
    -------
    float
        ``E[-g]``, so smaller is better.
    """
    share = share_probability(sigma, q, sigma_p)
    return _averaged_variance(share, delta**2 / 12, size, a_sd) + sigma**2


def expected_subject_utility(sigma, q=Q, sigma_p=SIGMA_P):
    """A subject's expected utility under the base utility and its rule.

    Unbounded as *sigma* falls: the subjects who share at a small noise scale
    are those who most dislike privacy loss, and the utility credits them
    ``-p / sigma`` for sharing anyway.

    Parameters
    ----------
    sigma, q, sigma_p : float
        As in :func:`calibration`.

    Returns
    -------
    float
    """
    threshold = q * sigma / sigma_p
    # E[p 1{p < k}] = -sigma_p phi(k / sigma_p) for p ~ N(0, sigma_p^2).
    return float(q * norm.cdf(threshold) + (sigma_p / sigma) * norm.pdf(threshold))


def bounded_subject_utility(sigma, sigma_p=SIGMA_P):
    """A subject's expected utility under the bounded, lognormal-cost utility.

    ``E[c (1 - exp(p) / (1 + sigma))]`` under :func:`bounded_sharing_rule`. Rises
    in *sigma* and is bounded above by the benefit of sharing, which is what
    makes the designer's objective have an interior maximum.

    Parameters
    ----------
    sigma, sigma_p : float
        As in :func:`calibration`.

    Returns
    -------
    float
    """
    threshold = np.log1p(sigma)
    # E[exp(p) 1{p < t}] = exp(sigma_p^2 / 2) Phi((t - sigma_p^2) / sigma_p).
    cost = np.exp(sigma_p**2 / 2) * norm.cdf((threshold - sigma_p**2) / sigma_p)
    return float(bounded_share_probability(sigma, sigma_p) - cost / (1 + sigma))


def designer_objective(sigma, size=SUBJECTS, sigma_p=SIGMA_P, delta=DELTA, a_sd=1.0):
    """``h = E[g] + E[u_i]``: the paper's example of a designer's objective.

    The analyst's accuracy plus a subject's welfare, under the bounded utility
    and the local model. Accuracy falls in *sigma* and welfare rises in it, so
    the maximum is interior: a designer weighing both chooses a noise scale that
    is neither zero nor unbounded, which is the paper's result.

    A designer with a different objective is a different function here, and that
    is the point of the argument rather than a limitation of it: what is being
    designed is a trade between the parties, so the trade has to be stated.

    Parameters
    ----------
    sigma, size, sigma_p, delta, a_sd : float
        As in :func:`calibration`.

    Returns
    -------
    float

    See Also
    --------
    bounded_subject_utility : the welfare term.
    local_error : the accuracy term, negated here.
    """
    share = bounded_share_probability(sigma, sigma_p)
    accuracy = _averaged_variance(share, sigma**2 + delta**2 / 12, size, a_sd)
    return bounded_subject_utility(sigma, sigma_p) - accuracy


def privacy_epsilon(sigma, delta=DELTA, dp_delta=1e-5):
    """The ``epsilon`` the Gaussian mechanism gives at noise scale *sigma*.

    What the designer's choice means as a privacy guarantee: the Gaussian
    mechanism is ``(epsilon, dp_delta)``-differentially private for
    ``epsilon = sensitivity * sqrt(2 log(1.25 / dp_delta)) / sigma``, and the
    sensitivity of a mean whose entries vary by at most *delta* is *delta*.

    Parameters
    ----------
    sigma : float
        The mechanism's noise scale.
    delta : float, optional
        The width of individual variation, which is the sensitivity.
    dp_delta : float, optional
        The mechanism's failure probability.

    Returns
    -------
    float
        Smaller is a stronger guarantee.
    """
    return float(delta * np.sqrt(2 * np.log(1.25 / dp_delta)) / sigma)
