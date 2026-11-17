"""Compare policies under contextual outcome guarantees.

:func:`contextual_payoff` fixes outcomes; :func:`compare_references` searches
settings; :func:`compare_outcome_sets` searches outcomes. :func:`minimal_subsets`
removes supersets. Callers supply the model, policies, and setting mapper.

Payoffs cover one block pass and discretize continuous shocks. Searches are
exponential. Multi-outcome searches share one set of guaranteed settings; see
:mod:`skagent.relevance` for graphical criteria.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from itertools import combinations
from typing import Any, TypeVar

from skagent.ground import Discretized, ExpectedPayoff, GroundedBlock

_T = TypeVar("_T", bound=Hashable)

__all__ = [
    "contextual_payoff",
    "minimal_subsets",
    "compare_references",
    "compare_outcome_sets",
]


# -- contextual intervention --------------------------------------------------


def contextual_payoff(
    ground: GroundedBlock,
    original_policy: Mapping[str, Callable[..., Any]],
    alternative_policy: Mapping[str, Callable[..., Any]],
    outcome: str | Iterable[str],
    guaranteed: Iterable[Hashable],
    setting_of: Callable[[Mapping[str, Any]], Hashable],
    *,
    states: Mapping[str, Any] | None = None,
    agent: str | None = None,
) -> ExpectedPayoff:
    """Evaluate an alternative with original outcomes fixed in selected settings.

    Fix outcomes where ``setting_of(shocks)`` belongs to *guaranteed*. Policies
    must cover every control. Payoffs use :class:`~skagent.ground.Discretized`.

    Parameters
    ----------
    ground : GroundedBlock
        Block and calibration.
    original_policy, alternative_policy : mapping of str to callable
        One decision rule per control.
    outcome : str or iterable of str
        Computed non-decision variables, including rewards.
    guaranteed : iterable of hashable
        Settings to guarantee outcomes in.
    setting_of : callable
        Maps shocks to a setting.
    states : mapping of str to Any, default=None
        Arrival-state values; None means none.
    agent : str, default=None
        Payee; None sums all rewards.

    Returns
    -------
    ExpectedPayoff
        Expected payoff under the guarantee.

    Raises
    ------
    ValueError
        If outcomes are empty, repeated, or not computed non-decision variables.

    References
    ----------
    Ward et al. (2024), "The Reasons that Agents Act: Intention and
    Instrumental Goals," Definition 5. :arxiv:`2402.07221`.

    """
    outcomes = (outcome,) if isinstance(outcome, str) else tuple(outcome)
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError("outcomes must be nonempty and distinct")
    # Block.transition cannot fix shocks or controls.
    for name in outcomes:
        if (
            name not in ground.block.get_dynamics()
            or name in ground.block.get_controls()
        ):
            raise ValueError(f"{name!r} must be a non-decision dynamics variable")

    selected = frozenset(guaranteed)
    # Keep measure metadata when nothing is fixed.
    if not selected:
        return ground.expected_payoff(
            alternative_policy, Discretized(), states=states, agent=agent
        )

    ground.block._require_paid_agent(agent)

    def at_setting(shocks: Mapping[str, Any]) -> Any:
        pre = {**ground.calibration, **(states or {}), **shocks}
        fixed = setting_of(shocks) in selected
        if fixed:
            original = ground.block.transition(pre, original_policy)
            pre.update((name, original[name]) for name in outcomes)
        # fix keeps pre-supplied outcomes from being recomputed.
        result = ground.block.transition(
            pre, alternative_policy, fix=outcomes if fixed else []
        )
        rewards = ground.block.calc_reward(
            result, agent=agent, fixed=outcomes if fixed else ()
        )
        return sum(rewards.values())

    return Discretized().reduce(ground, at_setting)


# -- subset minimality --------------------------------------------------------


def minimal_subsets(
    sets: Iterable[Iterable[_T]],
) -> tuple[frozenset[_T], ...]:
    """Return distinct candidates with no proper subset in *sets*.

    Parameters
    ----------
    sets : iterable of iterable of hashable
        Candidates in input order.

    Returns
    -------
    tuple of frozenset
        Minimal candidates in first-occurrence order.
    """
    candidates = tuple(dict.fromkeys(frozenset(candidate) for candidate in sets))
    return tuple(
        candidate
        for candidate in candidates
        if not any(other < candidate for other in candidates)
    )


# -- reference comparisons ----------------------------------------------------


def compare_references(
    ground: GroundedBlock,
    original_policy: Mapping[str, Callable[..., Any]],
    references: Mapping[Hashable, Mapping[str, Callable[..., Any]]],
    outcome: str | Iterable[str],
    settings: Iterable[Hashable],
    setting_of: Callable[[Mapping[str, Any]], Hashable],
    *,
    states: Mapping[str, Any] | None = None,
    agent: str | None = None,
    tolerance: float = 1e-10,
) -> dict[Hashable, tuple[ExpectedPayoff, tuple[frozenset[Hashable], ...]]]:
    """Find minimal settings that close each reference policy's payoff gap.

    Tests every subset of *settings* for the supplied outcome set.

    Parameters
    ----------
    ground : GroundedBlock
        Block and calibration.
    original_policy : mapping of str to callable
        Policy whose outcomes are guaranteed.
    references : mapping of hashable to mapping of str to callable
        Named alternative policies.
    outcome : str or iterable of str
        Computed non-decision variables to guarantee together.
    settings : iterable of hashable
        Distinct settings to search.
    setting_of : callable
        Maps shocks to a setting in *settings*.
    states : mapping of str to Any, default=None
        Arrival-state values; None means none.
    agent : str, default=None
        Payee; None sums all rewards.
    tolerance : float, default=1e-10
        Maximum payoff shortfall for a tie.

    Returns
    -------
    dict of hashable to tuple
        Each name maps to its unmodified ExpectedPayoff and a tuple of minimal
        guarantee sets.

    Raises
    ------
    ValueError
        If settings repeat or tolerance is negative.

    References
    ----------
    Ward et al. (2024), "The Reasons that Agents Act: Intention and
    Instrumental Goals," Definition 6. :arxiv:`2402.07221`.

    """
    outcome = outcome if isinstance(outcome, str) else tuple(outcome)
    settings = tuple(settings)
    if len(set(settings)) != len(settings):
        raise ValueError("settings must be distinct")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    # Each reference must reach the original payoff.
    baseline = float(
        ground.expected_payoff(
            original_policy, Discretized(), states=states, agent=agent
        )
    )
    # shortcut: exhaustive search grows exponentially, prune for many settings.
    subsets = tuple(
        frozenset(choice)
        for size in range(len(settings) + 1)
        for choice in combinations(settings, size)
    )
    comparison = {}
    for name, policy in references.items():
        payoffs = {
            subset: contextual_payoff(
                ground,
                original_policy,
                policy,
                outcome,
                subset,
                setting_of,
                states=states,
                agent=agent,
            )
            for subset in subsets
        }
        # A guarantee qualifies when it closes the gap.
        sufficient = [
            subset
            for subset, payoff in payoffs.items()
            if float(payoff) >= baseline - tolerance
        ]
        # Keep all incomparable minima.
        minimal = minimal_subsets(sufficient)
        comparison[name] = (payoffs[frozenset()], minimal)
    return comparison


def compare_outcome_sets(
    ground: GroundedBlock,
    original_policy: Mapping[str, Callable[..., Any]],
    references: Mapping[Hashable, Mapping[str, Callable[..., Any]]],
    outcomes: str | Iterable[str],
    guaranteed: Iterable[Hashable],
    setting_of: Callable[[Mapping[str, Any]], Hashable],
    *,
    states: Mapping[str, Any] | None = None,
    agent: str | None = None,
    tolerance: float = 1e-10,
) -> dict[Hashable, tuple[frozenset[str], ...]]:
    """Find minimal outcome sets that close each reference policy's payoff gap.

    Every outcome uses the same *guaranteed* settings. Settings are not searched.

    Parameters
    ----------
    ground : GroundedBlock
        Block and calibration.
    original_policy : mapping of str to callable
        Policy whose outcomes are guaranteed.
    references : mapping of hashable to mapping of str to callable
        Named alternative policies.
    outcomes : str or iterable of str
        Distinct computed non-decision variables to search.
    guaranteed : iterable of hashable
        Settings for every selected outcome.
    setting_of : callable
        Maps shocks to a setting.
    states : mapping of str to Any, default=None
        Arrival-state values; None means none.
    agent : str, default=None
        Payee; None sums all rewards.
    tolerance : float, default=1e-10
        Maximum payoff shortfall for a tie.

    Returns
    -------
    dict of hashable to tuple of frozenset
        Minimal sufficient sets per reference. An already tied policy gets
        ``(frozenset(),)``; one that cannot tie gets ``()``.

    Raises
    ------
    ValueError
        If outcomes are empty or repeated, or tolerance is negative.

    References
    ----------
    Ward et al. (2024), "The Reasons that Agents Act: Intention and
    Instrumental Goals," Definition 6. :arxiv:`2402.07221`.
    """
    outcomes = (outcomes,) if isinstance(outcomes, str) else tuple(outcomes)
    guaranteed = frozenset(guaranteed)
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError("outcomes must be nonempty and distinct")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    baseline = float(
        ground.expected_payoff(
            original_policy, Discretized(), states=states, agent=agent
        )
    )
    comparison = {}
    for name, policy in references.items():
        unmodified = contextual_payoff(
            ground,
            original_policy,
            policy,
            outcomes,
            (),
            setting_of,
            states=states,
            agent=agent,
        )
        if float(unmodified) >= baseline - tolerance:
            comparison[name] = (frozenset(),)
            continue

        # shortcut: exhaustive outcome subsets, use pruning for many outcomes.
        sufficient = (
            choice
            for size in range(1, len(outcomes) + 1)
            for choice in combinations(outcomes, size)
            if float(
                contextual_payoff(
                    ground,
                    original_policy,
                    policy,
                    choice,
                    guaranteed,
                    setting_of,
                    states=states,
                    agent=agent,
                )
            )
            >= baseline - tolerance
        )
        comparison[name] = minimal_subsets(sufficient)
    return comparison
