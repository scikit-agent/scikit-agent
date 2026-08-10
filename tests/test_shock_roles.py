"""Tests for the shock-role criterion of :mod:`skagent.relevance`.

Two layers, in increasing distance from the graph:

1. What the two graph repairs buy the criterion, on real blocks: without them a
   shock paid through the next period looks irrelevant. The repairs themselves are
   tested in ``test_influence``, as is the d-separation engine underneath.
2. The classification itself, pinned as a table over every benchmark and every
   conftest case. A regression then names the block and the shock rather than
   showing up as numeric drift in a solver test.
"""

import pytest

from skagent.block import Control, DBlock
from skagent.distributions import Bernoulli
from skagent.influence import LAG_SUFFIX
from skagent.relevance import (
    HIDDEN,
    MIXED,
    OBSERVED,
    classify_shock,
)
from skagent.model_analyzer import ModelAnalyzer
from skagent.models import benchmarks

import conftest


# --------------------------------------------------------- the graph repairs


def test_continuation_is_what_makes_a_survival_shock_reachable():
    """A shock reaching the objective only through the next period's value."""
    block = DBlock(
        name="survival",
        shocks={"live": (Bernoulli, {"p": "SurvivalProb"})},
        dynamics={
            "m": lambda a, R: a * R + 1,
            "c": Control(["m"], agent="consumer"),
            "u": lambda c: c,
            "a": lambda m, c: m - c,
            "liv": lambda liv, live: liv * live,
        },
        reward={"u": "consumer"},
    )
    analyzer = ModelAnalyzer(block, {"R": 1.03, "SurvivalProb": 0.98}).analyze()

    def role(scim):
        return classify_shock(scim, "live", "c")

    # Without the continuation node ``live`` influences nothing the diagram sees.
    assert role(analyzer.influence_graph(dynamic=False)) == OBSERVED
    assert role(analyzer.influence_graph(dynamic=True)) == HIDDEN


def test_lagged_arrivals_make_a_decisions_parents_its_iset():
    """An iset naming a reassigned variable resolves to that variable's ``*`` node."""
    block = DBlock(
        name="reassigned_iset",
        dynamics={
            "c": Control(["a"], agent="consumer"),
            "a": lambda a, c: a - c,
            "u": lambda c: c,
        },
        reward={"u": "consumer"},
    )
    analyzer = ModelAnalyzer(block, {}).analyze()

    plain = analyzer.influence_graph(dynamic=False)
    # The lag edge is dropped, so the decision has no parents at all.
    assert plain.parents("c") == []

    dynamic = analyzer.influence_graph(dynamic=True)
    assert set(dynamic.parents("c")) == {f"a{LAG_SUFFIX}"}
    # The plain node keeps the end-of-period value, so it is downstream of ``c``.
    assert set(dynamic.parents("a")) == {"c", f"a{LAG_SUFFIX}"}


# --------------------------------------------------------------- the table


#: Expected roles for every conftest case.
#:
#: ``case_3`` and ``case_4`` are the pair worth reading together: both derive a
#: pre-state ``m`` from a shock, but ``case_3`` declares ``m = a + theta`` before
#: its control and ``case_4`` declares it after. So ``case_3``'s control
#: conditions on a pre-state the shock feeds, while ``case_4``'s conditions on the
#: arrival value and the shock reaches only the next period. Declaration order is
#: the whole difference, and it flips the role.
CONFTEST_CASES = {
    "case_0": {"c": {}},
    "case_1": {"c": {"theta": OBSERVED}},
    "case_2": {"c": {"theta": HIDDEN}},
    "case_3": {"c": {"theta": OBSERVED, "psi": HIDDEN}},
    "case_4": {"c": {"theta": HIDDEN, "psi": HIDDEN}},
    "case_5": {"c": {"theta": HIDDEN}},
    "case_6": {"c": {"theta": HIDDEN}},
    "case_7": {"c": {"theta": HIDDEN}},
    "case_8": {"c": {"theta": HIDDEN}},
    "case_9": {"c": {}},
    "case_10": {"c": {}, "d": {}},
    "case_11": {"c": {"theta": OBSERVED}},
}


@pytest.mark.parametrize("name", sorted(CONFTEST_CASES))
def test_conftest_case_shock_roles(name):
    case = getattr(conftest, name)
    calibration = case.get("calibration", {"beta": 0.9})
    assert case["block"].shock_roles(calibration) == CONFTEST_CASES[name]


def test_control_with_no_reachable_objective_refuses():
    """A decision reaching neither a reward it owns nor a continuation."""
    block = DBlock(
        name="unrewarded_control",
        shocks={"psi": (Bernoulli, {"p": 0.5})},
        dynamics={
            # 'a' is never reassigned, so there is no route to a continuation
            # either, and 'u' belongs to a different agent.
            "c": Control(["a"], agent="planner"),
            "u": lambda c, psi: c + psi,
        },
        reward={"u": "consumer"},
    )
    with pytest.raises(ValueError, match="owns no reward downstream"):
        block.shock_roles({})


def test_continuation_alone_is_a_sufficient_objective():
    """A decision paid only through the next period still has a problem to solve."""
    block = DBlock(
        name="investment_only",
        shocks={"psi": (Bernoulli, {"p": 0.5})},
        dynamics={
            "c": Control(["a"], agent="planner"),
            "a": lambda a, c, psi: a - c + psi,
            "u": lambda c: c,
        },
        reward={"u": "consumer"},
    )
    assert block.shock_roles({}) == {"c": {"psi": HIDDEN}}


#: Every benchmark's expected classification, per control.
#:
#: The income shocks of U-1/U-2/U-3 are OBSERVED because each reaches the
#: objective only through the cash-on-hand pre-state ``m`` that ``c``'s
#: information set contains -- which is the point: none of them appears in an
#: ``iset``, and a syntactic test would call all three hidden. U-3 in particular
#: shows the criterion does not require the shock to be *recoverable* from ``m``:
#: ``m = R*a/psi + theta`` pins down neither ``psi`` nor ``theta`` individually.
#:
#: D-3's ``live`` is HIDDEN because it reaches the objective through next
#: period's survival state, not through ``m``.
BENCHMARK_ROLES = {
    "D-1": {"c": {}},
    "D-2": {"c": {}},
    "D-3": {"c": {"live": HIDDEN}},
    "D-4": {"c": {}},
    "U-1": {"c": {"eta": OBSERVED}},
    "U-2": {"c": {"psi": OBSERVED}},
    "U-3": {"c": {"psi": OBSERVED, "theta": OBSERVED}},
}


@pytest.mark.parametrize("model_id", sorted(BENCHMARK_ROLES))
def test_benchmark_shock_roles(model_id):
    block = benchmarks.get_benchmark_model(model_id)
    calibration = benchmarks.get_benchmark_calibration(model_id)
    assert block.shock_roles(calibration) == BENCHMARK_ROLES[model_id]


def test_no_benchmark_is_mixed():
    """MIXED means the solver must refuse; no benchmark should trip it."""
    for model_id in BENCHMARK_ROLES:
        block = benchmarks.get_benchmark_model(model_id)
        roles = block.shock_roles(benchmarks.get_benchmark_calibration(model_id))
        for control, shocks in roles.items():
            assert MIXED not in shocks.values(), (model_id, control, shocks)


# ------------------------------------------------------- the MIXED frontier


def test_mixed_when_a_shock_both_feeds_the_iset_and_bypasses_it():
    """A shock reaching the objective around its own pre-state is MIXED.

    Here ``psi`` feeds ``m``, which ``c`` conditions on, but it *also* enters the
    reward directly -- the shape of a normalized model whose utility was written
    in levels. The information set is partly informative about ``psi`` while the
    objective depends on it separately, so neither per-node solving nor
    integrating inside the maximization poses the declared problem.
    """
    block = DBlock(
        name="levels_utility_in_normalized_model",
        shocks={"psi": (Bernoulli, {"p": "SurvivalProb"})},
        dynamics={
            "m": lambda a, psi: a / psi,
            "c": Control(["m"], agent="consumer"),
            "u": lambda c, psi: psi * c,  # <- the bypassing route
            "a": lambda m, c: m - c,
        },
        reward={"u": "consumer"},
    )
    roles = block.shock_roles({"SurvivalProb": 0.9})
    assert roles["c"]["psi"] == MIXED


def test_roles_may_differ_between_controls_in_one_period():
    """A shock accounted for by one iset can be hidden to a narrower one."""
    block = DBlock(
        name="asymmetric_information",
        shocks={"psi": (Bernoulli, {"p": "SurvivalProb"})},
        dynamics={
            "m": lambda a, psi: a * psi,
            "c": Control(["m"], agent="consumer"),
            "d": Control([], agent="consumer"),  # sees nothing
            "u": lambda c, d: c + d,
            "a": lambda m, c, d: m - c - d,
        },
        reward={"u": "consumer"},
    )
    roles = block.shock_roles({"SurvivalProb": 0.9})
    assert roles["c"]["psi"] == OBSERVED
    assert roles["d"]["psi"] == HIDDEN
