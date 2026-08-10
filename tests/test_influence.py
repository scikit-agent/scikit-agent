"""Tests for :mod:`skagent.influence` -- the influence-diagram substrate.

Two layers:

1. The engine: ``d_connected`` against ``networkx.is_d_separator`` as an oracle,
   on random DAGs. This is what makes hand-rolling the traversal safe, and it is
   non-optional for any change to the sweep.
2. The transforms, in isolation, on hand-built graphs.
"""

import itertools
import random

import networkx as nx
import pytest

from skagent.influence import (
    CONTINUATION_PREFIX,
    DUMMY_PREFIX,
    LAG_SUFFIX,
    SCIM,
)


def _scim(graph, decisions=(), agent_utilities=None, decision_agent=None):
    """A SCIM over ``graph``; node roles default to empty for engine tests."""
    return SCIM(graph, decisions, agent_utilities or {}, decision_agent or {})


def _random_dag(n_nodes, edge_prob, rng):
    """A random DAG on ``0..n_nodes-1``, edges only from lower to higher."""
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for i, j in itertools.combinations(range(n_nodes), 2):
        if rng.random() < edge_prob:
            g.add_edge(i, j)
    return g


# ------------------------------------------------------------------- engine


@pytest.mark.parametrize("seed", range(25))
def test_d_connected_matches_networkx_oracle(seed):
    """``d_connected`` must agree with ``networkx.is_d_separator`` node by node."""
    rng = random.Random(seed)
    graph = _random_dag(rng.randint(3, 8), 0.35, rng)
    nodes = list(graph)

    targets = set(rng.sample(nodes, rng.randint(1, 2)))
    remaining = [n for n in nodes if n not in targets]
    given = (
        set(rng.sample(remaining, rng.randint(0, len(remaining))))
        if remaining
        else set()
    )

    reachable = _scim(graph).d_connected(targets, given)

    for node in nodes:
        if node in targets or node in given:
            # Excluded from the result by construction; d-separation of a node
            # from itself, or of a conditioned node, is not what is being asked.
            assert node not in reachable
            continue
        expected = not nx.is_d_separator(graph, {node}, targets, given)
        assert (node in reachable) == expected, (
            f"seed={seed} node={node} targets={targets} given={given}"
        )


def test_d_connected_blocks_through_conditioned_chain():
    #  s -> m -> u  is blocked by m, and open without it.
    scim = _scim(nx.DiGraph([("s", "m"), ("m", "u")]))
    assert "s" in scim.d_connected({"u"}, set())
    assert "s" not in scim.d_connected({"u"}, {"m"})


def test_d_connected_opens_collider_when_conditioned():
    #  a -> m <- b : conditioning on the collider makes a and b dependent.
    scim = _scim(nx.DiGraph([("a", "m"), ("b", "m")]))
    assert "b" not in scim.d_connected({"a"}, set())
    assert "b" in scim.d_connected({"a"}, {"m"})


def test_d_connected_no_targets_is_empty():
    scim = _scim(nx.DiGraph([("a", "b")]))
    assert scim.d_connected(set(), {"a"}) == set()
    assert scim.d_connected({"absent"}, set()) == set()


def test_d_connected_is_memoized_per_query():
    """Repeat queries must be served from the cache, not retraversed."""
    scim = _scim(nx.DiGraph([("s", "m"), ("m", "u")]))
    first = scim.d_connected({"u"}, {"m"})
    assert scim.d_connected(["u"], ["m"]) is first  # order and type are immaterial
    assert scim.d_connected({"u"}, set()) is not first


def test_ancestors_is_transitive_and_strict():
    scim = _scim(nx.DiGraph([("a", "b"), ("b", "c"), ("x", "c")]))
    assert scim.ancestors(["c"]) == {"a", "b", "x"}
    assert scim.ancestors(["a"]) == set()
    assert scim.ancestors(["absent"]) == set()


# --------------------------------------------------------------- vocabulary


def test_context_is_the_decision_family():
    scim = _scim(nx.DiGraph([("m", "c"), ("z", "c"), ("c", "u")]), decisions=["c"])
    assert scim.context("c") == {"m", "z", "c"}
    assert sorted(scim.parents("c")) == ["m", "z"]


def test_objectives_are_owned_and_downstream():
    graph = nx.DiGraph([("c", "u"), ("d", "v")])
    scim = _scim(
        graph,
        decisions=["c", "d"],
        agent_utilities={"consumer": ["u", "v"]},
        decision_agent={"c": "consumer", "d": "consumer"},
    )
    # ``v`` is owned by c's agent but is not downstream of c.
    assert scim.objectives("c") == {"u"}
    assert scim.objectives("d") == {"v"}


def test_objectives_empty_without_ownership():
    graph = nx.DiGraph([("c", "u")])
    scim = _scim(
        graph,
        decisions=["c"],
        agent_utilities={"planner": ["u"]},
        decision_agent={"c": "consumer"},
    )
    assert scim.objectives("c") == set()


# --------------------------------------------------------------- transforms


def test_with_continuation_adds_one_node_per_deciding_agent():
    graph = nx.DiGraph([("m", "c"), ("c", "a")])
    scim = _scim(graph, decisions=["c"], decision_agent={"c": "consumer"})
    out = scim.with_continuation(["a"])

    node = f"{CONTINUATION_PREFIX}consumer"
    assert node in out.graph
    assert out.parents(node) == ["a"]
    assert out.agent_utilities["consumer"] == [node]
    # The input is untouched.
    assert node not in graph
    assert scim.agent_utilities == {}


def test_with_continuation_preserves_existing_utilities():
    scim = _scim(
        nx.DiGraph([("c", "u"), ("c", "a")]),
        decisions=["c"],
        agent_utilities={"consumer": ["u"]},
        decision_agent={"c": "consumer"},
    )
    out = scim.with_continuation(["a"])
    assert out.agent_utilities["consumer"] == ["u", f"{CONTINUATION_PREFIX}consumer"]


def test_with_continuation_skips_ungraphed_arrival_states():
    scim = _scim(
        nx.DiGraph([("c", "a")]), decisions=["c"], decision_agent={"c": "consumer"}
    )
    out = scim.with_continuation(["a", "not_a_node"])
    assert set(out.parents(f"{CONTINUATION_PREFIX}consumer")) == {"a"}


def test_with_continuation_skips_agents_that_do_not_decide():
    """Only a value function needs continuing, so only deciding agents get one."""
    scim = _scim(
        nx.DiGraph([("c", "u"), ("c", "a")]),
        decisions=["c"],
        agent_utilities={"consumer": ["u"], "bystander": []},
        decision_agent={"c": "consumer"},
    )
    out = scim.with_continuation(["a"])
    assert f"{CONTINUATION_PREFIX}bystander" not in out.graph


def test_with_lagged_arrivals_is_a_noop_without_lag_dependencies():
    scim = _scim(nx.DiGraph([("m", "c")]))
    assert sorted(scim.with_lagged_arrivals([]).graph.edges) == [("m", "c")]


def test_with_lagged_arrivals_splits_the_arrival_value_out():
    graph = nx.DiGraph([("c", "a")])
    graph.nodes["a"]["kind"] = "chance"
    scim = _scim(graph, decisions=["c"])
    out = scim.with_lagged_arrivals([("c", "a")])

    lagged = f"a{LAG_SUFFIX}"
    assert out.parents("c") == [lagged]
    # The plain node keeps its in-period parent: it is the end-of-period value.
    assert out.parents("a") == ["c"]
    assert out.graph.nodes[lagged]["kind"] == "chance"


def test_with_dummy_parent_is_exogenous_and_fresh():
    graph = nx.DiGraph([("D", "Dp"), ("Dp", "U")])
    scim = _scim(graph, decisions=["D", "Dp"])
    out, dummy = scim.with_dummy_parent("Dp")

    assert dummy == f"{DUMMY_PREFIX}Dp"
    assert out.parents(dummy) == []
    assert dummy in out.parents("Dp")
    assert dummy not in graph  # the input is untouched


def test_with_dummy_parent_sidesteps_a_name_collision():
    """A real node named like the dummy must not be reused as the dummy."""
    graph = nx.DiGraph([("D", "Dp"), (f"{DUMMY_PREFIX}Dp", "U")])
    out, dummy = _scim(graph, decisions=["D", "Dp"]).with_dummy_parent("Dp")
    assert dummy != f"{DUMMY_PREFIX}Dp"
    assert out.parents("Dp") == ["D", dummy]


def test_transforms_do_not_inherit_a_stale_cache():
    """A transform changes the graph, so the new SCIM must retraverse it."""
    scim = _scim(nx.DiGraph([("s", "u")]), decisions=["u"])
    assert scim.d_connected({"u"}, set()) == {"s"}

    out = scim.with_lagged_arrivals([("u", "s")])
    assert out.d_connected({"u"}, set()) == {"s", f"s{LAG_SUFFIX}"}
