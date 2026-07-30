"""
Information-structure criteria on influence-diagram models.

Answers the question a solver must settle before it can pose a decision problem
correctly: at the moment a control is chosen, does the model's declared
information set already account for a given shock?

The criterion is a d-separation test on the influence-diagram (SCIM) view of a
block, not a syntactic test on the information set. A shock that appears in no
information set may still be fully accounted for, because it feeds a derived
pre-decision variable that an information set does contain.

Each shock is classified, per control, into one of three roles:

``OBSERVED``
    Every route from the shock to the objective is intercepted by the
    information set. Conditioning on the information set therefore leaves nothing
    about the shock for the objective to depend on, and a solver may grid the
    shock over its discretization nodes and solve per node.

``HIDDEN``
    The shock reaches the objective around the information set, which carries no
    information about it. An expectation over it belongs inside the maximization.

``MIXED``
    The shock reaches the objective around the information set, *and* the
    information set is partly informative about it. Computing the declared
    problem then requires the conditional law of the shock given the information
    set, so neither per-node solving nor integrating inside the maximization
    applies and a solver should refuse. This most often indicates that a reward
    or transition touches a shock the control's information set claims not to
    see, which is a modeling error rather than a solver limitation.

Two limits of the criterion:

1. d-separation is sound but not complete under determinism, and these models are
   largely deterministic functions of their parents. The tests are therefore
   conservative: a route whose functional effect vanishes still counts. Errors
   fall toward ``HIDDEN`` or ``MIXED``, never toward wrongly reporting a shock as
   accounted for. Sharpening this requires treating functionally determined nodes
   as observed.
2. A single-period diagram cannot express a payoff arriving through the next
   period's value, nor distinguish a variable's arrival value from the value it
   is reassigned to. Apply :func:`with_continuation` and
   :func:`with_lagged_arrivals` first; :meth:`ModelAnalyzer.influence_graph` does
   both on request.

This module depends only on ``networkx``, like :mod:`skagent.relevance`, so the
criteria can be developed and tested without constructing a model.

.. note::

   **Anticipated refactoring.** :func:`ancestors`, :func:`d_connected`,
   :func:`with_continuation`, :func:`with_lagged_arrivals` and :func:`objectives`
   are shared influence-diagram vocabulary rather than criteria, and
   :mod:`skagent.relevance` re-derives parts of them. The intended destination is
   a ``SCIM`` class owning that vocabulary and memoizing the traversals, with the
   criteria proper as thin functions over it -- which would also retire the
   many-positional-argument style both modules inherit from the current ``SCIM``
   namedtuple. Deferred: it is a refactor of working, tested code that nothing
   here needs yet.
"""

import networkx as nx

__all__ = [
    "OBSERVED",
    "HIDDEN",
    "MIXED",
    "CONTINUATION_PREFIX",
    "LAG_SUFFIX",
    "ancestors",
    "d_connected",
    "with_continuation",
    "with_lagged_arrivals",
    "objectives",
    "classify_shock",
    "shock_roles",
]


#: A shock the information set accounts for; may be gridded per node.
OBSERVED = "observed"
#: A shock the information set says nothing about; integrate inside the max.
HIDDEN = "hidden"
#: Partly informed *and* separately relevant; needs filtering, so refuse.
MIXED = "mixed"

#: Node-name prefix for the synthetic continuation utility.
CONTINUATION_PREFIX = "__continuation__"

#: Suffix marking a variable's arrival value, as distinct from the end-of-period
#: value the plain name carries.
LAG_SUFFIX = "*"

# Trail directions for the Bayes-Ball sweep in :func:`d_connected`.
_UP, _DOWN = True, False


def ancestors(graph, nodes):
    """Every strict ancestor of any node in *nodes*.

    One reverse multi-source traversal, rather than a :func:`networkx.ancestors`
    call per node.
    """
    seen = set()
    stack = [n for n in nodes if n in graph]
    while stack:
        for parent in graph.predecessors(stack.pop()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def d_connected(graph, targets, given):
    """Every node d-connected to some node of *targets*, conditioning on *given*.

    The complement is the d-separated set, so ``node not in d_connected(...)``
    certifies that no active trail carries influence from ``node`` to any target.
    One traversal answers this for every node at once, so callers should ask once
    per decision and test membership rather than calling
    :func:`networkx.is_d_separator` per candidate.

    Implemented as the Bayes-Ball / ``Reachable`` sweep (Shachter 1998; Koller &
    Friedman Alg. 3.1). The ancestral-moral-graph construction does not serve
    this signature: moralization is valid only over the ancestral closure of
    ``{candidate} | targets | given``, which differs per candidate, and widening
    it to the whole graph marries the parents of colliders carrying no evidence.

    *targets* and *given* are excluded from the result. The graph must be a DAG.
    """
    targets = {n for n in targets if n in graph}
    given = {n for n in given if n in graph}
    if not targets:
        return set()

    # Nodes in ``given`` or with a descendant there. At such a node a collision
    # is unblocked, so a trail arriving from a parent may turn and go back up.
    evidence_at_or_below = given | ancestors(graph, given)

    # Trails are explored as (node, direction) pairs, where UP means the trail
    # reached this node from one of its children.
    frontier = [(t, _UP) for t in targets]
    seen = set()
    reachable = set()

    while frontier:
        node, direction = frontier.pop()
        if (node, direction) in seen:
            continue
        seen.add((node, direction))
        if node not in given:
            reachable.add(node)

        if direction is _UP and node not in given:
            frontier.extend((p, _UP) for p in graph.predecessors(node))
            frontier.extend((c, _DOWN) for c in graph.successors(node))
        elif direction is _DOWN:
            if node not in given:
                frontier.extend((c, _DOWN) for c in graph.successors(node))
            if node in evidence_at_or_below:
                frontier.extend((p, _UP) for p in graph.predecessors(node))

    return reachable - targets - given


def with_lagged_arrivals(graph, lag_dependencies):
    """Split each variable's arrival value out into its own node.

    A single-period diagram carries one node per symbol, but a symbol reassigned
    within the period holds two values: the one it arrives with and the one it is
    reassigned to. This adds a source node ``<name>*`` for the arrival value,
    with an edge to each consumer that reads it. The plain node keeps its
    in-period parents and so denotes the end-of-period value, which is what the
    next period arrives with.

    A decision's parents in the result are exactly its information set, so
    callers may read the conditioning set off the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
    lag_dependencies : iterable of (consumer, source) pairs
        Dependencies that read a source's pre-period value -- the edges a
        single-period projection would otherwise drop.

    Returns
    -------
    networkx.DiGraph
        A new graph; the input is not modified.
    """
    graph = graph.copy()
    for consumer, source in lag_dependencies:
        if consumer not in graph or source not in graph:
            continue
        lagged = f"{source}{LAG_SUFFIX}"
        if lagged not in graph:
            attrs = dict(graph.nodes[source])
            attrs["kind"] = "chance"  # an arrival value is exogenous to the period
            graph.add_node(lagged, **attrs)
        graph.add_edge(lagged, consumer)
    return graph


def with_continuation(graph, arrival_states, agents, agent_utilities=None):
    """Add a synthetic continuation-value utility node per agent.

    A single-period diagram cannot express that a decision's payoff continues
    into the next period, so a shock reaching the objective only through the next
    period's value appears irrelevant. This adds, per agent, a utility node whose
    parents are the nodes named after the model's arrival-state variables -- for a
    variable that is also reassigned in-period, that node holds the reassigned
    value, which is what the next period arrives with.

    Parameters
    ----------
    graph : networkx.DiGraph
    arrival_states : iterable
        The model's arrival-state variable names.
    agents : iterable
        Agents to add a continuation node for.
    agent_utilities : mapping, optional
        Existing ``{agent: [utility nodes]}``, not modified.

    Returns
    -------
    tuple
        ``(graph, agent_utilities)``, both new objects.
    """
    graph = graph.copy()
    agent_utilities = {a: list(u) for a, u in (agent_utilities or {}).items()}

    parents = [s for s in arrival_states if s in graph]
    for agent in agents:
        node = f"{CONTINUATION_PREFIX}{agent}"
        while node in graph:  # pathological name collision
            node = "_" + node
        graph.add_node(node, kind="utility", agent=agent)
        graph.add_edges_from((p, node) for p in parents)
        agent_utilities.setdefault(agent, []).append(node)

    return graph, agent_utilities


def objectives(graph, decision, agent_utilities, decision_agent):
    """The utility nodes whose value *decision* is choosing over.

    Utilities owned by the deciding agent and downstream of the decision, the
    convention :func:`skagent.relevance.is_s_reachable` also uses. Includes the
    synthetic continuation node when :func:`with_continuation` has been applied
    and the decision can reach it.
    """
    owned = set(agent_utilities.get(decision_agent.get(decision), ()))
    return owned & nx.descendants(graph, decision)


def classify_shock(graph, shock, decision, targets):
    """Classify one *shock* for one *decision*: OBSERVED, HIDDEN, or MIXED.

    Thin wrapper over :func:`shock_roles`; *targets* is the decision's objective
    node set (see :func:`objectives`).
    """
    return shock_roles(graph, [shock], [decision], {decision: targets})[decision][shock]


def shock_roles(graph, shocks, decisions, targets_by_decision):
    """Classify every shock for every decision.

    Parameters
    ----------
    graph : networkx.DiGraph
        The influence-diagram view, after :func:`with_lagged_arrivals` and
        :func:`with_continuation`. A decision's parents are taken as its
        information set.
    shocks : iterable
        Shock variable names.
    decisions : iterable
        Decision node names.
    targets_by_decision : mapping
        ``targets_by_decision[decision]`` is its objective nodes.

    Returns
    -------
    dict
        ``{decision: {shock: role}}``.

    Raises
    ------
    ValueError
        If a decision has no objective nodes. Nothing is then reachable, so every
        shock would classify ``OBSERVED`` -- the one direction that must not be
        silent, since it invites a solver to condition on a shock the agent
        cannot see. It means the deciding agent owns no reward downstream of its
        own decision.

    Notes
    -----
    The classification is per decision and may legitimately differ between two
    controls in one period: a shock accounted for by a rich information set is
    hidden to a control that conditions on less. A solver that represents a shock
    one way for the whole period must check that the roles agree across controls.

    One traversal per decision, then membership tests, so the cost is linear in
    the graph per decision rather than per (shock, decision) pair.
    """
    shocks = list(shocks)
    roles = {}

    for decision in decisions:
        targets = set(targets_by_decision.get(decision, ()))
        if not targets:
            raise ValueError(
                f"Decision '{decision}' has no objective nodes, so every shock "
                "would be reported as accounted-for. The agent deciding "
                f"'{decision}' owns no reward downstream of it; note that a "
                "control with no declared agent does not own a reward assigned "
                "to a named one."
            )

        conditioned = set(graph.predecessors(decision))
        reachable = d_connected(graph, targets, conditioned | {decision})
        informative = ancestors(graph, conditioned)

        decision_roles = {}
        for shock in shocks:
            if shock in conditioned or shock not in reachable:
                # Conditioned on directly, absent from the diagram, or every
                # route to the objective is intercepted.
                decision_roles[shock] = OBSERVED
            elif shock in informative:
                decision_roles[shock] = MIXED
            else:
                decision_roles[shock] = HIDDEN
        roles[decision] = decision_roles

    return roles
