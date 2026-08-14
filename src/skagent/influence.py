"""
The influence-diagram substrate the graphical criteria are posed over.

A block can be read two ways: executably, by running its dynamics, and
structurally, by asking which symbol *can* influence which without evaluating
anything. The second reading is what d-separation answers, and d-separation is
only definable on a graph. This module owns that graph -- the SCIM view: chance,
decision and utility nodes with directed causal edges -- together with the
vocabulary and the traversal engine every criterion over it needs.

The name is Def. 4 of Everitt, Carey, Langlois, Ortega & Legg, "Agent
Incentives: A Causal Perspective" (AAAI-21, 35(13):11487-11495;
arXiv:2102.01685), where a *structural causal influence model* is an influence
diagram whose mechanisms are structural functions of their parents rather than
conditional probability tables. That is the form a block already takes.

The criteria themselves live in :mod:`skagent.relevance`. Construction of a
:class:`SCIM` from a block lives in :mod:`skagent.model_analyzer`. This module
depends only on ``networkx``, so the substrate and the criteria can be developed
and tested without constructing a model.

The engine is the Bayes-Ball / ``Reachable`` sweep (Shachter 1998; Koller &
Friedman Alg. 3.1) rather than an ancestral moral graph, because the query is
set-wide: :meth:`SCIM.d_connected` answers "which nodes are d-connected to these
targets" in one traversal, and moralization is valid only over the ancestral
closure of ``{candidate} | targets | given``, which differs per candidate.
Because the sweep is hand-rolled it is property-tested against
``networkx.is_d_separator`` on random DAGs; treat that oracle test as
non-optional for any change to it.

Two limits every criterion over this substrate inherits:

1. d-separation is sound but not complete under determinism, and these models are
   largely deterministic functions of their parents. Criteria are therefore
   conservative: a route whose functional effect vanishes still counts. The
   standard repair is Geiger-Verma-Pearl D-separation, treating functionally
   determined nodes as observed.
2. A single-period diagram cannot express a payoff arriving through the next
   period's value, nor distinguish a variable's arrival value from the value it is
   reassigned to. :meth:`SCIM.with_continuation` and
   :meth:`SCIM.with_lagged_arrivals` repair both.
"""

import networkx as nx

__all__ = ["SCIM", "CONTINUATION_PREFIX", "DUMMY_PREFIX", "LAG_SUFFIX"]


#: Node-name prefix for the synthetic continuation utility.
CONTINUATION_PREFIX = "__continuation__"

#: Node-name prefix for the synthetic parent of :meth:`SCIM.with_dummy_parent`.
DUMMY_PREFIX = "__hat__"

#: Suffix marking a variable's arrival value, as distinct from the end-of-period
#: value the plain name carries.
LAG_SUFFIX = "*"

# Trail directions for the Bayes-Ball sweep in :meth:`SCIM.d_connected`.
_UP, _DOWN = True, False


def _fresh_name(graph, base, prefix):
    """A node name built from *prefix* and *base* that is absent from *graph*."""
    name = f"{prefix}{base}"
    while name in graph:  # pathological name collision
        name = "_" + name
    return name


class SCIM:
    """The influence-diagram view of a model.

    Parameters
    ----------
    graph : networkx.DiGraph
        A DAG of chance / decision / utility nodes, each carrying a ``kind``
        attribute, with directed causal edges. Parameters must already be
        dropped: they are deterministic constants rather than random variables,
        and an un-conditioned fork through one opens spurious d-connection.
    decisions : iterable
        The decision nodes.
    agent_utilities : mapping
        ``agent_utilities[agent]`` is the utility nodes owned by *agent*.
    decision_agent : mapping
        ``decision_agent[decision]`` is the agent that owns each decision.

    Notes
    -----
    Traversals are memoized per instance, so a criterion may query freely, and
    every transform returns a new instance rather than mutating this one. Both
    rest on *graph* not changing after construction; mutate it and the caches go
    stale.
    """

    def __init__(self, graph, decisions, agent_utilities, decision_agent):
        self.graph = graph
        self.decisions = list(decisions)
        self.agent_utilities = {a: list(u) for a, u in agent_utilities.items()}
        self.decision_agent = dict(decision_agent)
        self._ancestors_cache = {}
        self._d_connected_cache = {}

    def __repr__(self):
        return (
            f"SCIM({self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"decisions={self.decisions})"
        )

    # -- vocabulary ----------------------------------------------------------

    def parents(self, node):
        """The parents of *node*, which for a decision are its information set."""
        return list(self.graph.predecessors(node))

    def context(self, decision):
        """The conditioning set ``Pa(D) | {D}`` every criterion conditions on.

        What the decision-maker knows when choosing, plus the choice itself.
        """
        return set(self.graph.predecessors(decision)) | {decision}

    def objectives(self, decision):
        """The utility nodes whose value *decision* is choosing over.

        Utilities owned by the deciding agent and downstream of the decision.
        Includes the synthetic continuation node when
        :meth:`with_continuation` has been applied and the decision reaches it.
        """
        owned = set(self.agent_utilities.get(self.decision_agent.get(decision), ()))
        return owned & nx.descendants(self.graph, decision)

    # -- engine --------------------------------------------------------------

    def ancestors(self, nodes):
        """Every strict ancestor of any node in *nodes*.

        One reverse multi-source traversal, rather than a
        :func:`networkx.ancestors` call per node.
        """
        key = frozenset(nodes)
        if key not in self._ancestors_cache:
            seen = set()
            stack = [n for n in key if n in self.graph]
            while stack:
                for parent in self.graph.predecessors(stack.pop()):
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
            self._ancestors_cache[key] = seen
        return self._ancestors_cache[key]

    def d_connected(self, targets, given):
        """Every node d-connected to some node of *targets*, conditioning on *given*.

        The complement is the d-separated set, so ``node not in
        d_connected(...)`` certifies that no active trail carries influence from
        ``node`` to any target. One traversal answers this for every node at
        once, so callers should ask once per decision and test membership rather
        than calling :func:`networkx.is_d_separator` per candidate.

        *targets* and *given* are excluded from the result.
        """
        key = (frozenset(targets), frozenset(given))
        if key in self._d_connected_cache:
            return self._d_connected_cache[key]

        target_set = {n for n in key[0] if n in self.graph}
        given_set = {n for n in key[1] if n in self.graph}
        if not target_set:
            self._d_connected_cache[key] = set()
            return self._d_connected_cache[key]

        # Nodes in ``given`` or with a descendant there. At such a node a
        # collision is unblocked, so a trail arriving from a parent may turn and
        # go back up.
        evidence_at_or_below = given_set | self.ancestors(given_set)

        # Trails are explored as (node, direction) pairs, where UP means the
        # trail reached this node from one of its children.
        frontier = [(t, _UP) for t in target_set]
        seen = set()
        reachable = set()

        while frontier:
            node, direction = frontier.pop()
            if (node, direction) in seen:
                continue
            seen.add((node, direction))
            if node not in given_set:
                reachable.add(node)

            if direction is _UP and node not in given_set:
                frontier.extend((p, _UP) for p in self.graph.predecessors(node))
                frontier.extend((c, _DOWN) for c in self.graph.successors(node))
            elif direction is _DOWN:
                if node not in given_set:
                    frontier.extend((c, _DOWN) for c in self.graph.successors(node))
                if node in evidence_at_or_below:
                    frontier.extend((p, _UP) for p in self.graph.predecessors(node))

        self._d_connected_cache[key] = reachable - target_set - given_set
        return self._d_connected_cache[key]

    # -- transforms ----------------------------------------------------------

    def _replace(self, graph, agent_utilities=None):
        """A new SCIM over *graph*, carrying this one's node roles forward."""
        return SCIM(
            graph,
            self.decisions,
            self.agent_utilities if agent_utilities is None else agent_utilities,
            self.decision_agent,
        )

    def with_lagged_arrivals(self, lag_dependencies):
        """Split each variable's arrival value out into its own node.

        A single-period diagram carries one node per symbol, but a symbol
        reassigned within the period holds two values: the one it arrives with
        and the one it is reassigned to. This adds a source node ``<name>*`` for
        the arrival value, with an edge to each consumer that reads it. The plain
        node keeps its in-period parents and so denotes the end-of-period value,
        which is what the next period arrives with.

        A decision's parents in the result are exactly its information set, so
        callers may read the conditioning set off the graph.

        Parameters
        ----------
        lag_dependencies : iterable of (consumer, source) pairs
            Dependencies that read a source's pre-period value -- the edges a
            single-period projection would otherwise drop.

        Returns
        -------
        SCIM
        """
        graph = self.graph.copy()
        for consumer, source in lag_dependencies:
            if consumer not in graph or source not in graph:
                continue
            lagged = f"{source}{LAG_SUFFIX}"
            if lagged not in graph:
                attrs = dict(graph.nodes[source])
                # An arrival value is exogenous to the period.
                attrs["kind"] = "chance"
                graph.add_node(lagged, **attrs)
            graph.add_edge(lagged, consumer)
        return self._replace(graph)

    def with_continuation(self, arrival_states):
        """Add a synthetic continuation-value utility node per deciding agent.

        A single-period diagram cannot express that a decision's payoff continues
        into the next period, so a shock reaching the objective only through the
        next period's value appears irrelevant. This adds, per agent that
        decides, a utility node whose parents are the nodes named after the
        model's arrival-state variables -- for a variable that is also reassigned
        in-period, that node holds the reassigned value, which is what the next
        period arrives with.

        Parameters
        ----------
        arrival_states : iterable
            The model's arrival-state variable names.

        Returns
        -------
        SCIM
        """
        graph = self.graph.copy()
        agent_utilities = {a: list(u) for a, u in self.agent_utilities.items()}

        parents = [s for s in arrival_states if s in graph]
        # Only agents that decide have a value function to continue.
        for agent in dict.fromkeys(self.decision_agent.values()):
            node = _fresh_name(graph, agent, CONTINUATION_PREFIX)
            graph.add_node(node, kind="utility", agent=agent)
            graph.add_edges_from((p, node) for p in parents)
            agent_utilities.setdefault(agent, []).append(node)

        return self._replace(graph, agent_utilities)

    def with_edge(self, source, target):
        """Add the directed edge ``source -> target``.

        Both endpoints must already be nodes, and the edge must not close a
        cycle: a criterion posed over a graph is not answerable on a graph that
        is no longer a DAG.

        Returns
        -------
        SCIM

        Raises
        ------
        ValueError
            If either endpoint is absent, or the edge would create a cycle.
        """
        for endpoint in (source, target):
            if endpoint not in self.graph:
                raise ValueError(f"{endpoint!r} is not a node of this diagram")
        if source == target or source in nx.descendants(self.graph, target):
            raise ValueError(
                f"the edge {source!r} -> {target!r} would create a cycle; "
                f"{target!r} already reaches {source!r}"
            )
        graph = self.graph.copy()
        graph.add_edge(source, target)
        return self._replace(graph)

    def without_edges(self, edges):
        """Drop *edges*, an iterable of ``(source, target)`` pairs.

        Edges that are not present are ignored, so the result is the diagram
        without any of them however many were there to begin with.

        Returns
        -------
        SCIM
        """
        graph = self.graph.copy()
        graph.remove_edges_from(edges)
        return self._replace(graph)

    def with_dummy_parent(self, node):
        """Add a fresh exogenous parent to *node*.

        The device s-reachability is defined by: a decision node's own value is
        not the object of interest, its decision rule is, and a synthetic parent
        stands in for that rule.

        Returns
        -------
        tuple
            ``(scim, dummy)`` -- a new SCIM, and the name of the added node.
        """
        graph = self.graph.copy()
        dummy = _fresh_name(graph, node, DUMMY_PREFIX)
        graph.add_node(dummy, kind="chance", agent=None)
        graph.add_edge(dummy, node)
        return self._replace(graph), dummy
