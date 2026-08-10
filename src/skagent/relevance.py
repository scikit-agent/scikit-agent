"""
Strategic relevance and relevance-graph analysis for influence-diagram models.

Implements the *s-reachability* graphical criterion of Koller & Milch,
"Multi-Agent Influence Diagrams for Representing and Solving Games"
(IJCAI-01; Games and Economic Behavior 45(1), 2003), Defs. 7-8:

  - Decision D strategically *relies on* decision D' iff D' is *s-reachable*
    from D.
  - The *relevance graph* is a directed graph over decision nodes with an edge
    D -> D' iff D relies on D' (equivalently, D' is s-reachable from D).
  - D' is s-reachable from D iff there is a utility node U owned by D's agent
    and descended from D such that, adding a fresh dummy parent to D', there is
    an active path (d-connection) from the dummy to U given Pa(D) u {D}.

The criterion is a thin function over a :class:`~skagent.influence.SCIM`, which
owns the graph, the conditioning-context and objective vocabulary, and the
d-separation engine. Construction of a ``SCIM`` from a scikit-agent Block lives
in :mod:`skagent.model_analyzer`; this module, like the substrate, depends only
on networkx so the criterion can be developed and tested in isolation.
"""

import networkx as nx

__all__ = ["is_s_reachable", "RelevanceGraph"]


def is_s_reachable(scim, d1, d2):
    """Is decision ``d2`` s-reachable from decision ``d1``?

    Equivalently: does ``d1`` strategically rely on ``d2`` (edge d1 -> d2 in the
    relevance graph)?

    Parameters
    ----------
    scim : skagent.influence.SCIM
        The influence-diagram view the decisions live in.
    d1, d2 : hashable
        Decision nodes in ``scim``.

    Returns
    -------
    bool
    """
    # A decision never strategically relies on itself.
    if d1 == d2:
        return False

    # The utilities d1 is choosing over: nothing to rely on without them.
    targets = scim.objectives(d1)
    if not targets:
        return False

    # d2's decision rule, not its value, is the object of interest, so the test
    # is run from a synthetic parent standing in for that rule.
    probe, dummy = scim.with_dummy_parent(d2)
    return dummy in probe.d_connected(targets, scim.context(d1))


class RelevanceGraph:
    """A relevance graph over decision nodes (edge d1 -> d2 iff d1 relies on d2).

    Wraps a ``networkx.DiGraph`` but never leaks it: all helpers return native
    Python types.
    """

    def __init__(self, graph):
        self._g = graph

    @classmethod
    def from_scim(cls, scim):
        """Build the relevance graph by testing s-reachability over all ordered
        pairs of ``scim``'s decisions.
        """
        rg = nx.DiGraph()
        rg.add_nodes_from(scim.decisions)
        for d1 in scim.decisions:
            for d2 in scim.decisions:
                if d1 == d2:
                    continue
                if is_s_reachable(scim, d1, d2):
                    rg.add_edge(d1, d2)
        return cls(rg)

    def _check_decision(self, name):
        if name not in self._g:
            raise ValueError(
                f"{name!r} is not a decision in this relevance graph; "
                f"known decisions are {sorted(map(str, self._g.nodes))}"
            )

    def relies_on(self, first, second):
        """True iff decision ``first`` strategically relies on ``second``."""
        self._check_decision(first)
        self._check_decision(second)
        return self._g.has_edge(first, second)

    def nodes(self):
        """The decision nodes, as a list."""
        return list(self._g.nodes)

    def edges(self):
        """The reliance edges (d1, d2) meaning "d1 relies on d2", as a list."""
        return list(self._g.edges)

    def is_acyclic(self):
        """True iff the relevance graph has no cycles."""
        return nx.is_directed_acyclic_graph(self._g)

    def sccs(self):
        """Strongly connected components, as a list of sets of decision nodes."""
        return [set(c) for c in nx.strongly_connected_components(self._g)]

    def condensation(self):
        """SCCs in backward-induction (solve) order.

        Returns a list of sets of decision nodes such that each component relies
        only on components appearing *earlier* in the list. Solving the game in
        this order (a la Koller & Milch Algorithm 1) means every decision an
        SCC relies on is already solved by the time the SCC is reached.
        """
        cond = nx.condensation(self._g)
        order = list(nx.topological_sort(cond))
        # Topological order points along reliance edges (a relies-on b => a
        # before b); reverse it so reliance targets are solved first.
        return [set(cond.nodes[n]["members"]) for n in reversed(order)]

    def draw(self):
        """Render the relevance graph to a ``pydot.Dot`` object.

        pydot is imported lazily so the core criterion has no hard dependency on
        the rendering stack.
        """
        import pydot

        dot = pydot.Dot(graph_type="digraph")
        for node in self._g.nodes:
            dot.add_node(pydot.Node(str(node), shape="box"))
        for src, tgt in self._g.edges:
            dot.add_edge(pydot.Edge(str(src), str(tgt)))
        return dot
