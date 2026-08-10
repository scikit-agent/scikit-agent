"""
What a decision must still account for, given what it already knows.

Every criterion here answers one question about a decision ``D`` in the
:class:`~skagent.influence.SCIM` view of a model: conditioning on what ``D``
observes, does some other node still reach ``D``'s objective? What varies is the
node asked about, and what a positive answer means for a solver.

**A decision** -- *s-reachability*, the criterion of Koller & Milch, "Multi-Agent
Influence Diagrams for Representing and Solving Games" (IJCAI-01; Games and
Economic Behavior 45(1), 2003), Defs. 7-8:

  - Decision D strategically *relies on* decision D' iff D' is *s-reachable*
    from D.
  - The *relevance graph* is a directed graph over decision nodes with an edge
    D -> D' iff D relies on D' (equivalently, D' is s-reachable from D).
  - D' is s-reachable from D iff there is a utility node U owned by D's agent
    and descended from D such that, adding a fresh dummy parent to D', there is
    an active path (d-connection) from the dummy to U given Pa(D) u {D}.

A decision node's own value is not what matters -- its decision rule is -- so the
test is run from a synthetic parent standing in for that rule. The reliance
ordering a relevance graph implies is what a best-response sweep solves a block
in.

**A shock** -- the same test, run from the shock itself, since an exogenous node
is already its own synthetic source. Here what varies is the reading, because a
solver needs to know not just whether the shock matters but how to integrate it:
see :data:`OBSERVED`, :data:`HIDDEN` and :data:`MIXED`.

Both are thin functions over a :class:`~skagent.influence.SCIM`, which owns the
graph, the conditioning-context and objective vocabulary, and the d-separation
engine. Construction of a ``SCIM`` from a scikit-agent Block lives in
:mod:`skagent.model_analyzer`; this module, like the substrate, depends only on
networkx so the criteria can be developed and tested in isolation.
"""

import networkx as nx

__all__ = [
    "OBSERVED",
    "HIDDEN",
    "MIXED",
    "is_s_reachable",
    "RelevanceGraph",
    "classify_shock",
    "shock_roles",
]


#: A shock the information set accounts for; may be gridded per node.
OBSERVED = "observed"
#: A shock the information set says nothing about; integrate inside the max.
HIDDEN = "hidden"
#: Partly informed *and* separately relevant; needs filtering, so refuse.
MIXED = "mixed"


# -- decisions ---------------------------------------------------------------


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


# -- shocks ------------------------------------------------------------------


def classify_shock(scim, shock, decision):
    """Classify one *shock* for one *decision*: OBSERVED, HIDDEN, or MIXED.

    Thin wrapper over :func:`shock_roles`.
    """
    return shock_roles(scim, [shock], decisions=[decision])[decision][shock]


def shock_roles(scim, shocks, decisions=None):
    """Classify every shock for every decision.

    Each shock takes one of three roles, per decision:

    :data:`OBSERVED`
        Every route from the shock to the objective is intercepted by the
        information set. Conditioning on the information set therefore leaves
        nothing about the shock for the objective to depend on, and a solver may
        grid the shock over its discretization nodes and solve per node.

    :data:`HIDDEN`
        The shock reaches the objective around the information set, which carries
        no information about it. An expectation over it belongs inside the
        maximization.

    :data:`MIXED`
        The shock reaches the objective around the information set, *and* the
        information set is partly informative about it. Computing the declared
        problem then requires the conditional law of the shock given the
        information set, so neither per-node solving nor integrating inside the
        maximization applies and a solver should refuse. This most often
        indicates that a reward or transition touches a shock the control's
        information set claims not to see, which is a modeling error rather than a
        solver limitation.

    The test is on the diagram, not on the syntax of the information set: a shock
    that appears in no information set may still be accounted for, because it
    feeds a derived pre-decision variable that an information set does contain.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        The influence-diagram view, after
        :meth:`~skagent.influence.SCIM.with_lagged_arrivals` and
        :meth:`~skagent.influence.SCIM.with_continuation`, so that a decision's
        parents are its information set.
    shocks : iterable
        Shock variable names.
    decisions : iterable, optional
        Decisions to classify for. Defaults to every decision in *scim*.

    Returns
    -------
    dict
        ``{decision: {shock: role}}``.

    Raises
    ------
    ValueError
        If a decision has no objective nodes. Nothing is then reachable, so every
        shock would classify :data:`OBSERVED` -- the one direction that must not
        be silent, since it invites a solver to condition on a shock the agent
        cannot see. It means the deciding agent owns no reward downstream of its
        own decision.

    Notes
    -----
    The classification is per decision and may legitimately differ between two
    controls in one period: a shock accounted for by a rich information set is
    hidden to a control that conditions on less. A solver that represents a shock
    one way for the whole period must check that the roles agree across controls.

    Errors fall toward :data:`HIDDEN` or :data:`MIXED`, never toward wrongly
    reporting a shock as accounted for.

    One traversal per decision, then membership tests, so the cost is linear in
    the graph per decision rather than per (shock, decision) pair.
    """
    shocks = list(shocks)
    roles = {}

    for decision in scim.decisions if decisions is None else decisions:
        targets = scim.objectives(decision)
        if not targets:
            raise ValueError(
                f"Decision '{decision}' has no objective nodes, so every shock "
                "would be reported as accounted-for. The agent deciding "
                f"'{decision}' owns no reward downstream of it; note that a "
                "control with no declared agent does not own a reward assigned "
                "to a named one."
            )

        conditioned = scim.parents(decision)
        reachable = scim.d_connected(targets, scim.context(decision))
        informative = scim.ancestors(conditioned)

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
