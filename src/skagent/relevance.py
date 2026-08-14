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

**A node that is neither** -- the four *incentive criteria* of Everitt, Carey,
Langlois, Ortega & Legg, "Agent Incentives: A Causal Perspective" (AAAI-21,
35(13):11487-11495; arXiv:2102.01685), which ask what one decision stands to
gain from a variable, or does to it:

  - :func:`admits_voi` (Def. 8, Thm. 9) -- would observing X raise the
    achievable payoff?
  - :func:`admits_ri` (Def. 10, Thm. 12) -- does every optimal policy respond to
    a change in X?
  - :func:`admits_voc` (Def. 15, Thm. 16) -- would setting X raise the
    achievable payoff?
  - :func:`admits_ici` (Def. 17, Thm. 18) -- does the decision reach its payoff
    *through* X?

Each is sound and complete, and each is stated for a diagram holding exactly one
decision; a diagram with more is refused rather than answered. Three of the four
run over the :func:`minimal_reduction`, the diagram with every observation
:func:`is_requisite` rejects unwired.

Completeness is a property of the graph. Under determinism d-separation
over-reports d-connection (see :mod:`skagent.influence`), and these mechanisms
are largely deterministic, so an incentive may be reported where the functional
effect vanishes -- never the reverse. For a safety criterion that is the safe
direction of error, but it is a direction.

Each is a thin function over a :class:`~skagent.influence.SCIM`, which owns the
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
    "shock_roles",
    "is_requisite",
    "minimal_reduction",
    "admits_voi",
    "admits_ri",
    "admits_voc",
    "admits_ici",
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


# -- everything else ---------------------------------------------------------


def _the_decision(scim, decision=None):
    """The one decision the incentive criteria are defined for.

    Validates the single-decision contract, and that *decision* names that
    decision when given.
    """
    if len(scim.decisions) != 1:
        raise ValueError(
            "the incentive criteria are stated for a diagram with exactly one "
            f"decision, but this one has {len(scim.decisions)}: "
            f"{sorted(map(str, scim.decisions))}. What a decision must account "
            "for in the presence of others is a different question, asked of "
            "is_s_reachable."
        )
    only = scim.decisions[0]
    if only not in scim.graph:
        raise ValueError(f"decision {only!r} is not a node of this diagram")
    if decision is not None and decision != only:
        raise ValueError(
            f"{decision!r} is not the decision of this diagram; {only!r} is"
        )
    return only


def _objectives(scim, decision):
    """``U_D``: the objectives the requisiteness test is run against."""
    targets = scim.objectives(decision)
    if not targets:
        raise ValueError(
            f"decision '{decision}' has no objective nodes, so every node would "
            "be reported as admitting no incentive. The agent deciding "
            f"'{decision}' owns no reward downstream of it; note that a control "
            "with no declared agent does not own a reward assigned to a named "
            "one."
        )
    return targets


def _utilities(scim, decision):
    """Every utility the deciding agent owns, downstream of the decision or not.

    The target set of the two control criteria, and wider than
    :func:`_objectives`: a variable can be worth controlling for the sake of a
    payoff the decision itself never reaches.
    """
    return set(scim.agent_utilities.get(scim.decision_agent.get(decision), ()))


def _check_node(scim, node):
    if node not in scim.graph:
        raise ValueError(f"{node!r} is not a node of this diagram")


def _reaches(graph, source, target_set):
    """Is there a directed path from *source* to some node of *target_set*?

    A path of length zero counts, so a node reaches itself. Reachability rather
    than path enumeration: in a DAG a directed path ``A -> B`` composed with one
    ``B -> C`` is itself a directed path, since a node shared by the two halves
    would put ``B`` on a cycle.
    """
    return bool(target_set & (nx.descendants(graph, source) | {source}))


def is_requisite(scim, decision, node):
    """Is the observation ``node`` requisite for ``decision``?

    Requisite means the decision rule may need to read it: ``node`` is still
    d-connected to some objective given everything else the decision observes,
    plus the decision itself. A nonrequisite observation (Def. 7; Lauritzen &
    Nilsson 2001) is one for which ``X`` is independent of ``U_D`` given
    ``Pa(D) u {D} \\ {X}``, so its information link carries nothing.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.
    decision : hashable
        That decision.
    node : hashable
        An observation of ``decision`` -- one of its parents.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the diagram holds more than one decision, if ``node`` is not an
        observation of ``decision``, or if the deciding agent owns no objective
        downstream of the decision.
    """
    decision = _the_decision(scim, decision)
    targets = _objectives(scim, decision)
    if node not in scim.parents(decision):
        raise ValueError(
            f"{node!r} is not an observation of {decision!r}; requisiteness is a "
            "property of an information link"
        )
    return node in scim.d_connected(targets, scim.context(decision) - {node})


def minimal_reduction(scim):
    """The diagram with every nonrequisite information link removed (Def. 11).

    Also known as the requisite graph, the d-reduction, or the trimmed graph.
    What is left of the decision's parents is what an optimal decision rule can
    depend on, which is why the response-incentive and value-of-control criteria
    are posed over this graph rather than the declared one.

    The links are found once and dropped together: with a single decision there
    is nothing for a second pass to find, since removing a link that carries no
    information cannot make another link stop carrying any.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.

    Returns
    -------
    skagent.influence.SCIM
        A new diagram; the input is untouched.
    """
    decision = _the_decision(scim)
    return scim.without_edges(
        (observation, decision)
        for observation in scim.parents(decision)
        if not is_requisite(scim, decision, observation)
    )


def admits_voi(scim, decision, node):
    """Does ``node`` have value of information for ``decision`` (Def. 8)?

    True when observing ``node`` could raise the achievable payoff. The
    criterion (Thm. 9) is that ``node`` is a requisite observation in the
    diagram with the information link ``node -> decision`` added.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.
    decision : hashable
        That decision.
    node : hashable
        A node outside ``Desc(decision) u {decision}``.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the diagram holds more than one decision, or if ``node`` is the
        decision or descends from it -- there is no diagram in which such a node
        is observed, since the added information link would close a cycle.
    """
    decision = _the_decision(scim, decision)
    _check_node(scim, node)
    if node == decision or node in nx.descendants(scim.graph, decision):
        raise ValueError(
            f"value of information is not defined for {node!r}: it is the "
            f"decision or descends from it, so the diagram in which it is "
            f"observed -- the one the criterion is posed over -- does not exist."
        )
    return is_requisite(scim.with_edge(node, decision), decision, node)


def admits_ri(scim, decision, node):
    """Does ``node`` admit a response incentive for ``decision`` (Def. 10)?

    True when every optimal policy responds to a change in ``node``. The
    criterion (Thm. 12) is a directed path from ``node`` to the decision in the
    :func:`minimal_reduction` -- a route by which the decision must hear about
    it, once the links that carry nothing are gone.

    A response incentive on a sensitive attribute means every optimal policy is
    counterfactually unfair in the sense of Kusner et al. (2017), by Thm. 14 of
    the same paper.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.
    decision : hashable
        That decision.
    node : hashable
        Any node other than the decision.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the diagram holds more than one decision, or if ``node`` is the
        decision, which the definition excludes.
    """
    decision = _the_decision(scim, decision)
    _check_node(scim, node)
    _objectives(scim, decision)
    if node == decision:
        raise ValueError(
            f"a response incentive is not defined for the decision {node!r} "
            "itself; it is a criterion on what the decision responds to"
        )
    return decision in nx.descendants(minimal_reduction(scim).graph, node)


def admits_voc(scim, decision, node):
    """Does ``node`` admit positive value of control (Def. 15)?

    True when being able to set ``node`` -- rather than take it as it comes --
    could raise the achievable payoff. The criterion (Thm. 16) is a directed
    path from ``node`` to a utility the deciding agent owns, in the
    :func:`minimal_reduction`. The path may run through the decision.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.
    decision : hashable
        That decision.
    node : hashable
        Any node other than the decision.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the diagram holds more than one decision, or if ``node`` is the
        decision, which the definition excludes: a decision is already under the
        agent's control.
    """
    decision = _the_decision(scim, decision)
    _check_node(scim, node)
    _objectives(scim, decision)
    if node == decision:
        raise ValueError(
            f"value of control is not defined for the decision {node!r} itself; "
            "it is already the agent's to set"
        )
    return _reaches(minimal_reduction(scim).graph, node, _utilities(scim, decision))


def admits_ici(scim, decision, node):
    """Does ``node`` admit an instrumental control incentive (Def. 17)?

    True when the decision reaches a payoff *through* ``node``, so an agent
    optimizing the decision has reason to move it. The criterion (Thm. 18) is a
    directed path ``decision -> ... -> node -> ... -> utility`` in the diagram as
    declared -- not in the :func:`minimal_reduction`, since what the decision can
    influence does not depend on what it observes.

    The decision itself, and any utility it reaches, lie on such a path
    trivially and so admit the incentive.

    Parameters
    ----------
    scim : skagent.influence.SCIM
        A diagram with exactly one decision.
    decision : hashable
        That decision.
    node : hashable
        Any node of the diagram.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If the diagram holds more than one decision.
    """
    decision = _the_decision(scim, decision)
    _check_node(scim, node)
    _objectives(scim, decision)
    if node != decision and node not in nx.descendants(scim.graph, decision):
        return False
    return _reaches(scim.graph, node, _utilities(scim, decision))
