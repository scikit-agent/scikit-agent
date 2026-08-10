"""
The shock-role criterion on influence-diagram models.

Answers the question a solver must settle before it can pose a decision problem
correctly: at the moment a control is chosen, does the model's declared
information set already account for a given shock?

The criterion is a d-separation test on the :class:`~skagent.influence.SCIM` view
of a block, not a syntactic test on the information set. A shock that appears in
no information set may still be fully accounted for, because it feeds a derived
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

The criterion is conservative in one direction only: errors fall toward
``HIDDEN`` or ``MIXED``, never toward wrongly reporting a shock as accounted for.
Its two structural limits, and the transforms that repair the second, are
documented on :mod:`skagent.influence`.
"""

__all__ = ["OBSERVED", "HIDDEN", "MIXED", "classify_shock", "shock_roles"]


#: A shock the information set accounts for; may be gridded per node.
OBSERVED = "observed"
#: A shock the information set says nothing about; integrate inside the max.
HIDDEN = "hidden"
#: Partly informed *and* separately relevant; needs filtering, so refuse.
MIXED = "mixed"


def classify_shock(scim, shock, decision):
    """Classify one *shock* for one *decision*: OBSERVED, HIDDEN, or MIXED.

    Thin wrapper over :func:`shock_roles`.
    """
    return shock_roles(scim, [shock], decisions=[decision])[decision][shock]


def shock_roles(scim, shocks, decisions=None):
    """Classify every shock for every decision.

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
