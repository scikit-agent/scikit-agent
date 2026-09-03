import inspect
import warnings

import numpy as np

import skagent.ann as ann
import skagent.algos.vfi as vfi
import skagent.bellman as bellman_module
import skagent.loss as loss_module
from skagent.block import Control, DBlock
from skagent.utils import param_names

#: How :func:`project` names the solved instance's symbols, and the others'.
ACTOR_SUFFIX = "_actor"
OTHER_SUFFIX = "_other"


def solve_multiple_controls(
    control_order, bellman_period, givens, calibration=None, epochs=200, loss=None
):
    """
    Solve a block with more than one control by training a policy network
    for each control in turn.

    Each control is given its own :class:`skagent.ann.BlockPolicyNet`. The
    networks are trained one at a time, in the order given by
    ``control_order``, with every network treating the other networks' current
    policies as fixed. A control may appear in ``control_order`` more than once
    to refine it after its neighbours have been updated (e.g.
    ``["c", "d", "c"]``), which is the multi-control analogue of a best-response
    sweep.

    Each network maximizes the payoff of the agent its control is attributed
    to, read off the block by
    :meth:`~skagent.block.Block.deciding_agent`. On a block whose utilities
    have more than one owner, a control carrying no attribution raises rather
    than being trained against someone else's objective.

    Currently restricted to single-period (non-recurring) reward objectives;
    by default the negative immediate reward
    (:class:`skagent.loss.StaticRewardLoss`) is maximized.

    Parameters
    ----------
    control_order : list of str
        Control symbols, in the order they should be solved. Symbols may repeat
        to schedule additional refinement passes.
    bellman_period : BellmanPeriod
        The model period whose controls are being solved.
    givens : skagent.grid.Grid
        Grid of arrival states and shock realizations to train over.
    calibration : dict, optional
        Deprecated. The period supplied as *bellman_period* already carries the
        calibration the losses are evaluated at, and that is the one used. If
        given, it must agree with the period's; a disagreement raises rather
        than silently evaluating a period's losses at parameters the period was
        not built with.
    epochs : int, optional
        Training epochs per pass. Default is 200.
    loss : type, optional
        A loss-function class with signature
        ``loss(bellman_period, *, agent=..., other_dr=...)``, which is the
        shape every loss in :mod:`skagent.loss` takes. The period carries the
        calibration the loss is evaluated at, so it is not passed separately.
        Defaults to :class:`skagent.loss.StaticRewardLoss`.

    Returns
    -------
    dict
        Mapping from each control symbol to its trained decision rule.

    Raises
    ------
    ValueError
        If *calibration* is given and disagrees with the period's, or if a
        control in *control_order* carries no agent attribution on a block
        whose utilities have several owners.
    """
    if calibration is not None:
        warnings.warn(
            "calibration is deprecated; the period passed as bellman_period "
            "already carries one, and that is what the losses are evaluated "
            "at.",
            DeprecationWarning,
            stacklevel=2,
        )
        differing = _differing_symbols(calibration, bellman_period.calibration)
        if differing:
            raise ValueError(
                f"calibration disagrees with bellman_period.calibration on "
                f"{differing}; the period is built from a calibration, so pass "
                "only the period"
            )

    # TODO: allow a variable 'loss function generator' once the API has
    # solidified.
    if loss is None:
        loss = loss_module.StaticRewardLoss

    # Control policy networks for each control in the block.
    cpns = {}

    # Invent Policy Neural Networks for each Control variable.
    for control_sym in bellman_period.get_controls():
        cpns[control_sym] = ann.BlockPolicyNet(bellman_period, control_sym=control_sym)

    dict_of_decision_rules = {
        k: v
        for d in [
            cpns[control_sym].get_decision_rule(length=givens.n())
            for control_sym in cpns
        ]
        for k, v in d.items()
    }

    for control_sym in control_order:
        ann.train_block_nn(
            cpns[control_sym],
            givens,
            loss(
                bellman_period,
                agent=bellman_period.block.deciding_agent(control_sym),
                other_dr=dict_of_decision_rules,
            ),
            epochs=epochs,
        )

    return dict_of_decision_rules


def _differing_symbols(first, second):
    """The symbols on which two calibrations disagree, sorted."""
    missing = object()
    return sorted(
        sym
        for sym in set(first) | set(second)
        if not np.array_equal(first.get(sym, missing), second.get(sym, missing))
    )


def _renamed(fn, mapping):
    """*fn* with its parameters renamed, still called positionally.

    Equations are invoked by the names in their own signature, so a copied
    equation has to advertise the copy's symbols rather than the original's.
    """
    if not callable(fn):
        return fn
    names = param_names(fn)
    if not any(name in mapping for name in names):
        return fn

    def renamed(*args):
        return fn(*args)

    renamed.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                mapping.get(name, name), inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            for name in names
        ]
    )
    return renamed


def _copy_control(control, mapping, agent_suffix):
    """One side's copy of a control: renamed information set, renamed agent."""
    return Control(
        [mapping.get(sym, sym) for sym in control.iset],
        lower_bound=_renamed(control.lower_bound, mapping),
        upper_bound=_renamed(control.upper_bound, mapping),
        agent=None if control.agent is None else control.agent + agent_suffix,
    )


def _joining_equation(actor_sym, other_sym, others_count):
    """``x = concat(x_actor, x_other)``: the only equation a projection writes.

    It reassembles the entity axis and nothing else. Whatever reduction the
    author wrote then runs on the result verbatim, so the projection never has
    to know whether the aggregate is a mean, a sum, a maximum or a masked mean.
    """

    def join(actor, other):
        import torch

        if isinstance(actor, torch.Tensor) or isinstance(other, torch.Tensor):
            # Either side may be a plain number -- a supplied constant rule is
            # one -- so both are lifted onto whichever side is already a tensor.
            reference = actor if isinstance(actor, torch.Tensor) else other
            a = torch.as_tensor(
                actor, dtype=reference.dtype, device=reference.device
            ).reshape(-1, 1)
            o = torch.as_tensor(
                other, dtype=reference.dtype, device=reference.device
            ).reshape(-1, 1)
            rows = max(a.shape[0], o.shape[0])
            return torch.cat([a.expand(rows, 1), o.expand(rows, others_count)], dim=-1)
        return np.concatenate(
            [
                np.atleast_1d(np.asarray(actor, dtype=float)),
                np.broadcast_to(np.asarray(other, dtype=float), (others_count,)),
            ]
        )

    join.__signature__ = inspect.Signature(
        [
            inspect.Parameter(actor_sym, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(other_sym, inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    return join


def _per_instance(equation, joined):
    """The author's aggregating equation, applied one instance-population at a time.

    An equation that reduces over an entity axis is written against that axis
    ALONE -- the simulator guarantees it by iterating the sample axis in Python,
    so a bare ``q.mean()`` means the mean over instances. A batched solver has
    no such loop, and the same ``q.mean()`` would reduce the batch as well,
    returning one number for the whole panel. That is a wrong answer that looks
    right whenever the panel is degenerate.

    ``torch.vmap`` is that loop, vectorized: it maps the equation over the batch
    so the equation sees an entity axis and nothing else. Gradients flow through
    it, so the neural path keeps its objective.
    """
    names = param_names(equation)

    def per_instance(*args):
        import torch

        batched = [
            isinstance(value, torch.Tensor) and value.ndim > 1 and name in joined
            for name, value in zip(names, args)
        ]
        if not any(batched):
            return equation(*args)
        in_dims = tuple(0 if is_batched else None for is_batched in batched)
        return torch.vmap(equation, in_dims=in_dims)(*args)

    per_instance.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in names
        ]
    )
    return per_instance


def project(ground, actor_suffix=ACTOR_SUFFIX, other_suffix=OTHER_SUFFIX):
    """One instance's problem, with the rest of its class beside it.

    The entity class is split in two -- the instance being solved, and the
    others -- and every per-instance equation is copied once per side under a
    suffixed name. For each symbol an aggregating equation reads over the class,
    ONE equation is synthesized that CONCATENATES the two sides back into the
    original symbol; the aggregating equation is then copied verbatim and reads
    it. So the projection reassembles the entity axis and never inspects the
    reduction, and a mean, a sum, a maximum or a masked mean all project alike.

    The two sides' rewards are attributed to suffixed agent roles, so a solver
    told which agent it serves maximizes one instance's payoff rather than the
    class's total.

    Two properties of this first scope, both narrower than the transform has to
    stay:

    - **The others are ONE rule, broadcast.** The projected block holds a single
      control for the whole remainder of the class, so the equilibrium sought is
      a symmetric one. A class of genuinely distinct rivals is expressible in
      this shape and is not built here.
    - **The projected block declares no entity.** The split lives in the shapes
      -- the solved instance is a scalar, the others broadcast to ``N - 1`` --
      rather than in two declarations, because the solvers refuse a block that
      declares an entity class at all, and re-keying that refusal is a separate
      decision.

    Parameters
    ----------
    ground : skagent.ground.GroundedBlock
        The population model and the calibration it is solved at. Its block must
        declare exactly one entity class of at least two instances, and the
        calibration must give that class's size under its own name.
    actor_suffix, other_suffix : str, optional
        How the two sides' symbols are named.

    Returns
    -------
    skagent.ground.GroundedBlock
        The projected problem, carrying two controls: the solved instance's and
        the others'.

    Raises
    ------
    ValueError
        If the block does not declare exactly one entity class, if the
        calibration does not size it, or if it holds fewer than two instances.
    """
    from skagent.ground import GroundedBlock

    block, calibration = ground.block, ground.calibration
    entities = block.entities()
    if len(entities) != 1:
        raise ValueError(
            f"projection needs exactly one entity class, and this block "
            f"declares {sorted(entities) if entities else 'none'}; a population "
            f"is what an instance is projected out of"
        )
    (entity,) = entities
    if entity not in calibration:
        raise ValueError(
            f"calibration gives no size for entity class {entity!r}, so there "
            f"is no population to split"
        )
    size = int(calibration[entity])
    if size < 2:
        raise ValueError(
            f"entity class {entity!r} holds {size} instance(s); a projection "
            f"separates one instance from the others, and there are no others"
        )

    signatures = block.signatures()
    per_instance = {sym for sym, axes in signatures.items() if entity in axes}
    crossings = block.crossings()
    actor = {sym: sym + actor_suffix for sym in per_instance}
    other = {sym: sym + other_suffix for sym in per_instance}

    dynamics, joined = {}, set()
    for sym, equation in block.get_dynamics().items():
        if sym in per_instance:
            for side, suffix in ((actor, actor_suffix), (other, other_suffix)):
                dynamics[side[sym]] = (
                    _copy_control(equation, side, suffix)
                    if isinstance(equation, Control)
                    else _renamed(equation, side)
                )
            continue
        # Axis-free. Rejoin whatever it reads out of the class, then copy it as
        # the author wrote it.
        for argument, _reduced, _broadcast in crossings.get(sym, []):
            if argument not in joined:
                dynamics[argument] = _joining_equation(
                    actor[argument], other[argument], size - 1
                )
                joined.add(argument)
        dynamics[sym] = (
            _per_instance(equation, joined) if sym in crossings else equation
        )

    projected = DBlock(
        name=f"{block.name}_projected",
        shocks={
            side[sym] if sym in per_instance else sym: declaration
            for sym, declaration in block.get_shocks().items()
            for side in ((actor, other) if sym in per_instance else (actor,))
        },
        dynamics=dynamics,
        reward={
            side[sym]: owner + suffix
            for sym, owner in block.reward.items()
            if sym in per_instance
            for side, suffix in ((actor, actor_suffix), (other, other_suffix))
        }
        | {
            sym: owner for sym, owner in block.reward.items() if sym not in per_instance
        },
    )
    return GroundedBlock(projected, dict(calibration), rng=ground.rng)


class NeuralBestResponse:
    """Best responses by training a policy network, and what that needs.

    A method object carries its own construction configuration beside its
    algorithm, so that a schedule can take any method without carrying every
    method's arguments on its own signature. This one needs a training panel
    and an epoch count; the exact backup needs a state grid and a continuation
    instead, and neither needs the other's.

    Parameters
    ----------
    ground : skagent.ground.GroundedBlock
        The problem being solved, already projected if it is a population.
    panel : skagent.grid.Grid
        What the network trains on and what two rules are compared over. Must
        carry every shock of the block, since the loss evaluates the whole
        period.
    epochs : int, optional
        Training epochs per best response.
    width : int, optional
        Hidden width of the policy network.
    """

    def __init__(self, ground, panel, epochs=200, width=32):
        self.ground = ground
        self.panel = panel
        self.epochs = epochs
        self.width = width
        self.period = bellman_module.BellmanPeriod(
            ground.block, None, ground.calibration
        )
        self.decisions = list(ground.block.get_controls())

    def best_response(self, decision, policies):
        """Train a network for *decision*, holding the rest of *policies* fixed."""
        net = ann.BlockPolicyNet(self.period, control_sym=decision, width=self.width)
        ann.train_block_nn(
            net,
            self.panel,
            loss_module.StaticRewardLoss(
                self.period,
                other_dr={
                    sym: rule for sym, rule in policies.items() if sym != decision
                },
                agent=self.ground.block.deciding_agent(decision),
            ),
            epochs=self.epochs,
        )
        return net.get_decision_rule(length=self.panel.n())[decision]

    def rule_distance(self, new_rule, old_rule, iset):
        """Supremum norm between two rules, evaluated on the training panel.

        A network has no cells to compare, so the comparison is over a common
        batch -- which is why the distance is the method's operation and not the
        schedule's.
        """
        return _sup_norm(new_rule, old_rule, [self.panel[sym] for sym in iset])


class ExactBestResponse:
    """Best responses by exact backup over a state grid.

    The method-object counterpart of :class:`NeuralBestResponse`: same two
    operations, entirely different construction configuration.

    Parameters
    ----------
    ground : skagent.ground.GroundedBlock
        The problem being solved, already projected if it is a population.
    state_grid : Mapping
        The grid the backup optimizes over, and where two rules are compared.
        An information-set variable must appear here rather than in *scope*,
        even as a single point, since a rule over it needs an axis to vary
        along.
    scope : Mapping, optional
        Shocks pinned to a fixed realization. Defaults to the calibration.
    continuation : Callable, optional
        The continuation value. Defaults to a terminal (zero) one, which is
        what makes the backup a single-period solve.
    disc_params : Mapping, optional
        Per-shock discretization arguments for the shocks integrated inside the
        maximization.
    """

    def __init__(
        self, ground, state_grid, scope=None, continuation=None, disc_params=None
    ):
        self.ground = ground
        self.state_grid = state_grid
        self.scope = ground.calibration if scope is None else scope
        self.continuation = (
            (lambda states, shocks, parameters: 0.0)
            if continuation is None
            else continuation
        )
        self.disc_params = {} if disc_params is None else disc_params
        self.period = bellman_module.BellmanPeriod(
            ground.block, None, ground.calibration
        )
        self.decisions = list(ground.block.get_controls())

    def best_response(self, decision, policies):
        """Back up *decision* alone, holding the rest of *policies* fixed."""
        rules, _value, _policy = vfi.solve_step(
            self.period,
            self.continuation,
            self.state_grid,
            scope=self.scope,
            agent=self.ground.block.deciding_agent(decision),
            control=decision,
            decision_rules={
                sym: rule for sym, rule in policies.items() if sym != decision
            },
            disc_params=self.disc_params,
        )
        return rules[decision]

    def rule_distance(self, new_rule, old_rule, iset):
        """Supremum norm between two rules over the grid they were solved on."""
        observed = [
            np.atleast_1d(np.asarray(self.state_grid[sym], dtype=float))
            if sym in self.state_grid
            else np.atleast_1d(np.asarray(self.scope[sym], dtype=float))
            for sym in iset
        ]
        return _sup_norm(new_rule, old_rule, observed)


def _sup_norm(first, second, observed):
    """The largest gap between two rules over a common set of observations."""
    import torch

    def values(rule):
        out = rule(*observed)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy()
        return np.asarray(out, dtype=float)

    with torch.no_grad():
        return float(np.max(np.abs(values(first) - values(second))))


def _swap_in(rule, iset):
    """The solved instance's rule, readable as the others' rule.

    The two sides' information sets differ only in their symbols' suffixes, so
    the swap is a rename: a rule is called positionally either way.
    """

    def swapped(*observed):
        return rule(*observed)

    swapped.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in iset
        ]
    )
    return swapped


def _blend(previous, response, damping, iset):
    """``(1 - damping) * previous + damping * response``, as a rule.

    Pointwise, so it serves any callable representation. It cannot move the
    fixed point -- a rule equal to its own blend is a rule equal to its own best
    response -- so it changes how the iteration travels and not where it stops.

    A representation with structure of its own, such as a rule tabulated over
    cells, is flattened to a plain callable by this and would need its own blend
    to keep that structure.
    """
    if damping == 1.0:
        return response

    def blended(*observed):
        return (1 - damping) * previous(*observed) + damping * response(*observed)

    blended.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in iset
        ]
    )
    return blended


def _constant_rule(value, iset):
    """A rule playing *value* whatever it observes."""

    def rule(*observed):
        return value

    rule.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in iset
        ]
    )
    return rule


def solve_symmetric_equilibrium(
    method,
    *,
    damping=1.0,
    tolerance=1e-3,
    max_iterations=20,
    initial=None,
):
    """A symmetric equilibrium, by iterated best response over a projection.

    Takes a projected problem -- one instance's decision beside the rest of its
    class, as :func:`project` builds -- solves the instance's decision against
    the others' current rule, swaps the solved rule in as the others', and
    repeats until the rule stops moving. A rule that is its own best response is
    an equilibrium of the projected game, and because the others play whatever
    the solved instance plays, it is a symmetric equilibrium of the population.

    The method supplies both the per-decision solve and the distance between two
    rules, since only it knows how a rule is represented. The schedule supplies
    the damping, the residual test and the swap.

    **Damping is a correctness requirement rather than a convergence aid.**
    Undamped iteration converges only where best response is a contraction;
    where its slope is -1 the iterates cycle between two points forever, and
    past that they diverge until the controls' bounds catch them. Both return a
    plausible number under an iteration cap, which is why the residual here is
    on the RULE and never on the count.

    Parameters
    ----------
    method : NeuralBestResponse or ExactBestResponse
        The per-decision solver, carrying its own configuration and the
        projected problem it solves. Any object with ``ground``,
        ``best_response(decision, policies)`` and
        ``rule_distance(new, old, iset)`` serves.
    damping : float, optional
        How far to move toward the best response each round, in ``(0, 1]``.
        Default 1.0, which is undamped.
    tolerance : float, optional
        The rule is converged when it moves less than this.
    max_iterations : int, optional
        Rounds before giving up. Reaching this is not convergence and is
        reported as such.
    initial : Callable, optional
        The others' rule on the first round. Defaults to a constant at the
        midpoint of the solved control's declared bounds.

    Returns
    -------
    rule : Callable
        The equilibrium decision rule, in the solved instance's symbols.
    info : dict
        ``converged``, ``iterations`` and ``distances``.

    Raises
    ------
    ValueError
        If the method's block does not carry a projected pair of controls.
    """
    block = method.ground.block
    solved = [sym for sym in block.get_controls() if sym.endswith(ACTOR_SUFFIX)]
    if len(solved) != 1:
        raise ValueError(
            f"expected one control named for the solved instance (ending "
            f"{ACTOR_SUFFIX!r}) and found {sorted(solved)}; the method's block "
            "should be one that project() built"
        )
    (decision,) = solved
    partner = decision[: -len(ACTOR_SUFFIX)] + OTHER_SUFFIX
    solved_iset = block.get_control(decision).iset
    partner_iset = block.get_control(partner).iset

    rule = (
        _swap_in(initial, partner_iset)
        if initial is not None
        else _constant_rule(_midpoint(block.get_control(decision)), partner_iset)
    )

    distances = []
    for _ in range(max_iterations):
        response = _swap_in(
            method.best_response(decision, {decision: rule, partner: rule}),
            partner_iset,
        )
        moved = method.rule_distance(response, rule, partner_iset)
        distances.append(moved)
        rule = _blend(rule, response, damping, partner_iset)
        if moved < tolerance:
            break
    return _swap_in(rule, solved_iset), {
        "converged": distances[-1] < tolerance,
        "iterations": len(distances),
        "distances": distances,
    }


def _midpoint(control):
    """A starting action: halfway between the control's declared bounds.

    A constant rather than an untrained network, so that where the iteration
    starts is a property of the model and not of a seed.
    """
    lower = 0.0 if control.lower_bound is None else float(control.lower_bound())
    upper = lower if control.upper_bound is None else float(control.upper_bound())
    return (lower + upper) / 2
