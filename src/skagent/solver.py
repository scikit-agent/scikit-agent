import inspect
import logging
import numbers
from itertools import groupby

import numpy as np

import skagent.ann as ann
import skagent.algos.vfi as vfi
import skagent.bellman as bellman_module
import skagent.loss as loss_module
from skagent.block import Control, DBlock, Entity, RBlock
from skagent.utils import param_names

logger = logging.getLogger(__name__)

#: How :func:`project` names the solved instance's symbols, and the others'.
ACTOR_SUFFIX = "_actor"
OTHER_SUFFIX = "_other"


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
        randomizes=control.randomizes,
    )


def _joining_equation(actor_sym, other_sym, others_count):
    """``x = concat(x_actor, x_other)``: the only equation a projection writes.

    It reassembles the entity axis and nothing else. Whatever reduction the
    author wrote then runs on the result verbatim, so the projection never has
    to know whether the aggregate is a mean, a sum, a maximum or a masked mean.

    **The entity axis is the LAST axis**, which is the convention
    :func:`_per_instance` already reads: a rejoined symbol of more than one
    dimension is a sample axis followed by an entity axis, and the aggregating
    equation is applied to each sample's population in turn. So the solved
    instance contributes exactly one slot on that axis and the others
    contribute ``others_count`` of them, whatever sample axis sits in front.

    Three rival shapes are meaningful, and the solved instance's shape is what
    says which one arrived, since it fixes the sample axis exactly:

    - a single number, or one value per sample. A rule constant across the
      class really does give every rival the same action, so broadcasting it
      is exact rather than an approximation.
    - one value per rival, with or without a sample axis in front. This is
      what lets rivals with private draws differ from one another.

    A shape that is neither raises, and so does the one case where both
    readings fit -- as many samples as there are rivals -- because the two are
    different models and a bare array carries no label saying which it is.
    """

    def rival_axis(value_shape, sample_shape):
        """Whether *other* needs an entity axis added before it broadcasts."""
        if value_shape != sample_shape or not sample_shape:
            return False
        if sample_shape == (others_count,):
            raise ValueError(
                f"{other_sym!r} has shape {value_shape}, and there are as many "
                f"samples as there are rivals ({others_count}), so it reads "
                f"equally as one value per sample and as one value per rival. "
                f"Give it an explicit entity axis -- shape "
                f"{sample_shape + (others_count,)} -- to say which it is."
            )
        return True

    def join(actor, other):
        import torch

        if isinstance(actor, torch.Tensor) or isinstance(other, torch.Tensor):
            # Either side may be a plain number -- a supplied constant rule is
            # one -- so both are lifted onto whichever side is already a tensor.
            reference = actor if isinstance(actor, torch.Tensor) else other
            a = torch.as_tensor(
                actor, dtype=reference.dtype, device=reference.device
            ).unsqueeze(-1)
            o = torch.as_tensor(other, dtype=reference.dtype, device=reference.device)
            if rival_axis(tuple(o.shape), tuple(a.shape[:-1])):
                o = o.unsqueeze(-1)
            o = torch.broadcast_to(o, a.shape[:-1] + (others_count,))
            return torch.cat([a, o], dim=-1)

        a = np.asarray(actor, dtype=float)[..., None]
        o = np.asarray(other, dtype=float)
        if rival_axis(o.shape, a.shape[:-1]):
            o = o[..., None]
        o = np.broadcast_to(o, a.shape[:-1] + (others_count,))
        return np.concatenate([a, o], axis=-1)

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
    alone, and the simulator guarantees as much by iterating the sample axis in
    Python, so a bare ``q.mean()`` means the mean over instances. A batched
    solver has no such loop, and the same ``q.mean()`` would reduce the batch as
    well, returning one number for the whole panel. That is a wrong answer that
    looks right whenever the panel is degenerate.

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


def _blocks_by_class(name, layout, shocks, rewards):
    """One block per contiguous run of symbols sharing an entity class.

    The inverse of :meth:`skagent.block.Block.signatures`: signatures read a
    symbol's entity class off the block tree, and this builds the tree a set of
    classified symbols implies. A block declares one class for everything in it,
    so symbols of different classes need different blocks, and the runs are kept
    CONTIGUOUS so that the order the caller laid out survives the split -- an
    equation that has to run before another still does, whichever class each of
    them belongs to.

    Parameters
    ----------
    name : str
        Stem for the sub-blocks' names, which are numbered from it.
    layout : sequence of (str, callable or None, str or None)
        Each symbol the tree declares, in the order it runs: its name, its
        equation -- ``None`` for a shock, which is declared rather than
        computed -- and the entity class it is one value per instance of, or
        ``None`` where it is a single value.
    shocks : Mapping
        Shock declarations, keyed by the names *layout* uses. Each is placed in
        the block that declares its symbol.
    rewards : Mapping
        Agent attributions, keyed by the names *layout* uses, for whichever
        symbols carry one.

    Returns
    -------
    list of skagent.block.DBlock
    """
    blocks = []
    for entity_class, run in groupby(layout, key=lambda item: item[2]):
        symbols = [(sym, equation) for sym, equation, _ in run]
        blocks.append(
            DBlock(
                name=f"{name}_{len(blocks)}",
                entity=Entity(entity_class) if entity_class else None,
                shocks={sym: shocks[sym] for sym, _ in symbols if sym in shocks},
                dynamics={
                    sym: equation for sym, equation in symbols if equation is not None
                },
                reward={sym: rewards[sym] for sym, _ in symbols if sym in rewards},
            )
        )
    return blocks


def project(ground, actor_suffix=ACTOR_SUFFIX, other_suffix=OTHER_SUFFIX):
    """One instance's problem, with the rest of its class beside it.

    The entity class is split in two -- the instance being solved, and the
    others -- and every per-instance equation is copied once per side under a
    suffixed name. For each symbol that an aggregating equation reads over the
    class, exactly one equation is synthesized, and it concatenates the two
    sides back into the original symbol; the aggregating equation is then copied
    verbatim and reads that symbol. The projection therefore reassembles the
    entity axis without inspecting the reduction, so a mean, a sum, a maximum
    and a masked mean all project alike.

    The two sides' rewards are attributed to suffixed agent roles, so a solver
    told which agent it serves maximizes one instance's payoff rather than the
    class's total.

    **The rest of the class stays a population.** The others are declared as an
    entity class of their own, named for the original and sized one short of
    it, so their symbols are one value per rival rather than one value shared
    by all of them. That is what lets rivals with private draws differ from one
    another: a rule they share is still evaluated once per rival, and the
    aggregate reads the spread of what they do. A rejoined symbol carries the
    class the author declared, at the size the author gave it, so the author's
    own aggregation reduces over the same class it always did.

    The solved instance's symbols carry no class, because it is one instance
    rather than a population of one, and that asymmetry is what the two sides
    are for.

    **The other instances share one rule**, so the equilibrium sought is a
    symmetric one. A class of genuinely distinct rules is expressible in this
    shape but is not built here.

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
        The projected problem, carrying two controls -- the solved instance's
        and the others' -- and a calibration that sizes the rivals' class.

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

    rivals = entity + other_suffix
    # Copied controls regenerate their correctly suffixed randomizers.
    declarations = block._get_declared_shocks()

    # Every reward the projection has, keyed by the symbol that carries it. The
    # two sides' rewards go to suffixed agent roles, so a solver told which
    # agent it serves maximizes one instance's payoff and not the class's total.
    rewards = {
        side[sym]: owner + suffix
        for sym, owner in block.reward.items()
        if sym in per_instance
        for side, suffix in ((actor, actor_suffix), (other, other_suffix))
    } | {sym: owner for sym, owner in block.reward.items() if sym not in per_instance}

    shocks = {
        other[sym]: declaration
        for sym, declaration in declarations.items()
        if sym in per_instance
    } | {
        (actor[sym] if sym in per_instance else sym): declaration
        for sym, declaration in declarations.items()
    }

    def copied(equation, side, suffix):
        """One side's copy of an author's per-instance equation."""
        return (
            _copy_control(equation, side, suffix)
            if isinstance(equation, Control)
            else _renamed(equation, side)
        )

    # The layout: every symbol the projection declares, in the order it has to
    # run, beside the entity class it belongs to. The rivals' symbols carry a
    # class of their own and the solved instance's carry none, so the two sides
    # cannot share a block; a rejoined symbol carries the class the author
    # declared. Shocks lead, since nothing they read is in the block.
    layout = [(other[sym], None, rivals) for sym in declarations if sym in per_instance]
    layout += [
        (actor[sym] if sym in per_instance else sym, None, None) for sym in declarations
    ]

    joined = set()
    for per_instance_run, run in groupby(
        block.get_dynamics().items(), key=lambda item: item[0] in per_instance
    ):
        run = list(run)
        if per_instance_run:
            # Both sides before whatever rejoins them, and each side in the
            # author's order, since a later equation may read an earlier one.
            layout += [
                (other[sym], copied(eq, other, other_suffix), rivals) for sym, eq in run
            ]
            layout += [
                (actor[sym], copied(eq, actor, actor_suffix), None) for sym, eq in run
            ]
            continue
        for sym, equation in run:
            # Rejoin whatever this equation reads out of the class, then copy it
            # as the author wrote it. A rejoined symbol is one value per member
            # of the WHOLE class, so it carries the author's own class at the
            # author's own size and the aggregate reduces over the same class it
            # always did.
            for argument, _reduced, _broadcast in crossings.get(sym, []):
                if argument in joined:
                    continue
                joined.add(argument)
                layout.append(
                    (
                        argument,
                        _joining_equation(actor[argument], other[argument], size - 1),
                        entity,
                    )
                )
            # A DECISION over the class needs no per-instance wrapper: its
            # rule is supplied rather than evaluated here, and the joins above
            # have already restored the symbols its information set names. Such
            # a rule reads a whole class, which is what no policy network in
            # this library is shaped for, so it is supplied rather than solved.
            crossed = sym in crossings and not isinstance(equation, Control)
            layout.append(
                (
                    sym,
                    _per_instance(equation, joined) if crossed else equation,
                    None,
                )
            )

    projected = RBlock(
        name=f"{block.name}_projected",
        blocks=_blocks_by_class(f"{block.name}_projected", layout, shocks, rewards),
    )
    return GroundedBlock(
        projected, dict(calibration) | {rivals: size - 1}, rng=ground.rng
    )


def _starting_policies(block):
    """A visibly provisional rule for every decision: a constant at mid-bounds.

    A starting profile is what an unsolved decision is held at, and it must not
    be mistakable for a solved one. An untrained policy network is exactly that
    mistake -- it is callable, it returns numbers, and nothing about it says it
    has not been trained -- so these are constants instead.
    """
    return {
        sym: _constant_rule(_midpoint(control), control.iset)
        for sym, control in block.get_controls().items()
    }


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

    def initial_policies(self):
        """A starting profile: every decision at a constant, none of them solved."""
        return _starting_policies(self.ground.block)

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
        An information-set variable must appear here rather than in *scope*:
        *scope* pins a shock to a realization, which leaves the rule no
        argument to be a function of. A single point is enough, and gives a
        rule constant along that variable; give it several to let the backup
        find whether the optimum varies along it.
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

    def initial_policies(self):
        """A starting profile: every decision at a constant, none of them solved."""
        return _starting_policies(self.ground.block)

    def rule_distance(self, new_rule, old_rule, iset):
        """Supremum norm between two rules over the grid they were solved on."""
        return _sup_norm(new_rule, old_rule, [self._observations(s) for s in iset])

    def _observations(self, sym):
        """The values *sym* takes when two rules are compared over it.

        A grid axis and a pinned realization each supply them directly. A shock
        that is neither has no single value to compare at and no axis to vary
        along -- a rival's private draw is the case -- so it is compared over
        its own discretization nodes, which are the points the backup
        integrates it over.
        """
        if sym in self.state_grid:
            return np.atleast_1d(np.asarray(self.state_grid[sym], dtype=float))
        if sym in self.scope:
            return np.atleast_1d(np.asarray(self.scope[sym], dtype=float))
        if sym in self.period.get_shocks():
            return vfi._shock_nodes(self.period, sym, self.disc_params)
        raise ValueError(
            f"{sym!r} is in a decision rule's information set, so two rules "
            f"have to be compared over it, and it is not a state-grid axis, a "
            f"symbol pinned in scope, or a shock of the block. Grid it, pin it, "
            f"or declare it."
        )


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
    Undamped iteration converges only where the best response is a contraction.
    Where its slope is -1 the iterates cycle between two points forever, and
    past that they diverge until the controls' bounds catch them. Both failures
    return a plausible number under an iteration cap, which is why the residual
    here is measured on the rule and never on the iteration count.

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


def solve_in_order(method, order, policies=None):
    """Solve the named decisions, one at a time, in the order given.

    This schedule takes its order from the caller rather than deriving one.
    Each decision is solved against the rules already in hand, so a symbol
    repeated in *order*, as in ``["c", "d", "c"]``, is refined after its
    neighbours have moved. The result is a best-response sweep run by hand.

    **There is no convergence test here.** The iteration stops because *order*
    ran out, and not because anything settled, so a repeated symbol buys a
    fixed number of refinement passes rather than a fixed point. Where a fixed
    point is wanted, use a schedule that measures one:
    :func:`solve_symmetric_equilibrium` iterates against a residual.

    A decision absent from *order* is returned at its starting rule, which
    means that it has not been solved. That is the caller's choice, since the
    caller writes the order, but the returned profile does not distinguish a
    solved rule from an unsolved one.

    Parameters
    ----------
    method : object
        A per-decision solver, as :class:`NeuralBestResponse`,
        :class:`ExactBestResponse` and
        :class:`skagent.algos.tabular.TabularBestResponseSolver` are. Needs
        ``best_response(decision, policies)`` and, when *policies* is omitted,
        ``initial_policies()``.
    order : sequence of str
        The decisions to solve, in order. Symbols may repeat.
    policies : Mapping[str, Callable], optional
        The profile to start from, with a rule for every decision. Defaults to
        the method's own starting profile.

    Returns
    -------
    dict
        A decision rule per control of the block.
    """
    policies = method.initial_policies() if policies is None else dict(policies)
    for decision in order:
        policies[decision] = method.best_response(decision, policies)
        logger.info("solved %s", decision)
    return policies


def solve_in_relevance_order(
    method, policies=None, *, max_iterations=25, tolerance=1e-6
):
    """Solve every decision in relevance-component order.

    This is a schedule rather than a method: it decides when each decision is
    solved, and it asks the method to carry out each solve. Acyclic components
    are solved once, after every decision rule they rely on has been computed.
    For a cyclic component, every decision's best response is computed against
    the same previous profile and installed simultaneously. This repeats until
    every policy is within *tolerance* of its own response.

    Parameters
    ----------
    method : object
        A per-decision solver carrying the problem, as
        :class:`NeuralBestResponse` and
        :class:`skagent.algos.tabular.TabularBestResponseSolver` do. Needs
        ``ground``, ``best_response(decision, policies)``,
        ``rule_distance(new, old, iset)`` and, when *policies* is omitted,
        ``initial_policies()``.
    policies : Mapping[str, Callable], optional
        Starting rules for the decisions, replaced one by one as they are
        solved. Defaults to the method's own starting profile.
    max_iterations : int, optional
        Maximum number of simultaneous best-response updates for a cyclic
        relevance component. Must be at least 1. Defaults to 25.
    tolerance : float, optional
        Maximum rule distance at convergence. Must be positive. Defaults to
        1e-6.

    Returns
    -------
    dict
        A decision rule per control of the block.

    Raises
    ------
    NotImplementedError
        If a cyclic component belongs to a recurring block, or if a decision
        relies on the other instances of its own entity class.
    RuntimeError
        If a cyclic component does not converge within *max_iterations*.
    ValueError
        If *max_iterations* is less than 1 or *tolerance* is not positive.
    TypeError
        If *max_iterations* is not an integer.
    """
    if not isinstance(max_iterations, numbers.Integral):
        raise TypeError(
            f"max_iterations must be an integer, got {type(max_iterations).__name__}"
        )
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")

    ground = method.ground
    policies = method.initial_policies() if policies is None else dict(policies)
    arrival_states = ground.block.get_arrival_states(ground.calibration)
    graph = ground.block.relevance_graph(ground.calibration)
    for component in graph.condensation():
        if len(component) == 1:
            (decision,) = component
            # A component of one is solved in one pass unless it relies on
            # ITSELF, which a decision taken by every instance of an entity
            # class does. That is a fixed point in one rule rather than an
            # order among decisions, so the count of members is not on its own
            # the cyclicity test.
            if graph.relies_on(decision, decision):
                plate = graph.plate(decision)
                raise NotImplementedError(
                    f"decision {decision!r} is one rule per instance of entity "
                    f"class {plate.entity!r}, and it relies on the other "
                    f"{plate.size - 1}, so there is no order that solves it: "
                    f"every instance responds to what the others do. Separate "
                    f"one instance from the rest with skagent.solver.project "
                    f"and solve the projection with solve_symmetric_equilibrium."
                )
            policies[decision] = method.best_response(decision, policies)
            logger.info("solved %s", decision)
            continue

        if arrival_states:
            raise NotImplementedError(
                "cyclic components in recurring blocks are not supported; "
                f"decisions {sorted(component)} depend on arrival states "
                f"{sorted(arrival_states)}"
            )

        for iteration in range(1, max_iterations + 1):
            responses = {
                decision: method.best_response(decision, policies)
                for decision in component
            }
            converged = all(
                method.rule_distance(
                    responses[decision],
                    policies[decision],
                    ground.block.get_control(decision).iset,
                )
                <= tolerance
                for decision in component
            )

            # Every response above was computed against the same profile; only
            # now replace the component's policies simultaneously.
            policies.update(responses)
            if converged:
                logger.info(
                    "solved cyclic component %s in %d iterations",
                    sorted(component),
                    iteration,
                )
                break
        else:
            raise RuntimeError(
                f"decisions {sorted(component)} did not converge within "
                f"{max_iterations} iterations"
            )
    return policies
