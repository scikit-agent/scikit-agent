"""
Best-response solving for blocks with several decisions.

Solves the decisions of a :class:`~skagent.block.DBlock` one at a time, in the
order given by the block's relevance graph, so that every decision rule a
decision strategically relies on is already computed when its turn comes. This
requires the relevance graph to be acyclic; a cyclic component is a set of
decisions that have to be solved jointly, which is a simultaneous-move
equilibrium problem and is not attempted here.

Each decision is solved *per information cell*: for every value of what the
decision-maker observes, the action maximizing their expected payoff conditional
on that observation. Expectations are estimated by drawing *shock_samples*
realizations of the block's shocks once, at construction, and reusing the same
draws for every candidate action (common random numbers), so that comparisons
between actions carry far less error than their levels do. Conditioning is done
by grouping the simulated samples, which makes
the beliefs at a decision the posterior induced by the other decision rules in
the profile.

The representation is tabular, which sets the limits of this module: a decision's
information set must take finitely many values under the profile being played,
since a cell with no repeats has no conditional expectation to estimate, and the
actions searched are a finite set. Shocks may be continuous -- they are
integrated over, never conditioned on. For continuous observations, or where a
differentiable policy is wanted, :func:`skagent.solver.solve_multiple_controls`
performs the same sweep with a policy network per control in place of a table.
"""

from collections import namedtuple
import logging
import warnings

import numpy as np

from skagent.algos.vfi import get_action_rule

logger = logging.getLogger(__name__)

__all__ = ["ConditionalPayoffs", "TabulatedRule", "TabularBestResponseSolver"]


#: Decimal places at which observed values are matched when grouping samples
#: into information cells.
CELL_PRECISION = 6


ConditionalPayoffs = namedtuple(
    "ConditionalPayoffs", ["cells", "counts", "actions", "payoff"]
)
ConditionalPayoffs.__doc__ = """Expected payoffs by information cell and action.

cells : numpy.ndarray
    Shape ``(n_cells, len(iset))``; the distinct observed values of the
    decision's information set. Shape ``(1, 0)`` for an empty information set.
counts : numpy.ndarray
    Shape ``(n_cells,)``; how many samples support each cell.
actions : numpy.ndarray
    Shape ``(n_actions,)``; the candidate actions searched.
payoff : numpy.ndarray
    Shape ``(n_cells, n_actions)``; the estimated expected payoff of the
    decision's agent in each cell for each action.
"""


class TabulatedRule:
    """A decision rule tabulated over the cells of an information set.

    Follows the library's decision-rule calling convention: the information-set
    values are passed as positional arguments in ``iset`` order. An observation
    is answered with the action of the nearest tabulated cell, so a rule remains
    total when queried away from the values it was tabulated on.

    For a rule tabulated on a full grid, and where interpolation between grid
    points is wanted, use :func:`skagent.algos.vfi.ar_from_data` instead.

    Parameters
    ----------
    iset : sequence of str
        The information set, in the order the cell columns are given.
    cells : array_like
        Shape ``(n_cells, len(iset))``; the tabulated observations.
    actions : array_like
        Shape ``(n_cells,)``; the action for each cell.
    """

    def __init__(self, iset, cells, actions):
        self.iset = list(iset)
        self.actions = np.asarray(actions, dtype=float)
        cells = np.asarray(cells, dtype=float)
        if cells.size != len(self.actions) * len(self.iset):
            raise ValueError(
                f"got {cells.size} cell coordinate(s) for {len(self.actions)} "
                f"action(s) over information set {self.iset}; there must be one "
                "action per cell"
            )
        # An empty information set is one cell with no coordinates.
        self.cells = cells.reshape(len(self.actions), len(self.iset))

    def __call__(self, *observed):
        if len(observed) != len(self.iset):
            raise TypeError(
                f"decision rule for information set {self.iset} expects "
                f"{len(self.iset)} positional argument(s), got {len(observed)}"
            )
        if not self.iset:
            return float(self.actions[0])

        columns = np.broadcast_arrays(*[np.asarray(o, dtype=float) for o in observed])
        query = np.stack([np.atleast_1d(c) for c in columns], axis=-1)
        nearest = np.abs(query[:, None, :] - self.cells[None, :, :]).sum(-1).argmin(1)
        chosen = self.actions[nearest]
        return chosen if np.ndim(columns[0]) else float(chosen[0])

    def to_dict(self, decimals=3):
        """The rule as ``{observed values: action}``, rounded for display."""
        return {
            tuple(np.round(cell, decimals)): round(float(action), decimals)
            for cell, action in zip(self.cells, self.actions)
        }


class TabularBestResponseSolver:
    """Solve a block's decisions by best response, in relevance-graph order.

    Parameters
    ----------
    block : skagent.block.DBlock
        The block to solve. Its dynamics must be declared in topological order,
        as the library requires of any block, and its controls must carry the
        ``agent`` attribution needed to tell whose payoff each maximizes when
        the block's utilities are owned by more than one agent.
    calibration : dict, optional
        Parameter values, used both to construct the block's shocks and as
        values for any parameter the dynamics refer to. Defaults to empty.
    actions : array_like, optional
        Candidate actions to search, shared by every decision. Defaults to
        ``action_count`` points spanning ``[0, 1]``.
    action_count : int, optional
        Number of candidate actions when *actions* is not given.
    shock_samples : int, optional
        Number of shock realizations drawn at construction. This is the axis the
        expectations are estimated over -- the integral over the block's
        declared shocks that a decision's objective is defined by, taken by
        Monte Carlo rather than by quadrature. Grouping splits these across a
        decision's information cells, so each cell's expectation rests on the
        samples that reached it rather than on all of them.
    rng : numpy.random.Generator, optional
        Generator for the shock draws.
    max_cells : int, optional
        Upper limit on the number of information cells a single decision may
        have. Exceeding it raises, since grouping samples by observed value only
        estimates a conditional expectation when observations repeat.

    Notes
    -----
    Constructing the solver draws the block's shocks against *calibration*. The
    block itself is left as its author wrote it, so one block may be solved at
    several calibrations.

    Expectations are Monte Carlo estimates; a solved rule is exact only up to
    sampling error, which falls as *shock_samples* rises.
    """

    def __init__(
        self,
        block,
        calibration=None,
        *,
        actions=None,
        action_count=21,
        shock_samples=100_000,
        rng=None,
        max_cells=1024,
        samples=None,
    ):
        if samples is not None:
            warnings.warn(
                "samples is deprecated; pass shock_samples, which says which "
                "axis it counts: realizations of the block's shocks, not "
                "independent trajectories.",
                DeprecationWarning,
                stacklevel=2,
            )
            shock_samples = samples
        self.block = block
        self.calibration = {} if calibration is None else dict(calibration)
        self.actions = (
            np.linspace(0.0, 1.0, action_count)
            if actions is None
            else np.asarray(actions, dtype=float)
        )
        self.shock_samples = shock_samples
        self.max_cells = max_cells
        self.rng = np.random.default_rng() if rng is None else rng

        self.decisions = list(block.get_controls())

        shocks = block.construct_shocks(self.calibration, rng=self.rng)
        for distribution in shocks.values():
            # Shocks the block declares as distribution instances rather than
            # as (class, arguments) pairs are not reached by construct_shocks'
            # rng argument. These are this solver's own copies, so seeding them
            # leaves the block's own distributions alone.
            distribution.rng = self.rng
        self.shocks = {
            sym: dist.draw(self.shock_samples) for sym, dist in shocks.items()
        }

    # -- inputs read off the block ------------------------------------------

    def _control(self, decision):
        if decision not in self.decisions:
            raise ValueError(
                f"{decision!r} is not a control of this block; the controls are "
                f"{self.decisions}"
            )
        return self.block.get_dynamics()[decision]

    def _agent(self, decision):
        """The agent whose payoff ``decision`` maximizes."""
        agent = self._control(decision).agent
        owners = set(self.block.reward.values())
        if agent is None:
            if len(owners) > 1:
                raise ValueError(
                    f"control {decision!r} has no agent attribution, but this "
                    f"block's utilities are owned by {sorted(map(str, owners))}; "
                    "attribute the control to say whose payoff it maximizes"
                )
            return None
        if agent not in owners:
            raise ValueError(
                f"agent {agent!r} of control {decision!r} owns no reward "
                f"variable in this block; its utilities are owned by "
                f"{sorted(map(str, owners))}"
            )
        return agent

    # -- policies ------------------------------------------------------------

    def mixed_rule(self, weights=None):
        """A full-support mixed rule: every action played on some samples.

        Held by decisions that are not yet solved, so that every information
        cell is reached and every conditional expectation is defined.

        Parameters
        ----------
        weights : array_like, optional
            Relative shares of the samples per action, one per candidate action.
            Defaults to equal shares.

        Returns
        -------
        callable
            A decision rule returning one action per sample.
        """
        if weights is None:
            weights = np.ones_like(self.actions)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.actions.shape:
            raise ValueError(
                f"got {weights.size} weights for {self.actions.size} candidate actions"
            )
        counts = np.maximum(1, np.round(self.shock_samples * weights / weights.sum()))
        drawn = np.repeat(self.actions, counts.astype(int))
        return get_action_rule(np.resize(drawn, self.shock_samples))

    def initial_policies(self):
        """A mixed rule for every decision in the block."""
        return {sym: self.mixed_rule() for sym in self.decisions}

    # -- payoffs and best responses -----------------------------------------

    def _vector(self, value):
        return np.broadcast_to(np.asarray(value, dtype=float), (self.shock_samples,))

    def payoff(self, vals, agent):
        """The sum of ``agent``'s utility nodes, per sample."""
        owned = self.block.calc_reward(vals, agent=agent).values()
        return self._vector(sum(owned))

    def conditional_payoffs(self, decision, policies):
        """Estimate ``decision``'s payoffs by information cell and action.

        Parameters
        ----------
        decision : str
            The control to evaluate.
        policies : Mapping[str, Callable]
            A decision rule for every control of the block, including
            *decision* itself: the profile the expectation is taken under.

        Returns
        -------
        ConditionalPayoffs
        """
        control = self._control(decision)
        agent = self._agent(decision)
        upstream = self.decisions[: self.decisions.index(decision)]

        # Simulate under the current profile to get the joint distribution of
        # everything this decision conditions on.
        base = self.block.transition(dict(self.calibration, **self.shocks), policies)

        # Hold the upstream decisions at their simulated values, so replacing
        # this decision's action does not disturb what precedes it. Downstream
        # decisions are left free: they react through their own decision rules.
        pre = dict(self.calibration, **self.shocks)
        pre.update({sym: self._vector(base[sym]) for sym in upstream})

        if control.iset:
            observed = np.stack(
                [np.round(self._vector(base[v]), CELL_PRECISION) for v in control.iset],
                axis=-1,
            )
        else:
            observed = np.zeros((self.shock_samples, 1))
        cells, inverse = np.unique(observed, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        if len(cells) > self.max_cells:
            raise ValueError(
                f"control {decision!r} takes {len(cells)} distinct values of its "
                f"information set {control.iset}, above max_cells="
                f"{self.max_cells}; a conditional expectation per cell needs "
                "observations that repeat, so discretize the information set or "
                "raise max_cells"
            )
        counts = np.bincount(inverse, minlength=len(cells))

        payoff = np.empty((len(cells), len(self.actions)))
        for k, action in enumerate(self.actions):
            trial = dict(policies, **{decision: get_action_rule(action)})
            vals = self.block.transition(pre, trial, fix=upstream)
            sample_payoff = self.payoff(vals, agent)
            # Mean payoff within each information cell.
            payoff[:, k] = (
                np.bincount(inverse, weights=sample_payoff, minlength=len(cells))
                / counts
            )

        return ConditionalPayoffs(
            cells=cells if control.iset else np.zeros((1, 0)),
            counts=counts,
            actions=self.actions,
            payoff=payoff,
        )

    def best_response(self, decision, policies):
        """The payoff-maximizing rule for ``decision`` against ``policies``.

        Parameters
        ----------
        decision : str
            The control to solve.
        policies : Mapping[str, Callable]
            A decision rule for every control of the block; the profile the best
            response is computed against.

        Returns
        -------
        TabulatedRule
            One action per information cell.
        """
        cells, _, actions, payoff = self.conditional_payoffs(decision, policies)
        return TabulatedRule(
            self._control(decision).iset, cells, actions[payoff.argmax(axis=1)]
        )

    def solve(self, policies=None):
        """Solve every decision, in relevance-graph order.

        Parameters
        ----------
        policies : Mapping[str, Callable], optional
            Starting rules for the decisions, replaced one by one as they are
            solved. Defaults to a mixed rule per decision, which is what makes
            the conditional expectation at every information cell defined.

        Returns
        -------
        dict
            A decision rule per control of the block.

        Raises
        ------
        NotImplementedError
            If the relevance graph has a cyclic component: those decisions rely
            on each other and admit no one-at-a-time order.
        """
        policies = self.initial_policies() if policies is None else dict(policies)
        for component in self.block.relevance_graph(self.calibration).condensation():
            if len(component) > 1:
                raise NotImplementedError(
                    f"decisions {sorted(component)} strategically rely on each "
                    "other and must be solved jointly, as a simultaneous-move "
                    "equilibrium; only acyclic relevance graphs are supported"
                )
            (decision,) = component
            policies[decision] = self.best_response(decision, policies)
            logger.info("solved %s", decision)
        return policies
