"""
Functions to support Monte Carlo simulation of models.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence, Union

import numpy as np

from skagent.distributions import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
)
from skagent.block import Aggregate
from skagent.block import DBlock, RBlock
from skagent.block import construct_shocks, simulate_dynamics


def draw_shocks(
    shocks: Mapping[str, Distribution],
    conditions: Sequence[int] = (),
    n=None,
    rng: np.random.Generator | None = None,
):
    """
    Draw from each shock distribution values, subject to given conditions.

    Parameters
    ------------
    shocks Mapping[str, Distribution]
        A dictionary-like mapping from shock names to distributions from which to draw

    conditions: Sequence[int]
        An array of conditions, one for each agent.
        Typically these will be agent ages.

    n : int (optional)
        Number of draws to do. An alternative to a conditions sequence.

    rng : np.random.Generator, optional
        Random number generator to use for drawing. If provided, will be used for
        distributions that support it.

    Returns
    -------
    draws : Mapping[str, Sequence]
        A mapping from shock names to drawn shock values.
    """
    draws = {}

    # TODO: generalize `conditions` to a wider range of inputs (currently
    # expected to be a per-agent sequence such as ages).
    if n is None:
        n = len(conditions)

    for shock_var in shocks:
        shock = shocks[shock_var]

        if isinstance(shock, (int, float)):
            draws[shock_var] = np.ones(n) * shock
        elif isinstance(shock, Aggregate):
            # For Aggregate shocks, set RNG if the distribution supports it
            if rng is not None and hasattr(shock.dist, "rng"):
                shock.dist.rng = rng
            draws[shock_var] = shock.dist.draw(1)[0]
        elif isinstance(shock, IndexDistribution) or isinstance(
            shock, TimeVaryingDiscreteDistribution
        ):
            ## TODO  his type test is awkward. They should share a superclass.
            # For index-varying distributions, set RNG if supported
            if rng is not None and hasattr(shock, "rng"):
                shock.rng = rng
            draws[shock_var] = shock.draw(conditions)
        else:
            # For regular distributions, set RNG if the distribution supports it
            if rng is not None and hasattr(shock, "rng"):
                shock.rng = rng
            draws[shock_var] = shock.draw(n)
            # this is hacky if there are no conditions.

    return draws


def _set_rng_recursive(obj, rng):
    """
    Recursively set the RNG on an object and its nested distributions.

    Parameters
    ----------
    obj : any
        An object that may have rng, dist, or distributions attributes
    rng : np.random.Generator
        The random number generator to set
    """
    if hasattr(obj, "rng"):
        obj.rng = rng
    if hasattr(obj, "dist") and hasattr(obj.dist, "rng"):
        obj.dist.rng = rng
    if hasattr(obj, "distributions"):
        for dist in obj.distributions:
            _set_rng_recursive(dist, rng)


class Simulator:
    """
    Base class for Monte Carlo simulation engines.

    Provides common functionality for simulation including:
    - RNG management and seeding
    - State variable tracking
    - History management
    - Simulation loop structure

    Parameters
    ----------
    calibration : Mapping[str, Any]
        Model calibration parameters
    block : DBlock or RBlock
        Has shocks, dynamics, and rewards
    dr : Mapping[str, Callable]
        Decision rules for control variables
    initial : dict
        Initial state distributions
    seed : int
        A seed for this instance's random number generator
    sample_count : int
        The number of independent trajectories to simulate. This is the
        replication axis of the histories, not a population: the trajectories
        do not interact, so a cross-sectional statistic taken over it is a
        Monte Carlo estimate whose error falls as *sample_count* rises.
    T_sim : int
        The number of periods to simulate
    """

    state_vars = []

    def __init__(
        self,
        calibration,
        block: Union[DBlock, RBlock],
        dr,
        initial,
        seed=0,
        sample_count=1,
        T_sim=10,
        agent_count=None,
    ):
        if agent_count is not None:
            warnings.warn(
                "agent_count is deprecated; pass sample_count, which names the "
                "axis this argument has always set: independent trajectories, "
                "not interacting agents.",
                DeprecationWarning,
                stacklevel=2,
            )
            sample_count = agent_count
        self.calibration = calibration
        self.block = block

        # shocks are exogenous but can depend on calibration
        raw_shocks = block.get_shocks()
        # Pass RNG to construct_shocks for deterministic distribution creation
        self.shocks = construct_shocks(
            raw_shocks, calibration, rng=np.random.default_rng(seed)
        )

        # Entity metadata: which symbols are per-instance attributes, and how
        # many instances each class has. A block declaring no entity resolves to
        # empty here, and every array keeps the single sample axis it has today.
        self.signatures = block.signatures()
        self.entities = block.entities()
        self.entity_sizes = self._resolve_entity_sizes(calibration)
        self.crossings = block.crossings()

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed
        self.sample_count = sample_count
        self.T_sim = T_sim

        # State tracking
        self.vars = block.get_vars()
        self.vars_now = {v: None for v in self.vars}
        self.vars_prev = self.vars_now.copy()

        # History tracking
        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}

        self.reset_rng()

    def reset_rng(self):
        """
        Reset the random number generator for this type.
        """
        self.RNG = np.random.default_rng(self.seed)
        self._set_rng_on_shocks()

    def _set_rng_on_shocks(self):
        """
        Set the simulator's RNG on all shock distributions that support it.
        This ensures deterministic behavior when the simulator's seed is set.
        """
        for shock in self.shocks.values():
            _set_rng_recursive(shock, self.RNG)
        for init_dist in self.initial.values():
            _set_rng_recursive(init_dist, self.RNG)

    def _resolve_entity_sizes(self, calibration):
        """How many instances each declared entity class has.

        An entity carries a name and no cardinality, so the count is a fact
        about the population rather than about the model, and is read from the
        calibration under a key equal to the entity's name.

        Raises
        ------
        ValueError
            If a declared entity has no cardinality in the calibration, or one
            that is not a positive integer.
        """
        sizes = {}
        for name in self.entities:
            if name not in calibration:
                raise ValueError(
                    f"the block declares an entity {name!r} but the calibration "
                    f"has no key {name!r} giving how many of them there are"
                )
            size = calibration[name]
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise ValueError(
                    f"entity {name!r} has cardinality {size!r}; a count of "
                    "instances must be an integer"
                )
            if size < 1:
                raise ValueError(
                    f"entity {name!r} has cardinality {size}; a population needs "
                    "at least one instance"
                )
            sizes[name] = int(size)
        return sizes

    @property
    def _has_entities(self):
        return bool(self.entity_sizes)

    def _entity_shape(self, var):
        """The entity axes of *var*: one length per class it is an attribute of.

        Empty for an axis-free variable, and empty for every variable of a block
        that declares no entity.
        """
        signature = self.signatures.get(var, frozenset())
        return tuple(self.entity_sizes[name] for name in sorted(signature))

    def _var_shape(self, var):
        """*var*'s shape for one period: the sample axis, then its entity axes."""
        return (self.sample_count,) + self._entity_shape(var)

    def _init_vars_array(self):
        """Initialize variable arrays with NaN values."""
        for var in self.vars:
            if self.vars_now[var] is None:
                self.vars_now[var] = np.full(self._var_shape(var), np.nan)

    def _init_newborn_history(self):
        """Initialize newborn history arrays."""
        for var_name in self.initial:
            self.newborn_init_history[var_name] = np.full(
                (self.T_sim,) + self._var_shape(var_name), np.nan
            )

    def initialize_sim(self):
        """
        Prepares for a new simulation. Resets the internal random number generator,
        makes initial states for all agents, clears histories of tracked variables.
        """
        if self.T_sim <= 0:
            raise ValueError(
                "T_sim represents the number of periods to simulate "
                "and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        self._init_vars_array()
        self.t_cycle = np.zeros(self.sample_count, dtype=int)
        self._init_newborn_history()

        all_agents = np.ones(self.sample_count, dtype=bool)
        self.sim_birth(all_agents)
        self.clear_history()
        return None

    def _advance_state(self):
        """Move current state to previous state and prepare for new values."""
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]
            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.full(self._var_shape(var), np.nan)
            # Else: Probably an aggregate variable set by Market

    def _get_shocks(self, conditions):
        """Draw shocks for the current period.

        A shock that is an attribute of an entity class is drawn once per
        instance per sample, so that instances are heterogeneous. An axis-free
        shock is drawn once per sample.
        """
        if not self._has_entities:
            return draw_shocks(self.shocks, conditions, rng=self.RNG)

        drawn = {}
        for sym, distribution in self.shocks.items():
            shape = self._var_shape(sym)
            count = int(np.prod(shape))
            values = np.asarray(
                draw_shocks({sym: distribution}, np.zeros(count), rng=self.RNG)[sym]
            )
            # An Aggregate draws one value for everyone, by construction.
            drawn[sym] = (
                values.reshape(shape)
                if values.size == count
                else np.broadcast_to(values, shape).copy()
            )
        return drawn

    def _simulate_entity_dynamics(self, pre):
        """Run the period's dynamics once per sample, then restack.

        An equation is written against the entity axes alone: a per-instance
        variable arrives as an array over instances, and an aggregation over one
        reads as ``q.mean()`` rather than having to name an axis. The sample axis
        is therefore iterated here rather than being handed to the equations,
        which is what keeps a reduction from silently averaging over samples as
        well as over instances.
        """
        shapes = {var: self._entity_shape(var) for var in self.vars}
        per_sample = []
        for s in range(self.sample_count):
            sliced = {
                sym: (value[s] if sym in self.vars else value)
                for sym, value in pre.items()
            }
            per_sample.append(
                simulate_dynamics(self.dynamics, sliced, self.dr, shapes=shapes)
            )

        post = {}
        for sym in per_sample[0]:
            post[sym] = (
                np.stack([one[sym] for one in per_sample])
                if sym in self.vars
                else per_sample[0][sym]
            )
        return post

    def _validate_period(self, post):
        """Check each symbol's value against its declared signature.

        An equation declared outside every entity class that returns an array
        over one is the error this whole feature exists to catch, and it is
        caught here rather than several steps downstream where the shape happens
        to stop broadcasting.
        """
        for var in self.vars:
            if var not in post:
                continue
            value = np.asarray(post[var])
            expected = self._entity_shape(var)
            actual = value.shape[1:] if value.ndim else ()
            # A per-instance equation may return one value, which broadcasts.
            if actual == expected or actual == ():
                continue
            entities = sorted(self.signatures.get(var, frozenset()))
            described = (
                f"an attribute of {entities}" if entities else "outside every entity"
            )
            raise ValueError(
                f"the equation for {var!r} is declared {described} but returned "
                f"shape {value.shape} per period; expected "
                f"{(self.sample_count,) + expected} or one value to broadcast"
            )

        for var, crossed in self.crossings.items():
            for argument, _reduced, _broadcast in crossed:
                if argument not in post:
                    continue
                value = np.asarray(post[argument], dtype=float)
                offending = int(np.count_nonzero(~np.isfinite(value)))
                if offending:
                    raise ValueError(
                        f"{argument!r} holds {offending} non-finite "
                        f"{'entry' if offending == 1 else 'entries'} and is "
                        f"reduced into {var!r}; one instance's bad value would "
                        "become every instance's"
                    )

    def _get_pre_state(self, shocks_now):
        """Build the pre-state dictionary for dynamics simulation."""
        pre = self.calibration.copy()
        pre.update(self.vars_prev)
        pre.update(shocks_now)
        return pre

    def _get_shock_conditions(self):
        """
        Get the conditions array passed to shock draws.

        The base class draws shocks unconditionally (a zero vector). The hook
        exists so a subclass can make shock draws depend on per-agent state.
        """
        return np.zeros(self.sample_count)

    def sim_one_period(self):
        """
        Simulates one period for this type.
        Subclasses may override to add mortality/aging logic.
        """
        self._advance_state()

        # Draw shocks using conditions from the subclass-overridable method
        shocks_now = self._get_shocks(self._get_shock_conditions())

        pre = self._get_pre_state(shocks_now)
        if self._has_entities:
            post = self._simulate_entity_dynamics(pre)
        else:
            post = simulate_dynamics(self.dynamics, pre, self.dr)
        self._validate_period(post)
        self.vars_now = post

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.sample_count indicating which agents should be "born".
        """
        born = int(which_agents.sum())
        if not self._has_entities:
            initial_vals = draw_shocks(self.initial, np.zeros(born), rng=self.RNG)
        else:
            # An arrival state that is a per-instance attribute needs one draw
            # per instance per newborn sample, not one per sample.
            initial_vals = {}
            for sym, distribution in self.initial.items():
                shape = (born,) + self._entity_shape(sym)
                count = int(np.prod(shape))
                values = np.asarray(
                    draw_shocks({sym: distribution}, np.zeros(count), rng=self.RNG)[sym]
                )
                initial_vals[sym] = (
                    values.reshape(shape)
                    if values.size == count
                    else np.broadcast_to(values, shape).copy()
                )

        if born > 0:
            for sym in initial_vals:
                self.vars_now[sym][which_agents] = initial_vals[sym]
                self.newborn_init_history[sym][self.t_sim, which_agents] = initial_vals[
                    sym
                ]

    def simulate(self, sim_periods=None):
        """
        Simulates this agent type for a given number of periods.
        Defaults to self.T_sim if no input.

        Parameters
        ----------
        sim_periods : int, optional
            Number of periods to simulate.

        Returns
        -------
        history : dict
            The history tracked during the simulation.
        """
        if not hasattr(self, "t_sim"):
            raise RuntimeError(
                "Simulation variables were not initialized before calling simulate(). "
                "Call initialize_sim() first."
            )
        if sim_periods is not None and self.T_sim < sim_periods:
            raise ValueError(
                "sim_periods must be <= T_sim. "
                "Increase T_sim and call initialize_sim() again."
            )

        # Ignore floating point "errors" that have well-defined answers
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if sim_periods is None:
                sim_periods = self.T_sim

            for _ in range(sim_periods):
                self.sim_one_period()

                # Track all the vars -- shocks and dynamics
                for var_name in self.vars:
                    self.history[var_name][self.t_sim] = self.vars_now[var_name]

                self.t_sim += 1

            return self.history

    def clear_history(self):
        """Clears the histories.

        A variable's history carries the period axis, then the sample axis, then
        its entity axes: ``(T_sim, sample_count)`` for an axis-free variable and
        ``(T_sim, sample_count, size)`` for an attribute of a class with *size*
        instances. A block declaring no entity therefore keeps the
        ``(T_sim, sample_count)`` histories it has always had.
        """
        for var_name in self.vars:
            self.history[var_name] = np.full(
                (self.T_sim,) + self._var_shape(var_name), np.nan
            )


# Alias for backward compatibility
MonteCarloSimulator = Simulator
