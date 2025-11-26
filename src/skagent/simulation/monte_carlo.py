"""
Functions to support Monte Carlo simulation of models.
"""

from copy import copy
from typing import Mapping, Sequence

import numpy as np

from skagent.distributions import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
)
from skagent.block import Aggregate
from skagent.block import DBlock
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
        # TODO: generalize this to wider range of inputs.

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


def calibration_by_age(ages, calibration):
    """
    Returns calibration for this model, but with vectorized
    values which map age-varying values to agent ages.

    Parameters
    ----------
    ages: np.array
        An array of agent ages.

    calibration: dict
        A calibration dictionary

    Returns
    --------
    aged_calibration: dict
        A dictionary of parameter values.
        If a parameter is age-varying, the value is a vector
        corresponding to the values for each input age.
    """

    def aged_param(ages, p_value):
        if isinstance(p_value, (float, int)) or callable(p_value):
            return p_value
        elif isinstance(p_value, list) and len(p_value) > 1:
            pv_array = np.array(p_value)

            return np.apply_along_axis(lambda a: pv_array[a], 0, ages)
        else:
            return np.empty(ages.size)

    return {p: aged_param(ages, calibration[p]) for p in calibration}


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
    block : DBlock
        Has shocks, dynamics, and rewards
    dr : Mapping[str, Callable]
        Decision rules for control variables
    initial : dict
        Initial state distributions
    seed : int
        A seed for this instance's random number generator
    agent_count : int
        The number of agents to simulate
    T_sim : int
        The number of periods to simulate
    """

    state_vars = []

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        self.calibration = calibration
        self.block = block

        # shocks are exogenous but can depend on calibration
        raw_shocks = block.get_shocks()
        # Pass RNG to construct_shocks for deterministic distribution creation
        self.shocks = construct_shocks(
            raw_shocks, calibration, rng=np.random.default_rng(seed)
        )

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed
        self.agent_count = agent_count
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

    def _init_vars_array(self):
        """Initialize variable arrays with NaN values."""
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan
        for var in self.vars:
            if self.vars_now[var] is None:
                self.vars_now[var] = copy(blank_array)

    def _init_newborn_history(self):
        """Initialize newborn history arrays."""
        for var_name in self.initial:
            self.newborn_init_history[var_name] = (
                np.zeros((self.T_sim, self.agent_count)) + np.nan
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
        self.t_cycle = np.zeros(self.agent_count, dtype=int)
        self._init_newborn_history()

        all_agents = np.ones(self.agent_count, dtype=bool)
        self.sim_birth(all_agents)
        self.clear_history()
        return None

    def _advance_state(self):
        """Move current state to previous state and prepare for new values."""
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]
            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.empty(self.agent_count)
                self.vars_now[var][:] = np.nan
            # Else: Probably an aggregate variable set by Market

    def _get_shocks(self, conditions):
        """Draw shocks for the current period."""
        return draw_shocks(self.shocks, conditions, rng=self.RNG)

    def _get_pre_state(self, shocks_now):
        """Build the pre-state dictionary for dynamics simulation."""
        pre = self.calibration.copy()
        pre.update(self.vars_prev)
        pre.update(shocks_now)
        return pre

    def _get_shock_conditions(self):
        """
        Get the conditions array for shock drawing.
        Base class uses zeros; subclasses may override (e.g., for age-varying shocks).
        """
        return np.zeros(self.agent_count)

    def sim_one_period(self):
        """
        Simulates one period for this type.
        Subclasses may override to add mortality/aging logic.
        """
        self._advance_state()

        # Draw shocks using conditions from the subclass-overridable method
        shocks_now = self._get_shocks(self._get_shock_conditions())

        pre = self._get_pre_state(shocks_now)
        post = simulate_dynamics(self.dynamics, pre, self.dr)
        self.vars_now = post

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.agent_count indicating which agents should be "born".
        """
        initial_vals = draw_shocks(
            self.initial, np.zeros(which_agents.sum()), rng=self.RNG
        )

        if np.sum(which_agents) > 0:
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
                    self.history[var_name][self.t_sim, :] = self.vars_now[var_name]

                self.t_sim += 1

            return self.history

    def clear_history(self):
        """Clears the histories."""
        for var_name in self.vars:
            self.history[var_name] = np.empty((self.T_sim, self.agent_count))
            self.history[var_name].fill(np.nan)


class AgentTypeMonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine for agent-based economic models with aging and mortality.

    This class extends the base Simulator with:
    - Age tracking for agents
    - Mortality handling (agents die when `live` state <= 0)
    - Age-varying calibration and decision rules
    - Pre-specified shock history support

    This simulator makes assumptions about population birth and mortality which
    are not generic. All agents are replaced with newborns when they expire.

    Parameters
    ----------
    calibration : Mapping[str, Any]
        Model calibration parameters
    block : DBlock
        Has shocks, dynamics, and rewards
    dr : Mapping[str, Callable]
        Decision rules for control variables
    initial : dict
        Initial state distributions
    seed : int
        A seed for this instance's random number generator
    agent_count : int
        The number of agents to simulate
    T_sim : int
        The number of periods to simulate
    """

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__(calibration, block, dr, initial, seed, agent_count, T_sim)
        self.read_shocks = False

    def initialize_sim(self):
        """
        Prepares for a new simulation with age tracking.
        """
        if self.T_sim <= 0:
            raise ValueError(
                "T_sim represents the number of periods to simulate "
                "and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        self._init_vars_array()

        # Age and cycle tracking
        self.t_age = np.zeros(self.agent_count, dtype=int)
        self.t_cycle = np.zeros(self.agent_count, dtype=int)

        # Handle pre-specified shocks
        if self.read_shocks and bool(self.newborn_init_history):
            for init_var_name in self.initial:
                self.vars_now[init_var_name] = self.newborn_init_history[init_var_name][
                    self.t_sim, :
                ]
        else:
            self._init_newborn_history()

        all_agents = np.ones(self.agent_count, dtype=bool)
        self.sim_birth(all_agents)
        self.clear_history()
        return None

    def _get_shock_conditions(self):
        """Override to use agent ages as conditions for shock drawing."""
        return self.t_age

    def sim_one_period(self):
        """
        Simulates one period with mortality and age-varying parameters.
        """
        # Mortality adjusts the agent population
        self.get_mortality()

        self._advance_state()

        # Get shocks (from history or draw new using age as conditions)
        if self.read_shocks:
            shocks_now = {
                var_name: self.shock_history[var_name][self.t_sim, :]
                for var_name in self.shocks
            }
        else:
            shocks_now = self._get_shocks(self._get_shock_conditions())

        # Age-varying calibration and decision rules
        pre = calibration_by_age(self.t_age, self.calibration)
        pre.update(self.vars_prev)
        pre.update(shocks_now)

        dr = calibration_by_age(self.t_age, self.dr)
        post = simulate_dynamics(self.dynamics, pre, dr)
        self.vars_now = post

        # Advance age for all agents
        self.t_age = self.t_age + 1

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.

        This method runs the simulation loop and stores shock values,
        then sets read_shocks to True so these shocks are used in future simulations.

        Returns
        -------
        shock_history : dict
            The subset of simulation history that are the shocks for each agent and time.
        """
        self.initialize_sim()
        self.simulate()

        for shock_name in self.shocks:
            self.shock_history[shock_name] = self.history[shock_name]

        self.read_shocks = True
        self.clear_history()

        return self.shock_history

    def get_mortality(self):
        """
        Simulates mortality or agent turnover.
        Agents die when their state `live` is less than or equal to zero.
        """
        who_dies = self.vars_now["live"] <= 0
        self.sim_birth(who_dies)
        self.who_dies = who_dies
        return None

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation with age tracking.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.agent_count indicating which agents should be "born".
        """
        if self.read_shocks:
            t = self.t_sim
            initial_vals = {
                init_var: self.newborn_init_history[init_var][t, which_agents]
                for init_var in self.initial
            }
        else:
            initial_vals = draw_shocks(
                self.initial, np.zeros(which_agents.sum()), rng=self.RNG
            )

        if np.sum(which_agents) > 0:
            for sym in initial_vals:
                self.vars_now[sym][which_agents] = initial_vals[sym]
                self.newborn_init_history[sym][self.t_sim, which_agents] = initial_vals[
                    sym
                ]

        self.t_age[which_agents] = 0
        self.t_cycle[which_agents] = 0


# Alias for backward compatibility
MonteCarloSimulator = Simulator
