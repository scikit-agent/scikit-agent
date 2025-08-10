"""
Functions to support Monte Carlo simulation of models.
"""

from copy import copy
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from itertools import product

from skagent.distributions import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
)
from skagent.model import Aggregate
from skagent.model import DBlock
from skagent.model import construct_shocks, simulate_dynamics


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


class Simulator:
    pass


class AgentTypeMonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based on the HARK.core.AgentType framework.

    Unlike HARK.core.AgentType, this class does not do any model solving,
    and depends on dynamic equations, shocks, and decision rules paased into it.

    The purpose of this class is to provide a way to simulate models without
    relying on inheritance from the AgentType class.

    This simulator makes assumptions about population birth and mortality which
    are not generic. All agents are replaced with newborns when they expire.

    Parameters
    ------------

    calibration: Mapping[str, Any]

    block : DBlock
        Has shocks, dynamics, and rewards

    dr: Mapping[str, Callable]

    initial: dict

    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    agent_count : int
        The number of agents of this type to use in simulation.

    T_sim : int
        The number of periods to simulate.
    """

    state_vars = []

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__()

        self.calibration = calibration
        self.block = block

        # shocks are exogenous (but for age) but can depend on calibration
        raw_shocks = block.get_shocks()
        # Pass RNG to construct_shocks for deterministic distribution creation
        self.shocks = construct_shocks(
            raw_shocks, calibration, rng=np.random.default_rng(seed)
        )

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed  # NOQA
        self.agent_count = agent_count
        self.T_sim = T_sim

        # changes here from HARK.core.AgentType
        self.vars = block.get_vars()

        self.vars_now = {v: None for v in self.vars}
        self.vars_prev = self.vars_now.copy()

        self.read_shocks = False  # NOQA
        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}

        self.reset_rng()  # NOQA

    def reset_rng(self):
        """
        Reset the random number generator for this type.
        """
        self.RNG = np.random.default_rng(self.seed)
        # Set RNG on shock distributions after RNG is reset
        self._set_rng_on_shocks()

    def _set_rng_on_shocks(self):
        """
        Set the simulator's RNG on all shock distributions that support it.
        This ensures deterministic behavior when the simulator's seed is set.
        """

        def _set_rng_recursive(obj):
            if hasattr(obj, "rng"):
                obj.rng = self.RNG
            if hasattr(obj, "dist") and hasattr(obj.dist, "rng"):
                obj.dist.rng = self.RNG
            if hasattr(obj, "distributions"):
                for dist in obj.distributions:
                    _set_rng_recursive(dist)

        for shock_name, shock in self.shocks.items():
            _set_rng_recursive(shock)

        for init_name, init_dist in self.initial.items():
            _set_rng_recursive(init_dist)

    def initialize_sim(self):
        """
        Prepares for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using sim_birth), clears histories of tracked variables.
        """
        if self.T_sim <= 0:
            raise Exception(
                "T_sim represents the largest number of observations "
                + "that can be simulated for an agent, and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        all_agents = np.ones(self.agent_count, dtype=bool)
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan
        for var in self.vars:
            if self.vars_now[var] is None:
                self.vars_now[var] = copy(blank_array)

        self.t_age = np.zeros(
            self.agent_count, dtype=int
        )  # Number of periods since agent entry
        self.t_cycle = np.zeros(
            self.agent_count, dtype=int
        )  # Which cycle period each agent is on

        # Get recorded newborn conditions or initialize blank history.
        if self.read_shocks and bool(self.newborn_init_history):
            for init_var_name in self.initial:
                self.vars_now[init_var_name] = self.newborn_init_history[init_var_name][
                    self.t_sim, :
                ]
        else:
            for var_name in self.initial:
                self.newborn_init_history[var_name] = (
                    np.zeros((self.T_sim, self.agent_count)) + np.nan
                )

        self.sim_birth(all_agents)

        self.clear_history()
        return None

    def sim_one_period(self):
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.
        """
        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]

            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.empty(self.agent_count)
                self.vars_now[var][:] = np.nan
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        shocks_now = {}

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            for var_name in self.shocks:
                shocks_now[var_name] = self.shock_history[var_name][self.t_sim, :]
        else:
            ### BIG CHANGES HERE from HARK.core.AgentType
            shocks_now = draw_shocks(self.shocks, self.t_age, rng=self.RNG)

        pre = calibration_by_age(self.t_age, self.calibration)

        pre.update(self.vars_prev)
        pre.update(shocks_now)
        # Won't work for 3.8: self.parameters | self.vars_prev | shocks_now

        # Age-varying decision rules captured here
        dr = calibration_by_age(self.t_age, self.dr)

        post = simulate_dynamics(self.dynamics, pre, dr)

        self.vars_now = post
        ### BIG CHANGES HERE

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period

        # What will we do with cycles?
        # self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        # self.t_cycle[
        #    self.t_cycle == self.T_cycle
        # ] = 0  # Resetting to zero for those who have reached the end

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock, a mapping from shock names to distributions.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shocks is stored in a T_sim x agent_count array in history dictionary self.history[X].
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

        Returns
        -------
        shock_history: dict
            The subset of simulation history that are the shocks for each agent and time.
        """
        # Re-initialize the simulation
        self.initialize_sim()
        self.simulate()

        for shock_name in self.shocks:
            self.shock_history[shock_name] = self.history[shock_name]

        # Flag that shocks can be read rather than simulated
        self.read_shocks = True
        self.clear_history()

        return self.shock_history

    def get_mortality(self):
        """
        Simulates mortality or agent turnover.
        Agents die when their states `live` is less than or equal to zero.
        """
        who_dies = self.vars_now["live"] <= 0

        self.sim_birth(who_dies)

        self.who_dies = who_dies
        return None

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.agent_count indicating which agents should be "born".

        Returns
        -------
        None
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

    def simulate(self, sim_periods=None):
        """
        Simulates this agent type for a given number of periods. Defaults to
        self.T_sim if no input.
        Records histories of attributes named in self.track_vars in
        self.history[sym].

        Parameters
        ----------
        None

        Returns
        -------
        history : dict
            The history tracked during the simulation.
        """
        if not hasattr(self, "t_sim"):
            raise Exception(
                "It seems that the simulation variables were not initialize before calling "
                + "simulate(). Call initialize_sim() to initialize the variables before calling simulate() again."
            )
        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception(
                "To simulate, sim_periods has to be larger than the maximum data set size "
                + "T_sim. Either increase the attribute T_sim of this agent type instance "
                + "and call the initialize_sim() method again, or set sim_periods <= T_sim."
            )

        # ignore floating point "errors". numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if sim_periods is None:
                sim_periods = self.T_sim

            for t in range(sim_periods):
                self.sim_one_period()

                # track all the vars -- shocks and dynamics
                for var_name in self.vars:
                    self.history[var_name][self.t_sim, :] = self.vars_now[var_name]

                self.t_sim += 1

            return self.history

    def clear_history(self):
        """
        Clears the histories.
        """
        for var_name in self.vars:
            self.history[var_name] = np.empty((self.T_sim, self.agent_count))
            self.history[var_name].fill(np.nan)


class MonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based.

    Unlike the AgentTypeMonteCarloSimulator HARK.core.AgentType,
    this class does make any assumptions about aging or mortality.
    It operates only on model information passed in as blocks.

    It also does not have read_shocks functionality;
    it is a strict subset of the AgentTypeMonteCarloSimulator functionality.

    Parameters
    ------------

    calibration: Mapping[str, Any]

    block : DBlock
        Has shocks, dynamics, and rewards

    dr: Mapping[str, Callable]

    initial: dict

    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    agent_count : int
        The number of agents of this type to use in simulation.

    T_sim : int
        The number of periods to simulate.
    """

    state_vars = []

    def __init__(
        self, calibration, block: DBlock, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__()

        self.calibration = calibration
        self.block = block

        # shocks are exogenous (but for age) but can depend on calibration
        raw_shocks = block.get_shocks()
        # Pass RNG to construct_shocks for deterministic distribution creation
        self.shocks = construct_shocks(
            raw_shocks, calibration, rng=np.random.default_rng(seed)
        )

        self.dynamics = block.get_dynamics()
        self.dr = dr
        self.initial = initial

        self.seed = seed  # NOQA
        self.agent_count = agent_count  # TODO: pass this in at block level
        self.T_sim = T_sim

        # changes here from HARK.core.AgentType
        self.vars = block.get_vars()

        self.vars_now = {v: None for v in self.vars}
        self.vars_prev = self.vars_now.copy()

        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}

        self.reset_rng()  # NOQA

    def reset_rng(self):
        """
        Reset the random number generator for this type.
        """
        self.RNG = np.random.default_rng(self.seed)
        # Set RNG on shock distributions after RNG is reset
        self._set_rng_on_shocks()

    def _set_rng_on_shocks(self):
        """
        Set the simulator's RNG on all shock distributions that support it.
        This ensures deterministic behavior when the simulator's seed is set.
        """

        def _set_rng_recursive(obj):
            if hasattr(obj, "rng"):
                obj.rng = self.RNG
            if hasattr(obj, "dist") and hasattr(obj.dist, "rng"):
                obj.dist.rng = self.RNG
            if hasattr(obj, "distributions"):
                for dist in obj.distributions:
                    _set_rng_recursive(dist)

        for shock_name, shock in self.shocks.items():
            _set_rng_recursive(shock)

        for init_name, init_dist in self.initial.items():
            _set_rng_recursive(init_dist)

    def initialize_sim(self):
        """
        Prepares for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using sim_birth), clears histories of tracked variables.
        """
        if self.T_sim <= 0:
            raise Exception(
                "T_sim represents the largest number of observations "
                + "that can be simulated for an agent, and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        all_agents = np.ones(self.agent_count, dtype=bool)
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan
        for var in self.vars:
            if self.vars_now[var] is None:
                self.vars_now[var] = copy(blank_array)

        self.t_cycle = np.zeros(
            self.agent_count, dtype=int
        )  # Which cycle period each agent is on

        for var_name in self.initial:
            self.newborn_init_history[var_name] = (
                np.zeros((self.T_sim, self.agent_count)) + np.nan
            )

        self.sim_birth(all_agents)

        self.clear_history()
        return None

    def sim_one_period(self):
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.
        """

        # state_{t-1}
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]

            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.empty(self.agent_count)
                self.vars_now[var][:] = np.nan
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        shocks_now = {}

        shocks_now = draw_shocks(
            self.shocks,
            np.zeros(self.agent_count),  # TODO: stupid hack to remove age calculations.
            # this needs a little more thought
            rng=self.RNG,
        )

        pre = self.calibration  # for AgentTypeMC, this is conditional on age
        # TODO: generalize indexing into calibration.

        pre.update(self.vars_prev)
        pre.update(shocks_now)

        # Won't work for 3.8: self.parameters | self.vars_prev | shocks_now

        dr = self.dr  # AgentTypeMC chooses rule by age;
        # that generalizes to age as a DR argument?

        post = simulate_dynamics(self.dynamics, pre, dr)

        # Rewards are computed as part of dynamics; reward mapping lists agent roles.
        # Do not treat reward mapping values as callables.
        self.vars_now = post

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.agent_count indicating which agents should be "born".

        Returns
        -------
        None
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
        Simulates this agent type for a given number of periods. Defaults to
        self.T_sim if no input.

        Records histories of attributes named in self.track_vars in
        self.history[symame].

        Parameters
        ----------
        sim_periods : int
            Number of periods to simulate.

        Returns
        -------
        history : dict
            The history tracked during the simulation.
        """
        if not hasattr(self, "t_sim"):
            raise Exception(
                "It seems that the simulation variables were not initialize before calling "
                + "simulate(). Call initialize_sim() to initialize the variables before calling simulate() again."
            )
        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception(
                "To simulate, sim_periods has to be larger than the maximum data set size "
                + "T_sim. Either increase the attribute T_sim of this agent type instance "
                + "and call the initialize_sim() method again, or set sim_periods <= T_sim."
            )

        # ignore floating point "errors". numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if sim_periods is None:
                sim_periods = self.T_sim

            for t in range(sim_periods):
                self.sim_one_period()

                # track all the vars -- shocks and dynamics
                for var_name in self.vars:
                    self.history[var_name][self.t_sim, :] = self.vars_now[var_name]

                self.t_sim += 1

            return self.history

    def clear_history(self):
        """
        Clears the histories.
        """
        for var_name in self.vars:
            self.history[var_name] = np.empty((self.T_sim, self.agent_count))
            self.history[var_name].fill(np.nan)


def ergodic_moments(
    history: dict[str, np.ndarray],
    *,
    variables: list[str] | None = None,
    burn_in: int | float = 0.5,
    stats: dict[str, callable] | None = None,
) -> dict[str, float]:
    """
    Compute ergodic-distribution moments from a Monte Carlo history.

    Inputs
    - history[var]: np.ndarray of shape (T_sim, agent_count)
    - variables: which variables to summarize; defaults to all array-valued keys
    - burn_in: fraction in [0,1] or absolute periods to drop from the start
    - stats: mapping stat_name -> function(np.ndarray -> float); defaults include
      mean, std, min, max, p10, p50, p90 (nan-safe)

    Returns flat dict like {'c_mean': ..., 'c_p50': ..., 'a_mean': ...}

    Notes
    -----
    This function flattens the post–burn-in sample along time and agent
    dimensions to approximate ergodic-distribution moments. Ensure T_sim is
    sufficiently large and burn-in is appropriate for your mixing time.
    """
    if not isinstance(history, dict) or not history:
        return {}

    # Choose default variable list as all 2D arrays in history
    if variables is None:
        variables = [
            k for k, v in history.items() if isinstance(v, np.ndarray) and v.ndim == 2
        ]

    # Default stats
    if stats is None:
        stats = {
            "mean": lambda x: float(np.nanmean(x)),
            "std": lambda x: float(np.nanstd(x)),
            "min": lambda x: float(np.nanmin(x)),
            "p10": lambda x: float(np.nanpercentile(x, 10)),
            "p50": lambda x: float(np.nanpercentile(x, 50)),
            "p90": lambda x: float(np.nanpercentile(x, 90)),
            "max": lambda x: float(np.nanmax(x)),
        }

    # Determine burn-in index
    if variables:
        any_key = variables[0]
    else:
        any_key = next(iter(history))
    T_sim = history[any_key].shape[0] if isinstance(history[any_key], np.ndarray) else 0

    if isinstance(burn_in, float):
        burn_idx = int(T_sim * burn_in)
    elif isinstance(burn_in, int):
        burn_idx = burn_in
    else:
        burn_idx = 0

    burn_idx = max(0, min(burn_idx, max(T_sim - 1, 0)))

    out: dict[str, float] = {}

    for var in variables:
        arr = history.get(var, None)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            continue
        # Slice post burn-in and flatten time x agents
        X = arr[burn_idx:, :].reshape(-1)
        for stat_name, fn in stats.items():
            try:
                out[f"{var}_{stat_name}"] = float(fn(X))
            except Exception:
                # Skip if stat fails
                continue

    return out


def sweep(
    *,
    block: DBlock,
    base_calibration: dict[str, object],
    dr: dict[str, object],
    initial: dict[str, object],
    param_grid: dict[str, list[object]] | object,
    agent_count: int = 1,
    T_sim: int = 2000,
    burn_in: int | float = 0.5,
    seed: int = 0,
    variables: list[str] | None = None,
    stats: dict[str, callable] | None = None,
) -> pd.DataFrame:
    """
    For each θ in Θ, run MonteCarloSimulator, compute ergodic moments, and return a DataFrame H.

    - param_grid can be dict of lists (cartesian product) or an object with .labels and .values
      (compatible with skagent.grid.Grid).
    - Per-θ RNG seed = seed + index for reproducibility.
    - variables: list of variable names to summarize; default is all array-valued history keys.
    - stats: mapping stat_name -> reducer(np.ndarray -> float); defaults in ergodic_moments.

    Returns:
      A pandas DataFrame with one row per θ; columns include parameter values and <var>_<stat>.

    Examples
    --------
    >>> from skagent.models.benchmarks import d3_block, d3_calibration
    >>> H = sweep(
    ...     block=d3_block,
    ...     base_calibration=d3_calibration,
    ...     dr={"c": lambda m: 0.3*m},
    ...     initial={"a": 0.5},
    ...     param_grid={"DiscFac": [0.94, 0.96], "CRRA": [1.5, 2.0]},
    ...     agent_count=100,
    ...     T_sim=2000,
    ...     burn_in=0.5,
    ...     variables=["a", "c", "m", "u"],
    ... )
    >>> set(["DiscFac", "CRRA"]).issubset(set(H.columns))
    True
    """
    # Build iterable of parameter points
    params_list: list[dict[str, object]] = []

    if isinstance(param_grid, dict):
        keys = sorted(param_grid.keys())
        values_lists = []
        for k in keys:
            v = param_grid[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                values_lists.append(list(v))
            else:
                values_lists.append([v])
        for combo in product(*values_lists):
            params_list.append(
                {
                    k: (c.item() if hasattr(c, "item") else c)
                    for k, c in zip(keys, combo)
                }
            )
    elif hasattr(param_grid, "labels") and hasattr(param_grid, "values"):
        labels = list(getattr(param_grid, "labels"))
        values = getattr(param_grid, "values")
        if isinstance(values, torch.Tensor):
            arr = values.detach().cpu().numpy()
        else:
            arr = np.asarray(values)
        for i in range(arr.shape[0]):
            row = arr[i]
            params_list.append(
                {
                    labels[j]: (row[j].item() if hasattr(row[j], "item") else row[j])
                    for j in range(len(labels))
                }
            )
    else:
        raise TypeError(
            "Unsupported param_grid type; use dict-of-lists or skagent.grid.Grid"
        )

    rows: list[dict[str, object]] = []

    for i, theta in enumerate(params_list):
        calibration_i = dict(base_calibration)
        calibration_i.update(theta)

        sim = MonteCarloSimulator(
            calibration_i,
            block,
            dr,
            initial,
            seed=seed + i,
            agent_count=agent_count,
            T_sim=T_sim,
        )
        sim.initialize_sim()
        history = sim.simulate()

        moments = ergodic_moments(
            history,
            variables=variables,
            burn_in=burn_in,
            stats=stats,
        )

        row = {**theta, **moments}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
