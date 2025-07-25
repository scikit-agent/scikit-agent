"""
Tools for crafting models.
"""

from dataclasses import dataclass, field, replace
from copy import copy, deepcopy
from skagent.distributions import (
    Distribution,
    DiscreteDistributionLabeled,
    combine_indep_dstns,
    expected,
)
from inspect import signature
import numpy as np
from skagent.parser import math_text_to_lambda
from typing import Any, Callable, Mapping, List, Union


class Aggregate:
    """
    Used to designate a shock as an aggregate shock.
    If so designated, draws from the shock will be scalar rather
    than array valued.
    """

    def __init__(self, dist: Distribution):
        self.dist = dist


class Control:
    """
    Used to designate a variable that is a control variable.

    Parameters
    ----------
    iset : list of str
        The labels of the variables that are in the information set of this control.

    lower_bound : function
        An 'equation function' which evaluates to the lower bound of the control variable.

    upper_bound : function
        An 'equation function' which evaluates to the upper bound of the control variable.

    agent : str
        A label identifying the agent role to which this control is attributed.
    """

    def __init__(self, iset, lower_bound=None, upper_bound=None, agent=None):
        self.iset = iset
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.agent = agent


def discretized_shock_dstn(shocks, disc_params):
    """
    Discretizes a collection of independent shocks and combines
    them into one DiscreteDistributionLabeled.

    Shocks are discretized only if they have a corresponding
    element of disc_params defined.

    Parameters
    -----------
    shocks: dict of Distribution
        A dictionary of Distributions, representing independent exogenous shocks.

    disc_params: dict of dict
        A dictionary of dictionaries with arguments to Distribution.discretize.
        Keys of this dictionary should be shared with the shocks argument.
    """
    dshocks = {}

    for shockn in shocks:
        if shockn == "live":  # hacky hack
            pass
        elif shockn in disc_params:
            dshocks[shockn] = DiscreteDistributionLabeled.from_unlabeled(
                shocks[shockn].discretize(**disc_params[shockn]), var_names=[shockn]
            )
        else:
            # assume already discrete
            dshocks[shockn] = DiscreteDistributionLabeled.from_unlabeled(
                shocks[shockn], var_names=[shockn]
            )

    all_shock_dstn = combine_indep_dstns(*dshocks.values())

    return all_shock_dstn


def construct_shocks(shock_data, scope, rng=None):
    """
    Returns a dictionary from shock labels to Distributions.

    When the corresponding value in shock_data contains
    a distribution constructor and input information,
    any symbolic expressions used in the inputs are
    evaluated in the provided scope.

    Parameters
    ------------

     shock_data: Mapping(str, Distribution or tuple)
        A mapping from variable names to Distribution objects,
        representing exogenous shocks.

        Optionally, the mapping can be to tuples of Distribution
        constructors and dictionary of input arguments.
        In this case, the dictionary can map argument names to
        numbers, or to strings. The strings are parsed as
        mathematical expressions and evaluated in the scope
        of a calibration dictionary.

    scope: dict(str, values)
        Variables assigned to numerical values.
        The scope in which expressions will be evaluated

    rng: np.random.Generator, optional
        Random number generator to pass to distribution constructors.
        If provided, distributions created from tuples will use this RNG.
    """
    sd = deepcopy(shock_data)

    for v in sd:
        if isinstance(sd[v], tuple):
            dist_class = sd[v][0]

            dist_args = sd[v][1]  # should be a dictionary

            for a in dist_args:
                if isinstance(dist_args[a], str):
                    arg_lambda = math_text_to_lambda(dist_args[a])
                    arg_value = arg_lambda(
                        *[scope[var] for var in signature(arg_lambda).parameters]
                    )

                    dist_args[a] = arg_value

            # Add RNG to distribution arguments if provided
            if rng is not None:
                dist_args = dist_args.copy()  # Don't modify original
                dist_args["rng"] = rng

            dist = dist_class(**dist_args)

            sd[v] = dist

    return sd


def simulate_dynamics(
    dynamics: Mapping[str, Union[Callable, Control]],
    pre: Mapping[str, Any],
    dr: Mapping[str, Callable],
):
    """
    From the beginning-of-period state (pre), follow the dynamics,
    including any decision rules, to compute the end-of-period state.

    Parameters
    ------------

    dynamics: Mapping[str, Callable]
        Maps variable names to functions from variables to values.
        Can include Controls
        ## TODO: Make collection of equations into a named type


    pre : Mapping[str, Any]
        Bound values for all variables that must be known before beginning the period's dynamics.


    dr : Mapping[str, Callable]
        Decision rules for all the Control variables in the dynamics.
    """
    vals = pre.copy()

    for sym in dynamics:
        # Using the fact that Python dictionaries are ordered
        feq = dynamics[sym]

        if isinstance(feq, Control):
            # This tests if the decision rule is age varying.
            # If it is, this will be a vector with the decision rule for each agent.
            if isinstance(dr[sym], np.ndarray):
                ## Now we have to loop through each agent, and apply the decision rule.
                ## This is quite slow.
                for i in range(dr[sym].size):
                    vals_i = {
                        var: (
                            vals[var][i]
                            if isinstance(vals[var], np.ndarray)
                            else vals[var]
                        )
                        for var in vals
                    }
                    vals[sym][i] = dr[sym][i](
                        *[vals_i[var] for var in signature(dr[sym][i]).parameters]
                    )
            else:
                vals[sym] = dr[sym](
                    *[vals[var] for var in signature(dr[sym]).parameters]
                )  # TODO: test for signature match with Control
        else:
            vals[sym] = feq(*[vals[var] for var in signature(feq).parameters])

    return vals


class Block:
    def get_attributions(self):
        """
        Return the agent assignments of variables as a dict of
        the form {"agent1" : ["var1", "var2", ... ], "agent2" : ["var3", "var4", ...]}
        """
        attributions = {}
        dyn = self.get_dynamics()

        for sym in self.get_controls():
            if dyn[sym].agent is not None:
                agent_name = dyn[sym].agent

                agent_attr = attributions.get(agent_name, [])
                agent_attr.append(sym)
                attributions[agent_name] = agent_attr

        for sym in self.reward:
            agent_name = self.reward[sym]

            agent_attr = attributions.get(agent_name, [])
            agent_attr.append(sym)
            attributions[agent_name] = agent_attr

        return attributions

    def get_controls(self):
        dyn = self.get_dynamics()

        return [sym for sym in dyn if isinstance(dyn[sym], Control)]


@dataclass
class DBlock(Block):
    """
    Represents a 'block' of model behavior.
    It prioritizes a representation of the dynamics of the block.
    Control variables are designated by the appropriate dynamic rule.

    Parameters
    ----------
    shocks: Mapping(str, Distribution or tuple)
        A mapping from variable names to Distribution objects,
        representing exogenous shocks.

        Optionally, the mapping can be to tuples of Distribution
        constructors and dictionary of input arguments.
        In this case, the dictionary can map argument names to
        numbers, or to strings. The strings are parsed as
        mathematical expressions and evaluated in the scope
        of a calibration dictionary.

    dynamics: Mapping(str, str or callable)
        A dictionary mapping variable names to mathematical expressions.
        These expressions can be simple functions, in which case the
        argument names should match the variable inputs.
        Or these can be strings, which are parsed into functions.

    reward: Mapping(str, str)
        A dictionary mapping variable names to agent role labels.
        The variable name will almost always appear in 'dynamics'.
        The agent role indicates which agent views the variable as a reward
        to optimize.

    """

    name: str = ""
    description: str = ""
    shocks: dict = field(default_factory=dict)
    dynamics: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)

    def construct_shocks(self, calibration, rng=None):
        """
        Constructs all shocks given calibration.
        This method mutates the DBlock.

        Parameters
        ----------
        calibration : dict
            Calibration parameters for shock construction
        rng : np.random.Generator, optional
            Random number generator to use for distribution construction
        """
        self.shocks = construct_shocks(self.shocks, calibration, rng=rng)

    def discretize(self, disc_params):
        """
        Returns a new DBlock which is a copy of this one, but with shock discretized.
        """

        disc_shocks = {}

        for shockn in self.shocks:
            if shockn in disc_params:
                disc_shocks[shockn] = self.shocks[shockn].discretize(
                    **disc_params[shockn]
                )
            else:
                disc_shocks[shockn] = deepcopy(self.shocks[shockn])

        # replace returns a modified copy
        new_dblock = replace(self, shocks=disc_shocks)

        return new_dblock

    def __post_init__(self):
        for v in self.dynamics:
            if isinstance(self.dynamics[v], str):
                self.dynamics[v] = math_text_to_lambda(self.dynamics[v])

        # --- this now has agent assignments.
        # for r in self.reward:
        #    if isinstance(self.reward[r], str):
        #        self.reward[r] = math_text_to_lambda(self.reward[r])

    def get_shocks(self):
        return self.shocks

    def get_dynamics(self):
        return self.dynamics

    def get_vars(self):
        """
        Returns the variables that are created/modified by the Block.
        Does *not* include variables that are only used as arguments to
        the dynamics. TODO: Get a way to find these.
        """
        return list(self.shocks.keys()) + list(self.dynamics.keys())

    def transition(self, pre, dr, screen=False, until=None, fix=None):
        if fix is None:
            fix = []
        """
        Computes the state variables following pre-given states,
        given a decision rule for all controls.

        Parameters
        -----------
        pre
        dr

        screen: Boolean
            If True, the remove any dynamics that are prior to the first given state.
            Defaults to False.

        until: str or None
            If not None, a symb which is the last symbol to simulate forward.
            Useful for not overwriting prestates with poststates.

        fix: list of string
            A list of symbols to make static, rather than dynamic.
            The symbol must appear in both dynamics and pre.
        """
        dyn = self.dynamics.copy()

        if screen:
            # don't simulate any states that are logically prior
            # to those that have already been given.
            met_pre = False  # this is a hack; really should use dependency graph
            for sym in list(dyn.keys()):
                if not met_pre:
                    if sym in pre:
                        met_pre = True
                        del dyn[sym]
                    elif sym not in pre and sym not in dr:
                        del dyn[sym]

            # this will break if there's a directly recursive label,
            # i.e. if dynamics at time t for variable 'a'
            # depend on state of 'a' at time t-1
            # This is a forbidden case in CDC's design.

        if until:
            # Validate that `until` is a valid symbol in `dyn`
            if until not in dyn:
                raise ValueError(
                    f"The `until` parameter ({until}) does not match any valid symbol in `dyn`. Available symbols: {list(dyn.keys())}"
                )
            # don't simulate any states that are logically after
            # the until state
            met_until = False  # this is a hack; really should use dependency graph
            for sym in list(dyn.keys()):
                if not met_until:
                    if sym == until:
                        met_until = True
                else:
                    del dyn[sym]

        for sym in fix:
            if sym in dyn and sym in pre:
                del dyn[sym]
            else:
                raise Exception(
                    f"Attempting to fix variable ({sym}) but it is not in either dyn ({sym in dyn}) or pre ({sym in pre})"
                )

        return simulate_dynamics(dyn, pre, dr)

    def calc_reward(self, vals):
        """
        Computes the reward for a given set of variable values
        """
        rvals = {}

        for sym in self.reward:
            feq = self.dynamics[sym]
            rvals[sym] = feq(*[vals[var] for var in signature(feq).parameters])

        return rvals

    def get_state_rule_value_function_from_continuation(
        self, continuation, screen=False
    ):
        """
        Given a continuation value function, returns a state-rule value
        function: the value for each state and decision rule.
        This value includes both the reward for executing the rule
        'this period', and the continuation value of the resulting states.
        """

        def state_rule_value_function(pre, dr):
            vals = self.transition(pre, dr, screen=screen)
            r = list(self.calc_reward(vals).values())[0]  # a hack; to be improved
            # this assumes a single reward variable; instead, a named could be passed in.
            cv = continuation(
                *[vals[var] for var in signature(continuation).parameters]
            )

            return r + cv

        return state_rule_value_function

    def get_decision_value_function(self, dr, continuation):
        """
        Given a decision rule and a continuation value function,
        return a function for the value at the decision step/tac,
        after the shock have been realized.

        ## TODO: it would be better to systematize these value functions per block
        ## better, then construct them with 'partial' methods
        """
        srvf = self.get_state_rule_value_function_from_continuation(continuation)

        def decision_value_function(shpre):
            return srvf(shpre, dr)

        return decision_value_function

    def get_arrival_value_function(self, disc_params, dr, continuation):
        """
        Returns an arrival value function, which is the value of the states
        upon arrival into the block.

        This involves taking an expectation over shocks (which must
        first be discretized), a decision rule, and a continuation
        value function.)
        """

        def arrival_value_function(arvs):
            dvf = self.get_decision_value_function(dr, continuation)

            ds = discretized_shock_dstn(self.shocks, disc_params)

            def mod_dvf(shock_value_array):
                shockvs = {
                    shn: shock_value_array[shn]
                    for i, shn in enumerate(list(ds.variables.keys()))
                }

                dvf_args = {}
                dvf_args.update(arvs)
                dvf_args.update(shockvs)

                return dvf(dvf_args)

            return expected(func=mod_dvf, dist=ds)

        return arrival_value_function

    # On DBlock class:
    def iter_dblocks(self):
        """A DBlock is its own leaf."""
        yield self


@dataclass
class RBlock(Block):
    """
    A recursive block.

    Parameters
    ----------
    ...
    """

    name: str = ""
    description: str = ""
    blocks: List[Block] = field(default_factory=list)

    def construct_shocks(self, calibration, rng=None):
        """
        Construct all shocks given a calibration dictionary.

        Parameters
        ----------
        calibration : dict
            Calibration parameters for shock construction
        rng : np.random.Generator, optional
            Random number generator to use for distribution construction
        """
        for b in self.blocks:
            b.construct_shocks(calibration, rng=rng)

    def discretize(self, disc_params):
        """
        Recursively discretizes all the blocks.
        It replaces any DBlocks with new blocks with discretized shocks.
        """
        cbs = copy(self.blocks)

        for i, b in list(enumerate(cbs)):
            if isinstance(b, DBlock):
                nb = b.discretize(disc_params)
                cbs[i] = nb
            elif isinstance(b, RBlock):
                b.discretize(disc_params)

        # returns a copy of the RBlock with the blocks replaced
        return replace(self, blocks=cbs)

    def get_shocks(self):
        ### TODO: Bug in here is causing AttributeError: 'set' object has no attribute 'draw'

        super_shocks = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.get_shocks().items():  # use d.iteritems() in python 2
                super_shocks[k] = v

        return super_shocks

    def get_controls(self):
        dyn = self.get_dynamics()

        return [sym for sym in dyn if isinstance(dyn[sym], Control)]

    def get_dynamics(self):
        super_dyn = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.get_dynamics().items():  # use d.iteritems() in python 2
                super_dyn[k] = v

        return super_dyn

    def get_vars(self):
        return list(self.get_shocks().keys()) + list(self.get_dynamics().keys())

    @property
    def reward(self):
        """
        The reward attributions for all subblocks.
        """
        super_rew = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.reward.items():  # use d.iteritems() in python 2
                super_rew[k] = v

        return super_rew

    def iter_dblocks(self):
        """Iterate over all DBlock leaves in this RBlock tree."""
        for block in self.blocks:
            if isinstance(block, DBlock):
                yield block
            elif isinstance(block, RBlock):
                # Recursively yield from nested RBlocks
                yield from block.iter_dblocks()
            else:
                raise ValueError(f"Unexpected block type: {type(block)}")
