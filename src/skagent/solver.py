from skagent.grid import Grid
import skagent.model as model
from skagent.simulation.monte_carlo import draw_shocks
import numpy as np
import torch
import skagent.utils as utils


def create_transition_function(block, state_syms):
    """
    block
    state_syms : list of string
        A list of symbols for 'state variables at time t', aka arrival states.
        # TODO: state variables should be derived from the block analysis.
    """

    def transition_function(states_t, shocks_t, controls_t, parameters):
        vals = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals, {}, fix=list(controls_t.keys()))

        return {sym: post[sym] for sym in state_syms}

    return transition_function


def create_decision_function(block, decision_rules):
    """
    block
    decision_rules
    """

    def decision_function(states_t, shocks_t, parameters):
        if parameters is None:
            parameters = {}
        vals = parameters | states_t | shocks_t
        post = block.transition(vals, decision_rules)

        return {sym: post[sym] for sym in decision_rules}

    return decision_function


def create_reward_function(block, agent=None):
    """
    block
    agent : optional, str
    """

    def reward_function(states_t, shocks_t, controls_t, parameters):
        vals_t = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals_t, {}, fix=list(controls_t.keys()))
        return {
            sym: post[sym]
            for sym in block.reward
            if agent is None or block.reward[sym] == agent
        }

    return reward_function


def simulate_forward(
    states_t,
    block: model.Block,
    decision_function: callable,
    parameters,
    big_t,
    # state_syms,
):
    if isinstance(states_t, Grid):
        n = states_t.n()
        states_t = states_t.to_dict()
    else:
        # kludge
        n = len(states_t[next(iter(states_t.keys()))])

    state_syms = list(states_t.keys())
    tf = create_transition_function(block, state_syms)

    for t in range(big_t):
        # TODO: make sure block shocks are 'constructed'
        # TODO: allow option for 'structured' draws, e.g. from exact discretization.
        shocks_t = draw_shocks(block.shocks, n=n)

        # this is cumbersome; probably can be solved deeper on the data structure level
        # note similarity to Grid.from_dict() reconciliation logic.
        states_template = states_t[next(iter(states_t.keys()))]
        shocks_t = {
            sym: utils.reconcile(states_template, shocks_t[sym]) for sym in shocks_t
        }

        controls_t = decision_function(states_t, shocks_t, parameters)

        states_t_plus_1 = tf(states_t, shocks_t, controls_t, parameters)
        states_t = states_t_plus_1

    return states_t_plus_1


def static_reward(
    block,
    dr,
    states,
    shocks={},
    parameters={},
    agent=None,
):
    """
    Returns the reward for an agent for a block, given a decision rule, states, shocks, and calibration.

    block
    dr - decision rules (dict of functions), or optionally a decision function (a function that returns the decisions)
    states - dict - initial states, symbols : values (scalars work; TODO: do vectors work here?)
    shocks- dict - sym : vector of shock values
        # TODO: Here the shocks are given. We will want to streamline a way of sampling here.
    parameters - optional - calibration parameters
    agent - optional - name of reference agent for rewards
    """
    if callable(dr):
        # assume a full decision function has been passed in
        df = dr
    else:
        # create a decision function from the decision rule
        df = create_decision_function(block, dr)

    rf = create_reward_function(block, agent)

    # this assumes only one reward is given.
    # can be generalized in the future.
    rsym = list(
        {sym for sym in block.reward if agent is None or block.reward[sym] == agent}
    )[0]

    controls = df(states, shocks, parameters)
    reward = rf(states, shocks, controls, parameters)

    # Maybe this can be less complicated because of unified array API
    if isinstance(reward[rsym], torch.Tensor) and torch.any(torch.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")
    if isinstance(reward[rsym], np.ndarray) and np.any(np.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")

    return reward[rsym]
