import numpy as np
import torch

"""
Tools for the implementation of the Maliar, Maliar, and Winant (JME '21) method.

This method relies on a simpler problem representation than that elaborated
by the skagent Block system.

"""


def create_transition_function(block, state_syms):
    """
    block
    state_syms : list of string
        A list of symbols for 'state variables at time t', aka arrival states.
    """

    def transition_function(states_t, controls_t, parameters={}):
        vals = parameters | states_t | controls_t
        post = block.transition(vals, {}, screen=True)

        return {sym: post[sym] for sym in state_syms}

    return transition_function


def create_decision_function(block, decision_rules):
    """
    block
    decision_rules
    """

    def decision_function(states_t, parameters={}):
        vals = parameters | states_t
        post = block.transition(vals, decision_rules)

        return {sym: post[sym] for sym in decision_rules}

    return decision_function


def create_reward_function(block):
    def reward_function(states_t, controls_t, parameters={}):
        vals_t = parameters | states_t | controls_t
        post = block.transition(vals_t, {}, screen=True)
        return {sym: post[sym] for sym in block.reward}

    return reward_function


def estimate_discounted_lifetime_reward(
    block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
):
    """
    block
    discount_factor - can be a number or a function of state variables
    dr - decision rule
    states_0 - initial states
    big_t - integer. Number of time steps to simulate forward
    parameters - optional - calibration parameters
    agent - optional - name of reference agent for rewards
    """
    states_t = states_0
    total_discounted_reward = 0

    tf = create_transition_function(block, list(states_0.keys()))

    # preparing decision function
    df = create_decision_function(block, dr)

    # preparing reward function
    # TODO: Can move this up to the function above.
    if agent is None:
        # assumes only one reward per block, not fully general
        # TODO: fix this to use all reward variables
        rsym = list(block.reward.keys())[0]
    else:
        # get the rsym corresponding to the given agent
        # again, assumes only one reward symbol.
        # TODO: make more general
        rsym = [rsym for rsym in block.reward if block.reward[rsym] == agent][0]

    rf = create_reward_function(block)

    if callable(discount_factor):
        raise Exception(
            "Currently only numerical, not state-dependent, discount factors are supported."
        )

    for t in range(big_t):
        controls_t = df(states_t, parameters=parameters)
        reward_t = rf(states_t, controls_t, parameters=parameters)

        # assumes torch
        if isinstance(reward_t[rsym], torch.Tensor) and torch.any(
            torch.isnan(reward_t[rsym])
        ):
            raise Exception(f"Calculated reward {[rsym]} is NaN: {reward_t}")
        if isinstance(reward_t[rsym], np.ndarray) and np.any(np.isnan(reward_t[rsym])):
            raise Exception(f"Calculated reward {[rsym]} is NaN: {reward_t}")

        total_discounted_reward += reward_t[rsym] * discount_factor**t
        # t + 1

        ###
        ### TODO: get these transition functions from the DBlock.
        ###
        states_t = tf(states_t, controls_t, parameters=parameters)

    return total_discounted_reward
