"""
Functions for creating and reasoning about Dynamic Stochastic Optimization Problems (DSOPs).

The Block data structure is rather general, and can be used to represent static problems.

Converting Block models into DSOPs involves identifying transition and reward functions,
and framing them in terms of arrival states, shocks, and decisions.

"""

import numpy as np
import torch
from torch.autograd import grad


def create_transition_function(block, state_syms, decision_rules=None):
    """
    block
    state_syms : list of string
        A list of symbols for 'state variables at time t', aka arrival states.
        # TODO: state variables should be derived from the block analysis.
    """
    decision_rules = {} if decision_rules is None else decision_rules

    def transition_function(states_t, shocks_t, controls_t, parameters):
        vals = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals, decision_rules, fix=list(controls_t.keys()))

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


def create_reward_function(block, agent=None, decision_rules=None):
    """
    block
    agent : optional, str
    decision_rules : optional, dict
        Decision rules of control variables that will _not_
        be given to the function.
    """
    decision_rules = {} if decision_rules is None else decision_rules

    def reward_function(states_t, shocks_t, controls_t, parameters):
        vals_t = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals_t, decision_rules, fix=list(controls_t.keys()))
        return {
            sym: post[sym]
            for sym in block.reward
            if agent is None or block.reward[sym] == agent
        }

    return reward_function


def get_grad_reward_function(block, agent=None, decision_rules=None):
    """
    Create a function to compute gradients of reward functions with respect to specified variables.

    This function returns a grad_reward_function that can compute dr/dx or dr/ds as needed,
    useful for constructing loss functions for the envelope condition.

    Parameters
    ----------
    block : DBlock
        The dynamic block containing reward specifications
    agent : str, optional
        If specified, only compute gradients for rewards belonging to this agent
    decision_rules : dict, optional
        Decision rules of control variables that will _not_ be given to the function

    Returns
    -------
    callable
        grad_reward_function(states, shocks, controls, parameters, wrt) that returns
        gradients of rewards with respect to variables specified in wrt
    """
    decision_rules = {} if decision_rules is None else decision_rules

    def grad_reward_function(states_t, shocks_t, controls_t, parameters, wrt):
        """
        Compute gradients of reward function with respect to specified variables.

        Parameters
        ----------
        states_t : dict
            State variables at time t
        shocks_t : dict
            Shock variables at time t
        controls_t : dict
            Control variables at time t
        parameters : dict
            Model parameters
        wrt : dict
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True

        Returns
        -------
        dict
            Nested dictionary of gradients for each reward symbol and variable:
            {reward_sym: {var_name: gradient}}
        """
        if parameters is None:
            parameters = {}

        # Combine all variables for block evaluation
        vals_t = parameters | states_t | shocks_t | controls_t

        # Compute rewards using block transition
        post = block.transition(vals_t, decision_rules, fix=list(controls_t.keys()))
        rewards = {
            sym: post[sym]
            for sym in block.reward
            if agent is None or block.reward[sym] == agent
        }

        # Compute gradients for each reward with respect to each variable in wrt
        gradients = {}
        for reward_sym in rewards:
            gradients[reward_sym] = {}
            for var_name, var_tensor in wrt.items():
                # Skip if variable doesn't require gradients
                if not var_tensor.requires_grad:
                    gradients[reward_sym][var_name] = None
                    continue

                # Compute gradient of this reward with respect to this variable
                reward_tensor = rewards[reward_sym]

                # For batched computations, we need to compute gradients for each element
                if reward_tensor.dim() > 0 and reward_tensor.numel() > 1:
                    # Handle batched case: compute gradients for each element in the batch
                    batch_gradients = []
                    for i in range(reward_tensor.shape[0]):
                        grad_result = grad(
                            reward_tensor[i],
                            var_tensor,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if grad_result[0] is not None:
                            batch_gradients.append(
                                grad_result[0][i]
                                if grad_result[0].dim() > 0
                                else grad_result[0]
                            )
                        else:
                            batch_gradients.append(None)

                    # Stack the gradients if they're not None
                    if all(g is not None for g in batch_gradients):
                        gradients[reward_sym][var_name] = torch.stack(batch_gradients)
                    else:
                        gradients[reward_sym][var_name] = None
                else:
                    # Handle scalar case
                    grad_result = grad(
                        reward_tensor, var_tensor, retain_graph=True, allow_unused=True
                    )
                    gradients[reward_sym][var_name] = (
                        grad_result[0] if grad_result[0] is not None else None
                    )

        return gradients

    return grad_reward_function


def estimate_discounted_lifetime_reward(
    block,
    discount_factor,
    dr,
    states_0,
    big_t,
    shocks_by_t=None,
    parameters={},
    agent=None,
):
    """
    Compute the discounted lifetime reward for a model given a fixed T of periods to simulate forward.

    MMW JME '21 for inspiration.

    block
    discount_factor - can be a number or a function of state variables
    dr - decision rules (dict of functions), or optionally a decision function (a function that returns the decisions)
    states_0 - dict - initial states, symbols : values (scalars work; TODO: do vectors work here?)
    shocks_by_t - dict - sym : big_t vector of shock values at each time period
    big_t - integer. Number of time steps to simulate forward
    parameters - optional - calibration parameters
    agent - optional - name of reference agent for rewards
    """
    states_t = states_0
    total_discounted_reward = 0

    tf = create_transition_function(block, list(states_0.keys()))

    if callable(dr):
        # assume a full decision function has been passed in
        df = dr
    else:
        # create a decision function from the decision rule
        df = create_decision_function(block, dr)

    rf = create_reward_function(block, agent)

    # Get all reward symbols for the agent
    reward_syms = list(
        {sym for sym in block.reward if agent is None or block.reward[sym] == agent}
    )

    if callable(discount_factor):
        raise Exception(
            "Currently only numerical, not state-dependent, discount factors are supported."
        )

    for t in range(big_t):
        if shocks_by_t is not None:
            shocks_t = {sym: shocks_by_t[sym][t] for sym in shocks_by_t}
        else:
            shocks_t = {}

        controls_t = df(states_t, shocks_t, parameters)
        reward_t = rf(states_t, shocks_t, controls_t, parameters)

        # Sum all rewards for this period
        period_reward = 0
        for rsym in reward_syms:
            # assumes torch
            if isinstance(reward_t[rsym], torch.Tensor) and torch.any(
                torch.isnan(reward_t[rsym])
            ):
                raise Exception(f"Calculated reward {rsym} is NaN: {reward_t}")
            if isinstance(reward_t[rsym], np.ndarray) and np.any(
                np.isnan(reward_t[rsym])
            ):
                raise Exception(f"Calculated reward {rsym} is NaN: {reward_t}")
            period_reward += reward_t[rsym]

        total_discounted_reward += period_reward * discount_factor**t

        # t + 1
        states_t = tf(states_t, shocks_t, controls_t, parameters)

    return total_discounted_reward


def estimate_bellman_residual(
    block,
    discount_factor,
    value_network,
    df,
    states_t,
    shocks,
    parameters={},
    agent=None,
):
    """
    Computes the Bellman equation residual for given states and shocks.

    The Bellman equation is: V(s) = max_c { u(s,c,ε) + β E_ε'[V(s')] }
    This function computes: V(s) - [u(s,c,ε) + β V(s')]
    where s' = f(s,c,ε) and V(s') is evaluated at a specific future shock realization ε'.

    Parameters
    ----------
    block : model.DBlock
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β
    value_network : callable
        A value function that takes state variables and returns value estimates
    df : callable
        Decision function that returns controls given states and shocks
    states_t : dict
        Current state variables
    shocks : dict
        Shock realizations for both periods:
        - {shock_sym}_0: period t shocks (for immediate reward and transitions)
        - {shock_sym}_1: period t+1 shocks (for continuation value evaluation)
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards

    Returns
    -------
    torch.Tensor
        Bellman equation residual
    """
    if callable(discount_factor):
        raise Exception(
            "Currently only numerical, not state-dependent, discount factors are supported."
        )

    # Get state variable names for transition
    state_variables = list(states_t.keys())

    # Get shock variable names
    shock_vars = block.get_shocks()
    shock_syms = list(shock_vars.keys())

    # Extract period-specific shocks from the combined shocks object
    shocks_t = {sym: shocks[f"{sym}_0"] for sym in shock_syms}
    shocks_t_plus_1 = {sym: shocks[f"{sym}_1"] for sym in shock_syms}

    # Get reward variables
    reward_vars = [
        sym for sym in block.reward if agent is None or block.reward[sym] == agent
    ]
    if len(reward_vars) == 0:
        raise Exception("No reward variables found in block")
    reward_sym = reward_vars[0]  # Assume single reward for now

    # Get current value estimates (using period t shocks)
    current_values = value_network(states_t, shocks_t, parameters)

    # Get controls from decision function (using period t shocks)
    controls_t = df(states_t, shocks_t, parameters)

    # Create transition and reward functions
    tf = create_transition_function(block, state_variables)
    rf = create_reward_function(block, agent)

    # Compute immediate reward (using period t shocks)
    immediate_reward = rf(states_t, shocks_t, controls_t, parameters)[reward_sym]

    # Compute next states (using period t shocks)
    next_states = tf(states_t, shocks_t, controls_t, parameters)

    # Compute continuation value using value network (using period t+1 shocks)
    continuation_values = value_network(next_states, shocks_t_plus_1, parameters)

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    return bellman_residual
