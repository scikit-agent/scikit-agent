import numpy as np
from skagent.algos.maliar import create_decision_function, create_reward_function
from skagent.grid import Grid
import torch


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
    other_dr - dict - decision rules for other controls to pass through.
    agent - optional - name of reference agent for rewards
    """
    if callable(dr):
        # assume a full decision function has been passed in
        df = dr
    else:
        # create a decision function from the decision rule
        df = create_decision_function(block, dr)

    rf = create_reward_function(block, agent, decision_rules=dr)

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


def get_static_reward_loss(state_variables, block, parameters, other_dr):
    # TODO: Should be able to get 'state variables' from block
    # Maybe with ZP's analysis modules

    shock_vars = block.get_shocks()

    def static_reward_loss(new_dr, input_grid: Grid):
        """
        dr - dict of callables
        """
        ## includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        shock_vals = {sym: input_grid[sym] for sym in shock_vars}

        # override any decision rules if necessary
        fresh_dr = {**other_dr, **new_dr}

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        r = static_reward(
            block,
            fresh_dr,
            {sym: given_vals[sym] for sym in state_variables},
            parameters=parameters,
            agent=None,  ## TODO: Pass through the agent?
            shocks=shock_vals,
            ## Handle multiple decision rules?
        )
        return -r

    return static_reward_loss
