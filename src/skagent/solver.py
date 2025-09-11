import functools
import skagent.ann as ann
import numpy as np
from skagent.bellman import BellmanPeriod
from skagent.grid import Grid
import torch


def solve_multiple_controls(
    control_order, bellman_period, givens, calibration, epochs=200, loss=None
):
    """
    Solves a block multiple times, once for each control in control_order.

    Currently restricted to static reward loss.

    TODO: all variable 'loss function generator' once API has solidified.

    Parameters
    ----------
    control_order: list
        List of control symbols in order to be solved
    bellman_period: BellmanPeriod
    """

    if loss is None:
        loss = StaticRewardLoss

    # Control policy networks for each control in the block.
    cpns = {}

    # Invent Policy Neural Networks for each Control variable.
    for control_sym in block.get_controls():
        cpns[control_sym] = ann.BlockPolicyNet(block, control_sym=control_sym)

    dict_of_decision_rules = {
        k: v
        for d in [
            cpns[control_sym].get_decision_rule(length=givens.n())
            for control_sym in cpns
        ]
        for k, v in d.items()
    }

    for control_sym in control_order:
        ann.train_block_policy_nn(
            cpns[control_sym],
            givens,
            loss(  # !!
                block,
                ["a"],  # !!
                calibration,
                dict_of_decision_rules,
            ),
            epochs=epochs,  # !!
        )

    return dict_of_decision_rules


def static_reward(
    bellman_period,
    dr,
    states,
    shocks={},
    parameters={},
    agent=None,
):
    """
    Returns the reward for an agent for a block, given a decision rule, states, shocks, and calibration.
     bellman_period
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
        controls = dr(states, shocks, parameters)
    else:
        controls = bellman_period.decision_function(states, shocks, parameters, decision_rules = dr)

    # this assumes only one reward is given.
    # can be generalized in the future.
    # -- logic should be moved into the BP object
    rsym = list(
        {sym for sym in bellman_period.block.reward if agent is None or block.reward[sym] == agent}
    )[0]

    
    reward = bellman_period.reward_function(states, shocks, controls, parameters, agent=agent, decision_rules=dr)

    # Maybe this can be less complicated because of unified array API
    if isinstance(reward[rsym], torch.Tensor) and torch.any(torch.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")
    if isinstance(reward[rsym], np.ndarray) and np.any(np.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")

    return reward[rsym]


class StaticRewardLoss:
    """
    A loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)
    """

    def __init__(self, bellman_period, parameters, other_dr=dict()):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.state_variables = self.bellman_period.get_arrival_states(calibration=parameters)
        self.other_dr = other_dr

    def __call__(self, new_dr, input_grid: Grid):
        """
        new_dr : dict of callable
        """
        ## includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        shock_vars = self.bellman_period.get_shocks()
        shock_vals = {sym: input_grid[sym] for sym in shock_vars}

        # override any decision rules if necessary
        fresh_dr = {**self.other_dr, **new_dr}

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        r = static_reward(
            bellman_period,
            fresh_dr,
            {sym: given_vals[sym] for sym in self.state_variables},
            parameters=self.parameters,
            agent=None,  ## TODO: Pass through the agent?
            shocks=shock_vals,
            ## Handle multiple decision rules?
        )
        return -r


class CustomLoss:
    """
    A custom loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)
    """

    def __init__(self, loss_function, bellman_period, parameters = None, other_dr=dict()):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.state_variables = self.bellman_period.arrival_states
        self.other_dr = other_dr
        self.loss_function = loss_function

    def __call__(self, new_dr, input_grid: Grid):
        """
        new_dr : dict of callable
        """
        ## includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        ## most variable part -- many uses of double shocks
        shock_vars = self.bellman_period.get_shocks()
        shock_vals = {sym: input_grid[sym] for sym in shock_vars}

        # override any decision rules if necessary
        fresh_dr = {**self.other_dr, **new_dr}

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        neg_loss = self.loss_function(
            self.bellman_period,
            fresh_dr,  # useful
            {
                sym: given_vals[sym] for sym in self.state_variables
            },  # replace with arrival states
            parameters=self.parameters,
            shocks=shock_vals,
        )
        return -neg_loss
