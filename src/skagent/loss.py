import numpy as np
from skagent.bellman import BellmanPeriod
from skagent.grid import Grid
import torch


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
    # move this logic to BellmanPeriod
    rsym = list(
        {sym for sym in bellman_period.block.reward if agent is None or bellman_period.block.reward[sym] == agent}
    )[0]

    reward = bellman_period.reward_function(states, shocks, controls, parameters, agent = agent, decision_rules=dr)

    # Maybe this can be less complicated because of unified array API
    if isinstance(reward[rsym], torch.Tensor) and torch.any(torch.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")
    if isinstance(reward[rsym], np.ndarray) and np.any(np.isnan(reward[rsym])):
        raise Exception(f"Calculated reward {[rsym]} is NaN: {reward}")

    return reward[rsym]


class CustomLoss:
    """
    A custom loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)

    TODO: leaving this as ambiguously about Blocks and BellmanPeriods for now
    """

    def __init__(self, loss_function, block, parameters = None, other_dr=dict()):
        self.block = block
        self.parameters = parameters
        self.state_variables = self.block.get_arrival_states(calibration=parameters)
        self.other_dr = other_dr
        self.loss_function = loss_function

    def __call__(self, new_dr, input_grid: Grid):
        """
        new_dr : dict of callable
        """
        ## includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        ## most variable part -- many uses of double shocks
        shock_vars = self.block.get_shocks()
        shock_vals = {sym: input_grid[sym] for sym in shock_vars}

        # override any decision rules if necessary
        fresh_dr = {**self.other_dr, **new_dr}

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        neg_loss = self.loss_function(
            self.block,
            fresh_dr,  # useful
            {
                sym: given_vals[sym] for sym in self.state_variables
            },  # replace with arrival states
            parameters=self.parameters,
            shocks=shock_vals,
        )
        return -neg_loss


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
            self.bellman_period,
            fresh_dr,
            {sym: given_vals[sym] for sym in self.state_variables},
            parameters=self.parameters,
            agent=None,  ## TODO: Pass through the agent?
            shocks=shock_vals,
            ## Handle multiple decision rules?
        )
        return -r


class EstimatedDiscountedLifetimeRewardLoss:
    """
    A loss function for a Block that computes the discounted lifetime reward for T time periods.

    Parameters
    -----------

    bellman_period
    discount_factor
    big_t: int
        The number of time steps to compute reward for
    parameters
    """

    def __init__(self, bellman_period, discount_factor, big_t, parameters):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.state_variables = self.bellman_period.get_arrival_states(calibration=parameters)
        self.discount_factor = discount_factor
        self.big_t = big_t

    def __call__(self, df: callable, input_grid: Grid):
        # convoluted
        shock_vars = self.bellman_period.get_shocks()
        big_t_shock_syms = sum(
            [
                [f"{sym}_{t}" for sym in list(shock_vars.keys())]
                for t in range(self.big_t)
            ],
            [],
        )
        # TODO: codify this encoding and decoding of the grid into a separate object
        # It is specifically the EDLR loss function that requires big_t of the shocks.
        # other AiO loss functions use 2 copies of the shocks only.

        # includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        shock_vals = {sym: given_vals[sym] for sym in big_t_shock_syms}
        shocks_by_t = {
            sym: torch.stack([shock_vals[f"{sym}_{t}"] for t in range(self.big_t)])
            for sym in shock_vars
        }

        # bellman_period, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        edlr = estimate_discounted_lifetime_reward(
            self.bellman_period,
            self.discount_factor,
            df,
            {sym: given_vals[sym] for sym in self.state_variables},
            self.big_t,
            parameters=self.parameters,
            agent=None,  # TODO: Pass through the agent?
            shocks_by_t=shocks_by_t,
            # Handle multiple decision rules?
        )
        return -edlr


class BellmanEquationLoss:
    """
    Creates a Bellman equation loss function for the Maliar method.

    The Bellman equation is: V(s) = max_c { u(s,c,ε) + β E_ε'[V(s')] }
    where s' = f(s,c,ε) is the next state given current state s, control c, and shock ε,
    and the expectation E_ε' is taken over future shock realizations ε'.

    This follows the same pattern as get_estimated_discounted_lifetime_reward_loss
    and is designed for use with the Maliar all-in-one approach.

    This function expects the input grid to contain two independent shock realizations:
    - {shock_sym}_0: shocks for period t (used for immediate reward and transitions)
    - {shock_sym}_1: shocks for period t+1 (used for continuation value evaluation)

    Parameters
    ----------
    state_variables : list of str
        List of state variable names (endogenous state variables)
    block : model.DBlock
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β
    value_network : callable
        A value function that takes state variables and returns value estimates
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards

    Returns
    -------
    callable
        A loss function that takes (decision_function, input_grid) and returns
        the Bellman equation residual loss
    """

    def __init__(
        self, block, discount_factor, value_network, parameters={}, agent=None
    ):
        self.block = block
        self.parameters = parameters
        self.arrival_variables = block.get_arrival_states(calibration=parameters)

        self.value_network = value_network

        self.discount_factor = discount_factor
        if callable(discount_factor):
            raise Exception(
                "Currently only numerical, not state-dependent, discount factors are supported."
            )

        # Get shock variables
        shock_vars = block.get_shocks()
        self.shock_syms = list(shock_vars.keys())

        self.agent = agent
        # Test reward variables
        reward_vars = [
            sym
            for sym in block.reward
            if agent is None or block.reward[sym] == self.agent
        ]
        if len(reward_vars) == 0:
            raise Exception("No reward variables found in block")

    def __call__(self, df, input_grid: Grid):
        """
        Bellman equation loss function.

        Parameters
        ----------
        df : callable
            Decision function from policy network
        input_grid : Grid
            Grid containing current states and two independent shock realizations:
            - {shock_sym}_0: period t shocks
            - {shock_sym}_1: period t+1 shocks (independent of period t)

        Returns
        -------
        torch.Tensor
            Bellman equation residual loss (squared)
        """
        given_vals = input_grid.to_dict()

        # Extract current states and both shock realizations
        states_t = {sym: given_vals[sym] for sym in self.arrival_variables}
        shocks = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
        shocks.update({f"{sym}_1": given_vals[f"{sym}_1"] for sym in self.shock_syms})

        # Use helper function to estimate Bellman residual with combined shock object
        bellman_residual = estimate_bellman_residual(
            self.block,
            self.discount_factor,
            self.value_network,
            df,
            states_t,
            shocks,
            self.parameters,
            self.agent,
        )

        # Return squared residual as loss
        return bellman_residual**2
