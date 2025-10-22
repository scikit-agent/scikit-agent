"""
Functions for creating and reasoning about Dynamic Stochastic Optimization Problems (DSOPs).

The Block data structure is rather general, and can be used to represent static problems.

Converting Block models into DSOPs involves identifying transition and reward functions,
and framing them in terms of arrival states, shocks, and decisions.

"""

import numpy as np
import torch
from torch.autograd import grad


class BellmanPeriod:
    """
    A class representing a period of a Bellman or Dynamic Stochastic Optimization Problem.

    TODO: Currently this is based on a Block, but I think BlockBellmanPeriod should be
    a subclass of an abstract class.
    """

    def __init__(self, block, calibration, decision_rules=None):
        self.block = block
        self.calibration = calibration
        self.decision_rules = decision_rules
        self.arrival_states = self.block.get_arrival_states(calibration)

    def get_arrival_states(self, calibration):
        return self.block.get_arrival_states(
            calibration if calibration else self.calibration
        )

    def get_controls(self):
        return self.block.get_controls()

    def get_shocks(self):
        return self.block.get_shocks()

    def transition_function(
        self, states_t, shocks_t, controls_t, parameters=None, decision_rules=None
    ):
        decision_rules = (
            decision_rules
            if decision_rules
            else (self.decision_rules if self.decision_rules else {})
        )
        parameters = (
            parameters if parameters else (self.calibration if self.calibration else {})
        )

        vals = parameters | states_t | shocks_t | controls_t
        post = self.block.transition(vals, decision_rules, fix=list(controls_t.keys()))

        return {sym: post[sym] for sym in self.arrival_states}

    def decision_function(
        self, states_t, shocks_t, parameters=None, decision_rules=None
    ):
        decision_rules = (
            decision_rules
            if decision_rules
            else (self.decision_rules if self.decision_rules else {})
        )
        parameters = (
            parameters if parameters else (self.calibration if self.calibration else {})
        )

        vals = parameters | states_t | shocks_t
        post = self.block.transition(vals, decision_rules)
        return {sym: post[sym] for sym in decision_rules}

    def reward_function(
        self,
        states_t,
        shocks_t,
        controls_t,
        parameters=None,
        agent=None,
        decision_rules=None,
    ):
        decision_rules = (
            decision_rules
            if decision_rules
            else (self.decision_rules if self.decision_rules else {})
        )
        parameters = (
            parameters if parameters else (self.calibration if self.calibration else {})
        )

        vals_t = parameters | states_t | shocks_t | controls_t
        post = self.block.transition(
            vals_t, decision_rules, fix=list(controls_t.keys())
        )
        return {
            sym: post[sym]
            for sym in self.block.reward
            if agent is None or self.block.reward[sym] == agent
        }

    def grad_reward_function(
        self,
        states_t,
        shocks_t,
        controls_t,
        parameters,
        wrt,
        agent=None,
        decision_rules=None,
    ):
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
        agent : str, optional
            If specified, only compute gradients for rewards belonging to this agent
        decision_rules : dict, optional
            Decision rules of control variables that will _not_ be given to the function

        Returns
        -------
        dict
            Nested dictionary of gradients for each reward symbol and variable:
            {reward_sym: {var_name: gradient}}
        """
        decision_rules = (
            decision_rules
            if decision_rules
            else (self.decision_rules if self.decision_rules else {})
        )
        parameters = (
            parameters if parameters else (self.calibration if self.calibration else {})
        )

        # Combine all variables for block evaluation
        vals_t = parameters | states_t | shocks_t | controls_t

        # Compute rewards using block transition
        post = self.block.transition(
            vals_t, decision_rules, fix=list(controls_t.keys())
        )

        # move this logic to BP
        rewards = {
            sym: post[sym]
            for sym in self.block.reward
            if agent is None or self.block.reward[sym] == agent
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


def estimate_discounted_lifetime_reward(
    bellman_period,
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

    bellman_period
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

    # Get all reward symbols for the agent
    # TODO: move logic to bellman period
    reward_syms = list(
        {
            sym
            for sym in bellman_period.block.reward
            if agent is None or bellman_period.block.reward[sym] == agent
        }
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

        if callable(dr):
            # assume a full decision function has been passed in
            controls_t = dr(states_t, shocks_t, parameters)
        else:
            # create a decision function from the decision rule
            controls_t = bellman_period.decision_function(
                states_t, shocks_t, parameters, decision_rules=dr
            )

        reward_t = bellman_period.reward_function(
            states_t, shocks_t, controls_t, parameters, agent=agent
        )

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
        states_t = bellman_period.transition_function(
            states_t, shocks_t, controls_t, parameters
        )

    return total_discounted_reward


def estimate_bellman_residual(
    bellman_period,
    discount_factor,
    value_function,
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
    bellman_period : BellmanPeriod
        The Bellman period with transitions, rewards, etc.
    discount_factor : float
        The discount factor β
    value_function : callable
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

    # Get shock variable names
    shock_vars = bellman_period.get_shocks()
    shock_syms = list(shock_vars.keys())

    # Extract period-specific shocks from the combined shocks object
    shocks_t = {sym: shocks[f"{sym}_0"] for sym in shock_syms}
    shocks_t_plus_1 = {sym: shocks[f"{sym}_1"] for sym in shock_syms}

    # Get reward variables
    # TODO: logic to BP
    reward_vars = [
        sym
        for sym in bellman_period.block.reward
        if agent is None or bellman_period.block.reward[sym] == agent
    ]
    if len(reward_vars) == 0:
        raise Exception("No reward variables found in block")
    reward_sym = reward_vars[0]  # Assume single reward for now

    # Get current value estimates (using period t shocks)
    current_values = value_function(states_t, shocks_t, parameters)

    # Get controls from decision function (using period t shocks)
    controls_t = df(states_t, shocks_t, parameters)

    # Compute immediate reward (using period t shocks)
    immediate_reward = bellman_period.reward_function(
        states_t, shocks_t, controls_t, parameters
    )[reward_sym]

    # Compute next states (using period t shocks)
    next_states = bellman_period.transition_function(
        states_t, shocks_t, controls_t, parameters
    )

    # Compute continuation value using value network (using period t+1 shocks)
    continuation_values = value_function(next_states, shocks_t_plus_1, parameters)

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    return bellman_residual


def estimate_euler_residual(
    bellman_period,
    discount_factor,
    df,
    states_t,
    shocks,
    parameters={},
    agent=None,
):
    """
    Computes the Euler equation residual for given states and shocks.

    The Euler equation is the first-order condition relating marginal utilities
    across periods. This function computes:

        f = u'(c_t, ...) - β * u'(c_{t+1}, ...)

    where the reward function u(...) and its derivatives are determined by the
    BellmanPeriod's block dynamics. Any model-specific factors (like gross returns,
    permanent income shocks, etc.) should be embedded in the reward function itself.

    Following Maliar et al. (2021) Definition 2.7 and equation (12), the AiO method
    approximates the squared expectation with two independent shock realizations:

        L(θ) = E[(f|_{ε=ε₁}) * (f|_{ε=ε₂})]

    where ε₁ and ε₂ are independent draws.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period with transitions, rewards, etc.
    discount_factor : float
        The discount factor β (time preference parameter).
    df : callable or dict
        Decision function that returns controls given states and shocks.
        Can be a callable with signature df(states_t, shocks_t, parameters) -> controls_t
        or a dict of decision rules.
    states_t : dict
        Current state variables (arrival states)
    shocks : dict
        Shock realizations for both periods:
        - {shock_sym}_0: period t shocks (for transitions to t+1)
        - {shock_sym}_1: period t+1 shocks (for transitions to t+2)
        This structure supports the AiO expectation operator.
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards

    Returns
    -------
    torch.Tensor
        Euler equation residual: u'(c_t, ...) - β * u'(c_{t+1}, ...)
        When using AiO operator, returns residuals for both shock realizations.

    Notes
    -----
    This implementation follows Maliar, Maliar, and Winant (2021, JME) Section 2.2.
    The AiO expectation operator (Definition 2.7) requires two independent shock
    realizations to approximate the squared expectation E[(E[f])²].

    Examples
    --------
    >>> # Euler equation automatically adapts to your model's reward structure
    >>> residual = estimate_euler_residual(
    ...     bp, discount_factor=0.95, df=my_policy,
    ...     states_t, shocks, parameters
    ... )
    """
    if callable(discount_factor):
        raise ValueError(
            "State-dependent discount factors not yet supported for Euler residuals. "
            "Please pass a numerical discount factor."
        )

    # Get shock variable names
    shock_vars = bellman_period.get_shocks()
    shock_syms = list(shock_vars.keys())

    # Extract period-specific shocks from the combined shocks object
    # shocks_0 are for transitions from t to t+1
    # shocks_1 are for transitions from t+1 to t+2 (independent realization)
    shocks_t = {sym: shocks[f"{sym}_0"] for sym in shock_syms}
    shocks_t_plus_1 = {sym: shocks[f"{sym}_1"] for sym in shock_syms}

    # Get reward variables (should have exactly one for standard Euler equation)
    reward_vars = [
        sym
        for sym in bellman_period.block.reward
        if agent is None or bellman_period.block.reward[sym] == agent
    ]
    if len(reward_vars) == 0:
        raise ValueError("No reward variables found in block for the specified agent")

    reward_sym = reward_vars[0]

    # Get controls from decision function for period t
    if callable(df):
        # Full decision function provided
        controls_t = df(states_t, shocks_t, parameters)
    else:
        # Dictionary of decision rules provided
        controls_t = bellman_period.decision_function(
            states_t, shocks_t, parameters, decision_rules=df
        )

    # Compute next period states (t+1) using first shock realization
    states_t_plus_1 = bellman_period.transition_function(
        states_t, shocks_t, controls_t, parameters
    )

    # Get controls for period t+1 using second independent shock realization
    if callable(df):
        controls_t_plus_1 = df(states_t_plus_1, shocks_t_plus_1, parameters)
    else:
        controls_t_plus_1 = bellman_period.decision_function(
            states_t_plus_1, shocks_t_plus_1, parameters, decision_rules=df
        )

    # Get control symbols to compute gradients with respect to
    control_syms = list(controls_t.keys())
    if len(control_syms) == 0:
        raise ValueError("No control variables found in decision function")
    if len(control_syms) > 1:
        raise NotImplementedError(
            "Euler residual estimation currently only supports single-control models. "
            f"Found controls: {control_syms}"
        )

    control_sym = control_syms[0]  # Assume single control for now

    # Make controls require gradients for automatic differentiation
    # We need to enable gradients on the control variables
    controls_t_grad = {
        sym: controls_t[sym].detach().requires_grad_(True) for sym in controls_t
    }
    controls_t_plus_1_grad = {
        sym: controls_t_plus_1[sym].detach().requires_grad_(True)
        for sym in controls_t_plus_1
    }

    # Compute marginal utility at t (u'(c_t))
    reward_gradients_t = bellman_period.grad_reward_function(
        states_t,
        shocks_t,
        controls_t_grad,
        parameters,
        wrt={control_sym: controls_t_grad[control_sym]},
        agent=agent,
    )
    marginal_utility_t = reward_gradients_t[reward_sym][control_sym]

    # Compute marginal utility at t+1 (u'(c_{t+1}))
    reward_gradients_t_plus_1 = bellman_period.grad_reward_function(
        states_t_plus_1,
        shocks_t_plus_1,
        controls_t_plus_1_grad,
        parameters,
        wrt={control_sym: controls_t_plus_1_grad[control_sym]},
        agent=agent,
    )
    marginal_utility_t_plus_1 = reward_gradients_t_plus_1[reward_sym][control_sym]

    # Euler equation first-order condition
    # Any model-specific factors (returns, shocks, etc.) should be in the reward function
    euler_residual = marginal_utility_t - discount_factor * marginal_utility_t_plus_1

    return euler_residual
