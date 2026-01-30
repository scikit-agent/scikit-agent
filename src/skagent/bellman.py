"""
Functions for creating and reasoning about Dynamic Stochastic Optimization Problems (DSOPs).

The Block data structure is rather general, and can be used to represent static problems.

Converting Block models into DSOPs involves identifying transition and reward functions,
and framing them in terms of arrival states, shocks, and decisions.

"""

import inspect

import numpy as np
import torch
from torch.autograd import grad

from skagent.utils import compute_gradients_for_tensors


class BellmanPeriod:
    """
    A class representing a period of a Bellman or Dynamic Stochastic Optimization Problem.

    TODO: Currently this is based on a Block, but I think BlockBellmanPeriod should be
    a subclass of an abstract class.

    Parameters
    -----------

    block: skagent.block.Block
        A block that represents a single time period of the Bellman problem

    discount_variable: str
        A variable name which represents the discount factor for future value streams

    calibration: dict
        Calibration dictionary for setting parameters.

    decision_rules: optional, dict
        A dictionary mapping control variable names to decision rules.

    """

    def __init__(self, block, discount_variable, calibration, decision_rules=None):
        self.block = block
        self.calibration = calibration
        self.discount_variable = discount_variable
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
        """
        TODO: refactor with post-function
        """
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
        """
        TODO: refactor with post-function
        """
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
        """
        TODO: refactor with post-function
        """
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

    def post_function(
        self,
        states_t,
        shocks_t,
        controls_t,
        parameters=None,
        agent=None,
        decision_rules=None,
    ):
        """
        Returns the full ex post variables for the period, given initial states,
        shocks, controls, and parameters.
        """
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
        return post

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
        # TODO: refactor with post-function

        # move this logic to BP
        rewards = {
            sym: post[sym]
            for sym in self.block.reward
            if agent is None or self.block.reward[sym] == agent
        }

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(rewards, wrt)

    def grad_transition_function(
        self,
        states_t,
        shocks_t,
        controls_t,
        parameters,
        wrt,
        decision_rules=None,
    ):
        """
        Compute gradients of transition function with respect to specified variables.

        This computes ∂s_{t+1}/∂x for each arrival state s_{t+1} and each variable x
        specified in wrt. This is needed for Euler equations where the gradient of
        future states with respect to current controls appears (e.g., ∂A_{t+1}/∂c_t = -R).

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
        decision_rules : dict, optional
            Decision rules of control variables that will _not_ be given to the function

        Returns
        -------
        dict
            Nested dictionary of gradients for each arrival state and variable:
            {state_sym: {var_name: gradient}}
        """
        # Use the existing transition_function method to compute next states
        next_states = self.transition_function(
            states_t, shocks_t, controls_t, parameters, decision_rules
        )

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(next_states, wrt)

    def grad_pre_state_function(
        self,
        states_t,
        shocks_t,
        parameters,
        wrt,
        control_sym=None,
    ):
        """
        Compute gradients of pre-decision state variables with respect to arrival states.

        This computes ∂m/∂s for each pre-decision state variable m and each arrival
        state s specified in wrt. This is needed for the envelope condition in dynamic
        programming, where the marginal value of an arrival state depends on how
        that state transforms through the dynamics before reaching the control.

        The "pre-decision state" (or "pre-state") is the state that exists immediately
        before the control decision is made. For example, cash-on-hand m = Ra + y is
        the pre-state before consumption c is chosen.

        By the envelope theorem:
            V'(s) = u'(c) * ∂m/∂s

        where m is the pre-decision state variable that the control depends on.

        For example, in a consumption-saving model with m = a*R + y:
            ∂m/∂a = R (the return on assets)

        Parameters
        ----------
        states_t : dict
            Arrival state variables (with requires_grad=True for gradient computation)
        shocks_t : dict
            Shock variables at time t
        parameters : dict
            Model parameters
        wrt : dict
            Dictionary of arrival states to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True
        control_sym : str, optional
            Name of the control variable whose info-set we want gradients for.
            If None, uses the first control found in the block.

        Returns
        -------
        dict
            Nested dictionary of gradients for each pre-state variable and arrival state:
            {pre_state_var: {state_sym: gradient}}
        """
        # Get the control's pre-state variables (stored as iset in the Control)
        if control_sym is None:
            # Find the first control in dynamics
            for sym, rule in self.block.dynamics.items():
                if hasattr(rule, "iset"):
                    control_sym = sym
                    break

        if control_sym is None:
            raise ValueError("No control with pre-state found in block dynamics")

        control_rule = self.block.dynamics.get(control_sym)
        if control_rule is None or not hasattr(control_rule, "iset"):
            raise ValueError(f"Control '{control_sym}' has no pre-state (iset) defined")

        pre_state_vars = control_rule.iset

        # Compute pre-state values using helper method
        pre_state_values = self._compute_pre_state_values(
            pre_state_vars, states_t, shocks_t, parameters
        )

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(pre_state_values, wrt)

    def _compute_pre_state_values(self, pre_state_vars, states_t, shocks_t, parameters):
        """
        Compute pre-decision state variable values.

        This is a helper method used by grad_pre_state_function and estimate_euler_residual
        to compute pre-state values without duplication.

        Parameters
        ----------
        pre_state_vars : list
            List of pre-state variable names (from Control.iset)
        states_t : dict
            Arrival state variables
        shocks_t : dict
            Shock variables at time t
        parameters : dict
            Model parameters

        Returns
        -------
        dict
            Dictionary mapping pre-state variable names to their computed values
        """
        # Build values dict with arrival states and parameters
        vals = {**states_t, **shocks_t, **parameters}

        # Compute pre-state variables by running dynamics up to the control
        pre_state_values = {}
        for var_name in pre_state_vars:
            if var_name in self.arrival_states:
                # Pre-state variable IS an arrival state, gradient is identity
                pre_state_values[var_name] = states_t[var_name]
            elif var_name in self.block.dynamics:
                # Compute the dynamics for this variable
                rule = self.block.dynamics[var_name]
                if callable(rule):
                    sig = inspect.signature(rule)
                    args = {p: vals[p] for p in sig.parameters if p in vals}
                    pre_state_values[var_name] = rule(**args)
                else:
                    pre_state_values[var_name] = rule
            else:
                raise ValueError(
                    f"Pre-state variable '{var_name}' not found in arrival_states or dynamics"
                )

        return pre_state_values


def estimate_discounted_lifetime_reward(
    bellman_period,
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

        post = bellman_period.post_function(
            states_t, shocks_t, controls_t, parameters, agent=agent
        )

        discount_factor = post[bellman_period.discount_variable]

        # TODO: can improve performance by consolidating multiple calls
        #       that simulation forward.
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
    value_function,
    df,
    states_t,
    shocks,
    parameters={},
    agent=None,
):
    r"""
    Computes the Bellman equation residual for given states and shocks.

    The Bellman equation is:

    .. math::

        V(s) = \\max_c \\{ u(s,c,\\varepsilon) + \\beta E_{\\varepsilon'}[V(s')] \\}

    This function computes the residual:

    .. math::

        f = V(s) - [u(s,c,\\varepsilon) + \\beta V(s')]

    where :math:`s' = f(s,c,\\varepsilon)` and :math:`V(s')` is evaluated at a
    specific future shock realization :math:`\\varepsilon'`.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period with transitions, rewards, etc.
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

    # TODO: this is all calling the forward simulation multiple times;
    #       can be made more efficient
    post = bellman_period.post_function(states_t, shocks_t, controls_t, parameters)
    discount_factor = post[bellman_period.discount_variable]

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
    parameters=None,
    agent=None,
):
    """
    Computes the Euler equation residual for given states and shocks.

    The Euler equation is the first-order condition from the Bellman equation,
    relating marginal utilities across periods. This function computes the Euler
    equation **residual**:

    .. math::

        f = u'(c_t) + \\beta \\cdot u'(c_{t+1}) \\cdot \\sum_s \\left[
            \\frac{\\partial s_{t+1}}{\\partial c_t} \\cdot \\frac{\\partial m'}{\\partial s_{t+1}}
        \\right]

    where :math:`f` is the Euler equation residual, :math:`s_{t+1}` is the next-period
    arrival state, and :math:`m'` is the pre-decision state (information set for the
    control). At optimality, :math:`f = 0` represents the first-order condition being
    satisfied.

    **Derivation:**

    The first-order condition from the Bellman equation
    :math:`V(s) = \\max_c \\{ u(c) + \\beta E[V(s')] \\}` is:

    .. math::

        u'(c_t) = -\\beta E\\left[V'(s_{t+1}) \\cdot \\frac{\\partial s_{t+1}}{\\partial c_t}\\right]

    By the envelope theorem, :math:`V'(s') = u'(c') \\cdot \\frac{\\partial m'}{\\partial s'}`,
    where :math:`m'` is the pre-decision state. Substituting:

    .. math::

        u'(c_t) = -\\beta E\\left[u'(c_{t+1}) \\cdot \\frac{\\partial m'}{\\partial s_{t+1}}
            \\cdot \\frac{\\partial s_{t+1}}{\\partial c_t}\\right]

    Rearranging to define the residual :math:`f`:

    .. math::

        f = u'(c_t) + \\beta E\\left[u'(c_{t+1}) \\cdot \\frac{\\partial m'}{\\partial s_{t+1}}
            \\cdot \\frac{\\partial s_{t+1}}{\\partial c_t}\\right] = 0

    **Example:**

    For a consumption-saving model with :math:`a_{t+1} = m_t - c_t` where
    :math:`m_t = R \\cdot a_t + y_t` (cash-on-hand):

    - Transition gradient: :math:`\\frac{\\partial a_{t+1}}{\\partial c_t} = -1`
    - Pre-state gradient: :math:`\\frac{\\partial m'}{\\partial a_{t+1}} = R`
    - Combined: :math:`\\frac{\\partial m'}{\\partial a_{t+1}} \\cdot \\frac{\\partial a_{t+1}}{\\partial c_t} = R \\cdot (-1) = -R`

    Substituting into the residual:

    .. math::

        f = u'(c_t) + \\beta \\cdot u'(c_{t+1}) \\cdot (-R) = u'(c_t) - \\beta R \\cdot u'(c_{t+1})

    At optimality, :math:`f = 0` gives the standard Euler equation
    :math:`u'(c_t) = \\beta R E[u'(c_{t+1})]`.

    **Notation:**

    Following Maliar et al. (2021) Definition 2.7, this function computes a single
    Euler equation residual using two independent shock realizations. The residual
    :math:`f` uses shocks :math:`\\varepsilon_0` for transitions from :math:`t` to
    :math:`t+1` and :math:`\\varepsilon_1` for transitions from :math:`t+1` to
    :math:`t+2`.

    The loss function then computes :math:`L(\\theta) = E[f^2]` where the squared
    residual approximates the squared expectation operator from the paper.

    Note: We use :math:`\\varepsilon` (epsilon) to denote exogenous shocks, which may
    differ slightly from Maliar et al.'s notation but represents the same concept of
    stochastic disturbances in the model.

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
        Euler equation residual computed using two independent shock realizations
        (one for each period transition). Returns one residual value per state sample.

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
    if parameters is None:
        parameters = {}

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
    # For deterministic models (no shocks), shock_syms will be empty and these dicts will be empty
    for sym in shock_syms:
        if f"{sym}_0" not in shocks:
            raise KeyError(
                f"Missing shock '{sym}_0' in shocks dict. For models with shocks, "
                f"provide two independent realizations: '{sym}_0' (period t) and '{sym}_1' (period t+1)."
            )
        if f"{sym}_1" not in shocks:
            raise KeyError(
                f"Missing shock '{sym}_1' in shocks dict. For models with shocks, "
                f"provide two independent realizations: '{sym}_0' (period t) and '{sym}_1' (period t+1)."
            )
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

    # For training, we need to compute marginal utilities while keeping the policy
    # in the computation graph. We use create_graph=True to allow backprop through
    # the gradient computation itself.
    #
    # The approach:
    # 1. Compute reward with controls that require grad
    # 2. Use autograd to get u'(c) w.r.t. c, with create_graph=True
    # 3. This keeps the policy network in the graph for end-to-end training

    # Enable gradients on the original controls (not detached copies)
    c_t = controls_t[control_sym]
    c_t_plus_1 = controls_t_plus_1[control_sym]

    # Ensure controls require gradients for computing marginal utility
    if not c_t.requires_grad:
        c_t = c_t.detach().requires_grad_(True)
    if not c_t_plus_1.requires_grad:
        c_t_plus_1 = c_t_plus_1.detach().requires_grad_(True)

    # Build controls dicts with grad-enabled tensors
    controls_t_grad = {**controls_t, control_sym: c_t}
    controls_t_plus_1_grad = {**controls_t_plus_1, control_sym: c_t_plus_1}

    # Compute reward at t and get marginal utility u'(c_t)
    rewards_t = bellman_period.reward_function(
        states_t, shocks_t, controls_t_grad, parameters, agent=agent
    )
    reward_t = rewards_t[reward_sym]

    # Compute marginal utility at t: u'(c_t) = ∂u/∂c
    # Use create_graph=True to keep policy in computation graph for training
    marginal_utility_t = grad(
        reward_t.sum(),
        c_t,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if marginal_utility_t is None:
        raise ValueError(
            f"Could not compute marginal utility: reward '{reward_sym}' "
            f"does not depend on control '{control_sym}'"
        )

    # Compute reward at t+1 and get marginal utility u'(c_{t+1})
    rewards_t_plus_1 = bellman_period.reward_function(
        states_t_plus_1,
        shocks_t_plus_1,
        controls_t_plus_1_grad,
        parameters,
        agent=agent,
    )
    reward_t_plus_1 = rewards_t_plus_1[reward_sym]

    marginal_utility_t_plus_1 = grad(
        reward_t_plus_1.sum(),
        c_t_plus_1,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]

    if marginal_utility_t_plus_1 is None:
        raise ValueError(
            f"Could not compute marginal utility at t+1: reward '{reward_sym}' "
            f"does not depend on control '{control_sym}'"
        )

    # Compute transition gradients: ∂s_{t+1}/∂c_t for all arrival states
    # This captures how today's control affects tomorrow's state (e.g., ∂a'/∂c = -1)
    # We need create_graph=True here too for end-to-end training
    next_states = bellman_period.transition_function(
        states_t, shocks_t, controls_t_grad, parameters
    )

    transition_gradients = {}
    for state_sym in bellman_period.arrival_states:
        next_state = next_states[state_sym]
        # Check if this state depends on the control by attempting to compute ∂s_{t+1}/∂c_t.
        # Using allow_unused=True ensures grad returns None (not an error) when there's no
        # dependency, which is more robust than checking requires_grad. A tensor can have
        # requires_grad=False even when computed from gradients (e.g., after certain ops).
        trans_grad = grad(
            next_state.sum(),
            c_t,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        # If grad returns None, the state doesn't depend on control (e.g., p' = p * psi)
        transition_gradients[state_sym] = trans_grad

    # Compute the return factor from the dynamics: ∂m'/∂s' (how pre-state depends on arrival state)
    # By the envelope theorem: V'(s') = u'(c') * ∂m'/∂s'
    # For consumption-saving with m = a*R + y, this gives ∂m/∂a = R
    #
    # For this, we need to compute gradients of pre-state variables w.r.t. arrival states
    # Make arrival states at t+1 require gradients
    states_t_plus_1_grad = {}
    for sym in bellman_period.arrival_states:
        s = states_t_plus_1[sym]
        if not s.requires_grad:
            s = s.detach().requires_grad_(True)
        states_t_plus_1_grad[sym] = s

    # Get the control's pre-state variables
    control_rule = bellman_period.block.dynamics.get(control_sym)
    if control_rule is not None and hasattr(control_rule, "iset"):
        pre_state_vars = control_rule.iset
    else:
        # Fall back: assume control depends on arrival states directly
        pre_state_vars = list(bellman_period.arrival_states)

    # Compute pre-state values using the helper method (avoids code duplication)
    pre_state_values = bellman_period._compute_pre_state_values(
        pre_state_vars, states_t_plus_1_grad, shocks_t_plus_1, parameters
    )

    # Compute ∂(pre_state)/∂(arrival_state) using utility function with create_graph=True
    pre_state_gradients = compute_gradients_for_tensors(
        pre_state_values, states_t_plus_1_grad, create_graph=True
    )

    # Compute ∂(pre_state)/∂(arrival_state) * ∂(arrival_state)/∂c summed over states
    # This gives the total derivative of pre-state w.r.t. control through all paths
    return_factor_sum = torch.zeros_like(marginal_utility_t_plus_1)

    for state_sym in bellman_period.arrival_states:
        trans_grad = transition_gradients[state_sym]
        if trans_grad is None:
            continue

        # Sum over all pre-state variables
        for pre_state_var, state_grads in pre_state_gradients.items():
            pre_state_grad = state_grads.get(state_sym)
            if pre_state_grad is not None:
                # Chain rule: dm/dc = dm/ds * ds/dc
                return_factor_sum = return_factor_sum + pre_state_grad * trans_grad

    # Compute the Euler equation residual
    # The first-order condition is: u'(c_t) = β * V'(a') * |∂a'/∂c|
    # where V'(a') = u'(c') * ∂m'/∂a' (envelope theorem)
    #
    # In residual form with proper signs:
    # f = u'(c_t) + β * u'(c_{t+1}) * (∂a'/∂c) * (∂m'/∂a')
    #
    # For consumption-saving: ∂a'/∂c = -1, ∂m'/∂a' = R
    # So: f = u'(c) + β * u'(c') * (-1) * R = u'(c) - βR*u'(c')

    # return_factor_sum already contains (∂a'/∂c * ∂m'/∂a') summed over states
    euler_residual = (
        marginal_utility_t
        + discount_factor * marginal_utility_t_plus_1 * return_factor_sum
    )

    return euler_residual
