"""
Functions for creating and reasoning about Dynamic Stochastic Optimization Problems (DSOPs).

The Block data structure is rather general, and can be used to represent static problems.

Converting Block models into DSOPs involves identifying transition and reward functions,
and framing them in terms of arrival states, shocks, and decisions.

"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from skagent.utils import compute_gradients_for_tensors

if TYPE_CHECKING:
    from skagent.block import Block


class BellmanPeriod:
    """
    A class representing a period of a Bellman or Dynamic Stochastic Optimization Problem.

    This class wraps a Block model with calibration parameters and decision rules,
    providing methods for computing transitions, decisions, rewards, and their gradients.

    Parameters
    ----------
    block : Block
        The underlying block model containing dynamics, shocks, and reward definitions.
    discount_variable : str
        A variable name which represents the discount factor for future value streams.
    calibration : dict[str, Any]
        Dictionary of calibration parameters for the model.
    decision_rules : dict[str, Callable] | None, optional
        Dictionary mapping control variable names to decision rule functions.

    Attributes
    ----------
    block : Block
        The underlying block model.
    discount_variable : str
        The name of the discount factor variable.
    calibration : dict[str, Any]
        The calibration parameters.
    decision_rules : dict[str, Callable] | None
        The decision rules for control variables.
    arrival_states : set[str]
        The set of arrival state variable names.

    Notes
    -----
    Future versions may introduce an abstract base class to support different
    block types beyond DBlock/RBlock.
    """

    block: Block
    discount_variable: str
    calibration: dict[str, Any]
    decision_rules: dict[str, Callable] | None
    arrival_states: set[str]

    def __init__(
        self,
        block: Block,
        discount_variable: str,
        calibration: dict[str, Any],
        decision_rules: dict[str, Callable] | None = None,
    ) -> None:
        self.block = block
        self.calibration = calibration
        self.discount_variable = discount_variable
        self.decision_rules = decision_rules
        self.arrival_states = self.block.get_arrival_states(calibration)

    def _resolve_decision_rules(
        self, decision_rules: dict[str, Callable] | None
    ) -> dict[str, Callable]:
        """Resolve decision rules with fallback to instance attribute then empty dict."""
        if decision_rules is not None:
            return decision_rules
        if self.decision_rules is not None:
            return self.decision_rules
        return {}

    def _resolve_parameters(self, parameters: dict[str, Any] | None) -> dict[str, Any]:
        """Resolve parameters with fallback to instance calibration then empty dict."""
        if parameters is not None:
            return parameters
        if self.calibration is not None:
            return self.calibration
        return {}

    def get_arrival_states(self, calibration: dict[str, Any] | None = None) -> set[str]:
        """Get arrival state variable names for given calibration."""
        return self.block.get_arrival_states(
            calibration if calibration is not None else self.calibration
        )

    def get_controls(self) -> dict[str, Any]:
        """Get control variables from the block."""
        return self.block.get_controls()

    def get_shocks(self) -> dict[str, Any]:
        """Get shock distributions from the block."""
        return self.block.get_shocks()

    def get_reward_syms(self, agent: str | None = None) -> list[str]:
        """Return all reward symbols for *agent* (or all agents if *agent* is None).

        Parameters
        ----------
        agent : str | None, optional
            If specified, only return reward symbols for this agent.

        Raises
        ------
        ValueError
            If no reward variables match the given agent.
        """
        reward_vars = [
            sym
            for sym in self.block.reward
            if agent is None or self.block.reward[sym] == agent
        ]
        if not reward_vars:
            raise ValueError(
                f"No reward variables found in block for agent '{agent}'"
                if agent is not None
                else "No reward variables found in block"
            )
        return reward_vars

    def get_reward_sym(self, agent: str | None = None) -> str:
        """Return the first reward symbol for *agent* (or any agent if *agent* is None).

        If multiple reward symbols match, only the first is returned.
        Models with multiple rewards per agent are not currently supported.

        Raises
        ------
        ValueError
            If no reward variables match the given agent.
        """
        return self.get_reward_syms(agent)[0]

    def compute_controls(
        self,
        df: dict[str, Callable] | Callable,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Compute control variable values from a decision function or decision rules.

        This generalises ``decision_function`` to also accept an external callable
        with signature ``df(states_t, shocks_t, parameters) -> controls_t``.

        Parameters
        ----------
        df : dict[str, Callable] | Callable
            A callable decision function, or a dict of decision rules passed
            through to ``decision_function``.
        states_t : dict[str, Any]
            Current state variables.
        shocks_t : dict[str, Any]
            Current shock realizations.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).

        Returns
        -------
        dict[str, Any]
            Control variable values.

        Raises
        ------
        TypeError
            If *df* is neither callable nor a dict.
        """
        if callable(df):
            params = parameters if parameters is not None else {}
            return df(states_t, shocks_t, params)
        if not isinstance(df, dict):
            raise TypeError(
                f"df must be a callable decision function or a dict of decision rules, "
                f"got {type(df).__name__!r}"
            )
        return self.decision_function(states_t, shocks_t, parameters, decision_rules=df)

    def transition_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        controls_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute the transition to next-period arrival states.

        Parameters
        ----------
        states_t : dict[str, Any]
            Current state variables.
        shocks_t : dict[str, Any]
            Current shock realizations.
        controls_t : dict[str, Any]
            Current control variable values.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            Next-period arrival state values.
        """
        decision_rules = self._resolve_decision_rules(decision_rules)
        parameters = self._resolve_parameters(parameters)

        vals = parameters | states_t | shocks_t | controls_t
        post = self.block.transition(vals, decision_rules, fix=list(controls_t.keys()))

        return {sym: post[sym] for sym in self.arrival_states}

    def decision_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute control variable values from decision rules.

        Parameters
        ----------
        states_t : dict[str, Any]
            Current state variables.
        shocks_t : dict[str, Any]
            Current shock realizations.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            Control variable values computed from decision rules.
        """
        decision_rules = self._resolve_decision_rules(decision_rules)
        parameters = self._resolve_parameters(parameters)

        vals = parameters | states_t | shocks_t
        post = self.block.transition(vals, decision_rules)
        return {sym: post[sym] for sym in decision_rules}

    def reward_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        controls_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute reward values for the current period.

        Parameters
        ----------
        states_t : dict[str, Any]
            Current state variables.
        shocks_t : dict[str, Any]
            Current shock realizations.
        controls_t : dict[str, Any]
            Current control variable values.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        agent : str | None, optional
            If specified, only return rewards for this agent.
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            Reward values for the period.
        """
        decision_rules = self._resolve_decision_rules(decision_rules)
        parameters = self._resolve_parameters(parameters)

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
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        controls_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Return the full ex post variables for the period.

        Parameters
        ----------
        states_t : dict[str, Any]
            Current state variables.
        shocks_t : dict[str, Any]
            Current shock realizations.
        controls_t : dict[str, Any]
            Current control variable values.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        agent : str | None, optional
            Agent identifier (currently unused, reserved for future use).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            All computed variables from the block transition.
        """
        decision_rules = self._resolve_decision_rules(decision_rules)
        parameters = self._resolve_parameters(parameters)

        vals_t = parameters | states_t | shocks_t | controls_t
        post = self.block.transition(
            vals_t, decision_rules, fix=list(controls_t.keys())
        )
        return post

    def grad_reward_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        controls_t: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """
        Compute gradients of reward function with respect to specified variables.

        Parameters
        ----------
        states_t : dict[str, Any]
            State variables at time t.
        shocks_t : dict[str, Any]
            Shock variables at time t.
        controls_t : dict[str, Any]
            Control variables at time t.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        agent : str | None, optional
            If specified, only compute gradients for rewards belonging to this agent.
        decision_rules : dict[str, Callable] | None, optional
            Decision rules of control variables that will _not_ be given to the function.
        create_graph : bool, optional
            If True, the graph of the derivative is constructed, allowing higher-order
            derivatives and end-to-end training through the gradient computation.

        Returns
        -------
        dict[str, dict[str, torch.Tensor | None]]
            Nested dictionary of gradients for each reward symbol and variable:
            {reward_sym: {var_name: gradient}}. Gradient is None if the reward
            does not depend on the variable.
        """
        decision_rules = self._resolve_decision_rules(decision_rules)
        parameters = self._resolve_parameters(parameters)

        # Combine all variables for block evaluation
        vals_t = parameters | states_t | shocks_t | controls_t

        # Compute rewards using block transition
        post = self.block.transition(
            vals_t, decision_rules, fix=list(controls_t.keys())
        )
        # TODO: refactor with post-function

        # Filter rewards by agent
        rewards = {sym: post[sym] for sym in self.get_reward_syms(agent)}

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(rewards, wrt, create_graph=create_graph)

    def grad_transition_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        controls_t: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """
        Compute gradients of transition function with respect to specified variables.

        This computes ∂s_{t+1}/∂x for each arrival state s_{t+1} and each variable x
        specified in wrt. This is needed for Euler equations where the gradient of
        future states with respect to current controls appears (e.g., ∂A_{t+1}/∂c_t = -R).

        Parameters
        ----------
        states_t : dict[str, Any]
            State variables at time t.
        shocks_t : dict[str, Any]
            Shock variables at time t.
        controls_t : dict[str, Any]
            Control variables at time t.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules of control variables that will _not_ be given to the function.
        create_graph : bool, optional
            If True, the graph of the derivative is constructed, allowing higher-order
            derivatives and end-to-end training through the gradient computation.

        Returns
        -------
        dict[str, dict[str, torch.Tensor | None]]
            Nested dictionary of gradients for each arrival state and variable:
            {state_sym: {var_name: gradient}}.
        """
        # Use the existing transition_function method to compute next states
        next_states = self.transition_function(
            states_t, shocks_t, controls_t, parameters, decision_rules
        )

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(
            next_states, wrt, create_graph=create_graph
        )

    def grad_pre_state_function(
        self,
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
        control_sym: str | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
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
        states_t : dict[str, Any]
            Arrival state variables (with requires_grad=True for gradient computation).
        shocks_t : dict[str, Any]
            Shock variables at time t.
        wrt : dict[str, torch.Tensor]
            Dictionary of arrival states to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        control_sym : str | None, optional
            Name of the control variable whose info-set we want gradients for.
            If None, uses the first control found in the block.
        create_graph : bool, optional
            If True, the graph of the derivative is constructed, allowing higher-order
            derivatives and end-to-end training through the gradient computation.

        Returns
        -------
        dict[str, dict[str, torch.Tensor | None]]
            Nested dictionary of gradients for each pre-state variable and arrival state:
            {pre_state_var: {state_sym: gradient}}.
        """
        parameters = self._resolve_parameters(parameters)

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
        if control_rule is not None and hasattr(control_rule, "iset"):
            pre_state_vars = control_rule.iset
        else:
            # Fall back: assume control depends on arrival states directly
            pre_state_vars = list(self.arrival_states)

        # Compute pre-state values using helper method
        pre_state_values = self._compute_pre_state_values(
            pre_state_vars, states_t, shocks_t, parameters
        )

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(
            pre_state_values, wrt, create_graph=create_graph
        )

    def _compute_pre_state_values(
        self,
        pre_state_vars: list[str],
        states_t: dict[str, Any],
        shocks_t: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Compute pre-decision state variable values.

        This is a helper method used by grad_pre_state_function and estimate_euler_residual
        to compute pre-state values without duplication.

        Parameters
        ----------
        pre_state_vars : list[str]
            List of pre-state variable names (from Control.iset).
        states_t : dict[str, Any]
            Arrival state variables.
        shocks_t : dict[str, Any]
            Shock variables at time t.
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).

        Returns
        -------
        dict[str, Any]
            Dictionary mapping pre-state variable names to their computed values.
        """
        parameters = self._resolve_parameters(parameters)

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


def _extract_period_shocks(
    bellman_period: BellmanPeriod,
    shocks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and split combined shocks into period-t and period-(t+1) dicts.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        Used to look up the model's shock variable names.
    shocks : dict[str, Any]
        Combined shock dict with keys ``{sym}_0`` (period t) and ``{sym}_1``
        (period t+1) for each shock symbol.

    Returns
    -------
    shocks_t, shocks_t_plus_1 : tuple[dict, dict]

    Raises
    ------
    KeyError
        If a required shock key is missing.
    """
    shock_syms = list(bellman_period.get_shocks())
    for sym in shock_syms:
        if f"{sym}_0" not in shocks:
            raise KeyError(
                f"Missing shock '{sym}_0' in shocks dict. For models with shocks, "
                f"provide two independent realizations: '{sym}_0' (period t) "
                f"and '{sym}_1' (period t+1)."
            )
        if f"{sym}_1" not in shocks:
            raise KeyError(
                f"Missing shock '{sym}_1' in shocks dict. For models with shocks, "
                f"provide two independent realizations: '{sym}_0' (period t) "
                f"and '{sym}_1' (period t+1)."
            )
    shocks_t = {sym: shocks[f"{sym}_0"] for sym in shock_syms}
    shocks_t_plus_1 = {sym: shocks[f"{sym}_1"] for sym in shock_syms}
    return shocks_t, shocks_t_plus_1


def _ensure_grad(
    controls: dict[str, Any], sym: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ensure the control tensor for *sym* has ``requires_grad=True``.

    If the tensor already tracks gradients it is returned as-is.  Otherwise it
    is detached and re-attached so that ``torch.autograd.grad`` can
    differentiate through it.

    Returns the (possibly new) tensor and an updated copy of *controls*.
    """
    c = controls[sym]
    if not c.requires_grad:
        c = c.detach().requires_grad_(True)
    return c, {**controls, sym: c}


def estimate_discounted_lifetime_reward(
    bellman_period: BellmanPeriod,
    dr: dict[str, Callable] | Callable,
    states_0: dict[str, Any],
    big_t: int,
    shocks_by_t: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    agent: str | None = None,
) -> float | torch.Tensor:
    """
    Compute the discounted lifetime reward for a model given a fixed T of periods to simulate forward.

    Based on Maliar, Maliar, and Winant (2021, JME).

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period object containing the model. The discount factor is
        extracted from the post-transition variables via ``bellman_period.discount_variable``.
    dr : dict[str, Callable] | Callable
        Decision rules (dict of functions), or a decision function that
        returns the decisions given states, shocks, and parameters.
    states_0 : dict[str, Any]
        Initial states as a dictionary mapping symbols to values.
        Both scalar and vector values are supported.
    big_t : int
        Number of time steps to simulate forward.
    shocks_by_t : dict[str, Any] | None, optional
        Dictionary mapping shock symbols to arrays of shock values at each
        time period. Shape should be (big_t, ...) for each shock.
    parameters : dict[str, Any] | None, optional
        Calibration parameters (defaults to empty dict).
    agent : str | None, optional
        Name of reference agent for rewards. If None, all rewards are summed.

    Returns
    -------
    float | torch.Tensor
        The total discounted lifetime reward.
    """
    states_t = states_0
    total_discounted_reward = 0
    cumulative_discount = 1  # Π_{τ=0}^{t-1} β_τ

    reward_syms = bellman_period.get_reward_syms(agent)

    for t in range(big_t):
        if shocks_by_t is not None:
            shocks_t = {sym: shocks_by_t[sym][t] for sym in shocks_by_t}
        else:
            shocks_t = {}

        controls_t = bellman_period.compute_controls(dr, states_t, shocks_t, parameters)

        # TODO: can improve performance by consolidating multiple calls
        #       that simulate forward.
        post = bellman_period.post_function(
            states_t, shocks_t, controls_t, parameters, agent=agent
        )
        discount_factor = post[bellman_period.discount_variable]

        reward_t = bellman_period.reward_function(
            states_t, shocks_t, controls_t, parameters, agent=agent
        )

        period_reward = 0
        for rsym in reward_syms:
            if isinstance(reward_t[rsym], torch.Tensor) and torch.any(
                torch.isnan(reward_t[rsym])
            ):
                raise ValueError(f"Calculated reward {rsym} is NaN: {reward_t}")
            if isinstance(reward_t[rsym], np.ndarray) and np.any(
                np.isnan(reward_t[rsym])
            ):
                raise ValueError(f"Calculated reward {rsym} is NaN: {reward_t}")
            period_reward += reward_t[rsym]

        total_discounted_reward += period_reward * cumulative_discount
        cumulative_discount = cumulative_discount * discount_factor

        states_t = bellman_period.transition_function(
            states_t, shocks_t, controls_t, parameters
        )

    return total_discounted_reward


def estimate_bellman_residual(
    bellman_period: BellmanPeriod,
    value_function: Callable,
    df: dict[str, Callable] | Callable,
    states_t: dict[str, Any],
    shocks: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    agent: str | None = None,
) -> torch.Tensor:
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
        The Bellman period with transitions, rewards, etc. The discount factor is
        extracted from the post-transition variables via ``bellman_period.discount_variable``.
    value_function : Callable
        A value function that takes state variables and returns value estimates.
    df : dict[str, Callable] | Callable
        Decision function that returns controls given states and shocks.
        Can be a callable with signature df(states_t, shocks_t, parameters) -> controls_t
        or a dict of decision rules.
    states_t : dict[str, Any]
        Current state variables.
    shocks : dict[str, Any]
        Shock realizations for both periods:
        - {shock_sym}_0: period t shocks (for immediate reward and transitions)
        - {shock_sym}_1: period t+1 shocks (for continuation value evaluation)
    parameters : dict[str, Any] | None, optional
        Model parameters for calibration (defaults to empty dict).
    agent : str | None, optional
        Agent identifier for rewards.

    Returns
    -------
    torch.Tensor
        Bellman equation residual.

    Raises
    ------
    ValueError
        If no reward variables are found in the block.
    KeyError
        If required shock variables are missing from the shocks dict.
    """
    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)

    reward_sym = bellman_period.get_reward_sym(agent)

    # value_function is an external callable that may not handle None
    params_ext = parameters if parameters is not None else {}

    # Get current value estimates (using period t shocks)
    current_values = value_function(states_t, shocks_t, params_ext)

    # Get controls from decision function (using period t shocks)
    controls_t = bellman_period.compute_controls(df, states_t, shocks_t, parameters)

    # Compute immediate reward (using period t shocks)
    immediate_reward = bellman_period.reward_function(
        states_t, shocks_t, controls_t, parameters
    )[reward_sym]

    # Compute next states (using period t shocks)
    next_states = bellman_period.transition_function(
        states_t, shocks_t, controls_t, parameters
    )

    # Compute continuation value using value network (using period t+1 shocks)
    continuation_values = value_function(next_states, shocks_t_plus_1, params_ext)

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
    bellman_period: BellmanPeriod,
    discount_factor: float,
    df: dict[str, Callable] | Callable,
    states_t: dict[str, Any],
    shocks: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    agent: str | None = None,
) -> torch.Tensor:
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

    **Limitations:**

    Currently only supports single-control models. Multi-control models will raise
    NotImplementedError.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period with transitions, rewards, etc.
    discount_factor : float
        The discount factor β (time preference parameter). Must be numerical
        (state-dependent discount factors are not yet supported).
    df : dict[str, Callable] | Callable
        Decision function that returns controls given states and shocks.
        Can be a callable with signature df(states_t, shocks_t, parameters) -> controls_t
        or a dict of decision rules.
    states_t : dict[str, Any]
        Current state variables (arrival states).
    shocks : dict[str, Any]
        Shock realizations for both periods:
        - {shock_sym}_0: period t shocks (for transitions to t+1)
        - {shock_sym}_1: period t+1 shocks (for transitions to t+2)
        This structure supports the AiO expectation operator.
    parameters : dict[str, Any] | None, optional
        Model parameters for calibration (defaults to empty dict).
    agent : str | None, optional
        Agent identifier for rewards.

    Returns
    -------
    torch.Tensor
        Euler equation residual computed using two independent shock realizations
        (one for each period transition). Returns one residual value per state sample.

    Raises
    ------
    ValueError
        If discount_factor is callable, no control or reward variables found,
        or marginal utility cannot be computed.
    NotImplementedError
        If the model has multiple control variables.
    KeyError
        If required shock variables are missing from the shocks dict.

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

    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)

    reward_sym = bellman_period.get_reward_sym(agent)

    # Get controls from decision function for period t
    controls_t = bellman_period.compute_controls(df, states_t, shocks_t, parameters)

    # Compute next period states (t+1) using first shock realization
    states_t_plus_1 = bellman_period.transition_function(
        states_t, shocks_t, controls_t, parameters
    )

    # Get controls for period t+1 using second independent shock realization
    controls_t_plus_1 = bellman_period.compute_controls(
        df, states_t_plus_1, shocks_t_plus_1, parameters
    )

    # Get control symbols to compute gradients with respect to
    control_syms = list(controls_t)
    if len(control_syms) == 0:
        raise ValueError("No control variables found in decision function")
    if len(control_syms) > 1:
        raise NotImplementedError(
            "Euler residual estimation currently only supports single-control models. "
            f"Found controls: {control_syms}"
        )

    control_sym = control_syms[0]

    # Ensure requires_grad on control tensors for gradient computation.
    # Detach and reattach if the policy network didn't set requires_grad.
    c_t, controls_t_grad = _ensure_grad(controls_t, control_sym)
    c_t1, controls_t1_grad = _ensure_grad(controls_t_plus_1, control_sym)

    # Marginal utilities at t and t+1 via grad_reward_function.
    # create_graph=True keeps the policy network in the computation graph
    # for end-to-end training.
    reward_grads_t = bellman_period.grad_reward_function(
        states_t,
        shocks_t,
        controls_t_grad,
        wrt={control_sym: c_t},
        parameters=parameters,
        agent=agent,
        create_graph=True,
    )
    marginal_utility_t = reward_grads_t[reward_sym][control_sym]
    if marginal_utility_t is None:
        raise ValueError(
            f"Could not compute marginal utility at period t: "
            f"reward '{reward_sym}' does not depend on control '{control_sym}'"
        )

    reward_grads_t1 = bellman_period.grad_reward_function(
        states_t_plus_1,
        shocks_t_plus_1,
        controls_t1_grad,
        wrt={control_sym: c_t1},
        parameters=parameters,
        agent=agent,
        create_graph=True,
    )
    marginal_utility_t_plus_1 = reward_grads_t1[reward_sym][control_sym]
    if marginal_utility_t_plus_1 is None:
        raise ValueError(
            f"Could not compute marginal utility at period t+1: "
            f"reward '{reward_sym}' does not depend on control '{control_sym}'"
        )

    # Transition gradients: ∂s_{t+1}/∂c_t for all arrival states.
    # This captures how today's control affects tomorrow's state (e.g., ∂a'/∂c = -1).
    trans_grads_nested = bellman_period.grad_transition_function(
        states_t,
        shocks_t,
        controls_t_grad,
        wrt={control_sym: c_t},
        parameters=parameters,
        create_graph=True,
    )
    transition_gradients = {
        state_sym: trans_grads_nested[state_sym][control_sym]
        for state_sym in bellman_period.arrival_states
    }

    # Pre-state gradients: ∂m'/∂s' (how pre-state depends on arrival state).
    # By the envelope theorem: V'(s') = u'(c') * ∂m'/∂s'.
    # For consumption-saving with m = a*R + y, this gives ∂m/∂a = R.
    #
    # Make arrival states at t+1 require gradients for differentiation.
    states_t_plus_1_grad = {}
    for sym in bellman_period.arrival_states:
        s = states_t_plus_1[sym]
        if not s.requires_grad:
            s = s.detach().requires_grad_(True)
        states_t_plus_1_grad[sym] = s

    pre_state_gradients = bellman_period.grad_pre_state_function(
        states_t_plus_1_grad,
        shocks_t_plus_1,
        wrt=states_t_plus_1_grad,
        parameters=parameters,
        control_sym=control_sym,
        create_graph=True,
    )

    # Chain rule: ∂m'/∂c = Σ_s [∂m'/∂s' * ∂s'/∂c]
    # This gives the total derivative of pre-state w.r.t. control through all paths
    return_factor_sum = torch.zeros_like(marginal_utility_t_plus_1)

    for state_sym in bellman_period.arrival_states:
        trans_grad = transition_gradients[state_sym]
        if trans_grad is None:
            continue

        for _pre_state_var, state_grads in pre_state_gradients.items():
            pre_state_grad = state_grads.get(state_sym)
            if pre_state_grad is not None:
                return_factor_sum = return_factor_sum + pre_state_grad * trans_grad

    # Euler equation residual.
    # FOC: u'(c_t) = -β * V'(s') * ∂s'/∂c
    # Envelope: V'(s') = u'(c') * ∂m'/∂s'
    # Residual: f = u'(c_t) + β * u'(c_{t+1}) * Σ_s [∂m'/∂s' * ∂s'/∂c] = 0
    #
    # For consumption-saving: ∂a'/∂c = -1, ∂m'/∂a' = R
    # So: f = u'(c) + β * u'(c') * (-R) = u'(c) - βR*u'(c')

    euler_residual = (
        marginal_utility_t
        + discount_factor * marginal_utility_t_plus_1 * return_factor_sum
    )

    return euler_residual
