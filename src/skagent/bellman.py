"""
Dynamic Stochastic Optimization Problems (DSOPs) built on Block models.

Bellman timing within a period:

    [arrival] + [shock] -> [pre] -> [control] -> [post] -> [arrival']

- [arrival]  ``s``  state on arrival, before any shock
- [shock]    ``e``  exogenous random variable
- [pre]      ``m``  pre-decision state (the control's iset)
- [control]  ``c``  chosen by the decision rule on m
- [post]            post-transition output: the bag of variables
                    realized in the period (m, c, u, b, s'),
                    returned by :meth:`BellmanPeriod.post_function`
- [arrival'] ``s'`` next-period arrival state

Reward ``u`` and discount ``b`` are realized between [control] and
[arrival'].

State-variable naming (long / short / informal):

- pre-decision: ``pre_decision_state`` / ``pre_state`` / ``iset``

The Bellman-timing distinction between the post-decision *state* (a
single timing point) and the post-transition *bag* is conflated in
``post_function`` for now, and will be split in a future PR.

A ``_rule`` is a user-supplied callable on pre-decision variables; a
``_function`` is a callable on arrival states. Module-level ``df``
and ``vf`` are the decision and value callables; each accepts a
single ``Callable`` or a ``dict[str, Callable]`` (``df`` keyed by
control symbol; ``vf`` by agent name).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from skagent.utils import compute_gradients_for_tensors

if TYPE_CHECKING:
    from skagent.block import Block


class BellmanPeriod:
    """A period of a Bellman / DSOP, wrapping a Block with calibration
    and decision rules. See module docstring for the timing convention.

    Parameters
    ----------
    block : Block
        The underlying block model.
    discount_variable : str
        Name of the variable holding the discount factor.
    calibration : dict[str, Any]
        Calibration parameters.
    decision_rules : dict[str, Callable] | None, optional
        Mapping from control symbol to decision rule callable.
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

    def _resolve_inputs(
        self,
        shocks: dict[str, Any] | None,
        decision_rules: dict[str, Callable] | None,
        parameters: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Callable], dict[str, Any]]:
        """Resolve ``(shocks, decision_rules, parameters)``, replacing ``None``
        with defaults: ``{}`` for shocks; instance ``decision_rules`` then
        ``{}`` for decision_rules; instance calibration for parameters.
        """
        if decision_rules is None:
            decision_rules = self.decision_rules
        if decision_rules is None:
            decision_rules = {}
        return (
            shocks if shocks is not None else {},
            decision_rules,
            parameters if parameters is not None else self.calibration,
        )

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

    def compute_pre_state(
        self,
        control_sym: str,
        states: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the pre-decision state values for *control_sym* as
        ``{var: value}``. The variables are those in the control's
        information set (``iset``).

        If the pre-decision state variables are already in *states* or
        *shocks*, they are taken from there directly. Otherwise block
        dynamics are run from arrival states up to the control to
        produce them.
        """
        iset = self.block.dynamics[control_sym].iset
        shocks = shocks if shocks is not None else {}
        params = parameters if parameters is not None else self.calibration
        vals = params | states | shocks

        if all(isym in vals for isym in iset):
            return {isym: vals[isym] for isym in iset}

        drs = {cs: (lambda: 1) for cs in self.get_controls()}
        out = self.block.transition(vals, drs, until=control_sym)
        return {isym: out[isym] for isym in iset}

    def compute_controls(
        self,
        df: dict[str, Callable] | Callable,
        states: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Compute control values at *states*.

        Accepts a callable (invoked directly as ``df(states, shocks, params)``)
        or a dict of decision rules keyed by control symbol (passed through
        to :meth:`decision_function`).
        """
        shocks = shocks if shocks is not None else {}
        params = parameters if parameters is not None else self.calibration

        if callable(df):
            return df(states, shocks, params)
        if not isinstance(df, dict):
            raise TypeError(
                f"df must be a callable decision function or a dict of decision rules, "
                f"got {type(df).__name__!r}"
            )
        return self.decision_function(
            states, shocks=shocks, parameters=params, decision_rules=df
        )

    def compute_value(
        self,
        vf: dict[str, Callable] | Callable,
        states: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> Any:
        """
        Compute value-function output at *states*, parallel to
        :meth:`compute_controls`.

        Accepts a single callable, or a dict ``{agent: callable}`` from
        which *agent* selects an entry. The selected callable receives
        arrival states; any pre-decision (iset) computation it needs is
        the callable's responsibility.
        """
        shocks = shocks if shocks is not None else {}
        params = parameters if parameters is not None else self.calibration

        if callable(vf):
            return vf(states, shocks, params)
        if not isinstance(vf, dict):
            raise TypeError(
                f"vf must be a callable value function or a dict mapping "
                f"agent name to a callable, got {type(vf).__name__!r}"
            )
        if agent is None:
            raise ValueError(
                "vf is a dict (per-agent value functions); the 'agent' "
                f"argument must be specified. Available agents: {sorted(vf)}."
            )
        if agent not in vf:
            raise KeyError(
                f"vf has no entry for agent '{agent}'. Available agents: {sorted(vf)}."
            )
        return vf[agent](states, shocks, params)

    def transition_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute the transition to next-period arrival states.

        Parameters
        ----------
        states : dict[str, Any]
            Current state variables.
        controls : dict[str, Any]
            Current control variable values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            Next-period arrival state values.
        """
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks | controls
        post = self.block.transition(vals, decision_rules, fix=list(controls.keys()))

        return {sym: post[sym] for sym in self.arrival_states}

    def decision_function(
        self,
        states: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute control variable values from decision rules.

        Parameters
        ----------
        states : dict[str, Any]
            Current state variables.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to instance decision_rules).

        Returns
        -------
        dict[str, Any]
            Control variable values computed from decision rules.
        """
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks
        post = self.block.transition(vals, decision_rules)
        return {sym: post[sym] for sym in decision_rules}

    def reward_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """
        Compute reward values for the current period.

        Parameters
        ----------
        states : dict[str, Any]
            Current state variables.
        controls : dict[str, Any]
            Current control variable values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
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
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks | controls
        post = self.block.transition(vals, decision_rules, fix=list(controls.keys()))
        return {sym: post[sym] for sym in self.get_reward_syms(agent)}

    def post_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
    ) -> dict[str, Any]:
        """Return the post-transition output for the period (every
        variable realized by ``block.transition``). The ``agent``
        parameter is reserved for future per-agent filtering.
        """
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks | controls
        post = self.block.transition(vals, decision_rules, fix=list(controls.keys()))
        return post

    def grad_reward_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        decision_rules: dict[str, Callable] | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """
        Compute gradients of reward function with respect to specified variables.

        Parameters
        ----------
        states : dict[str, Any]
            State variables.
        controls : dict[str, Any]
            Control variables.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        shocks : dict[str, Any] | None, optional
            Shock variables (defaults to empty dict).
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
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks | controls
        # Calls block.transition directly (rather than post_function) to keep
        # the exact computation graph needed for autograd differentiation.
        post = self.block.transition(vals, decision_rules, fix=list(controls.keys()))
        rewards = {sym: post[sym] for sym in self.get_reward_syms(agent)}
        return compute_gradients_for_tensors(rewards, wrt, create_graph=create_graph)

    def grad_transition_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """Gradients ∂s_{t+1}/∂x for each next-period arrival state and
        each tensor in *wrt*. ``create_graph=True`` enables higher-order
        derivatives.
        """
        next_states = self.transition_function(
            states,
            controls,
            shocks=shocks,
            parameters=parameters,
            decision_rules=decision_rules,
        )
        return compute_gradients_for_tensors(
            next_states, wrt, create_graph=create_graph
        )

    def grad_pre_state_function(
        self,
        states: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        control_sym: str | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """Gradients ∂m/∂s of pre-decision state variables m with respect
        to arrival states s in *wrt*. Used for the envelope condition

            V'(s) = u'(c) · ∂m/∂s.

        If *control_sym* is None, uses the first control with an iset.
        """
        if control_sym is None:
            for sym, rule in self.block.dynamics.items():
                if hasattr(rule, "iset"):
                    control_sym = sym
                    break
        if control_sym is None:
            raise ValueError("No control with pre-state found in block dynamics")
        control_rule = self.block.dynamics.get(control_sym)
        if control_rule is None or not hasattr(control_rule, "iset"):
            raise ValueError(
                f"Control '{control_sym}' has no pre-state (iset) attribute defined. "
                f"Ensure the Control object in block.dynamics['{control_sym}'] is "
                "constructed with an explicit 'iset' argument specifying which "
                "variables the control depends on."
            )

        pre_state_values = self.compute_pre_state(
            control_sym, states, shocks=shocks, parameters=parameters
        )

        return compute_gradients_for_tensors(
            pre_state_values, wrt, create_graph=create_graph
        )

    def resolve_discount_factor(self, post: dict[str, Any]) -> Any:
        """Return ``post[self.discount_variable]``, raising ``KeyError``
        with a diagnostic message if the discount variable is missing.
        Expects the post-transition output returned by
        :meth:`post_function`.
        """
        dv = self.discount_variable
        if dv not in post:
            raise KeyError(
                f"Discount variable '{dv}' not found in post-transition output. "
                f"Available variables: {sorted(post.keys())}. "
                "Ensure the discount variable is defined in block.dynamics "
                "or passed in calibration."
            )
        return post[dv]


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

    Notes
    -----
    For deterministic models with no shocks, ``shock_syms`` is empty,
    no keys are required in *shocks*, and both returned dicts are empty.
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
    if not isinstance(c, torch.Tensor):
        raise TypeError(
            f"Control '{sym}' must be a torch.Tensor for gradient computation, "
            f"got {type(c).__name__}"
        )
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
    r"""
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
        time period. The first axis must have length ``big_t``; remaining
        axes are batch dimensions (e.g., shape ``(big_t, n_samples)``).
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
    total_discounted_reward = 0.0
    cumulative_discount = 1.0  # Π_{τ=0}^{t-1} β_τ

    reward_syms = bellman_period.get_reward_syms(agent)

    for t in range(big_t):
        if shocks_by_t is not None:
            shocks_t = {sym: shocks_by_t[sym][t] for sym in shocks_by_t}
        else:
            shocks_t = {}

        controls_t = bellman_period.compute_controls(
            dr, states_t, shocks=shocks_t, parameters=parameters
        )

        # TODO: can improve performance by consolidating multiple calls
        #       that simulate forward.
        post = bellman_period.post_function(
            states_t, controls_t, shocks=shocks_t, parameters=parameters, agent=agent
        )
        discount_factor = bellman_period.resolve_discount_factor(post)

        reward_t = bellman_period.reward_function(
            states_t, controls_t, shocks=shocks_t, parameters=parameters, agent=agent
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
            states_t, controls_t, shocks=shocks_t, parameters=parameters
        )

    return total_discounted_reward


def estimate_bellman_residual(
    bellman_period: BellmanPeriod,
    vf: dict[str, Callable] | Callable,
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
    vf : dict[str, Callable] | Callable
        Value function ``vf(states_t, shocks_t, parameters) -> tensor``
        on arrival states, or a dict mapping ``agent`` name to such a
        callable for multi-agent models (in which case ``agent`` must be
        specified). Any pre-decision (iset) computation the underlying
        approximator needs is the callable's responsibility.
    df : dict[str, Callable] | Callable
        Decision callable ``df(states_t, shocks_t, parameters) -> controls_t``
        on arrival states, or a dict of decision rules keyed by control
        symbol (callables on the iset).
    states_t : dict[str, Any]
        Current arrival state variables.
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

    Notes
    -----
    Single-reward, multi-control: this function returns a single residual
    tensor, evaluated against the first reward variable matching ``agent``.
    For multi-control models it complements
    :func:`estimate_euler_residual` (which returns one residual per
    control) and :func:`estimate_bellman_foc_residual` (which returns one
    FOC residual per control by differentiating the value callable).
    """
    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)

    reward_sym = bellman_period.get_reward_sym(agent)

    # V(s_t) — value at the period-t arrival state
    current_values = bellman_period.compute_value(
        vf, states_t, shocks=shocks_t, parameters=parameters, agent=agent
    )

    # Controls from decision callable (also takes arrival states)
    controls_t = bellman_period.compute_controls(
        df, states_t, shocks=shocks_t, parameters=parameters
    )

    # Immediate reward at period t
    immediate_reward = bellman_period.reward_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )[reward_sym]

    # Next-period arrival state from the transition
    next_states = bellman_period.transition_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )

    # V(s_{t+1}) — continuation value at the next-period arrival state
    # using the second independent shock draw
    continuation_values = bellman_period.compute_value(
        vf,
        next_states,
        shocks=shocks_t_plus_1,
        parameters=parameters,
        agent=agent,
    )

    # TODO: this is all calling the forward simulation multiple times;
    #       can be made more efficient
    post = bellman_period.post_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )
    discount_factor = bellman_period.resolve_discount_factor(post)

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    if torch.any(torch.isnan(bellman_residual)) or torch.any(
        torch.isinf(bellman_residual)
    ):
        # Provide detailed diagnostics to help locate the source
        def _range_str(t):
            if not isinstance(t, torch.Tensor):
                return str(t)
            return f"[{t.min().item():.2e}, {t.max().item():.2e}]"

        raise ValueError(
            "Bellman residual contains NaN or Inf. "
            f"immediate_reward range: {_range_str(immediate_reward)}, "
            f"discount_factor: {_range_str(discount_factor)}, "
            f"continuation_values range: {_range_str(continuation_values)}, "
            f"current_values range: {_range_str(current_values)}."
        )

    return bellman_residual


def _chain_rule_return_factor(
    bellman_period: BellmanPeriod,
    control_sym: str,
    transition_gradients: dict[str, torch.Tensor | None],
    pre_state_gradients: dict[str, dict[str, torch.Tensor | None]],
    like: torch.Tensor,
) -> torch.Tensor:
    r"""Sum the chain-rule product :math:`\sum_s \partial m'/\partial s' \cdot \partial s'/\partial c`.

    Here :math:`s'` indexes the next-period arrival states, :math:`m'` is
    the next-period pre-decision state (the variable in the control's
    information set), and :math:`c` is the period-:math:`t` control. The
    sum is the envelope-condition return factor used in the Euler
    residual.

    Raises ``ValueError`` if no chain-rule path contributes.
    """
    if torch.any(torch.isnan(like)) or torch.any(torch.isinf(like)):
        raise ValueError(
            f"Euler residual: marginal_reward_t1 contains NaN or Inf for "
            f"control '{control_sym}'. Cannot compute chain-rule return factor."
        )

    total = torch.zeros_like(like)

    for state_sym in bellman_period.arrival_states:
        trans_grad = transition_gradients[state_sym]
        if trans_grad is None:
            logging.debug(
                "Transition gradient d(%s)/d(%s) is None (no computational path). "
                "If unexpected, check that control '%s' tensors require_grad.",
                state_sym,
                control_sym,
                control_sym,
            )
            continue
        for state_grads in pre_state_gradients.values():
            pre_state_grad = state_grads.get(state_sym)
            if pre_state_grad is not None:
                total = total + pre_state_grad * trans_grad

    if torch.any(torch.isnan(total)):
        raise ValueError(
            f"Euler residual: return_factor_sum contains NaN for "
            f"control '{control_sym}'. This indicates ill-conditioned "
            "transition or pre-state gradients. Check block dynamics for "
            "numerical stability."
        )
    if torch.any(torch.isinf(total)):
        raise ValueError(
            f"Euler residual: return_factor_sum contains Inf for "
            f"control '{control_sym}'. This indicates ill-conditioned "
            "transition or pre-state gradients. Check block dynamics for "
            "numerical stability."
        )
    if not torch.any(total != 0):
        raise ValueError(
            "Euler residual: return_factor_sum is zero for all arrival states. "
            "No arrival state depends on the control through the transition "
            "and pre-state gradients. Check that the block dynamics correctly "
            f"connect the control '{control_sym}' to the arrival states "
            f"{sorted(bellman_period.arrival_states)} and that the Control "
            "object has a properly defined 'iset'."
        )
    return total


def _euler_residual_single_control(
    bellman_period: BellmanPeriod,
    discount_factor: Any,
    control_sym: str,
    reward_sym: str,
    states_t: dict[str, Any],
    controls_t: dict[str, Any],
    states_t_plus_1: dict[str, Any],
    controls_t_plus_1: dict[str, Any],
    shocks_t: dict[str, Any],
    shocks_t_plus_1: dict[str, Any],
    parameters: dict[str, Any] | None,
    agent: str | None,
) -> torch.Tensor:
    """Compute the Euler residual for a single control variable.

    This is the inner workhorse called once per control by
    ``estimate_euler_residual``.  Factored out to support multi-control models.
    """
    c_t, controls_t_grad = _ensure_grad(controls_t, control_sym)
    c_t1, controls_t1_grad = _ensure_grad(controls_t_plus_1, control_sym)

    # ∂u/∂c at period t
    grads_t = bellman_period.grad_reward_function(
        states_t,
        controls_t_grad,
        wrt={control_sym: c_t},
        shocks=shocks_t,
        parameters=parameters,
        agent=agent,
        create_graph=True,
    )
    marginal_reward_t = grads_t[reward_sym][control_sym]
    if marginal_reward_t is None:
        raise ValueError(
            f"Could not compute marginal reward at period t: "
            f"reward '{reward_sym}' does not depend on control '{control_sym}'"
        )

    # ∂u/∂c at period t+1
    grads_t1 = bellman_period.grad_reward_function(
        states_t_plus_1,
        controls_t1_grad,
        wrt={control_sym: c_t1},
        shocks=shocks_t_plus_1,
        parameters=parameters,
        agent=agent,
        create_graph=True,
    )
    marginal_reward_t1 = grads_t1[reward_sym][control_sym]
    if marginal_reward_t1 is None:
        raise ValueError(
            f"Could not compute marginal reward at period t+1: "
            f"reward '{reward_sym}' does not depend on control '{control_sym}'"
        )

    # Transition gradients: ∂s_{t+1}/∂c_t for all arrival states.
    trans_grads_nested = bellman_period.grad_transition_function(
        states_t,
        controls_t_grad,
        wrt={control_sym: c_t},
        shocks=shocks_t,
        parameters=parameters,
        create_graph=True,
    )
    transition_gradients = {
        state_sym: trans_grads_nested[state_sym][control_sym]
        for state_sym in bellman_period.arrival_states
    }

    # Pre-state gradients: ∂m'/∂s' (envelope condition).
    states_t1_grad = {
        sym: s if s.requires_grad else s.detach().requires_grad_(True)
        for sym, s in states_t_plus_1.items()
        if sym in bellman_period.arrival_states
    }

    pre_state_gradients = bellman_period.grad_pre_state_function(
        states_t1_grad,
        wrt=states_t1_grad,
        shocks=shocks_t_plus_1,
        parameters=parameters,
        control_sym=control_sym,
        create_graph=True,
    )

    return_factor = _chain_rule_return_factor(
        bellman_period,
        control_sym,
        transition_gradients,
        pre_state_gradients,
        like=marginal_reward_t1,
    )

    # f = u'(c_t) + β * u'(c_{t+1}) * Σ_s [∂m'/∂s' * ∂s'/∂c] = 0
    return marginal_reward_t + discount_factor * marginal_reward_t1 * return_factor


def estimate_euler_residual(
    bellman_period: BellmanPeriod,
    df: dict[str, Callable] | Callable,
    states_t: dict[str, Any],
    shocks: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    agent: str | None = None,
    controls_t: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    r"""Compute the Euler equation residual for given states and shocks.

    The Euler equation is the first-order condition from the Bellman equation,
    relating marginal rewards across periods.  For each control variable
    :math:`c_j`, this function computes the residual:

    .. math::

        f_j = u'(c_{j,t}) + \beta \cdot u'(c_{j,t+1}) \cdot \sum_s \left[
            \frac{\partial s_{t+1}}{\partial c_{j,t}}
            \cdot \frac{\partial m'_j}{\partial s_{t+1}}
        \right]

    At optimality :math:`f_j = 0` for every control :math:`j`.

    The discount factor :math:`\beta` is obtained from the model via
    ``bellman_period.discount_variable``.

    Following Maliar et al. (2021, JME) Definition 2.7, this function uses two
    independent shock realizations (AiO expectation operator).

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period with transitions, rewards, etc.  The discount factor
        is extracted from the post-transition variables via
        ``bellman_period.discount_variable``.
    df : dict[str, Callable] | Callable
        Decision function or dict of decision rules.
    states_t : dict[str, Any]
        Current state variables (arrival states).
    shocks : dict[str, Any]
        Shock realizations for both periods (``{sym}_0`` and ``{sym}_1``).
    parameters : dict[str, Any] | None, optional
        Model parameters for calibration.
    agent : str | None, optional
        Agent identifier for rewards.
    controls_t : dict[str, Any] | None, optional
        Pre-computed period-t controls. When provided, the function skips its
        internal ``compute_controls`` call for period t.  This is used by
        ``EulerEquationLoss`` to share the same control tensors between the
        residual computation and the constraint slack computation.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from each control symbol to its Euler residual tensor.
    """
    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)

    reward_sym = bellman_period.get_reward_sym(agent)

    # Period-t controls and transition
    if controls_t is None:
        controls_t = bellman_period.compute_controls(
            df, states_t, shocks=shocks_t, parameters=parameters
        )
    states_t_plus_1 = bellman_period.transition_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )

    # Period-(t+1) controls (second independent shock draw — AiO)
    controls_t_plus_1 = bellman_period.compute_controls(
        df, states_t_plus_1, shocks=shocks_t_plus_1, parameters=parameters
    )

    control_syms = list(controls_t)
    if len(control_syms) == 0:
        raise ValueError("No control variables found in decision function")

    # Resolve discount factor from model
    post = bellman_period.post_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )
    discount_factor = bellman_period.resolve_discount_factor(post)

    # Compute Euler residual for each control
    residuals = {}
    for control_sym in control_syms:
        residuals[control_sym] = _euler_residual_single_control(
            bellman_period,
            discount_factor,
            control_sym,
            reward_sym,
            states_t,
            controls_t,
            states_t_plus_1,
            controls_t_plus_1,
            shocks_t,
            shocks_t_plus_1,
            parameters,
            agent,
        )

    return residuals


def estimate_bellman_foc_residual(
    bellman_period: BellmanPeriod,
    vf: dict[str, Callable] | Callable,
    df: dict[str, Callable] | Callable,
    states_t: dict[str, Any],
    shocks: dict[str, Any],
    parameters: dict[str, Any] | None = None,
    agent: str | None = None,
) -> dict[str, torch.Tensor]:
    r"""Compute the first-order condition (FOC) residual from the Bellman equation.

    The Bellman equation is:

    .. math::

        V(s) = \max_c \{ u(s,c,\varepsilon) + \beta E_{\varepsilon'}[V(s')] \}

    The FOC w.r.t. each control :math:`c_j` is:

    .. math::

        \frac{\partial u}{\partial c_j}
        + \beta \sum_s \frac{\partial V(s')}{\partial s'_s}
        \cdot \frac{\partial s'_s}{\partial c_j} = 0

    Adding a weighted FOC term to the Bellman loss improves convergence
    (Maliar et al. 2021, equation 14).

    Unlike :func:`estimate_euler_residual`, which replaces :math:`V'(s')` with
    the envelope condition :math:`u'(c') \cdot \partial m'/\partial s'` (where
    :math:`m'` is the next-period pre-decision state), this function
    differentiates the value callable directly.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period.
    vf : dict[str, Callable] | Callable
        Value function on arrival states; either a single callable or a
        per-agent dict. See :func:`estimate_bellman_residual` for the
        full contract.
    df : dict[str, Callable] | Callable
        Decision callable on arrival states, or dict of decision rules.
    states_t : dict[str, Any]
        Current arrival state variables.
    shocks : dict[str, Any]
        Shock realizations with ``{sym}_0`` and ``{sym}_1`` keys.
    parameters : dict[str, Any] | None, optional
        Model parameters.
    agent : str | None, optional
        Agent identifier.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from each control symbol to its FOC residual tensor.
    """
    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)
    reward_sym = bellman_period.get_reward_sym(agent)

    controls_t = bellman_period.compute_controls(
        df, states_t, shocks=shocks_t, parameters=parameters
    )
    control_syms = list(controls_t)

    post = bellman_period.post_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )
    discount_factor = bellman_period.resolve_discount_factor(post)

    residuals = {}
    for control_sym in control_syms:
        c_t, controls_t_grad = _ensure_grad(controls_t, control_sym)

        # u'(c_t) — marginal reward at period t
        reward_grads = bellman_period.grad_reward_function(
            states_t,
            controls_t_grad,
            wrt={control_sym: c_t},
            shocks=shocks_t,
            parameters=parameters,
            agent=agent,
            create_graph=True,
        )
        mr_t = reward_grads[reward_sym][control_sym]
        if mr_t is None:
            raise ValueError(
                f"Could not compute marginal reward: "
                f"reward '{reward_sym}' does not depend on control '{control_sym}'"
            )

        # s' = f(s, c, ε₀) — transition preserving autograd graph through c_t
        next_states = bellman_period.transition_function(
            states_t, controls_t_grad, shocks=shocks_t, parameters=parameters
        )

        # V(s', ε₁) — continuation value with second independent shock draw,
        # evaluated on next-period arrival states
        v_next = bellman_period.compute_value(
            vf,
            next_states,
            shocks=shocks_t_plus_1,
            parameters=parameters,
            agent=agent,
        )

        # ∂V/∂c via autograd chain rule: ∂V/∂s' * ∂s'/∂c
        dv_dc = torch.autograd.grad(
            v_next.sum(),
            c_t,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if dv_dc is None:
            # Continuation value has no differentiable dependence on this
            # control; with allow_unused=True this is a legitimate outcome
            # for multi-control models where only some controls affect V(s').
            # Treat it as a zero gradient and continue.
            logging.debug(
                "Autograd returned None for dV/d%s — treating as zero gradient.",
                control_sym,
            )
            dv_dc = torch.zeros_like(c_t)
        if torch.any(torch.isnan(dv_dc)):
            raise ValueError(
                f"Autograd gradient dV/d{control_sym} is NaN. "
                "Check that vf is properly initialized and numerically stable."
            )

        # FOC: u'(c) + β * ∂V(s',ε')/∂c = 0
        residuals[control_sym] = mr_t + discount_factor * dv_dc

    return residuals
