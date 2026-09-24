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

from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
import torch

from skagent.ground import GroundedBlock
from skagent.utils import any_nan, compute_gradients_for_tensors, tracked

if TYPE_CHECKING:
    from skagent.block import Block


class BellmanPeriod(GroundedBlock):
    """
    A class representing a period of a Bellman or Dynamic Stochastic Optimization Problem.

    A :class:`~skagent.ground.GroundedBlock` (a block read against a
    calibration) that additionally carries a discount factor, arrival states
    and decision rules, and provides methods for computing transitions,
    decisions, rewards, and their gradients.

    Parameters
    ----------
    block : Block
        The underlying block model containing dynamics, shocks, and reward definitions.
    discount_variable : str or None
        A variable name which represents the discount factor for future value
        streams, resolved out of the post-transition values. Pass ``None`` for a
        static block, where nothing is discounted and there is no discount factor
        to name; :meth:`resolve_discount_factor` then returns ``1.0``.
    calibration : dict[str, Any]
        Dictionary of calibration parameters for the model.
    rng : numpy.random.Generator, optional
        Generator for this period's shock draws. Since the shocks are resolved
        against this period's calibration, it is the period rather than the
        block that owns their sample path.

    Attributes
    ----------
    block : Block
        The underlying block model.
    discount_variable : str or None
        The name of the discount factor variable, or ``None`` when nothing is
        discounted.
    calibration : dict[str, Any]
        The calibration parameters.
    arrival_states : set[str]
        The set of arrival state variable names.
    rng : numpy.random.Generator | None
        The generator this period's shock draws come from.

    Notes
    -----
    Future versions may introduce an abstract base class to support different
    block types beyond DBlock/RBlock.
    """

    discount_variable: str | None
    arrival_states: set[str]

    def __init__(
        self,
        block: Block,
        discount_variable: str | None,
        calibration: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(block, calibration, rng=rng)
        self.discount_variable = discount_variable
        self.arrival_states = self.block.get_arrival_states(calibration)

    def _resolve_inputs(
        self,
        shocks: dict[str, Any] | None,
        decision_rules: dict[str, Callable] | None,
        parameters: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Callable], dict[str, Any]]:
        """Resolve ``(shocks, decision_rules, parameters)``, replacing ``None``
        with defaults: ``{}`` for shocks and for decision_rules; instance
        calibration for parameters.

        Decision rules are supplied per call rather than held on the period: a
        rule is a solution, and which rules the other controls are held at is a
        property of the question being asked rather than of the model.
        """
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
        shocks, _, params = self._resolve_inputs(shocks, None, parameters)
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
        Compute control variable values from a decision function or decision rules.

        This generalises ``decision_function`` to also accept an external callable
        with signature ``df(states, shocks, parameters) -> controls``.

        Parameters
        ----------
        df : dict[str, Callable] | Callable
            A callable decision function, or a dict of decision rules passed
            through to ``decision_function``.
        states : dict[str, Any]
            Current state values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
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
        shocks, _, params = self._resolve_inputs(shocks, None, parameters)

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
        shocks, _, params = self._resolve_inputs(shocks, None, parameters)

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
            Current state values.
        controls : dict[str, Any]
            Current control variable values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to empty dict).

        Returns
        -------
        dict[str, Any]
            Next-period arrival state values.
        """
        post = self.post_function(
            states,
            controls,
            shocks=shocks,
            parameters=parameters,
            decision_rules=decision_rules,
        )
        return self.select_arrival_states(post)

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
            Current state values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to empty dict).

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
            Current state values.
        controls : dict[str, Any]
            Current control variable values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        agent : str | None, optional
            If specified, only return rewards for this agent.
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to empty dict).

        Returns
        -------
        dict[str, Any]
            Reward values for the period.
        """
        post = self.post_function(
            states,
            controls,
            shocks=shocks,
            parameters=parameters,
            agent=agent,
            decision_rules=decision_rules,
        )
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
        """
        Return the full ex post variables for the period.

        Parameters
        ----------
        states : dict[str, Any]
            Current state values.
        controls : dict[str, Any]
            Current control variable values.
        shocks : dict[str, Any] | None, optional
            Current shock realizations (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        agent : str | None, optional
            Agent identifier (currently unused, reserved for future use).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules (defaults to empty dict).

        Returns
        -------
        dict[str, Any]
            All computed variables from the block transition.
        """
        shocks, decision_rules, parameters = self._resolve_inputs(
            shocks, decision_rules, parameters
        )

        vals = parameters | states | shocks | controls
        post = self.block.transition(vals, decision_rules, fix=list(controls.keys()))
        return post

    def grad_post_function(
        self,
        states: dict[str, Any],
        controls: dict[str, Any],
        wrt: dict[str, torch.Tensor],
        symbols: Iterable[str],
        *,
        shocks: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        decision_rules: dict[str, Callable] | None = None,
        create_graph: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Compute gradients of named ex post variables from one block pass.

        Every symbol in *symbols* is read off the same :meth:`post_function`
        result, so asking for reward and arrival-state gradients together costs
        one pass over the block dynamics.

        Parameters
        ----------
        states : dict[str, Any]
            State values.
        controls : dict[str, Any]
            Control values.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        symbols : Iterable[str]
            Names of the ex post variables to differentiate.
        shocks : dict[str, Any] | None, optional
            Shock values (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules of control variables that will _not_ be given to the function.
        create_graph : bool, optional
            If True, the graph of the derivative is constructed, allowing higher-order
            derivatives and end-to-end training through the gradient computation.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Nested dictionary of gradients for each symbol and variable:
            {symbol: {var_name: gradient}}. The gradient is a zero tensor
            when the symbol does not depend on the variable.
        """
        post = self.post_function(
            states,
            controls,
            shocks=shocks,
            parameters=parameters,
            decision_rules=decision_rules,
        )
        return compute_gradients_for_tensors(
            {sym: post[sym] for sym in symbols}, wrt, create_graph=create_graph
        )

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
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Compute gradients of reward function with respect to specified variables.

        Parameters
        ----------
        states : dict[str, Any]
            State values.
        controls : dict[str, Any]
            Control values.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        shocks : dict[str, Any] | None, optional
            Shock values (defaults to empty dict).
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
        dict[str, dict[str, torch.Tensor]]
            Nested dictionary of gradients for each reward symbol and variable:
            {reward_sym: {var_name: gradient}}. The gradient is a zero tensor
            when the reward does not depend on the variable.
        """
        return self.grad_post_function(
            states,
            controls,
            wrt,
            self.get_reward_syms(agent),
            shocks=shocks,
            parameters=parameters,
            decision_rules=decision_rules,
            create_graph=create_graph,
        )

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
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Compute gradients of transition function with respect to specified variables.

        This computes ∂s_{t+1}/∂x for each arrival state s_{t+1} and each variable x
        specified in wrt. This is needed for Euler equations where the gradient of
        future states with respect to current controls appears (e.g., ∂a_{t+1}/∂c_t = -1
        for the budget constraint a_{t+1} = m_t - c_t).

        Parameters
        ----------
        states : dict[str, Any]
            State values.
        controls : dict[str, Any]
            Control values.
        wrt : dict[str, torch.Tensor]
            Dictionary of variables to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        shocks : dict[str, Any] | None, optional
            Shock values (defaults to empty dict).
        parameters : dict[str, Any] | None, optional
            Model parameters (defaults to instance calibration).
        decision_rules : dict[str, Callable] | None, optional
            Decision rules of control variables that will _not_ be given to the function.
        create_graph : bool, optional
            If True, the graph of the derivative is constructed, allowing higher-order
            derivatives and end-to-end training through the gradient computation.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Nested dictionary of gradients for each arrival state and variable:
            {state_sym: {var_name: gradient}}. The gradient is a zero tensor
            when the arrival state does not depend on the variable.
        """
        return self.grad_post_function(
            states,
            controls,
            wrt,
            self.arrival_states,
            shocks=shocks,
            parameters=parameters,
            decision_rules=decision_rules,
            create_graph=create_graph,
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
    ) -> dict[str, dict[str, torch.Tensor]]:
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
        states : dict[str, Any]
            Arrival state values (with requires_grad=True for gradient computation).
        wrt : dict[str, torch.Tensor]
            Dictionary of arrival states to compute gradients with respect to.
            Keys are variable names, values are tensors with requires_grad=True.
        shocks : dict[str, Any] | None, optional
            Shock values (defaults to empty dict).
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
        dict[str, dict[str, torch.Tensor]]
            Nested dictionary of gradients for each pre-state variable and arrival state:
            {pre_state_var: {state_sym: gradient}}. The gradient is a zero
            tensor when the pre-state variable does not depend on the
            arrival state.
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
            raise ValueError(
                f"Control '{control_sym}' has no pre-state (iset) attribute defined. "
                f"Ensure the Control object in block.dynamics['{control_sym}'] is "
                "constructed with an explicit 'iset' argument specifying which "
                "variables the control depends on."
            )

        pre_state_values = self.compute_pre_state(
            control_sym, states, shocks=shocks, parameters=parameters
        )

        # Use utility function to compute gradients
        return compute_gradients_for_tensors(
            pre_state_values, wrt, create_graph=create_graph
        )

    def select_arrival_states(self, post: dict[str, Any]) -> dict[str, Any]:
        """Return the next-period arrival states from a ``post_function`` result.

        Parameters
        ----------
        post : dict[str, Any]
            The post-transition output returned by :meth:`post_function`.

        Returns
        -------
        dict[str, Any]
            The entries of *post* named by :attr:`arrival_states`.
        """
        return {sym: post[sym] for sym in self.arrival_states}

    def resolve_discount_factor(self, post: dict[str, Any]) -> Any:
        """Return ``post[self.discount_variable]``, raising ``KeyError``
        with a diagnostic message if the discount variable is missing.
        Expects the post-transition output returned by
        :meth:`post_function`.

        Returns ``1.0`` when the period names no discount variable, which is the
        static case: there is no future value stream to discount.
        """
        dv = self.discount_variable
        if dv is None:
            return 1.0
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
        for key in (f"{sym}_0", f"{sym}_1"):
            if key not in shocks:
                raise KeyError(
                    f"Missing shock '{key}' in shocks dict. For models with "
                    f"shocks, provide two independent realizations: '{sym}_0' "
                    f"(period t) and '{sym}_1' (period t+1)."
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
    c = tracked(c)
    return c, {**controls, sym: c}


def _payoff_marginal(
    grads: dict[str, dict[str, torch.Tensor]],
    reward_syms: list[str],
    control_sym: str,
    where: str = "",
) -> torch.Tensor:
    """Sum the reward symbols' marginals in *control_sym*; refuse an all-zero sum.

    The payoff is the SUM of the agent's reward symbols, so its marginal is the
    sum of theirs. The test is on that sum, since a decomposed utility may have
    a part that does not depend on the control.
    """
    marginal = sum(grads[sym][control_sym] for sym in reward_syms)
    if not torch.any(marginal != 0):
        raise ValueError(
            f"Marginal reward{where} is zero at every sample point: the payoff "
            f"{reward_syms} is structurally independent of control "
            f"'{control_sym}', or its gradient vanishes on the whole batch"
        )
    return marginal


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
        Calibration parameters (defaults to the period's calibration).
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

        # One block pass per period: the ex post values carry the period's
        # rewards, its discount factor and the next arrival states alike.
        post = bellman_period.post_function(
            states_t, controls_t, shocks=shocks_t, parameters=parameters, agent=agent
        )
        discount_factor = bellman_period.resolve_discount_factor(post)
        reward_t = {rsym: post[rsym] for rsym in reward_syms}

        period_reward = 0
        for rsym in reward_syms:
            if any_nan(reward_t[rsym]):
                raise ValueError(f"Calculated reward {rsym} is NaN: {reward_t}")
            period_reward += reward_t[rsym]

        total_discounted_reward += period_reward * cumulative_discount
        cumulative_discount = cumulative_discount * discount_factor

        states_t = bellman_period.select_arrival_states(post)

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
        Current arrival state values.
    shocks : dict[str, Any]
        Shock realizations for both periods:
        - {shock_sym}_0: period t shocks (for immediate reward and transitions)
        - {shock_sym}_1: period t+1 shocks (for continuation value evaluation)
    parameters : dict[str, Any] | None, optional
        Model parameters for calibration (defaults to the period's calibration).
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
    Multi-control: this function returns a single residual tensor, evaluated
    against the sum of the reward variables ``agent`` owns.
    For multi-control models it complements
    :func:`estimate_euler_residual` (which returns one residual per
    control) and :func:`estimate_bellman_foc_residual` (which returns one
    FOC residual per control by differentiating the value callable).
    """
    shocks_t, shocks_t_plus_1 = _extract_period_shocks(bellman_period, shocks)

    reward_syms = bellman_period.get_reward_syms(agent)

    # V(s_t), the value at the period-t arrival state
    current_values = bellman_period.compute_value(
        vf, states_t, shocks=shocks_t, parameters=parameters, agent=agent
    )

    # Controls from decision callable (also takes arrival states)
    controls_t = bellman_period.compute_controls(
        df, states_t, shocks=shocks_t, parameters=parameters
    )

    # One block pass: the ex post values carry the immediate reward, the
    # discount factor and the next-period arrival state alike.
    post = bellman_period.post_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )
    immediate_reward = sum(post[sym] for sym in reward_syms)
    discount_factor = bellman_period.resolve_discount_factor(post)
    next_states = bellman_period.select_arrival_states(post)

    # V(s_{t+1}), the continuation value at the next-period arrival state,
    # using the second independent shock draw
    continuation_values = bellman_period.compute_value(
        vf,
        next_states,
        shocks=shocks_t_plus_1,
        parameters=parameters,
        agent=agent,
    )

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    if not torch.isfinite(bellman_residual).all():
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
    transition_gradients: dict[str, torch.Tensor],
    pre_state_gradients: dict[str, dict[str, torch.Tensor]],
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
    if not torch.isfinite(like).all():
        raise ValueError(
            f"Euler residual: marginal_reward_t1 contains NaN or Inf for "
            f"control '{control_sym}'. Cannot compute chain-rule return factor."
        )

    total = torch.zeros_like(like)

    for state_sym in bellman_period.arrival_states:
        trans_grad = transition_gradients[state_sym]
        if not torch.any(trans_grad != 0):
            # An all-zero transition gradient means the arrival state is
            # independent of the control; skipping the term also avoids
            # 0 * Inf = NaN from an ill-conditioned pre-state factor.
            continue
        for state_grads in pre_state_gradients.values():
            total = total + state_grads[state_sym] * trans_grad

    if not torch.isfinite(total).all():
        raise ValueError(
            f"Euler residual: return_factor_sum contains NaN or Inf for "
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
    reward_syms: list[str],
    states_t: dict[str, Any],
    controls_t: dict[str, Any],
    states_t_plus_1: dict[str, Any],
    controls_t_plus_1: dict[str, Any],
    shocks_t: dict[str, Any],
    shocks_t_plus_1: dict[str, Any],
    parameters: dict[str, Any] | None,
) -> torch.Tensor:
    """Compute the Euler residual for a single control variable.

    This is the inner workhorse called once per control by
    ``estimate_euler_residual``.  Factored out to support multi-control models.
    """
    c_t, controls_t_grad = _ensure_grad(controls_t, control_sym)
    c_t1, controls_t1_grad = _ensure_grad(controls_t_plus_1, control_sym)

    # ∂u/∂c and ∂s_{t+1}/∂c at period t, read off one block pass
    grads_t = bellman_period.grad_post_function(
        states_t,
        controls_t_grad,
        {control_sym: c_t},
        [*reward_syms, *bellman_period.arrival_states],
        shocks=shocks_t,
        parameters=parameters,
        create_graph=True,
    )
    marginal_reward_t = _payoff_marginal(
        grads_t, reward_syms, control_sym, " at period t"
    )

    # ∂u/∂c at period t+1
    grads_t1 = bellman_period.grad_post_function(
        states_t_plus_1,
        controls_t1_grad,
        {control_sym: c_t1},
        reward_syms,
        shocks=shocks_t_plus_1,
        parameters=parameters,
        create_graph=True,
    )
    marginal_reward_t1 = _payoff_marginal(
        grads_t1, reward_syms, control_sym, " at period t+1"
    )

    transition_gradients = {
        state_sym: grads_t[state_sym][control_sym]
        for state_sym in bellman_period.arrival_states
    }

    # Pre-state gradients: ∂m'/∂s' (envelope condition).
    states_t1_grad = {sym: tracked(s) for sym, s in states_t_plus_1.items()}

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
        Current state values (arrival states).
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

    reward_syms = bellman_period.get_reward_syms(agent)

    # Period-t controls and transition
    if controls_t is None:
        controls_t = bellman_period.compute_controls(
            df, states_t, shocks=shocks_t, parameters=parameters
        )
    # One block pass: the ex post values carry the next-period arrival states
    # and the discount factor alike.
    post = bellman_period.post_function(
        states_t, controls_t, shocks=shocks_t, parameters=parameters
    )
    states_t_plus_1 = bellman_period.select_arrival_states(post)

    # Period-(t+1) controls (second independent shock draw, AiO)
    controls_t_plus_1 = bellman_period.compute_controls(
        df, states_t_plus_1, shocks=shocks_t_plus_1, parameters=parameters
    )

    control_syms = list(controls_t)
    if len(control_syms) == 0:
        raise ValueError("No control variables found in decision function")

    discount_factor = bellman_period.resolve_discount_factor(post)

    # Compute Euler residual for each control
    residuals = {}
    for control_sym in control_syms:
        residuals[control_sym] = _euler_residual_single_control(
            bellman_period,
            discount_factor,
            control_sym,
            reward_syms,
            states_t,
            controls_t,
            states_t_plus_1,
            controls_t_plus_1,
            shocks_t,
            shocks_t_plus_1,
            parameters,
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
        Current arrival state values.
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
    reward_syms = bellman_period.get_reward_syms(agent)

    controls_t = bellman_period.compute_controls(
        df, states_t, shocks=shocks_t, parameters=parameters
    )
    residuals = {}
    for control_sym in controls_t:
        c_t, controls_t_grad = _ensure_grad(controls_t, control_sym)

        # One pass at the grad-tracking control yields u'(c_t), the discount
        # factor, and the next-period arrival states, whose graph still runs
        # through c_t.
        post_grad = bellman_period.post_function(
            states_t, controls_t_grad, shocks=shocks_t, parameters=parameters
        )
        discount_factor = bellman_period.resolve_discount_factor(post_grad)
        reward_grads = compute_gradients_for_tensors(
            {sym: post_grad[sym] for sym in reward_syms},
            {control_sym: c_t},
            create_graph=True,
        )
        mr_t = _payoff_marginal(reward_grads, reward_syms, control_sym)

        # V(s', ε₁), the continuation value with the second independent shock
        # draw, evaluated on next-period arrival states
        v_next = bellman_period.compute_value(
            vf,
            bellman_period.select_arrival_states(post_grad),
            shocks=shocks_t_plus_1,
            parameters=parameters,
            agent=agent,
        )

        # ∂V/∂c via autograd chain rule: ∂V/∂s' * ∂s'/∂c. A control that V(s')
        # does not depend on, as in a multi-control model, gets a zero.
        dv_dc = compute_gradients_for_tensors(
            {"v": v_next}, {control_sym: c_t}, create_graph=True
        )["v"][control_sym]
        if not torch.isfinite(dv_dc).all():
            raise ValueError(
                f"Autograd gradient dV/d{control_sym} contains NaN or Inf. "
                "Check that vf is properly initialized and numerically stable."
            )

        # FOC: u'(c) + β * ∂V(s',ε')/∂c = 0
        residuals[control_sym] = mr_t + discount_factor * dv_dc

    return residuals
