from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch

from skagent.bellman import (
    BellmanPeriod,
    _extract_period_shocks,
    estimate_bellman_foc_residual,
    estimate_bellman_residual,
    estimate_discounted_lifetime_reward,
    estimate_euler_residual,
)
from skagent.grid import Grid
from skagent.utils import any_nan, fischer_burmeister, reconcile

logger = logging.getLogger(__name__)


def static_reward(
    bellman_period,
    dr,
    states,
    shocks=None,
    parameters=None,
    agent=None,
):
    """
    Returns the reward for an agent for a block, given a decision rule, states, shocks, and calibration.

    The reward is the SUM of the reward symbols in scope: an agent owning
    several reward variables has a payoff of all of them, not of one.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The Bellman period object containing the model.
    dr : dict or callable
        Decision rules (dict of functions), or a decision function.
    states : dict
        Initial states, symbols to values.
    shocks : dict, optional
        Shock variable values.
    parameters : dict, optional
        Calibration parameters (defaults to the period's calibration).
    agent : str or None, optional
        Name of reference agent for rewards. When omitted, every reward symbol
        in the block is summed, which is one agent's payoff only if the block
        has one agent; on a block whose utilities have several owners it is the
        sum of their payoffs and no agent's objective.

    Raises
    ------
    ValueError
        If any reward symbol in scope is NaN.
    """
    controls = bellman_period.compute_controls(
        dr, states, shocks=shocks, parameters=parameters
    )

    reward = bellman_period.reward_function(
        states,
        controls,
        shocks=shocks,
        parameters=parameters,
        agent=agent,
        decision_rules=dr,
    )

    total_reward = 0
    for rsym, value in reward.items():
        if any_nan(value):
            raise ValueError(f"Calculated reward {rsym} is NaN: {reward}")
        total_reward = total_reward + value

    return total_reward


def _prepare_loss_inputs(
    model_obj,
    input_grid: Grid,
    state_variables: set[str],
    other_dr: dict,
    new_dr: dict,
) -> tuple[dict, dict, dict]:
    """Extract states, shocks, and merged decision rules from an input grid.

    *new_dr* takes precedence over *other_dr* for any overlapping keys.
    """
    given_vals = input_grid.to_dict()
    shock_vals = {sym: given_vals[sym] for sym in model_obj.get_shocks()}
    states = {sym: given_vals[sym] for sym in state_variables}
    fresh_dr = {**other_dr, **new_dr}
    return states, shock_vals, fresh_dr


class CustomLoss:
    """
    A custom loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)

    Parameters
    ----------
    loss_function : callable
        ``loss_function(bellman_period, dr, states, shocks=, parameters=,
        agent=)`` returning a per-sample reward, whose negative is the loss.
    bellman_period : BellmanPeriod
        The period being trained. Its calibration is what the loss is evaluated
        at.
    agent : str, optional
        Whose payoff to maximize. See :class:`StaticRewardLoss`.
    other_dr : dict of callable, optional
        Decision rules for the controls this loss is not training, held fixed.
    """

    def __init__(self, loss_function, bellman_period, *, agent=None, other_dr=None):
        self.bellman_period = bellman_period
        self.parameters = bellman_period.calibration
        self.arrival_variables = bellman_period.arrival_states
        self.other_dr = other_dr if other_dr is not None else {}
        self.loss_function = loss_function
        self.agent = agent

    def __call__(self, dr, input_grid: Grid):
        """*dr* maps each control symbol to a decision RULE -- a function over
        that control's information set -- and is merged over *other_dr*. A
        decision FUNCTION, which takes the arrival states, shocks and
        calibration in total, is not accepted here: there is nothing to merge
        one into.
        """
        states, shock_vals, fresh_dr = _prepare_loss_inputs(
            self.bellman_period,
            input_grid,
            self.arrival_variables,
            self.other_dr,
            dr,
        )

        neg_loss = self.loss_function(
            self.bellman_period,
            fresh_dr,
            states,
            parameters=self.parameters,
            shocks=shock_vals,
            agent=self.agent,
        )
        return -neg_loss


class StaticRewardLoss(CustomLoss):
    """
    A loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The period whose reward is maximized.
    other_dr : dict of callable, optional
        Decision rules for the controls this loss is not training, held fixed.
    agent : str, optional
        Whose payoff to maximize: the sum of the reward symbols that agent
        owns. Required on a block whose utilities have more than one owner,
        since without it the loss maximizes the sum of every agent's reward --
        a planner's objective, and no player's. Use
        :meth:`skagent.block.Block.deciding_agent` to read it off the control
        being trained.
    """

    def __init__(self, bellman_period, *, agent=None, other_dr=None):
        super().__init__(static_reward, bellman_period, agent=agent, other_dr=other_dr)


class EstimatedDiscountedLifetimeRewardLoss:
    """
    A loss function for a Block that computes the discounted lifetime reward for T time periods.

    Parameters
    -----------

    bellman_period : BellmanPeriod
        The period being trained. Its calibration is what the loss is evaluated
        at.
    big_t : int
        The number of time steps to compute reward for.
    agent : str, optional
        Whose payoff to maximize: the sum of the reward symbols that agent owns,
        discounted over *big_t* periods. Required on a block whose utilities
        have more than one owner, since without it the loss maximizes the sum of
        every agent's reward -- a planner's objective, and no player's.
    """

    def __init__(self, bellman_period, *, big_t, agent=None):
        self.bellman_period = bellman_period
        self.parameters = bellman_period.calibration
        self.arrival_variables = self.bellman_period.arrival_states
        self.big_t = big_t
        self.agent = agent

    def __call__(self, df: Callable, input_grid: Grid):
        # TODO: codify this encoding and decoding of the grid into a separate object
        # It is specifically the EDLR loss function that requires big_t of the shocks.
        # other AiO loss functions use 2 copies of the shocks only.

        # includes the values of state_0 variables, and shocks {sym}_{t}.
        given_vals = input_grid.to_dict()

        shocks_by_t = {
            sym: torch.stack([given_vals[f"{sym}_{t}"] for t in range(self.big_t)])
            for sym in self.bellman_period.get_shocks()
        }

        edlr = estimate_discounted_lifetime_reward(
            self.bellman_period,
            df,
            {sym: given_vals[sym] for sym in self.arrival_variables},
            self.big_t,
            parameters=self.parameters,
            agent=self.agent,
            shocks_by_t=shocks_by_t,
        )
        return -edlr


class _EquationLossBase(ABC):
    """
    Private base class for Bellman and Euler equation losses.

    Stores shared setup (bellman_period, arrival_variables, shock_syms, reward
    validation) and provides ``_extract_states_and_shocks`` to avoid duplicate
    grid-extraction logic in subclass ``__call__`` methods.
    """

    def __init__(
        self,
        bellman_period: BellmanPeriod,
        *,
        agent: str | None = None,
    ) -> None:
        if not isinstance(bellman_period, BellmanPeriod):
            raise TypeError(
                f"bellman_period must be a BellmanPeriod, "
                f"got {type(bellman_period).__name__}"
            )
        self.bellman_period = bellman_period
        self.parameters = bellman_period.calibration
        # Defensive copy to prevent external mutation of arrival_states
        self.arrival_variables: set[str] = set(bellman_period.arrival_states)

        shock_vars = self.bellman_period.get_shocks()
        self.shock_syms: list[str] = list(shock_vars.keys())

        self.agent: str | None = agent

        # Validate that reward variables exist (raises ValueError with agent context)
        bellman_period.get_reward_syms(agent)

    @abstractmethod
    def __call__(self, df: Callable, input_grid: Grid) -> torch.Tensor: ...

    def _redraw_next_shocks(self, states_t: dict, shocks: dict) -> dict:
        """Return *shocks* with an independent second draw of the ``{sym}_1`` keys.

        The input grid supplies the first next-period draw; the all-in-one
        operator (MMW 2021, Def. 2.7) multiplies residuals at two independent
        ones. For a deterministic model there is nothing to draw, and the two
        coincide. The batch comes from the states, or from the grid's shocks
        when the period has no arrival states; an aggregate shock draws one
        value, which the whole batch shares.
        """
        template = next(iter({**states_t, **shocks}.values()), None)
        if template is None:
            return shocks
        n = template.shape[0]
        draws = self.bellman_period.draw_shocks(n)
        return shocks | {
            f"{s}_1": reconcile(template, np.full(n, v) if np.ndim(v) == 0 else v)
            for s, v in draws.items()
        }

    def _extract_states_and_shocks(
        self, input_grid: Grid
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract states and two-period shocks from *input_grid*.

        Returns
        -------
        states_t : dict
            Arrival state variables extracted from the grid.
        shocks : dict
            Combined shock dict with keys ``{sym}_0`` and ``{sym}_1``.
        """
        given_vals = input_grid.to_dict()
        return self._arrival_states_from(given_vals), self._shocks_from(given_vals)

    def _arrival_states_from(self, given_vals: dict[str, Any]) -> dict[str, Any]:
        """Select the arrival states from *given_vals*, raising if any is missing."""
        missing_states = [
            sym for sym in self.arrival_variables if sym not in given_vals
        ]
        if missing_states:
            raise KeyError(
                f"Missing arrival state variable(s) {missing_states} in input_grid. "
                f"Expected: {sorted(self.arrival_variables)}."
            )
        return {sym: given_vals[sym] for sym in self.arrival_variables}

    def _shocks_from(self, given_vals: dict[str, Any]) -> dict[str, Any]:
        """Select both realizations of every shock, keyed ``{sym}_0``/``{sym}_1``."""
        shock_keys = [f"{sym}_{i}" for sym in self.shock_syms for i in (0, 1)]
        missing_shocks = [k for k in shock_keys if k not in given_vals]
        if missing_shocks:
            raise KeyError(
                f"Missing shock variable(s) {missing_shocks} in input_grid. "
                f"Expected two independent realizations per shock: "
                f"{shock_keys}."
            )
        return {k: given_vals[k] for k in shock_keys}


class BellmanEquationLoss(_EquationLossBase):
    """
    Creates a Bellman equation loss function for the Maliar method.

    The Bellman equation is: V(s) = max_c { u(s,c,ε) + β E_ε'[V(s')] }
    where s' = f(s,c,ε) is the next state given current state s, control c, and shock ε,
    and the expectation E_ε' is taken over future shock realizations ε'.

    This function expects the input grid to contain two independent shock realizations:
    - {shock_sym}_0: shocks for period t (used for immediate reward and transitions)
    - {shock_sym}_1: shocks for period t+1 (used for continuation value evaluation)

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model block containing dynamics, rewards, and shocks
    value_function : callable
        A value function that takes state variables and returns value estimates
    agent : str, optional
        Agent identifier for rewards
    foc_weight : float, optional
        Weight on the squared first-order-condition residual added to the loss
        (default 0.0, which leaves the FOC term out).
    """

    def __init__(
        self,
        bellman_period: BellmanPeriod,
        *,
        value_function: dict[str, Callable] | Callable,
        agent: str | None = None,
        foc_weight: float = 0.0,
    ) -> None:
        super().__init__(bellman_period, agent=agent)
        if not callable(value_function) and not isinstance(value_function, dict):
            raise TypeError(
                "value_function must be a callable or a dict mapping agent "
                f"name to a callable, got {type(value_function).__name__}"
            )
        if foc_weight < 0:
            raise ValueError(f"foc_weight must be >= 0, got {foc_weight}")
        self.value_function = value_function
        self.foc_weight = foc_weight

    def __call__(self, df: Callable, input_grid: Grid) -> torch.Tensor:
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
            All-in-one estimate of the squared expected Bellman residual, plus
            ``foc_weight`` times that of the FOC residuals: each is the product
            of the residuals at the grid's next-period draw and at a second,
            independent one (MMW 2021, Definition 2.10, eq. 15). Squaring one
            draw would add the variance of the continuation value to the
            objective.
        """
        states_t, shocks = self._extract_states_and_shocks(input_grid)
        draws = (shocks, self._redraw_next_shocks(states_t, shocks))
        args = (self.bellman_period, self.value_function, df, states_t)

        res_a, res_b = (
            estimate_bellman_residual(*args, s, self.parameters, self.agent)
            for s in draws
        )
        loss = res_a * res_b

        if self.foc_weight > 0:
            foc_a, foc_b = (
                estimate_bellman_foc_residual(*args, s, self.parameters, self.agent)
                for s in draws
            )
            foc_loss = sum((foc_a[c] * foc_b[c] for c in foc_a), 0.0)
            loss = loss + self.foc_weight * foc_loss

        return loss


def _complementarity_residual(f, slack_lower, slack_upper):
    r"""Smooth complementarity residual for a control's Euler residual ``f``.

    Combines whichever bounds are present using the sign convention that the
    residual ``f`` is :math:`\geq 0` when an upper bound binds and
    :math:`\leq 0` when a lower bound binds:

    - upper only: :math:`\text{FB}(f, s_u)`
    - lower only: :math:`\text{FB}(-f, s_l)`
    - both: :math:`\text{FB}(s_u, -\text{FB}(s_l, -f))`, a two-sided form that
      reduces to either one-sided residual when the opposite bound is slack
      (so a control with a non-binding floor and a binding ceiling matches the
      upper-only residual, leaving upper-bound benchmarks unchanged).

    Parameters
    ----------
    f : torch.Tensor
        The Euler equation residual for one control.
    slack_lower, slack_upper : torch.Tensor or None
        Lower slack ``x - lb`` and upper slack ``ub - x``; ``None`` when the
        corresponding bound is absent.

    Returns
    -------
    torch.Tensor
        The complementarity residual. A control with neither bound gets the
        one-sided fallback ``relu(-f)``, which penalizes only violations of
        ``f >= 0``.
    """
    if slack_upper is not None and slack_lower is not None:
        inner = fischer_burmeister(slack_lower, -f)
        return fischer_burmeister(slack_upper, -inner)
    if slack_upper is not None:
        return fischer_burmeister(f, slack_upper)
    if slack_lower is not None:
        return fischer_burmeister(-f, slack_lower)
    return torch.relu(-f)


def _call_bound(bound: Callable, param_names: list[str], pre_state: dict) -> Any:
    """Evaluate a control bound on the pre-state variables it names."""
    return bound(**{k: pre_state[k] for k in param_names if k in pre_state})


class EulerEquationLoss(_EquationLossBase):
    """
    Creates an Euler equation loss function for the Maliar method.

    The Euler equation is the first-order condition from the Bellman equation,
    relating marginal rewards across periods. For a DSOP with control :math:`x_t`,
    arrival states :math:`s_t`, and pre-decision states :math:`m_t`, this loss
    function computes the Euler equation **residual**:

    .. math::

        f = u'(x_t) + \\beta \\cdot u'(x_{t+1}) \\cdot \\sum_s \\left[
            \\frac{\\partial s_{t+1}}{\\partial x_t} \\cdot \\frac{\\partial m'}{\\partial s_{t+1}}
        \\right]

    where :math:`f` is the residual that equals zero at optimality, :math:`s_{t+1}` is
    the next-period arrival state, and :math:`m'` is the pre-decision state.

    The discount factor :math:`\\beta` is obtained from the ``BellmanPeriod`` via
    ``bellman_period.discount_variable``, so it adapts to the model's calibration.

    **Multi-control support:**

    For models with :math:`J` control variables, a separate Euler residual is
    computed per control.  The loss sums over all controls:
    :math:`L = \\sum_j w \\cdot f_j^2`.

    **Handling Inequality Constraints (Fischer-Burmeister):**

    When ``constrained=True``, a control's declared bounds are turned into a
    smooth complementarity residual via the Fischer-Burmeister function
    (Maliar et al. 2021, equation 25), :math:`\\text{FB}(a, b) = a + b -
    \\sqrt{a^2 + b^2 + \\varepsilon}`, which is zero (up to the regularizer
    :math:`\\sqrt{\\varepsilon}`) exactly where :math:`a \\geq 0`,
    :math:`b \\geq 0`, :math:`a \\cdot b = 0`.

    The sign convention is that the Euler residual :math:`f` is :math:`\\geq 0`
    when an upper bound binds and :math:`\\leq 0` when a lower bound binds.
    Writing :math:`s_u = \\overline{x} - x` and :math:`s_l = x - \\underline{x}`
    for the upper and lower slacks, the per-control residual is

    - upper bound only: :math:`\\text{FB}(f, s_u)`;
    - lower bound only: :math:`\\text{FB}(-f, s_l)`;
    - both bounds: :math:`\\text{FB}(s_u, -\\text{FB}(s_l, -f))`, a two-sided
      form that reduces to either one-sided residual when the opposite bound is
      slack, so a control with a non-binding floor and a binding ceiling matches
      the upper-only residual;
    - no bound: the one-sided fallback :math:`\\text{relu}(-f)`, penalizing only
      violations of :math:`f \\geq 0`.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model block containing dynamics, rewards, and shocks.
    agent : str, optional
        Agent identifier for rewards.
    weight : float, optional
        Exogenous weight for combining multiple optimality conditions (default: 1.0).
        This corresponds to the vector :math:`v` in equation (12) of the paper.
    constrained : bool, optional
        If True, turn each control's declared bounds, lower, upper or both,
        into the Fischer-Burmeister residual above (default: False).

    Examples
    --------
    >>> bp = BellmanPeriod(block, "beta", calibration={"R": 1.04, "beta": 0.95})
    >>> loss_fn = EulerEquationLoss(bp)
    """

    def __init__(
        self,
        bellman_period: BellmanPeriod,
        *,
        agent: str | None = None,
        weight: float = 1.0,
        constrained: bool = False,
    ) -> None:
        super().__init__(bellman_period, agent=agent)

        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self.weight = weight
        self.constrained = constrained
        # Cache bound parameter names so the slack helpers do not call
        # inspect.signature on every forward pass.
        self._upper_bound_params: dict[str, list[str]] = {}
        self._lower_bound_params: dict[str, list[str]] = {}
        if self.constrained:
            for sym, control in bellman_period.get_controls().items():
                if control.upper_bound is not None:
                    self._upper_bound_params[sym] = list(
                        inspect.signature(control.upper_bound).parameters
                    )
                if control.lower_bound is not None:
                    self._lower_bound_params[sym] = list(
                        inspect.signature(control.lower_bound).parameters
                    )
            if not self._upper_bound_params and not self._lower_bound_params:
                logger.warning(
                    "constrained=True but no Control in the block has a "
                    "lower_bound or upper_bound. The loss will fall back to "
                    "the one-sided residual relu(-f), combined across the two "
                    "independent shock draws as the all-in-one product "
                    "relu(-f_a) * relu(-f_b), which penalizes only violations "
                    "of f >= 0. Define a bound on Control objects to enable "
                    "the Fischer-Burmeister formulation."
                )

    def _slacks(
        self, control_sym: str, controls_t: dict, states_t: dict, shocks_t: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the lower and upper slacks ``x - lb`` and ``ub - x``.

        Either is ``None`` when the control has no bound on that side. Both
        bounds read one pre-decision state, computed once.
        """
        lower_params = self._lower_bound_params.get(control_sym)
        upper_params = self._upper_bound_params.get(control_sym)
        if lower_params is None and upper_params is None:
            return None, None

        control_obj = self.bellman_period.block.dynamics[control_sym]
        pre_state = self.bellman_period.compute_pre_state(
            control_sym, states_t, shocks=shocks_t, parameters=self.parameters
        )
        x = controls_t[control_sym]
        slack_lower = None
        if lower_params is not None:
            slack_lower = x - _call_bound(
                control_obj.lower_bound, lower_params, pre_state
            )
        slack_upper = None
        if upper_params is not None:
            slack_upper = (
                _call_bound(control_obj.upper_bound, upper_params, pre_state) - x
            )
        return slack_lower, slack_upper

    def _aio_residual_pair(self, df: Callable, states_t: dict, shocks: dict):
        """Two Euler residuals sharing the current control, at two independent
        next-period shock draws (MMW JME'21 all-in-one operator, Def. 2.7).

        The squared expected residual is estimated by the *product* of two
        residuals evaluated at independent draws of the next-period shock,
        which is unbiased: ``E[f_a f_b] = (E[f])**2``. Squaring a single
        residual would instead add ``Var(f) >= 0``, biasing the solution of any
        stochastic model. For deterministic models the two draws coincide and
        the product reduces to ``f**2``.

        Returns ``(res_a, res_b, controls_t, shocks_t)``.
        """
        # The input grid's draws are the first factor's shocks as given.
        shocks_t, _ = _extract_period_shocks(self.bellman_period, shocks)
        # Current control: computed once and shared by both factors so the
        # all-in-one product cancels the cross terms to (E[f])**2.
        controls_t = self.bellman_period.compute_controls(
            df, states_t, shocks=shocks_t, parameters=self.parameters
        )
        shocks_b = self._redraw_next_shocks(states_t, shocks)
        res_a = estimate_euler_residual(
            self.bellman_period,
            df,
            states_t,
            shocks,
            self.parameters,
            self.agent,
            controls_t=controls_t,
        )
        res_b = estimate_euler_residual(
            self.bellman_period,
            df,
            states_t,
            shocks_b,
            self.parameters,
            self.agent,
            controls_t=controls_t,
        )
        return res_a, res_b, controls_t, shocks_t

    def __call__(self, df: Callable, input_grid: Grid) -> torch.Tensor:
        """
        Euler equation loss function using the AiO expectation operator.

        Parameters
        ----------
        df : callable
            Decision function from policy network.
            Signature: df(states_t, shocks_t, parameters) -> controls_t
        input_grid : Grid
            Grid containing current states and two independent shock realizations:
            - {shock_sym}_0: shocks for transitions t → t+1
            - {shock_sym}_1: shocks for transitions t+1 → t+2 (independent)

        Returns
        -------
        torch.Tensor
            Weighted all-in-one estimate of the squared expected Euler
            residual, summed over controls.

        Notes
        -----
        Each residual f is computed with ε₀ for transitions from t to t+1 and
        an independent ε₁ for transitions from t+1 to t+2 (Maliar et al. 2021,
        Definition 2.7). The loss is the product of two such residuals that
        share ε₀ and draw ε₁ independently, L = f(ε₀, ε₁ᵃ) f(ε₀, ε₁ᵇ), whose
        expectation is the squared expected residual.
        """
        states_t, shocks = self._extract_states_and_shocks(input_grid)

        # All-in-one operator: form the product of two residuals at independent
        # next-period draws, never the square of a single draw (MMW eq. 12).
        res_a, res_b, controls_t, shocks_t = self._aio_residual_pair(
            df, states_t, shocks
        )

        if self.constrained:
            total = 0.0
            for ctrl_sym in res_a:
                slacks = self._slacks(ctrl_sym, controls_t, states_t, shocks_t)
                total = total + _complementarity_residual(
                    res_a[ctrl_sym], *slacks
                ) * _complementarity_residual(res_b[ctrl_sym], *slacks)
            return self.weight * total

        # Unconstrained loss: mean of the product estimates (E[f])**2.
        return self.weight * sum((res_a[c] * res_b[c] for c in res_a), 0.0)
