from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from skagent.bellman import (
    _extract_period_shocks,
    estimate_bellman_foc_residual,
    estimate_bellman_residual,
    estimate_discounted_lifetime_reward,
    estimate_euler_residual,
)
from skagent.grid import Grid
from skagent.utils import fischer_burmeister

if TYPE_CHECKING:
    from skagent.bellman import BellmanPeriod

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
        Calibration parameters.
    agent : str or None, optional
        Name of reference agent for rewards.
    """
    if shocks is None:
        shocks = {}
    if parameters is None:
        parameters = {}
    if callable(dr):
        controls = dr(states, shocks, parameters)
    else:
        controls = bellman_period.decision_function(
            states, shocks=shocks, parameters=parameters, decision_rules=dr
        )

    rsym = bellman_period.get_reward_sym(agent)

    reward = bellman_period.reward_function(
        states,
        controls,
        shocks=shocks,
        parameters=parameters,
        agent=agent,
        decision_rules=dr,
    )

    if isinstance(reward[rsym], torch.Tensor) and torch.any(torch.isnan(reward[rsym])):
        raise ValueError(f"Calculated reward {rsym} is NaN: {reward}")
    if isinstance(reward[rsym], np.ndarray) and np.any(np.isnan(reward[rsym])):
        raise ValueError(f"Calculated reward {rsym} is NaN: {reward}")

    return reward[rsym]


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
    shock_vals = {sym: input_grid[sym] for sym in model_obj.get_shocks()}
    states = {sym: given_vals[sym] for sym in state_variables}
    fresh_dr = {**other_dr, **new_dr}
    return states, shock_vals, fresh_dr


class CustomLoss:
    """
    A custom loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)

    TODO: leaving this as ambiguously about Blocks and BellmanPeriods for now
    """

    def __init__(self, loss_function, block, parameters=None, other_dr=None):
        self.block = block
        self.parameters = parameters
        self.arrival_variables = self.block.arrival_states
        self.other_dr = other_dr if other_dr is not None else {}
        self.loss_function = loss_function

    def __call__(self, new_dr, input_grid: Grid):
        """
        new_dr : dict of callable
        """
        states, shock_vals, fresh_dr = _prepare_loss_inputs(
            self.block, input_grid, self.arrival_variables, self.other_dr, new_dr
        )

        neg_loss = self.loss_function(
            self.block,
            fresh_dr,
            states,
            parameters=self.parameters,
            shocks=shock_vals,
        )
        return -neg_loss


class StaticRewardLoss:
    """
    A loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)
    """

    def __init__(self, bellman_period, parameters, other_dr=None):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.arrival_variables = self.bellman_period.arrival_states
        self.other_dr = other_dr if other_dr is not None else {}

    def __call__(self, new_dr, input_grid: Grid):
        """
        new_dr : dict of callable
        """
        states, shock_vals, fresh_dr = _prepare_loss_inputs(
            self.bellman_period,
            input_grid,
            self.arrival_variables,
            self.other_dr,
            new_dr,
        )

        r = static_reward(
            self.bellman_period,
            fresh_dr,
            states,
            parameters=self.parameters,
            agent=None,  ## TODO: Pass through the agent?
            shocks=shock_vals,
        )
        return -r


class EstimatedDiscountedLifetimeRewardLoss:
    """
    A loss function for a Block that computes the discounted lifetime reward for T time periods.

    Parameters
    -----------

    bellman_period
    big_t: int
        The number of time steps to compute reward for
    parameters
    """

    def __init__(self, bellman_period, big_t, parameters):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.arrival_variables = self.bellman_period.arrival_states
        self.big_t = big_t

    def __call__(self, df: Callable, input_grid: Grid):
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

        edlr = estimate_discounted_lifetime_reward(
            self.bellman_period,
            df,
            {sym: given_vals[sym] for sym in self.arrival_variables},
            self.big_t,
            parameters=self.parameters,
            agent=None,  # TODO: Pass through the agent?
            shocks_by_t=shocks_by_t,
            # Handle multiple decision rules?
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
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        from skagent.bellman import BellmanPeriod as _BellmanPeriod

        if not isinstance(bellman_period, _BellmanPeriod):
            raise TypeError(
                f"bellman_period must be a BellmanPeriod, "
                f"got {type(bellman_period).__name__}"
            )
        self.bellman_period = bellman_period
        self.parameters = parameters
        # Defensive copy to prevent external mutation of arrival_states
        self.arrival_variables: set[str] = set(bellman_period.arrival_states)

        shock_vars = self.bellman_period.get_shocks()
        self.shock_syms: list[str] = list(shock_vars.keys())

        self.agent: str | None = agent

        # Validate that reward variables exist (raises ValueError with agent context)
        bellman_period.get_reward_sym(agent)

    @abstractmethod
    def __call__(self, df: Callable, input_grid: Grid) -> torch.Tensor: ...

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

        missing_states = [
            sym for sym in self.arrival_variables if sym not in given_vals
        ]
        if missing_states:
            raise KeyError(
                f"Missing arrival state variable(s) {missing_states} in input_grid. "
                f"Expected: {sorted(self.arrival_variables)}."
            )
        states_t = {sym: given_vals[sym] for sym in self.arrival_variables}

        # Build the combined shock dict — let _extract_period_shocks validate
        # and produce informative error messages for missing keys
        shock_keys = [f"{sym}_{i}" for sym in self.shock_syms for i in (0, 1)]
        missing_shocks = [k for k in shock_keys if k not in given_vals]
        if missing_shocks:
            raise KeyError(
                f"Missing shock variable(s) {missing_shocks} in input_grid. "
                f"Expected two independent realizations per shock: "
                f"{shock_keys}."
            )
        shocks = {k: given_vals[k] for k in shock_keys}

        return states_t, shocks


class BellmanEquationLoss(_EquationLossBase):
    """
    Creates a Bellman equation loss function for the Maliar method.

    The Bellman equation is: V(s) = max_c { u(s,c,ε) + β E_ε'[V(s')] }
    where s' = f(s,c,ε) is the next state given current state s, control c, and shock ε,
    and the expectation E_ε' is taken over future shock realizations ε'.

    When ``foc_weight > 0``, the loss includes a first-order condition (FOC) term
    derived from the Bellman equation (equation 14 of Maliar et al. 2021):

    .. math::

        L = f_{\\text{Bellman}}^2 + v \\cdot f_{\\text{FOC}}^2

    where :math:`f_{\\text{FOC}}` is the FOC residual
    :math:`u'(c) + \\beta \\, \\partial V(s')/\\partial c = 0`.

    This function expects the input grid to contain two independent shock realizations:
    - {shock_sym}_0: shocks for period t (used for immediate reward and transitions)
    - {shock_sym}_1: shocks for period t+1 (used for continuation value evaluation)

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model block containing dynamics, rewards, and shocks
    value_network : callable
        A value function that takes state variables and returns value estimates
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards
    foc_weight : float, optional
        Weight :math:`v` on the FOC residual term (default: 0.0, pure Bellman).
        Positive values add the FOC loss to improve convergence.
    """

    def __init__(
        self,
        bellman_period: BellmanPeriod,
        value_network: Callable,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        foc_weight: float = 0.0,
    ) -> None:
        super().__init__(bellman_period, parameters=parameters, agent=agent)
        if not callable(value_network):
            raise TypeError(
                f"value_network must be callable, got {type(value_network).__name__}"
            )
        if foc_weight < 0:
            raise ValueError(f"foc_weight must be >= 0, got {foc_weight}")
        self.value_network = value_network
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
            Bellman equation residual loss (squared), optionally with FOC term.
        """
        states_t, shocks = self._extract_states_and_shocks(input_grid)

        bellman_residual = estimate_bellman_residual(
            self.bellman_period,
            self.value_network,
            df,
            states_t,
            shocks,
            self.parameters,
            self.agent,
        )

        loss = bellman_residual**2

        if self.foc_weight > 0:
            foc_residuals = estimate_bellman_foc_residual(
                self.bellman_period,
                self.value_network,
                df,
                states_t,
                shocks,
                self.parameters,
                self.agent,
            )
            foc_loss = sum((r**2 for r in foc_residuals.values()), 0.0)
            loss = loss + self.foc_weight * foc_loss

        return loss


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

    When ``constrained=True`` and a control has an ``upper_bound`` defined on its
    ``Control`` object, the complementarity conditions

    .. math::

        f \\geq 0, \\quad s \\geq 0, \\quad f \\cdot s = 0

    (where :math:`s` is the constraint slack) are replaced by the smooth
    Fischer-Burmeister equation (Maliar et al. 2021, equation 25):

    .. math::

        \\text{FB}(f, s) = f + s - \\sqrt{f^2 + s^2} = 0

    For controls without an explicit ``upper_bound``, the loss falls back to the
    one-sided :math:`\\text{relu}(-f)^2` formulation.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model block containing dynamics, rewards, and shocks.
    parameters : dict, optional
        Model parameters for calibration.
    agent : str, optional
        Agent identifier for rewards.
    weight : float, optional
        Exogenous weight for combining multiple optimality conditions (default: 1.0).
        This corresponds to the vector :math:`v` in equation (12) of the paper.
    constrained : bool, optional
        If True, use Fischer-Burmeister or one-sided loss for upper-bound
        constrained controls (default: False).

    Examples
    --------
    >>> bp = BellmanPeriod(block, "beta", calibration={"R": 1.04, "beta": 0.95})
    >>> loss_fn = EulerEquationLoss(bp, parameters={"R": 1.04, "beta": 0.95})
    """

    def __init__(
        self,
        bellman_period: BellmanPeriod,
        parameters: dict[str, Any] | None = None,
        agent: str | None = None,
        weight: float = 1.0,
        constrained: bool = False,
    ) -> None:
        super().__init__(bellman_period, parameters=parameters, agent=agent)

        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self.weight = weight
        self.constrained = constrained

        if self.constrained:
            has_upper_bound = any(
                hasattr(rule, "upper_bound") and rule.upper_bound is not None
                for rule in bellman_period.block.dynamics.values()
                if hasattr(rule, "iset")
            )
            if not has_upper_bound:
                logger.warning(
                    "constrained=True but no Control in the block has an upper_bound. "
                    "The one-sided loss will use relu(-f)^2 as a fallback. "
                    "Define upper_bound on Control objects to enable the "
                    "Fischer-Burmeister formulation."
                )

    def _compute_slack(
        self, control_sym: str, controls_t: dict, states_t: dict, shocks_t: dict
    ) -> torch.Tensor | None:
        """Compute constraint slack for a control with an upper bound.

        Returns ``upper_bound - control_value``, or ``None`` if the control
        has no upper bound defined.
        """
        control_obj = self.bellman_period.block.dynamics.get(control_sym)
        if (
            control_obj is None
            or not hasattr(control_obj, "upper_bound")
            or control_obj.upper_bound is None
        ):
            return None

        pre_state = self.bellman_period._compute_pre_state_values(
            control_obj.iset,
            states_t,
            shocks=shocks_t,
            parameters=self.parameters,
        )

        sig = inspect.signature(control_obj.upper_bound)
        ub_args = {k: pre_state[k] for k in sig.parameters if k in pre_state}
        ub_value = control_obj.upper_bound(**ub_args)

        return ub_value - controls_t[control_sym]

    def __call__(self, df: Callable, input_grid: Grid) -> torch.Tensor:
        """
        Euler equation loss function using the AiO expectation operator.

        Parameters
        ----------
        df : callable
            Decision function from policy network.
        input_grid : Grid
            Grid containing current states and two independent shock realizations.

        Returns
        -------
        torch.Tensor
            Weighted squared Euler equation residual.
        """
        states_t, shocks = self._extract_states_and_shocks(input_grid)

        if self.constrained:
            # Compute controls once and share them between the residual
            # computation and the slack computation so both use the same
            # autograd graph.
            shocks_t, _ = _extract_period_shocks(self.bellman_period, shocks)
            controls_t = self.bellman_period.compute_controls(
                df, states_t, shocks=shocks_t, parameters=self.parameters
            )

            euler_residuals = estimate_euler_residual(
                self.bellman_period,
                df,
                states_t,
                shocks,
                self.parameters,
                self.agent,
                controls_t=controls_t,
            )

            total = 0.0
            for ctrl_sym, residual in euler_residuals.items():
                slack = self._compute_slack(ctrl_sym, controls_t, states_t, shocks_t)
                if slack is not None:
                    total = total + fischer_burmeister(residual, slack) ** 2
                else:
                    total = total + torch.relu(-residual) ** 2
            return self.weight * total

        # Unconstrained loss
        euler_residuals = estimate_euler_residual(
            self.bellman_period,
            df,
            states_t,
            shocks,
            self.parameters,
            self.agent,
        )
        return self.weight * sum((r**2 for r in euler_residuals.values()), 0.0)
