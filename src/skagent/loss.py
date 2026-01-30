import numpy as np
from skagent.bellman import (
    estimate_bellman_residual,
    estimate_discounted_lifetime_reward,
    estimate_euler_residual,
)
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
        controls = bellman_period.decision_function(
            states, shocks, parameters, decision_rules=dr
        )

    # this assumes only one reward is given.
    # can be generalized in the future.
    # move this logic to BellmanPeriod
    rsym = list(
        {
            sym
            for sym in bellman_period.block.reward
            if agent is None or bellman_period.block.reward[sym] == agent
        }
    )[0]

    reward = bellman_period.reward_function(
        states, shocks, controls, parameters, agent=agent, decision_rules=dr
    )

    # Maybe this can be less complicated because of unified array API
    if isinstance(reward[rsym], torch.Tensor) and torch.any(torch.isnan(reward[rsym])):
        raise ValueError(f"Calculated reward {[rsym]} is NaN: {reward}")
    if isinstance(reward[rsym], np.ndarray) and np.any(np.isnan(reward[rsym])):
        raise ValueError(f"Calculated reward {[rsym]} is NaN: {reward}")

    return reward[rsym]


class CustomLoss:
    """
    A custom loss function that computes the negative reward for a block,
    assuming it is executed just once (a non-dynamic model)

    TODO: leaving this as ambiguously about Blocks and BellmanPeriods for now
    """

    def __init__(self, loss_function, block, parameters=None, other_dr=None):
        self.block = block
        self.parameters = parameters
        self.state_variables = self.block.arrival_states
        self.other_dr = other_dr if other_dr is not None else {}
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

    def __init__(self, bellman_period, parameters, other_dr=None):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.state_variables = self.bellman_period.arrival_states
        self.other_dr = other_dr if other_dr is not None else {}

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
    big_t: int
        The number of time steps to compute reward for
    parameters
    """

    def __init__(self, bellman_period, big_t, parameters):
        self.bellman_period = bellman_period
        self.parameters = parameters
        self.state_variables = self.bellman_period.arrival_states
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

    Supports two modes:

    - Standard mode (aio=False): Uses 2 shock copies ({shock_sym}_0 and {shock_sym}_1).
      Returns f^2 where f is the Bellman residual.
    - AiO mode (aio=True): Uses 3 shock copies ({shock_sym}_0, {shock_sym}_1, {shock_sym}_2).
      Computes two residuals f1 and f2 sharing period-t shocks (_0) but with independent
      period t+1 shocks (_1 and _2). Returns f1 * f2, providing an unbiased estimate of
      E[(E[f])^2] per Maliar et al. (2021, Definition 2.7).

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
    aio : bool, optional
        If True, use the all-in-one (AiO) expectation operator with 3 shock copies.
        Default is False (standard squared-residual loss with 2 shock copies).
    """

    def __init__(
        self, bellman_period, value_network, parameters=None, agent=None, aio=False
    ):
        self.bellman_period = bellman_period
        self.parameters = parameters if parameters is not None else {}
        self.arrival_variables = bellman_period.arrival_states

        self.value_network = value_network
        self.aio = aio

        # Get shock variables
        shock_vars = self.bellman_period.get_shocks()
        self.shock_syms = list(shock_vars.keys())

        self.agent = agent
        # Test reward variables
        # TODO: move this to BP
        reward_vars = [
            sym
            for sym in self.bellman_period.block.reward
            if agent is None or self.bellman_period.block.reward[sym] == self.agent
        ]
        if len(reward_vars) == 0:
            raise ValueError("No reward variables found in block")

    def __call__(self, df, input_grid: Grid):
        """
        Bellman equation loss function.

        Parameters
        ----------
        df : callable
            Decision function from policy network
        input_grid : Grid
            Grid containing current states and shock realizations:
            - Standard mode (aio=False): requires {shock_sym}_0 and {shock_sym}_1
            - AiO mode (aio=True): requires {shock_sym}_0, {shock_sym}_1, and {shock_sym}_2

        Returns
        -------
        torch.Tensor
            Bellman equation residual loss.
            Standard mode: f^2 (squared residual)
            AiO mode: f1 * f2 (product of two residuals with independent shocks)
        """
        given_vals = input_grid.to_dict()

        # Extract current states
        states_t = {sym: given_vals[sym] for sym in self.arrival_variables}

        # Validate shock keys exist in grid
        required_copies = 3 if self.aio else 2
        for sym in self.shock_syms:
            for i in range(required_copies):
                key = f"{sym}_{i}"
                if key not in given_vals:
                    expected = [f"{sym}_{j}" for j in range(required_copies)]
                    found = [k for k in given_vals if k.startswith(f"{sym}_")]
                    raise KeyError(
                        f"Missing '{key}' in input_grid. "
                        f"{'AiO mode requires 3' if self.aio else 'Standard mode requires 2'} "
                        f"independent shock realizations per shock symbol. "
                        f"Expected keys: {expected}, found: {found}. "
                        f"Use generate_givens_from_states(states, block, shock_copies={required_copies}) "
                        f"to create the input grid."
                    )

        if not self.aio:
            # Standard mode: 2 shock copies, return f^2
            shocks = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks.update(
                {f"{sym}_1": given_vals[f"{sym}_1"] for sym in self.shock_syms}
            )

            bellman_residual = estimate_bellman_residual(
                self.bellman_period,
                self.value_network,
                df,
                states_t,
                shocks,
                self.parameters,
                self.agent,
            )
            return bellman_residual**2
        else:
            # AiO mode: 3 shock copies, compute two residuals sharing _0, return f1 * f2
            # Residual 1 uses shocks _0 (period t) and _1 (period t+1)
            shocks_1 = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks_1.update(
                {f"{sym}_1": given_vals[f"{sym}_1"] for sym in self.shock_syms}
            )

            # Residual 2 uses shocks _0 (period t) and _2 (period t+1, independent)
            shocks_2 = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks_2.update(
                {f"{sym}_1": given_vals[f"{sym}_2"] for sym in self.shock_syms}
            )

            residual_1 = estimate_bellman_residual(
                self.bellman_period,
                self.value_network,
                df,
                states_t,
                shocks_1,
                self.parameters,
                self.agent,
            )

            residual_2 = estimate_bellman_residual(
                self.bellman_period,
                self.value_network,
                df,
                states_t,
                shocks_2,
                self.parameters,
                self.agent,
            )

            return residual_1 * residual_2


class EulerEquationLoss:
    """
    Creates an Euler equation loss function for the Maliar method.

    The Euler equation is the first-order condition from the Bellman equation,
    relating marginal utilities across periods. This loss function computes the
    Euler equation **residual**:

    .. math::

        f = u'(c_t) + \\beta \\cdot u'(c_{t+1}) \\cdot \\sum_s \\left[
            \\frac{\\partial s_{t+1}}{\\partial c_t} \\cdot \\frac{\\partial m'}{\\partial s_{t+1}}
        \\right]

    where :math:`f` is the residual that equals zero at optimality, :math:`s_{t+1}` is
    the next-period arrival state, and :math:`m'` is the pre-decision state.

    **Derivation:**

    The first-order condition from the Bellman equation is:

    .. math::

        u'(c_t) = -\\beta E\\left[V'(s_{t+1}) \\cdot \\frac{\\partial s_{t+1}}{\\partial c_t}\\right]

    By the envelope theorem, :math:`V'(s') = u'(c') \\cdot \\frac{\\partial m'}{\\partial s'}`.

    For a consumption-saving model with :math:`a_{t+1} = m_t - c_t` where
    :math:`m_t = R \\cdot a_t + y_t` (cash-on-hand):

    - Transition gradient: :math:`\\frac{\\partial a_{t+1}}{\\partial c_t} = -1`
    - Pre-state gradient: :math:`\\frac{\\partial m'}{\\partial a_{t+1}} = R`
    - Combined: :math:`R \\cdot (-1) = -R`

    This gives :math:`f = u'(c_t) - \\beta R \\cdot u'(c_{t+1}) = 0` at optimality,
    which is the standard Euler equation :math:`u'(c_t) = \\beta R E[u'(c_{t+1})]`.

    **Methodology:**

    This follows the Maliar et al. (2021) methodology (Definition 2.7, equations 9-12)
    and is designed for use with the all-in-one (AiO) expectation operator.

    The implementation computes the Euler residual using two independent shock
    realizations (:math:`\\varepsilon_0` for period :math:`t` and :math:`\\varepsilon_1`
    for period :math:`t+1`), then squares it. The loss is:

    .. math::

        L(\\theta) = E[f^2]

    where :math:`f` is the Euler equation residual computed with both shock realizations.

    **Notation:**

    We use :math:`\\varepsilon` (epsilon) to denote exogenous shocks, following standard
    convention for stochastic disturbances. This is functionally equivalent to the shock
    notation in Maliar et al. (2021).

    **Input Grid Structure:**

    This function expects the input grid to contain two independent shock realizations:

    - ``{shock_sym}_0``: shocks for transitions from t to t+1
    - ``{shock_sym}_1``: shocks for transitions from t+1 to t+2 (independent of period t)

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β (time preference parameter)
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards
    weight : float, optional
        Exogenous weight for combining multiple optimality conditions (default: 1.0)
        This corresponds to the vector v in equation (12) of the paper.

    Returns
    -------
    callable
        A loss function that takes (decision_function, input_grid) and returns
        the Euler equation residual loss

    Notes
    -----
    This implementation follows Maliar, Maliar, and Winant (2021, JME) Section 2.2
    and Section 4.4 for the consumption-saving problem with Kuhn-Tucker conditions.

    **Current Limitations:**

    - Only single-control models are currently supported. Models with multiple
      control variables will raise a ``NotImplementedError``.
    - State-dependent discount factors are not yet supported.

    The Euler equation automatically adapts to your model's structure. For example:

    - Consumption-saving: :math:`u(c) = \\log(c)`, with :math:`a_{t+1} = m_t - c_t`
      where :math:`m_t = R \\cdot a_t + y_t`. Transition gradient
      :math:`\\partial a_{t+1}/\\partial c_t = -1` and pre-state gradient
      :math:`\\partial m'/\\partial a' = R` combine to give :math:`-R`.
      Euler equation becomes: :math:`u'(c_t) = \\beta R u'(c_{t+1})`
    - With permanent income: :math:`u(c/P)` where :math:`P` evolves with shocks.
      Multiple transition gradients are summed automatically.

    Examples
    --------
    >>> # Euler equation adapts to your block's reward structure
    >>> block = DBlock(
    ...     shocks={"income": Normal(mu=1.0, sigma=0.1)},
    ...     dynamics={
    ...         "consumption": Control(iset=["wealth"]),
    ...         "wealth": lambda wealth, income, consumption, R:
    ...             R * (wealth - consumption) + income,
    ...         "utility": lambda consumption: torch.log(consumption),
    ...     },
    ...     reward={"utility": "consumer"}
    ... )
    >>> bp = BellmanPeriod(block, parameters={"R": 1.04})
    >>> loss_fn = EulerEquationLoss(bp, discount_factor=0.95, parameters={"R": 1.04})
    >>>
    >>> # Create input grid with two shock realizations
    >>> input_grid = Grid.from_dict({
    ...     "wealth": torch.linspace(1.0, 10.0, 50),
    ...     "income_0": torch.ones(50),      # First shock realization
    ...     "income_1": torch.ones(50) * 1.1 # Second independent realization
    ... })
    >>>
    >>> # Compute loss for a decision function
    >>> loss = loss_fn(my_decision_function, input_grid)
    """

    def __init__(
        self,
        bellman_period,
        discount_factor,
        parameters=None,
        agent=None,
        weight=1.0,
        aio=False,
    ):
        self.bellman_period = bellman_period
        self.parameters = parameters if parameters is not None else {}
        self.arrival_variables = bellman_period.arrival_states

        self.discount_factor = discount_factor
        if callable(discount_factor):
            raise ValueError(
                "Currently only numerical, not state-dependent, discount factors are supported."
            )

        # Get shock variables
        shock_vars = self.bellman_period.get_shocks()
        self.shock_syms = list(shock_vars.keys())

        self.agent = agent
        self.weight = weight
        self.aio = aio

        # Test reward variables
        reward_vars = [
            sym
            for sym in self.bellman_period.block.reward
            if agent is None or self.bellman_period.block.reward[sym] == self.agent
        ]
        if len(reward_vars) == 0:
            raise ValueError("No reward variables found in block")

    def __call__(self, df, input_grid: Grid):
        """
        Euler equation loss function.

        Parameters
        ----------
        df : callable
            Decision function from policy network.
            Signature: df(states_t, shocks_t, parameters) -> controls_t
        input_grid : Grid
            Grid containing current states and shock realizations:
            - Standard mode (aio=False): requires {shock_sym}_0 and {shock_sym}_1
            - AiO mode (aio=True): requires {shock_sym}_0, {shock_sym}_1, and {shock_sym}_2

        Returns
        -------
        torch.Tensor
            Standard mode: weight * f^2 (squared residual)
            AiO mode: weight * f1 * f2 (product of two residuals with independent shocks)

        Notes
        -----
        Standard mode computes a single Euler residual f using shock copies _0 and _1,
        then returns weight * f^2.

        AiO mode (all-in-one expectation operator from Maliar et al. 2021, Definition 2.7)
        computes two Euler residuals f1 and f2 that share the period-t shocks (_0) but
        use independent period t+1 shocks (_1 and _2 respectively). The loss is
        weight * f1 * f2, which provides an unbiased estimate of E[(E[f])^2].
        """
        given_vals = input_grid.to_dict()

        # Extract current states
        states_t = {sym: given_vals[sym] for sym in self.arrival_variables}

        # Validate shock keys exist in grid
        required_copies = 3 if self.aio else 2
        for sym in self.shock_syms:
            for i in range(required_copies):
                key = f"{sym}_{i}"
                if key not in given_vals:
                    expected = [f"{sym}_{j}" for j in range(required_copies)]
                    found = [k for k in given_vals if k.startswith(f"{sym}_")]
                    raise KeyError(
                        f"Missing '{key}' in input_grid. "
                        f"{'AiO mode requires 3' if self.aio else 'Standard mode requires 2'} "
                        f"independent shock realizations per shock symbol. "
                        f"Expected keys: {expected}, found: {found}. "
                        f"Use generate_givens_from_states(states, block, shock_copies={required_copies}) "
                        f"to create the input grid."
                    )

        if not self.aio:
            # Standard mode: 2 shock copies, return weight * f^2
            shocks = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks.update(
                {f"{sym}_1": given_vals[f"{sym}_1"] for sym in self.shock_syms}
            )

            euler_residual = estimate_euler_residual(
                self.bellman_period,
                self.discount_factor,
                df,
                states_t,
                shocks,
                self.parameters,
                self.agent,
            )

            return self.weight * euler_residual**2
        else:
            # AiO mode: 3 shock copies, compute two residuals sharing _0, return weight * f1 * f2
            # Residual 1 uses shocks _0 (period t) and _1 (period t+1)
            shocks_1 = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks_1.update(
                {f"{sym}_1": given_vals[f"{sym}_1"] for sym in self.shock_syms}
            )

            # Residual 2 uses shocks _0 (period t) and _2 (period t+1, independent)
            shocks_2 = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in self.shock_syms}
            shocks_2.update(
                {f"{sym}_1": given_vals[f"{sym}_2"] for sym in self.shock_syms}
            )

            residual_1 = estimate_euler_residual(
                self.bellman_period,
                self.discount_factor,
                df,
                states_t,
                shocks_1,
                self.parameters,
                self.agent,
            )

            residual_2 = estimate_euler_residual(
                self.bellman_period,
                self.discount_factor,
                df,
                states_t,
                shocks_2,
                self.parameters,
                self.agent,
            )

            return self.weight * residual_1 * residual_2
