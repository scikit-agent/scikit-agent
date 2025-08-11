import numpy as np
import skagent.ann as ann
from skagent.grid import Grid
import skagent.model as model
from skagent.simulation.monte_carlo import draw_shocks
import torch
import skagent.utils as utils
import math

"""
Tools for the implementation of the Maliar, Maliar, and Winant (JME '21) method.

This method relies on a simpler problem representation than that elaborated
by the skagent Block system.

"""


def create_transition_function(block, state_syms):
    """
    block
    state_syms : list of string
        A list of symbols for 'state variables at time t', aka arrival states.
        # TODO: state variables should be derived from the block analysis.
    """

    def transition_function(states_t, shocks_t, controls_t, parameters):
        vals = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals, {}, fix=list(controls_t.keys()))

        return {sym: post[sym] for sym in state_syms}

    return transition_function


def create_decision_function(block, decision_rules):
    """
    block
    decision_rules
    """

    def decision_function(states_t, shocks_t, parameters):
        if parameters is None:
            parameters = {}
        vals = parameters | states_t | shocks_t
        post = block.transition(vals, decision_rules)

        return {sym: post[sym] for sym in decision_rules}

    return decision_function


def create_reward_function(block, agent=None):
    """
    block
    agent : optional, str
    """

    def reward_function(states_t, shocks_t, controls_t, parameters):
        vals_t = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals_t, {}, fix=list(controls_t.keys()))
        return {
            sym: post[sym]
            for sym in block.reward
            if agent is None or block.reward[sym] == agent
        }

    return reward_function


def estimate_discounted_lifetime_reward(
    block,
    discount_factor,
    dr,
    states_0,
    big_t,
    shocks_by_t=None,
    parameters={},
    agent=None,
):
    """
    block
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

    tf = create_transition_function(block, list(states_0.keys()))

    if callable(dr):
        # assume a full decision function has been passed in
        df = dr
    else:
        # create a decision function from the decision rule
        df = create_decision_function(block, dr)

    rf = create_reward_function(block, agent)

    # Get all reward symbols for the agent
    reward_syms = list(
        {sym for sym in block.reward if agent is None or block.reward[sym] == agent}
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

        controls_t = df(states_t, shocks_t, parameters)
        reward_t = rf(states_t, shocks_t, controls_t, parameters)

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
        states_t = tf(states_t, shocks_t, controls_t, parameters)

    return total_discounted_reward


def get_estimated_discounted_lifetime_reward_loss(
    state_variables, block, discount_factor, big_t, parameters
):
    # TODO: Should be able to get 'state variables' from block
    # Maybe with ZP's analysis modules

    # convoluted
    # TODO: codify this encoding and decoding of the grid into a separate object
    # It is specifically the EDLR loss function that requires big_t of the shocks.
    # other AiO loss functions use 2 copies of the shocks only.
    shock_vars = block.get_shocks()
    big_t_shock_syms = sum(
        [[f"{sym}_{t}" for sym in list(shock_vars.keys())] for t in range(big_t)], []
    )

    def estimated_discounted_lifetime_reward_loss(df: callable, input_grid: Grid):
        # includes the values of state_0 variables, and shocks.
        given_vals = input_grid.to_dict()

        shock_vals = {sym: given_vals[sym] for sym in big_t_shock_syms}
        shocks_by_t = {
            sym: torch.stack([shock_vals[f"{sym}_{t}"] for t in range(big_t)])
            for sym in shock_vars
        }

        # block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        edlr = estimate_discounted_lifetime_reward(
            block,
            discount_factor,
            df,
            {sym: given_vals[sym] for sym in state_variables},
            big_t,
            parameters=parameters,
            agent=None,  # TODO: Pass through the agent?
            shocks_by_t=shocks_by_t,
            # Handle multiple decision rules?
        )
        return -edlr

    return estimated_discounted_lifetime_reward_loss


def generate_givens_from_states(states: Grid, block: model.Block, shock_copies: int):
    """
    Generates omega_i values of the MMW JME '21 method.

    states : a grid of starting state values (exogenous and endogenous)
    block: block information (used to get the shock names)
    shock_copies : int - number of copies of the shocks to be included.
    """

    # get the length of the states vectors -- N
    n = states.n()
    new_shock_values = {}

    for i in range(shock_copies):
        # relies on constructed shocks
        # required
        shock_values = draw_shocks(block.shocks, n=n)
        new_shock_values.update(
            {f"{sym}_{i}": shock_values[sym] for sym in shock_values}
        )

    givens = states.update_from_dict(new_shock_values)

    return givens


############################
# Euler residual loss (MMW)
############################


def get_euler_fb_loss(state_variables, block: model.DBlock, parameters: dict):
    """
    Model-agnostic all-in-one Euler residual loss with optional Fischer–Burmeister term.

    This routine derives everything it can from the DBlock, and accepts optional
    callables to inject model-specific components (pricing term and complementarity term).

    Expectations are computed via the Maliar all-in-one trick using two independent
    shock copies per exogenous shock: for each shock name z, columns z_0 and z_1
    must exist in the input Grid.

    Parameters
    - state_variables: list of state variable names used by the policy and transitions
    - block: DBlock
    - discount_factor: β
    - parameters: calibration dictionary used by block dynamics
    - pricing_term: callable(states_t, states_next, parameters) -> tensor factor. If None, uses 1.
    - consumption_var: name of the consumption variable in block dynamics. If None, try detect
      from the reward equation arguments.
    - multiplier_control: optional control name to subtract from the Euler term (e.g., 'h'). If None,
      no subtraction is performed.
    - fb_term: optional callable(states_t, controls_t, parameters) -> tensor, the FB residual (R2).
      If None, no FB penalty is added.
    - agent: optional agent filter for reward selection (unused for derivative, but kept for symmetry)
    - weight_fb: multiplier on FB penalty squared.
    """

    shock_syms = list(block.get_shocks().keys())

    # Require custom residuals on the block
    if not (
        hasattr(block, "resid")
        and isinstance(block.resid, dict)
        and len(block.resid) > 0
    ):
        raise Exception(
            "get_euler_fb_loss requires block.resid to be provided (dict of residual functions)"
        )

    def loss(decision_function, input_grid: Grid):
        vals = input_grid.to_dict()
        # extract current states
        states_t = {sym: vals[sym] for sym in state_variables}
        # controls at t
        controls_t = decision_function(states_t, {}, parameters)
        # Expect each residual function rf(states_t, controls_t, shocks, parameters) -> tensor
        e0 = {sym: vals[f"{sym}_0"] for sym in shock_syms}
        e1 = {sym: vals[f"{sym}_1"] for sym in shock_syms}
        losses = []
        for name, rf in block.resid.items():
            # Support rf with 4 or 5 arguments (optionally decision_function)
            try:
                r0 = rf(states_t, controls_t, e0, parameters, decision_function)
            except TypeError:
                r0 = rf(states_t, controls_t, e0, parameters)
            try:
                r1 = rf(states_t, controls_t, e1, parameters, decision_function)
            except TypeError:
                r1 = rf(states_t, controls_t, e1, parameters)
            if not torch.is_tensor(r0):
                r0 = torch.as_tensor(
                    r0,
                    dtype=next(iter(states_t.values())).dtype,
                    device=next(iter(states_t.values())).device,
                )
            if not torch.is_tensor(r1):
                r1 = torch.as_tensor(r1, dtype=r0.dtype, device=r0.device)
            losses.append(r0 * r1)
        return sum(losses)

    return loss


def generate_random_ergodic_state_grid(
    calibration: dict,
    n: int,
    state_variables: list[str],
    ergodic_specs: dict[str, dict] | None = None,
) -> Grid:
    """
    Model-agnostic random state generator.

    For each state symbol in state_variables, draw i.i.d. samples using the following rules:
    - If ergodic_specs provides {"dist": callable, "kwargs": {...}} for the symbol, call it to draw.
    - Else if both sigma_{sym} and rho_{sym} exist in calibration, draw from N(0, sigma/sqrt(1-rho^2)).
    - Else if bounds {sym_min, sym_max} exist in calibration, draw Uniform[min,max].
    - Else, default to N(0,1).
    """

    draws = {}
    rng_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def erg_std(sigma, rho):
        return sigma / math.sqrt(max(1e-24, 1.0 - rho * rho))

    for sym in state_variables:
        if ergodic_specs and sym in ergodic_specs:
            spec = ergodic_specs[sym]
            dist = spec.get("dist")
            kwargs = spec.get("kwargs", {})
            if dist is None:
                raise ValueError(f"Missing 'dist' in ergodic_specs for {sym}")
            sample = dist(**kwargs)
            sample = sample if torch.is_tensor(sample) else torch.as_tensor(sample)
            if sample.numel() != n:
                sample = sample.reshape(-1)[:n]
            draws[sym] = sample.to(rng_device)
            continue

        sigma_key = f"sigma_{sym}"
        rho_key = f"rho_{sym}"
        min_key = f"{sym}_min"
        max_key = f"{sym}_max"

        if sigma_key in calibration and rho_key in calibration:
            std = erg_std(float(calibration[sigma_key]), float(calibration[rho_key]))
            draws[sym] = torch.normal(mean=0.0, std=std, size=(n,), device=rng_device)
        elif min_key in calibration and max_key in calibration:
            a = float(calibration[min_key])
            b = float(calibration[max_key])
            draws[sym] = (
                torch.distributions.Uniform(low=a, high=b).sample((n,)).to(rng_device)
            )
        else:
            draws[sym] = torch.normal(mean=0.0, std=1.0, size=(n,), device=rng_device)

    return Grid.from_dict(draws)


def prepare_aio_training_inputs(
    block: model.DBlock,
    calibration: dict,
    n: int,
    seed: int | None = None,
    shock_copies: int = 2,
    state_variables: list[str] | None = None,
    ergodic_specs: dict[str, dict] | None = None,
):
    """
    Construct seeded shocks on the block, build a random ergodic state grid (model-agnostic),
    and append independent shock copies for all-in-one training.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    block.construct_shocks(calibration, rng=rng)
    if state_variables is None:
        # default to the information set of the first control
        cs = block.get_controls()
        if len(cs) == 0:
            raise Exception(
                "State variables not provided and block has no controls to infer from"
            )
        iset = block.dynamics[cs[0]].iset
        state_variables = list(iset)
    states = generate_random_ergodic_state_grid(
        calibration, n, state_variables, ergodic_specs
    )
    givens = generate_givens_from_states(states, block, shock_copies=shock_copies)
    return givens


def simulate_forward(
    states_t,
    block: model.Block,
    decision_function: callable,
    parameters,
    big_t,
    # state_syms,
):
    if isinstance(states_t, Grid):
        n = states_t.n()
        states_t = states_t.to_dict()
    else:
        # kludge
        n = len(states_t[next(iter(states_t.keys()))])

    state_syms = list(states_t.keys())
    tf = create_transition_function(block, state_syms)

    for t in range(big_t):
        # TODO: make sure block shocks are 'constructed'
        # TODO: allow option for 'structured' draws, e.g. from exact discretization.
        shocks_t = draw_shocks(block.shocks, n=n)

        # this is cumbersome; probably can be solved deeper on the data structure level
        # note similarity to Grid.from_dict() reconciliation logic.
        states_template = states_t[next(iter(states_t.keys()))]
        shocks_t = {
            sym: utils.reconcile(states_template, shocks_t[sym]) for sym in shocks_t
        }

        controls_t = decision_function(states_t, shocks_t, parameters)

        states_t_plus_1 = tf(states_t, shocks_t, controls_t, parameters)
        states_t = states_t_plus_1

    return states_t_plus_1


def maliar_training_loop(
    block,
    loss_function,
    states_0_n: Grid,
    parameters,
    shock_copies=2,
    max_iterations=5,
    random_seed=None,
    simulation_steps=1,
):
    """
    block - a model definition
    loss_function : callable((df, input_vector) -> loss vector
    states_0_n : Grid a panel of starting states
    parameters : dict : given parameters for the model

    shock_copies: int : number of copies of shocks to include in the training set omega
                        must match expected number of shock copies in the loss function
                        TODO: make this better, less ad hoc

    loss_function is the "empirical risk Xi^n" in MMW JME'21.

    max_iterations: int
        Number of times to perform the training loop, if there is no convergence.
    simulation_steps : int
        The number of time steps to simulate forward when determining the next omega set for training
    """

    # Step 1. Initialize the algorithm:

    # i). construct theoretical risk Xi(θ ) = Eω [ξ (ω; θ )] (lifetime reward, Euler/Bellmanequations);
    # ii). deﬁne empirical risk Xi^n (θ ) = 1n ni=1 ξ (ωi ; θ );
    loss_function  # This is provided as an argument.

    # iii). deﬁne a topology of neural network ϕ (·, θ );
    # iv). ﬁx initial vector of the coeﬃcients θ .

    if random_seed is not None:
        torch.manual_seed(random_seed)

    bpn = ann.BlockPolicyNet(block, width=16)

    states = states_0_n  # V) Create initial panel of agents/starting states.

    # Step 2. Train the machine, i.e., ﬁnd θ that minimizes theempirical risk Xi^n (θ ):
    for i in range(max_iterations):
        # i). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        givens = generate_givens_from_states(states_0_n, block, shock_copies)

        # ii). construct the gradient ∇ Xi^n (θ ) = 1n ni=1 ∇ ξ (ωi ; θ );
        # iii). update the coeﬃcients θ_hat = θ − λk ∇ Xi^n (θ ) and go to step 2.i);
        # TODO how many epochs? What Adam scale? Passing through variables
        ann.train_block_policy_nn(bpn, givens, loss_function, epochs=250)

        # i/iv). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        next_states = simulate_forward(
            states, block, bpn.get_decision_function(), parameters, simulation_steps
        )

        states = Grid.from_dict(next_states)

        # End Step 2 if the convergence criterion || θ_hat − θ ||  < ε is satisﬁed.
        # TODO: test for difference.. how? This effects the FOR (/while) loop above.

    # Step 3. Assess the accuracy of constructed approximation ϕ (·, θ ) on a new sample.
    return bpn, states


def estimate_bellman_residual(
    block,
    discount_factor,
    value_network,
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
    block : model.DBlock
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β
    value_network : callable
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

    # Get state variable names for transition
    state_variables = list(states_t.keys())

    # Get shock variable names
    shock_vars = block.get_shocks()
    shock_syms = list(shock_vars.keys())

    # Extract period-specific shocks from the combined shocks object
    shocks_t = {sym: shocks[f"{sym}_0"] for sym in shock_syms}
    shocks_t_plus_1 = {sym: shocks[f"{sym}_1"] for sym in shock_syms}

    # Get reward variables
    reward_vars = [
        sym for sym in block.reward if agent is None or block.reward[sym] == agent
    ]
    if len(reward_vars) == 0:
        raise Exception("No reward variables found in block")
    reward_sym = reward_vars[0]  # Assume single reward for now

    # Get current value estimates (using period t shocks)
    current_values = value_network(states_t, shocks_t, parameters)

    # Get controls from decision function (using period t shocks)
    controls_t = df(states_t, shocks_t, parameters)

    # Create transition and reward functions
    tf = create_transition_function(block, state_variables)
    rf = create_reward_function(block, agent)

    # Compute immediate reward (using period t shocks)
    immediate_reward = rf(states_t, shocks_t, controls_t, parameters)[reward_sym]

    # Compute next states (using period t shocks)
    next_states = tf(states_t, shocks_t, controls_t, parameters)

    # Compute continuation value using value network (using period t+1 shocks)
    continuation_values = value_network(next_states, shocks_t_plus_1, parameters)

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    return bellman_residual


def get_bellman_equation_loss(
    state_variables, block, discount_factor, value_network, parameters={}, agent=None
):
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
    if callable(discount_factor):
        raise Exception(
            "Currently only numerical, not state-dependent, discount factors are supported."
        )

    # Get shock variables
    shock_vars = block.get_shocks()
    shock_syms = list(shock_vars.keys())

    # Get control variables
    control_vars = block.get_controls()
    if len(control_vars) == 0:
        raise Exception("No control variables found in block")

    # Get reward variables
    reward_vars = [
        sym for sym in block.reward if agent is None or block.reward[sym] == agent
    ]
    if len(reward_vars) == 0:
        raise Exception("No reward variables found in block")
    reward_vars[0]  # Assume single reward for now

    def bellman_equation_loss(df, input_grid: Grid):
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
        states_t = {sym: given_vals[sym] for sym in state_variables}
        shocks = {f"{sym}_0": given_vals[f"{sym}_0"] for sym in shock_syms}
        shocks.update({f"{sym}_1": given_vals[f"{sym}_1"] for sym in shock_syms})

        # Use helper function to estimate Bellman residual with combined shock object
        bellman_residual = estimate_bellman_residual(
            block,
            discount_factor,
            value_network,
            df,
            states_t,
            shocks,
            parameters,
            agent,
        )

        # Return squared residual as loss
        return bellman_residual**2

    return bellman_equation_loss
