import logging
import numpy as np
import skagent.ann as ann
from skagent.grid import Grid
import skagent.model as model
from skagent.simulation.monte_carlo import draw_shocks
import torch
import skagent.utils as utils

"""
Tools for the implementation of the Maliar, Maliar, and Winant (JME '21) method.

This method relies on a simpler problem representation than that elaborated
by the skagent Block system.

"""


def create_transition_function(block, state_syms, decision_rules=None):
    """
    block
    state_syms : list of string
        A list of symbols for 'state variables at time t', aka arrival states.
        # TODO: state variables should be derived from the block analysis.
    """
    decision_rules = {} if decision_rules is None else decision_rules

    def transition_function(states_t, shocks_t, controls_t, parameters):
        vals = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals, decision_rules, fix=list(controls_t.keys()))

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


def create_reward_function(block, agent=None, decision_rules=None):
    """
    block
    agent : optional, str
    decision_rules : optional, dict
        Decision rules of control variables that will _not_
        be given to the function.
    """
    decision_rules = {} if decision_rules is None else decision_rules

    def reward_function(states_t, shocks_t, controls_t, parameters):
        vals_t = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals_t, decision_rules, fix=list(controls_t.keys()))
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
    state_variables, block, discount_factor, big_t, parameters={}, agent=None
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
            agent=agent,
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


def _extract_network_parameters(network):
    """Extract all parameters from a network into a flat tensor.

    Helper function used by both training loops for convergence checking.

    Parameters
    ----------
    network : torch.nn.Module
        Neural network to extract parameters from

    Returns
    -------
    torch.Tensor
        Flattened parameter tensor
    """
    params = []
    for param in network.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params) if params else torch.tensor([])


def _compute_parameter_difference(params1, params2):
    """Compute the L2 norm of the difference between two parameter vectors.

    Helper function used by both training loops for convergence checking.

    Parameters
    ----------
    params1 : torch.Tensor
        First parameter vector
    params2 : torch.Tensor
        Second parameter vector

    Returns
    -------
    float
        L2 norm of the difference, or infinity if shapes don't match
    """
    if len(params1) != len(params2):
        return float("inf")
    return torch.norm(params1 - params2).item()


def maliar_training_loop(
    block,
    loss_function,
    states_0_n: Grid,
    parameters,
    shock_copies=2,
    max_iterations=5,
    tolerance=1e-6,
    random_seed=None,
    simulation_steps=1,
    epochs=250,
):
    """
    Maliar training loop for policy network training only.

    This function trains only a policy network using lifetime reward loss functions.
    It cannot use Bellman residual loss since it does not create or track value networks.
    For joint training of both policy and value networks, use bellman_training_loop instead.

    This implements policy-only optimization of θ₂ where:
    - θ₂: Decision rule parameters φ(·; θ₂)

    Parameters
    ----------
    block : DBlock
        The model definition
    loss_function : callable
        Loss function for policy training with signature (decision_function, input_grid) -> loss.
        Must be a lifetime reward loss function. Cannot use Bellman residual loss since
        this function only trains policy networks and does not create/track value networks.
        Created by get_estimated_discounted_lifetime_reward_loss().
    states_0_n : Grid
        Initial panel of starting states
    parameters : dict
        Model parameters
    shock_copies : int, optional
        Number of shock copies for training, by default 2
    max_iterations : int, optional
        Maximum training iterations, by default 5
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
    random_seed : int, optional
        Random seed, by default None
    simulation_steps : int, optional
        Steps to simulate forward between iterations, by default 1
    epochs : int, optional
        Number of training epochs per iteration, by default 250

    Returns
    -------
    tuple
        (trained_policy_net, final_states)

    Notes
    -----
    loss_function is the "empirical risk Xi^n" from MMW JME'21 for policy training.
    shock_copies must match expected number of shock copies in the loss function.
    This function only supports lifetime reward loss, not Bellman residual loss.
    Implements Algorithm 1 from MMW JME'21 for policy network optimization.
    """

    # Step 1. Initialize the algorithm:

    # i). construct theoretical risk Xi(θ ) = Eω [ξ (ω; θ )] (lifetime reward, Euler/Bellmanequations);
    # ii). deﬁne empirical risk Xi^n (θ ) = 1n ni=1 ξ (ωi ; θ );
    loss_function  # This is provided as an argument.

    # iii). deﬁne a topology of neural network ϕ (·, θ );
    # iv). ﬁx initial vector of the coeﬃcients θ .

    if random_seed is not None:
        torch.manual_seed(random_seed)

    policy_net = ann.BlockPolicyNet(block)

    states = states_0_n  # V) Create initial panel of agents/starting states.

    # Step 2. Train the machine, i.e., ﬁnd θ that minimizes theempirical risk Xi^n (θ ):
    iteration = 0
    converged = False

    while iteration < max_iterations and not converged:
        # Store current parameters before training
        prev_params = _extract_network_parameters(policy_net)

        # i). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        givens = generate_givens_from_states(states_0_n, block, shock_copies)

        # ii). construct the gradient ∇ Xi^n (θ ) = 1n ni=1 ∇ ξ (ωi ; θ );
        # iii). update the coeﬃcients θ_hat = θ − λk ∇ Xi^n (θ ) and go to step 2.i);
        ann.train_block_policy_nn(policy_net, givens, loss_function, epochs=epochs)

        # Extract parameters after training
        curr_params = _extract_network_parameters(policy_net)

        # Check for convergence: || θ_hat − θ ||  < ε
        param_diff = _compute_parameter_difference(prev_params, curr_params)

        if param_diff < tolerance:
            converged = True
            logging.info(
                f"Converged after {iteration + 1} iterations. Parameter difference: {param_diff:.2e}"
            )
        else:
            logging.info(
                f"Iteration {iteration + 1}: Parameter difference: {param_diff:.2e}"
            )

        # i/iv). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        next_states = simulate_forward(
            states,
            block,
            policy_net.get_decision_function(),
            parameters,
            simulation_steps,
        )

        states = Grid.from_dict(next_states)
        iteration += 1

    if not converged:
        logging.warning(
            f"Training completed without convergence after {max_iterations} iterations."
        )

    # Step 3. Assess the accuracy of constructed approximation ϕ (·, θ ) on a new sample.
    return policy_net, states


def estimate_bellman_residual(
    block,
    discount_factor,
    vf,
    dr,
    states_t,
    shocks,
    parameters={},
    agent=None,
):
    """
    Computes the Bellman equation residual for given states and shocks.

    This is a DIFFERENT approach from lifetime reward estimation.

    The Bellman equation is: V(s) = max_c { u(s,c,ε) + β E_ε'[V(s')] }
    This function computes: V(s) - [u(s,c,ε) + β V(s')]
    where:
    - V(s) and V(s') come from a separate value function (e.g., value network)
    - This is NOT estimated by forward simulation

    Parameters
    ----------
    block : model.DBlock
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β
    value_function : callable
        A value function that takes (states_t, shocks_t, parameters) and returns values
        This should be a proper value estimator (like a value network's value function)
    dr : callable or dict
        Decision rules (dict of functions), or optionally a decision function
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

    # Create helper functions early (following estimate_discounted_lifetime_reward pattern)
    tf = create_transition_function(block, state_variables)
    rf = create_reward_function(block, agent)

    # Handle decision function conversion (following estimate_discounted_lifetime_reward pattern)
    if callable(dr):
        # assume a full decision function has been passed in
        df = dr
    else:
        # create a decision function from the decision rule
        df = create_decision_function(block, dr)

    # Get all reward symbols for the agent (following estimate_discounted_lifetime_reward pattern)
    reward_syms = list(
        {sym for sym in block.reward if agent is None or block.reward[sym] == agent}
    )
    if len(reward_syms) == 0:
        raise Exception("No reward variables found in block")

    # Get controls from decision function (using period t shocks)
    controls_t = df(states_t, shocks_t, parameters)

    # Compute immediate reward (using period t shocks)
    reward_t = rf(states_t, shocks_t, controls_t, parameters)

    # Sum all rewards for this period (following estimate_discounted_lifetime_reward pattern)
    immediate_reward = 0
    for rsym in reward_syms:
        # Add NaN checking (following estimate_discounted_lifetime_reward pattern)
        if isinstance(reward_t[rsym], torch.Tensor) and torch.any(
            torch.isnan(reward_t[rsym])
        ):
            raise Exception(f"Calculated reward {rsym} is NaN: {reward_t}")
        if isinstance(reward_t[rsym], np.ndarray) and np.any(np.isnan(reward_t[rsym])):
            raise Exception(f"Calculated reward {rsym} is NaN: {reward_t}")
        immediate_reward += reward_t[rsym]

    # Compute next states (using period t shocks)
    next_states = tf(states_t, shocks_t, controls_t, parameters)

    # # Compute next period controls
    # next_controls_t = df(next_states, shocks_t_plus_1, parameters)

    # Get current value V(s) from the value function
    current_values = vf(states_t, shocks_t, parameters)

    # Get continuation value V(s') from the value function
    continuation_values = vf(next_states, shocks_t_plus_1, parameters)

    # Bellman equation: V(s) = u(s,c,ε) + β E_ε'[V(s')]
    bellman_rhs = immediate_reward + discount_factor * continuation_values

    # Return residual: V(s) - [u(s,c,ε) + β V(s')]
    bellman_residual = current_values - bellman_rhs

    return bellman_residual


def get_bellman_equation_loss(
    state_variables, block, discount_factor, parameters={}, agent=None, nu=1.0
):
    """
    Creates a unified Bellman equation loss function implementing MMW Definition 2.10.

    This implements the "Bellman-residual minimization with all-in-one expectation operator"
    for joint training of both value function V(·; θ₁) and decision rule φ(·; θ₂).

    This function expects the input grid to contain two independent shock realizations:
    - {shock_sym}_0: shocks ε₁ for first Bellman residual
    - {shock_sym}_1: shocks ε₂ for second Bellman residual

    Parameters
    ----------
    state_variables : list of str
        List of state variable names (endogenous state variables)
    block : model.DBlock
        The model block containing dynamics, rewards, and shocks
    discount_factor : float
        The discount factor β
    parameters : dict, optional
        Model parameters for calibration
    agent : str, optional
        Agent identifier for rewards
    nu : float, optional
        Weight parameter ν > 0 for derivative terms, by default 1.0

    Returns
    -------
    callable
        Unified loss function for joint training with signature:
        loss_function(value_function, decision_function, input_grid) -> loss
        Designed for use with bellman_training_loop and train_block_value_and_policy_nn

    Notes
    -----
    This implements MMW Definition 2.10 for joint training of both networks.
    The loss function is compatible with existing training infrastructure.
    """
    if callable(discount_factor):
        raise Exception(
            "Currently only numerical, not state-dependent, discount factors are supported."
        )

    # Get shock variables
    # shock_vars = block.get_shocks()
    # shock_syms = list(shock_vars.keys())

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

    def bellman_loss(vf, df, input_grid: Grid):
        """
        Bellman loss function implementing simplified Bellman residual computation.

        Computes the Bellman equation residual: V(s) - [u(s,c,ε) + β V(s')]
        where V(s) comes from the actual value network being trained.

        This is a working implementation but simplified compared to full MMW Definition 2.10
        which would include product of residuals under two shock realizations plus derivative terms.

        Parameters
        ----------
        vf : callable
            Value function from value network
        df : callable
            Decision function from policy network
        input_grid : Grid
            Grid containing states and two independent shock realizations

        Returns
        -------
        torch.Tensor
            Bellman residual loss (squared)
        """
        # Use estimate_bellman_residual for computation
        given_vals = input_grid.to_dict()

        # Extract current states and combined shocks for estimate_bellman_residual
        states_t = {sym: given_vals[sym] for sym in state_variables}
        shocks = given_vals  # Contains both _0 and _1 shock realizations

        # Compute Bellman residual using the existing function with actual value function
        bellman_residual = estimate_bellman_residual(
            block=block,
            discount_factor=discount_factor,
            vf=vf,
            dr=df,
            states_t=states_t,
            shocks=shocks,
            parameters=parameters,
            agent=agent,
        )

        # Return squared residual as loss
        return bellman_residual**2

    return bellman_loss


def bellman_training_loop(
    block,
    loss_function,
    states_0_n: Grid,
    parameters,
    shock_copies=2,
    max_iterations=5,
    tolerance=1e-6,
    random_seed=None,
    simulation_steps=1,
    epochs=250,
):
    """
    Bellman training loop for joint policy and value network training.

    This function performs joint training of both policy and value networks using
    the full Bellman-residual minimization with all-in-one expectation operator.
    For policy-only training with lifetime reward loss, use maliar_training_loop instead.

    This implements joint optimization of θ ≡ (θ₁, θ₂) where:
    - θ₁: Value function parameters V(·; θ₁)
    - θ₂: Decision rule parameters φ(·; θ₂)

    Parameters
    ----------
    block : DBlock
        The model definition
    loss_function : callable
        A unified loss function with signature (value_function, decision_function, input_grid) -> loss.
        Should implement Definition 2.10 for joint training of both networks.
        Typically created by get_bellman_equation_loss().
    states_0_n : Grid
        Initial panel of starting states
    parameters : dict
        Model parameters
    shock_copies : int, optional
        Number of shock copies for training, by default 2
    max_iterations : int, optional
        Maximum training iterations, by default 5
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
    random_seed : int, optional
        Random seed, by default None
    simulation_steps : int, optional
        Steps to simulate forward between iterations, by default 1
    epochs : int, optional
        Number of training epochs per iteration, by default 250

    Returns
    -------
    tuple
        (trained_policy_net, trained_value_net, final_states)

    Notes
    -----
    loss_function implements MMW Definition 2.10 from MMW JME'21 for joint training.
    shock_copies must match expected number of shock copies in the loss function.
    This function only supports Bellman residual loss for joint network training.
    Implements Definition 2.10 from MMW JME'21 for joint policy and value optimization.
    """

    # Initialize the algorithm - follows maliar_training_loop pattern
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Create both networks (unlike maliar_training_loop which only creates policy)
    policy_net = ann.BlockPolicyNet(block)
    value_net = ann.BlockValueNet(block)

    states = states_0_n
    iteration = 0
    converged = False

    while iteration < max_iterations and not converged:
        # Store current parameters before training
        prev_policy_params = _extract_network_parameters(policy_net)
        prev_value_params = _extract_network_parameters(value_net)

        # Generate training data - matches maliar_training_loop pattern
        givens = generate_givens_from_states(states_0_n, block, shock_copies)

        # Joint training using MMW Definition 2.10 unified loss
        trained_policy, trained_value = ann.train_block_value_and_policy_nn(
            policy_net,
            value_net,
            givens,
            loss_function,
            epochs=epochs,
        )

        # Extract parameters after training
        curr_policy_params = _extract_network_parameters(policy_net)
        curr_value_params = _extract_network_parameters(value_net)

        # Check for convergence - simpler logic matching maliar_training_loop
        policy_diff = _compute_parameter_difference(
            prev_policy_params, curr_policy_params
        )
        value_diff = _compute_parameter_difference(prev_value_params, curr_value_params)
        total_diff = max(policy_diff, value_diff)

        if total_diff < tolerance:
            converged = True
            logging.info(
                f"Converged after {iteration + 1} iterations. "
                f"Policy diff: {policy_diff:.2e}, Value diff: {value_diff:.2e}"
            )
        else:
            logging.info(
                f"Iteration {iteration + 1}: "
                f"Policy diff: {policy_diff:.2e}, Value diff: {value_diff:.2e}"
            )

        # Simulate forward for next iteration - matches maliar_training_loop pattern
        next_states = simulate_forward(
            states,
            block,
            policy_net.get_decision_function(),
            parameters,
            simulation_steps,
        )

        states = Grid.from_dict(next_states)
        iteration += 1

    if not converged:
        logging.warning(
            f"Training completed without convergence after {max_iterations} iterations."
        )

    return policy_net, value_net, states
