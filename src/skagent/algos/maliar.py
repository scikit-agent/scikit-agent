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
    tolerance: float
        Convergence tolerance. Training stops when the L2 norm of parameter changes
        is below this threshold.
    simulation_steps : int
        The number of time steps to simulate forward when determining the next omega set for training
    """

    def extract_parameters(network):
        """Extract all parameters from the network into a flat tensor."""
        params = []
        for param in network.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params) if params else torch.tensor([])

    def compute_parameter_difference(params1, params2):
        """Compute the L2 norm of the difference between two parameter vectors."""
        if len(params1) != len(params2):
            return float("inf")
        return torch.norm(params1 - params2).item()

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
    iteration = 0
    converged = False

    while iteration < max_iterations and not converged:
        # Store current parameters before training
        prev_params = extract_parameters(bpn)

        # i). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        givens = generate_givens_from_states(states_0_n, block, shock_copies)

        # ii). construct the gradient ∇ Xi^n (θ ) = 1n ni=1 ∇ ξ (ωi ; θ );
        # iii). update the coeﬃcients θ_hat = θ − λk ∇ Xi^n (θ ) and go to step 2.i);
        # TODO how many epochs? What Adam scale? Passing through variables
        ann.train_block_policy_nn(bpn, givens, loss_function, epochs=250)

        # Extract parameters after training
        curr_params = extract_parameters(bpn)

        # Check for convergence: || θ_hat − θ ||  < ε
        param_diff = compute_parameter_difference(prev_params, curr_params)

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
            states, block, bpn.get_decision_function(), parameters, simulation_steps
        )

        states = Grid.from_dict(next_states)
        iteration += 1

    if not converged:
        logging.warning(
            f"Training completed without convergence after {max_iterations} iterations."
        )

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
