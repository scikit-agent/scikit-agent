import logging
import skagent.ann as ann
from skagent.grid import Grid
import skagent.model as model
from skagent.simulation.monte_carlo import draw_shocks
import torch
import skagent.utils as utils
from skagent.utils import extract_parameters, compute_parameter_difference

"""
Tools for the implementation of the Maliar, Maliar, and Winant (JME '21) method.

This method relies on a simpler problem representation than that elaborated
by the skagent Block system.

"""


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
    bellman_period,
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

    for t in range(big_t):
        # TODO: make sure block shocks are 'constructed'
        # TODO: allow option for 'structured' draws, e.g. from exact discretization.
        # this breaks the BP abstraction somewhat; BP should have a wrapper method
        shocks_t = draw_shocks(bellman_period.block.shocks, n=n)

        # this is cumbersome; probably can be solved deeper on the data structure level
        # note similarity to Grid.from_dict() reconciliation logic.
        states_template = states_t[next(iter(states_t.keys()))]
        shocks_t = {
            sym: utils.reconcile(states_template, shocks_t[sym]) for sym in shocks_t
        }

        controls_t = decision_function(states_t, shocks_t, parameters)

        states_t_plus_1 = bellman_period.transition_function(
            states_t, shocks_t, controls_t, parameters
        )
        states_t = states_t_plus_1

    return states_t_plus_1


def maliar_training_loop(
    bellman_period,
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
    bellman_period - a model definition
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
        Convergence tolerance. Training stops when either the L2 norm of parameter changes
        or the absolute difference in loss is below this threshold.
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

    bpn = ann.BlockPolicyNet(bellman_period, width=16)

    states = states_0_n  # V) Create initial panel of agents/starting states.

    # Step 2. Train the machine, i.e., ﬁnd θ that minimizes theempirical risk Xi^n (θ ):
    iteration = 0
    converged = False
    prev_loss = None

    while iteration < max_iterations and not converged:
        # Store current parameters before training
        prev_params = extract_parameters(bpn)

        # i). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        # TODO: this breaks the bellman period abstraction slightly. consider refactoring generate-givens
        # to use BP instead.
        givens = generate_givens_from_states(
            states_0_n, bellman_period.block, shock_copies
        )

        # ii). construct the gradient ∇ Xi^n (θ ) = 1n ni=1 ∇ ξ (ωi ; θ );
        # iii). update the coeﬃcients θ_hat = θ − λk ∇ Xi^n (θ ) and go to step 2.i);
        # TODO how many epochs? What Adam scale? Passing through variables
        bpn, current_loss = ann.train_block_nn(bpn, givens, loss_function, epochs=250)

        # Extract parameters after training
        curr_params = extract_parameters(bpn)

        # Check for parameter convergence
        param_diff = compute_parameter_difference(prev_params, curr_params)
        param_converged = param_diff < tolerance

        # Check for loss convergence
        loss_converged = False
        if prev_loss is not None:
            loss_diff = abs(current_loss - prev_loss)
            loss_converged = loss_diff < tolerance

        # Convergence if either parameter or loss criteria are met
        converged = param_converged or loss_converged

        # Logging
        if converged:
            if param_converged:
                logging.info(
                    f"Converged after {iteration + 1} iterations by parameters. Parameter difference: {param_diff:.2e}"
                )
            if loss_converged:
                logging.info(
                    f"Converged after {iteration + 1} iterations by loss. Loss difference: {loss_diff:.2e}"
                )
        else:
            log_msg = f"Iteration {iteration + 1}: Parameter difference: {param_diff:.2e}, Loss: {current_loss:.2e}"
            if prev_loss is not None:
                log_msg += f" (loss diff: {abs(current_loss - prev_loss):.2e})"
            logging.info(log_msg)

        # Update previous loss for next iteration
        prev_loss = current_loss

        # i/iv). simulate the model to produce data {ωi }ni=1 by using the decision rule ϕ (·, θ );
        # todo: same thing about breaking the BellmanPeriod abstraction
        next_states = simulate_forward(
            states,
            bellman_period,
            bpn.get_decision_function(),
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
    return bpn, states
