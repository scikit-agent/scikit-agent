import logging
import skagent.ann as ann
from skagent.grid import Grid
import skagent.block as model
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
    if big_t < 0:
        raise ValueError(f"big_t must be non-negative, got {big_t}")

    # When no forward simulation is requested, return initial states unchanged.
    if big_t == 0:
        if isinstance(states_t, Grid):
            return states_t.to_dict()
        return states_t

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
    nn_width=16,
    training_epochs=250,
    learning_rate=0.01,
    gradient_clip_norm=None,
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
        Convergence tolerance. Training stops when both the L2 norm of parameter changes
        and the absolute difference in loss are below this threshold. Requiring both
        criteria prevents premature termination when parameters stabilize but loss
        continues improving, or vice versa.
    simulation_steps : int
        The number of time steps to simulate forward when determining the next omega set for training
    nn_width : int
        Width of the hidden layers in the policy neural network. Default is 16.
    training_epochs : int
        Number of gradient descent epochs per outer iteration. Default is 250.
    learning_rate : float
        Learning rate for the Adam optimizer. Default is 0.01.
    gradient_clip_norm : float or None
        If not None, clips gradient norm to this value during training.
    """

    # Step 1. Initialize the algorithm:

    # i). construct theoretical risk Xi(θ ) = Eω [ξ (ω; θ )] (lifetime reward, Euler/Bellmanequations);
    # ii). deﬁne empirical risk Xi^n (θ ) = 1n ni=1 ξ (ωi ; θ );
    # loss_function is provided as an argument.

    # iii). deﬁne a topology of neural network ϕ (·, θ );
    # iv). ﬁx initial vector of the coeﬃcients θ .

    if random_seed is not None:
        torch.manual_seed(random_seed)

    bpn = ann.BlockPolicyNet(bellman_period, width=nn_width)

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
        givens = generate_givens_from_states(states, bellman_period.block, shock_copies)

        # ii). construct the gradient ∇ Xi^n (θ ) = 1n ni=1 ∇ ξ (ωi ; θ );
        # iii). update the coeﬃcients θ_hat = θ − λk ∇ Xi^n (θ ) and go to step 2.i);
        bpn, current_loss = ann.train_block_nn(
            bpn,
            givens,
            loss_function,
            epochs=training_epochs,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
        )

        # Extract parameters after training
        curr_params = extract_parameters(bpn)

        # Check for parameter convergence
        param_diff = compute_parameter_difference(prev_params, curr_params)
        param_converged = param_diff < tolerance

        # Check for loss convergence (skip on first iteration when no prior loss exists)
        if prev_loss is not None:
            loss_diff = abs(current_loss - prev_loss)
            loss_converged = loss_diff < tolerance
        else:
            loss_converged = True

        # Convergence requires both parameter and loss criteria to be met
        converged = param_converged and loss_converged

        # Logging
        if converged:
            if param_converged:
                logging.info(
                    f"Converged after {iteration + 1} iterations by parameters. Parameter difference: {param_diff:.2e}"
                )
            if loss_converged and prev_loss is not None:
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

        # Detach tensors from the computation graph so each outer iteration trains
        # fresh, without gradient dependencies on previous iterations' forward passes.
        # Without this, PyTorch would attempt to backpropagate through the entire
        # simulation history, causing "backward through the graph a second time" errors.
        detached_states = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in next_states.items()
        }
        states = Grid.from_dict(detached_states)
        iteration += 1

    if not converged:
        logging.warning(
            f"Training completed without convergence after {max_iterations} iterations. "
            f"Final parameter difference: {param_diff:.2e}, "
            f"Final loss: {current_loss:.2e} (tolerance: {tolerance:.2e})."
        )

    # Step 3. Assess the accuracy of constructed approximation ϕ (·, θ ) on a new sample.
    return bpn, states, converged
