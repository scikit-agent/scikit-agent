"""
Tools for the implementation of the Maliar, Maliar, and Winant (JME '21) method.

This method relies on a simpler problem representation than that elaborated
by the skagent Block system.

.. note::
   ``generate_givens_from_states`` currently accesses ``bellman_period.block``
   directly rather than working through the BellmanPeriod interface. A future
   refactoring could route shock generation through BellmanPeriod itself.
   Similarly, shock draws are currently Monte Carlo only; structured draws
   (e.g. exact discretizations) could be supported via BellmanPeriod.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

import skagent.ann as ann
import skagent.block as block
import skagent.utils as utils
from skagent.grid import Grid
from skagent.simulation.monte_carlo import draw_shocks

if TYPE_CHECKING:
    from skagent.bellman import BellmanPeriod


def generate_givens_from_states(
    states: Grid, model_block: block.Block, shock_copies: int
) -> Grid:
    """
    Generate omega_i values of the MMW JME '21 method.

    Parameters
    ----------
    states : Grid
        A grid of starting state values (exogenous and endogenous).
    model_block : block.Block
        Block information (used to get the shock names).
    shock_copies : int
        Number of copies of the shocks to be included. Must be >= 1.

    Returns
    -------
    Grid
        Grid containing states augmented with shock copies.
    """
    n = states.n()
    new_shock_values = {}

    for i in range(shock_copies):
        shock_values = draw_shocks(model_block.shocks, n=n)
        new_shock_values.update(
            {f"{sym}_{i}": shock_values[sym] for sym in shock_values}
        )

    givens = states.update_from_dict(new_shock_values)

    return givens


def simulate_forward(
    states_t: Grid | dict,
    bellman_period: BellmanPeriod,
    decision_function: Callable,
    parameters: dict,
    big_t: int,
) -> dict:
    """
    Simulate the model forward for a specified number of periods.

    Parameters
    ----------
    states_t : Grid or dict
        Initial state values.
    bellman_period : BellmanPeriod
        The Bellman period containing model dynamics.
    decision_function : Callable
        Function mapping (states, shocks, parameters) to controls.
    parameters : dict
        Model parameters.
    big_t : int
        Number of time periods to simulate forward. If 0, returns the
        initial states unchanged.

    Returns
    -------
    dict
        Final state values after big_t periods.

    Raises
    ------
    ValueError
        If big_t < 0 or if states_t is an empty dict.
    """
    if big_t < 0:
        raise ValueError(f"big_t must be non-negative, got {big_t}")

    if isinstance(states_t, Grid):
        n = states_t.n()
        states_t = states_t.to_dict()
    else:
        if not states_t:
            raise ValueError("states_t cannot be an empty dict")
        n = len(states_t[next(iter(states_t.keys()))])

    if big_t == 0:
        return states_t

    for t in range(big_t):
        shocks_t = draw_shocks(bellman_period.block.shocks, n=n)

        # Reconcile shock dimensions with state dimensions (see Grid.from_dict())
        states_template = states_t[next(iter(states_t.keys()))]
        shocks_t = {
            sym: utils.reconcile(states_template, shocks_t[sym]) for sym in shocks_t
        }

        controls_t = decision_function(states_t, shocks_t, parameters)

        states_t = bellman_period.transition_function(
            states_t, controls_t, shocks=shocks_t, parameters=parameters
        )

    return states_t


def maliar_training_loop(
    bellman_period: BellmanPeriod,
    loss_function: Callable,
    states_0_n: Grid,
    parameters: dict,
    shock_copies: int = 2,
    max_iterations: int = 5,
    tolerance: float = 1e-6,
    random_seed: Optional[int] = None,
    simulation_steps: int = 1,
    network_width: int = 16,
    epochs_per_iteration: int = 250,
    value_network: Optional[object] = None,
    value_loss_function: Optional[Callable] = None,
) -> tuple:
    """
    Run the Maliar, Maliar, and Winant (JME '21) training loop.

    This implements the machine learning method for solving dynamic economic models
    by training a neural network policy to minimize empirical risk (loss). The
    network architecture (width, training epochs per iteration) is configurable
    via optional parameters.

    When ``value_network`` and ``value_loss_function`` are both provided, the loop
    trains the policy and value networks jointly using
    :func:`~skagent.ann.train_block_value_and_policy_nn`.  This is the
    Bellman-equation approach where both a policy and a value approximation are
    updated simultaneously (Maliar et al. 2021, Section 2.3).

    Parameters
    ----------
    bellman_period : BellmanPeriod
        A model definition containing block dynamics and transitions.
    loss_function : Callable
        The empirical risk function Xi^n from MMW JME'21. This function is
        passed to the neural network training routine as
        ``loss_function(decision_function, input_grid) -> loss_tensor``.
    states_0_n : Grid
        A panel of starting states for training. Must contain at least one state.
    parameters : dict
        Given parameters for the model.
    shock_copies : int, optional
        Number of copies of shocks to include in the training set omega.
        Must match expected number of shock copies in the loss function.
        Must be >= 1. Default is 2.
    max_iterations : int, optional
        Maximum number of training loop iterations before stopping.
        Must be >= 1. Default is 5.
    tolerance : float, optional
        Convergence tolerance. Training stops when either the L2 norm of
        parameter changes or the absolute difference in loss is below this
        threshold. Satisfying either criterion alone is sufficient.
        Must be > 0. Default is 1e-6.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    simulation_steps : int, optional
        Number of time steps to simulate forward when determining the next
        omega set for training. Higher values let the training states
        explore more of the state space at higher computational cost.
        Must be >= 1. Default is 1.
    network_width : int, optional
        Width of hidden layers in the policy neural network.
        Must be >= 1. Default is 16.
    epochs_per_iteration : int, optional
        Number of training epochs per iteration.
        Must be >= 1. Default is 250.
    value_network : BlockValueNet, optional
        A pre-constructed value network for joint training. When provided
        together with ``value_loss_function``, the loop trains both
        networks simultaneously.
    value_loss_function : Callable, optional
        Loss function for the value network. Signature matches
        ``value_loss_function(value_function, input_grid) -> loss_tensor``.
        Required when ``value_network`` is provided.

    Returns
    -------
    tuple
        Without joint training: ``(trained_policy_network, training_states)``.
        With joint training: ``(trained_policy_network, trained_value_network,
        training_states)``.

    Raises
    ------
    ValueError
        If max_iterations < 1, tolerance <= 0, shock_copies < 1,
        simulation_steps < 1, network_width < 1, epochs_per_iteration < 1,
        states_0_n contains no states, or only one of value_network /
        value_loss_function is provided.
    TypeError
        If bellman_period is None or loss_function is not callable.
    """
    # Validate object-type parameters
    if bellman_period is None:
        raise TypeError("bellman_period cannot be None")
    if not callable(loss_function):
        raise TypeError(
            f"loss_function must be callable, got {type(loss_function).__name__}"
        )
    if parameters is None:
        raise TypeError(
            "parameters cannot be None; pass an empty dict if no parameters are needed"
        )

    # Validate joint-training parameters
    joint_training = value_network is not None or value_loss_function is not None
    if joint_training:
        if value_network is None:
            raise ValueError(
                "value_loss_function provided without value_network. "
                "Both must be specified for joint training."
            )
        if value_loss_function is None:
            raise ValueError(
                "value_network provided without value_loss_function. "
                "Both must be specified for joint training."
            )
        if not callable(value_loss_function):
            raise TypeError(
                "value_loss_function must be callable, "
                f"got {type(value_loss_function).__name__}"
            )

    # Validate numeric parameters
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    if shock_copies < 1:
        raise ValueError(f"shock_copies must be >= 1, got {shock_copies}")
    if simulation_steps < 1:
        raise ValueError(f"simulation_steps must be >= 1, got {simulation_steps}")
    if network_width < 1:
        raise ValueError(f"network_width must be >= 1, got {network_width}")
    if epochs_per_iteration < 1:
        raise ValueError(
            f"epochs_per_iteration must be >= 1, got {epochs_per_iteration}"
        )
    if states_0_n.n() < 1:
        raise ValueError("states_0_n must contain at least one state")

    # Step 1. Initialize the algorithm:
    # i). Theoretical risk Xi(θ) = E_ω[ξ(ω;θ)] provided as loss_function
    # ii). Define neural network topology ϕ(·,θ)
    # iii). Fix initial coefficients θ

    if random_seed is not None:
        torch.manual_seed(random_seed)

    bpn = ann.BlockPolicyNet(bellman_period, width=network_width)
    states = states_0_n

    # Step 2. Train the machine to minimize empirical risk Xi^n(θ)
    iteration = 0
    converged = False
    prev_loss = None

    while iteration < max_iterations and not converged:
        # Store current parameters before training
        prev_params = utils.extract_parameters(bpn)

        # i). Simulate model to produce data {ω_i}_{i=1}^n using decision rule ϕ(·,θ)
        givens = generate_givens_from_states(states, bellman_period.block, shock_copies)

        # ii). Construct gradient ∇Xi^n(θ) and update coefficients
        if joint_training:
            bpn, value_network = ann.train_block_value_and_policy_nn(
                bpn,
                value_network,
                givens,
                loss_function,
                value_loss_function,
                epochs=epochs_per_iteration,
            )
            # Use policy loss for convergence tracking
            current_loss = 0.0  # joint training doesn't return a single loss
        else:
            bpn, current_loss = ann.train_block_nn(
                bpn, givens, loss_function, epochs=epochs_per_iteration
            )

        # Extract parameters after training
        curr_params = utils.extract_parameters(bpn)

        # Check for parameter convergence
        param_diff = utils.compute_parameter_difference(prev_params, curr_params)
        param_converged = param_diff < tolerance

        # Check for loss convergence (only for policy-only training)
        loss_converged = False
        if prev_loss is not None and not joint_training:
            loss_diff = abs(current_loss - prev_loss)
            loss_converged = loss_diff < tolerance

        # Convergence if either parameter or loss criteria are met
        converged = param_converged or loss_converged

        # Logging
        if converged:
            if param_converged:
                logging.info(
                    f"Converged after {iteration + 1} iterations by parameters. "
                    f"Parameter difference: {param_diff:.2e}"
                )
            if loss_converged:
                logging.info(
                    f"Converged after {iteration + 1} iterations by loss. "
                    f"Loss difference: {loss_diff:.2e}"
                )
        else:
            log_msg = (
                f"Iteration {iteration + 1}: "
                f"Parameter difference: {param_diff:.2e}, Loss: {current_loss:.2e}"
            )
            if prev_loss is not None and not joint_training:
                log_msg += f" (loss diff: {abs(current_loss - prev_loss):.2e})"
            logging.info(log_msg)

        # Update previous loss for next iteration
        prev_loss = current_loss

        # Simulate forward to get next omega set for training
        next_states = simulate_forward(
            states,
            bellman_period,
            bpn.get_decision_function(),
            parameters,
            simulation_steps,
        )

        # Detach from the computational graph so the next iteration's
        # backward pass does not attempt to traverse a freed graph.
        detached = {k: v.detach() for k, v in next_states.items()}
        states = Grid.from_dict(detached)
        iteration += 1

    if not converged:
        logging.warning(
            f"Training completed without convergence after {max_iterations} iterations."
        )

    # Step 3. Return trained approximation ϕ(·,θ) for accuracy assessment
    if joint_training:
        return bpn, value_network, states
    return bpn, states
