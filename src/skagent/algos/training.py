import numpy as np
import torch
from skagent.grid import Grid
import skagent.model as model
from skagent.simulation.monte_carlo import draw_shocks
import skagent.utils as utils
import skagent.ann as ann
from inspect import signature
from skagent.model import Control, construct_shocks

"""
Training utilities for neural network-based economic models.

This module provides training grid generation and simulation functions used in
the Maliar, Maliar, and Winant (2021) framework for solving dynamic economic
models with neural networks.

Key Functions
=============

Training Grid Generation:
- `generate_bellman_training_grid()`: Creates 2-shock grids for Bellman training
- `generate_euler_training_grid()`: Creates 2-shock grids for Euler training (alias)
- `generate_ergodic_training_grid()`: Automatic ergodic sampling for AR(1) processes

Simulation and Training:
- `generate_givens_from_states()`: Generates omega_i values for MMW method
- `simulate_forward()`: Forward simulation functionality
- `maliar_training_loop()`: Complete MMW training loop implementation

Grid Types
===========

1. **2-Shock Grids**: Used by Bellman and Euler methods
   - State variables: Cartesian product over specified ranges
   - Shock variables: Independent draws for periods t and t+1
   - Computationally efficient for expectation computation

2. **Ergodic Grids**: Used for MMW reproduction
   - Automatically detects AR(1) processes from DBlock dynamics
   - Samples from stationary distributions
   - Handles bounded state variables

3. **Multi-Shock Grids**: Used by EDLR method
   - Supports big_t shock copies for forward simulation
   - Required for Expected Discounted Lifetime Reward computation

Usage
=====

The training functions integrate with the MMW loss functions to provide complete
training infrastructure for neural network-based economic models.

Example:
--------
```python
from skagent.algos.training import generate_bellman_training_grid, maliar_training_loop
from skagent.algos.maliar import get_bellman_residual_loss

# Generate training grid
state_config = {"assets": {"min": 0.1, "max": 10.0, "count": 50}}
training_grid = generate_bellman_training_grid(
    state_config, block, parameters=calibration
)

# Create loss function
bellman_loss = get_bellman_residual_loss(["assets"], block, 0.96, calibration)

# Run complete training loop
trained_ann, final_states = maliar_training_loop(
    block, bellman_loss, training_grid, calibration, max_iterations=10
)
```

References
==========
Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic
economic models. Journal of Monetary Economics, 122, 76-101.
"""


def generate_bellman_training_grid(state_config, block, n_samples=100, parameters=None):
    """
    Generate training grid for Bellman residual loss function.

    Creates a Cartesian product grid of state variables combined with 2 copies
    of shock realizations (for periods t and t+1) as required by the Bellman
    equation approach. This grid structure enables proper computation of
    expectations E[V(s_{t+1}) | s_t, c_t] in the Bellman residual loss.

    Grid Structure:
    - State variables: Cartesian product over specified ranges
    - Shock variables: Independent draws with suffixes "_0" (period t) and "_1" (period t+1)
    - Total grid points: product of state counts

    The 2-shock structure is computationally efficient compared to the EDLR
    approach which requires big_t shock copies for forward simulation.

    Parameters
    -----------
    state_config : dict
        Grid configuration for state variables. Each key is a variable name,
        each value is a dict with "min", "max", "count" specifying the grid range.
        Example: {"assets": {"min": 0.1, "max": 10.0, "count": 50}}
    block : DBlock
        Model block containing shock distribution information via get_shocks().
        Shock distributions will be used to draw realizations for periods t and t+1.
    n_samples : int, optional
        Number of shock samples to draw for each state point. Default is 100.
        Note: Currently not used as we draw one shock per state point.
    parameters : dict, optional
        Model calibration parameters required for constructing shock distributions
        when block contains shock specifications as tuples. If None and block
        requires parameters, will raise an error.

    Returns
    --------
    Grid
        Training grid with combined state variables and shock realizations.
        Grid.labels contains all variable names including shock suffixes.
        Grid.values contains the full training data as a PyTorch tensor.
    """
    # Create base state grid
    states_grid = Grid.from_config(state_config)

    # Get number of state points
    n_states = len(states_grid.values)

    # Get device from states_grid
    device = (
        states_grid.values.device
        if hasattr(states_grid.values, "device")
        else torch.device("cpu")
    )

    # Draw shocks for t and t+1
    shock_vars = block.get_shocks()
    new_shock_values = {}

    # Only add shocks if the block has any
    if shock_vars:
        # Construct actual distributions if needed (handle tuples)
        if any(isinstance(shock_vars[var], tuple) for var in shock_vars):
            # Need calibration parameters to construct shocks
            if parameters is None:
                raise ValueError(
                    "Block contains shock specifications that require calibration parameters, "
                    "but no parameters were provided. Please pass parameters argument with "
                    "calibration values needed to construct shock distributions."
                )
            constructed_shocks = construct_shocks(shock_vars, parameters)
        else:
            constructed_shocks = shock_vars

        for period in [0, 1]:  # t and t+1
            shocks = draw_shocks(
                constructed_shocks, np.zeros(n_states)
            )  # conditions not used for most shocks
            for shock_name, shock_values in shocks.items():
                new_shock_values[f"{shock_name}_{period}"] = torch.tensor(
                    shock_values, dtype=torch.float32, device=device
                )

    # Combine state grid with shock values
    states_dict = states_grid.to_dict()
    combined_dict = {**states_dict, **new_shock_values}

    # Create new grid with combined data
    combined_labels = list(combined_dict.keys())
    combined_values = torch.stack([combined_dict[label] for label in combined_labels]).T

    # Create a new Grid-like structure
    result_grid = Grid(combined_labels, combined_values)

    return result_grid


def generate_euler_training_grid(state_config, block, n_samples=100, parameters=None):
    """
    Generate training grid for Euler residual loss function.

    This is an alias for generate_bellman_training_grid() since both Euler and
    Bellman methods use the same 2-shock structure (periods t and t+1).

    The grid contains:
    - State variables: Cartesian product over specified ranges
    - Shock variables: Independent draws with suffixes "_0" (period t) and "_1" (period t+1)

    This efficient 2-shock structure is computationally superior to the EDLR
    approach which requires big_t shock copies for forward simulation.

    Parameters
    -----------
    state_config : dict
        Grid configuration for state variables. Each key is a variable name,
        each value is a dict with "min", "max", "count" specifying the grid range.
        Example: {"assets": {"min": 0.1, "max": 10.0, "count": 50}}
    block : DBlock
        Model block containing shock distribution information via get_shocks().
        Shock distributions will be used to draw realizations for periods t and t+1.
    n_samples : int, optional
        Number of shock samples to draw for each state point. Default is 100.
        Note: Currently not used as we draw one shock per state point.
    parameters : dict, optional
        Model calibration parameters required for constructing shock distributions
        when block contains shock specifications as tuples. If None and block
        requires parameters, will raise an error.

    Returns
    --------
    Grid
        Training grid with combined state variables and shock realizations.
        Grid.labels contains all variable names including shock suffixes.
        Grid.values contains the full training data as a PyTorch tensor.

    See Also
    --------
    generate_bellman_training_grid : Same function for Bellman method
    """
    return generate_bellman_training_grid(state_config, block, n_samples, parameters)


def generate_givens_from_states(states: Grid, block: model.Block, shock_copies: int):
    """
    Generates omega_i values of the MMW JME '21 method.

    Creates training data by combining state variables with multiple copies of
    shock realizations. This is used in the MMW training loop to generate the
    empirical risk function inputs.

    Parameters
    -----------
    states : Grid
        A grid of starting state values (exogenous and endogenous)
    block : model.Block
        Block information (used to get the shock names and distributions)
    shock_copies : int
        Number of copies of the shocks to be included in the training data

    Returns
    --------
    Grid
        Extended grid with state variables and shock copies for training
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
    decision_function,
    parameters,
    big_t,
):
    """
    Simulate economic model forward in time using given decision function.

    Performs forward simulation of the economic model by iteratively applying
    the transition function with stochastic shocks and policy decisions. This
    is used in the MMW training loop to generate new training states.

    Parameters
    -----------
    states_t : Grid or dict
        Initial state variables
    block : model.Block
        Economic model block containing dynamics
    decision_function : callable
        Policy function that maps (states, shocks, parameters) -> controls
    parameters : dict
        Model calibration parameters
    big_t : int
        Number of time steps to simulate forward

    Returns
    --------
    dict
        Final state variables after big_t periods of simulation
    """
    if isinstance(states_t, Grid):
        n = states_t.n()
        states_t = states_t.to_dict()
    else:
        # kludge
        n = len(states_t[next(iter(states_t.keys()))])

    state_syms = list(states_t.keys())

    # Import create_transition_function from parent module
    from . import maliar

    tf = maliar.create_transition_function(block, state_syms)

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
    Complete MMW JME '21 training loop implementation.

    Implements the full Maliar training algorithm with iterative neural network
    training and forward simulation. This follows the MMW JME '21 Algorithm 1
    for solving dynamic economic models with neural networks.

    Algorithm Steps:
    1. Initialize neural network policy
    2. For each iteration:
       a. Generate training data (omega_i values)
       b. Train neural network on empirical risk
       c. Simulate forward to get new states
       d. Check convergence (optional)
    3. Return trained network and final states

    Parameters
    -----------
    block : DBlock
        A model definition containing dynamics, shocks, and rewards
    loss_function : callable
        Loss function with signature (policy_function, input_grid) -> loss_tensor
        This is the "empirical risk Xi^n" in MMW JME'21
    states_0_n : Grid
        A panel of starting states for the agents
    parameters : dict
        Given parameters for the model (calibration)
    shock_copies : int, optional
        Number of copies of shocks to include in the training set omega.
        Must match expected number of shock copies in the loss function.
        Default is 2.
    max_iterations : int, optional
        Number of times to perform the training loop if no convergence.
        Default is 5.
    random_seed : int, optional
        Random seed for reproducible results. If None, no seed is set.
    simulation_steps : int, optional
        The number of time steps to simulate forward when determining the
        next omega set for training. Default is 1.

    Returns
    --------
    tuple[ann.BlockPolicyNet, Grid]
        - Trained neural network policy
        - Final states after training iterations
    """
    # Step 1. Initialize the algorithm:

    # i). construct theoretical risk Xi(θ) = Eω [ξ (ω; θ)] (lifetime reward, Euler/Bellman equations);
    # ii). define empirical risk Xi^n (θ) = 1/n Σ_i=1^n ξ (ωi ; θ);
    loss_function  # This is provided as an argument.

    # iii). define a topology of neural network φ (·, θ);
    # iv). fix initial vector of the coefficients θ.

    if random_seed is not None:
        torch.manual_seed(random_seed)

    bpn = ann.BlockPolicyNet(block, width=16)

    states = states_0_n  # V) Create initial panel of agents/starting states.

    # Step 2. Train the machine, i.e., find θ that minimizes the empirical risk Xi^n (θ):
    for i in range(max_iterations):
        # i). simulate the model to produce data {ωi}_{i=1}^n by using the decision rule φ (·, θ);
        givens = generate_givens_from_states(states_0_n, block, shock_copies)

        # ii). construct the gradient ∇ Xi^n (θ) = 1/n Σ_i=1^n ∇ ξ (ωi ; θ);
        # iii). update the coefficients θ_hat = θ − λk ∇ Xi^n (θ) and go to step 2.i);
        # TODO how many epochs? What Adam scale? Passing through variables
        ann.train_block_policy_nn(bpn, givens, loss_function, epochs=250)

        # i/iv). simulate the model to produce data {ωi}_{i=1}^n by using the decision rule φ (·, θ);
        next_states = simulate_forward(
            states, block, bpn.get_decision_function(), parameters, simulation_steps
        )

        states = Grid.from_dict(next_states)

        # End Step 2 if the convergence criterion || θ_hat − θ || < ε is satisfied.
        # TODO: test for difference.. how? This affects the FOR (/while) loop above.

    # Step 3. Assess the accuracy of constructed approximation φ (·, θ) on a new sample.
    return bpn, states


def generate_ergodic_training_grid(
    block, calibration, n_samples=1000, state_bounds=None
):
    """
    Generate training grid by sampling from ergodic distributions of AR(1) processes.

    This function automatically detects AR(1) processes from DBlock dynamics and samples
    from their stationary distributions. It infers persistence and volatility parameters
    from the function signatures and shock specifications, making it fully generic to
    any DBlock configuration.

    This is particularly useful for MMW-style reproduction where training data should
    be sampled from the ergodic distributions of exogenous processes rather than
    using fixed grids.

    Parameters
    -----------
    block : DBlock
        Economic model block containing dynamics and shock specifications.
        The function analyzes the dynamics to detect AR(1) patterns of the form:
        x_{t+1} = ρ * x_t + ε_{t+1}
    calibration : dict
        Model parameters including persistence (ρ) and volatility (σ) parameters.
        Parameter names are inferred from function signatures.
    n_samples : int, optional
        Number of samples to draw from each ergodic distribution. Default is 1000.
    state_bounds : dict, optional
        Bounds for non-AR(1) state variables (e.g., wealth, capital).
        Format: {"var_name": (min_val, max_val)}
        These variables will be sampled uniformly within the specified bounds.

    Returns
    --------
    Grid
        Training grid with samples from ergodic distributions.
        AR(1) variables are sampled from N(0, σ_ergodic) where σ_ergodic = σ/(1-ρ²)^0.5
        Bounded variables are sampled uniformly within specified bounds.

    Examples
    --------
    >>> # MMW consumption-savings model
    >>> mmw_grid = generate_ergodic_training_grid(
    ...     block=mmw_block,
    ...     calibration={"rho_r": 0.2, "sigma_r": 0.001, "rho_delta": 0.2, ...},
    ...     n_samples=1000,
    ...     state_bounds={"w": (0.1, 4.0)}  # Wealth bounds from MMW
    ... )

    Notes
    -----
    The function detects AR(1) processes by analyzing function signatures in block.dynamics.
    A variable is considered AR(1) if:
    1. The function depends on the variable itself (lagged value)
    2. The function depends on a shock variable
    3. The function has a parameter that appears in the calibration (persistence)

    For the MMW notebook reproduction, this eliminates the need for manual specification
    of ergodic standard deviations and makes the sampling fully automatic.
    """
    # Get dynamics and shocks from the block
    dynamics = block.get_dynamics()
    shock_vars = block.get_shocks()
    constructed_shocks = construct_shocks(shock_vars, calibration)

    ergodic_data = {}

    # Analyze dynamics to detect AR(1) processes
    for var_name, var_func in dynamics.items():
        if isinstance(var_func, Control):
            continue  # Skip control variables

        # Get function signature to analyze dependencies
        sig = signature(var_func)
        param_names = list(sig.parameters.keys())

        # Check if this looks like an AR(1) process
        # Pattern: f(x_t, shock, persistence_param, ...)
        has_self_dependence = var_name in param_names
        shock_param = None
        persistence_param = None

        # Find shock parameter
        for param in param_names:
            if param in constructed_shocks:
                shock_param = param
                break

        # Find persistence parameter (appears in calibration but isn't the variable itself)
        for param in param_names:
            if param in calibration and param != var_name and param != shock_param:
                # Check if this looks like a persistence parameter (typically 0 < ρ < 1)
                param_value = calibration[param]
                if isinstance(param_value, (int, float)) and 0 < abs(param_value) < 1:
                    persistence_param = param
                    break

        # If we found AR(1) pattern, compute ergodic distribution
        if has_self_dependence and shock_param and persistence_param:
            rho = calibration[persistence_param]

            # Get shock distribution properties
            shock_dist = constructed_shocks[shock_param]

            # Extract volatility from shock distribution
            if hasattr(shock_dist, "sigma"):
                sigma = (
                    shock_dist.sigma()
                    if callable(shock_dist.sigma)
                    else shock_dist.sigma
                )
            elif hasattr(shock_dist, "std"):
                sigma = shock_dist.std() if callable(shock_dist.std) else shock_dist.std
            elif hasattr(shock_dist, "scale"):
                sigma = (
                    shock_dist.scale()
                    if callable(shock_dist.scale)
                    else shock_dist.scale
                )
            else:
                # Try to get from distribution parameters
                if hasattr(shock_dist, "args") and len(shock_dist.args) >= 2:
                    sigma = shock_dist.args[1]  # Second argument often std dev
                else:
                    print(
                        f"Warning: Could not extract volatility for {shock_param}, using default 0.01"
                    )
                    sigma = 0.01

            # Compute ergodic standard deviation: σ_ergodic = σ / √(1 - ρ²)
            if abs(rho) < 1:  # Ensure stationarity
                ergodic_std = sigma / (1 - rho**2) ** 0.5
                ergodic_data[var_name] = torch.randn(n_samples) * ergodic_std
                print(
                    f"Detected AR(1): {var_name} ~ N(0, {ergodic_std:.6f}) [ρ={rho}, σ={sigma}]"
                )
            else:
                print(
                    f"Warning: Non-stationary AR(1) process detected for {var_name} (ρ={rho})"
                )

    # Add bounded state variables (like wealth, capital, etc.)
    if state_bounds:
        for var_name, (min_val, max_val) in state_bounds.items():
            if var_name not in ergodic_data:  # Don't override AR(1) variables
                ergodic_data[var_name] = (
                    torch.rand(n_samples) * (max_val - min_val) + min_val
                )
                print(f"Bounded variable: {var_name} ~ Uniform({min_val}, {max_val})")

    # Add current period shock realizations if needed
    for shock_name, shock_dist in constructed_shocks.items():
        if shock_name not in ergodic_data:  # Don't override if already sampled
            # Draw from shock distribution
            try:
                shock_draws = draw_shocks({shock_name: shock_dist}, np.zeros(n_samples))
                ergodic_data[shock_name] = torch.tensor(
                    shock_draws[shock_name], dtype=torch.float32
                )
                print(f"Shock variable: {shock_name} sampled from distribution")
            except Exception as e:
                print(f"Warning: Could not sample {shock_name}: {e}")

    if not ergodic_data:
        raise ValueError(
            "No variables detected for ergodic sampling. Check block dynamics and calibration."
        )

    print(
        f"Generated ergodic training grid with {n_samples} samples and {len(ergodic_data)} variables"
    )

    # Create Grid using the correct constructor
    # Convert dict of tensors to Grid format
    var_names = list(ergodic_data.keys())
    values_tensor = torch.stack([ergodic_data[name] for name in var_names], dim=0).T
    
    # Create proper Grid object
    return Grid(var_names, values_tensor)


def train_block_policy_nn(
    policy_net,
    loss_function,
    epochs,
    input_grid,
    lr=1e-3,
    print_loss=True,
    loss_frequency=10,
    gpu=True,
    stochastic=False,
    batch_size=None,
):
    """
    Train a neural network policy using a given loss function.

    This is the primary training routine for neural network policies in skagent.
    It provides a standardized interface for optimizing policy functions using
    various loss formulations (Euler, Bellman, etc.).

    Parameters
    ----------
    policy_net : nn.Module
        Neural network policy to train. Should implement a decision_function method.
    loss_function : callable
        Loss function with signature (policy_function, grid) -> loss_tensor
    epochs : int
        Number of training epochs
    input_grid : Grid or callable
        Training data grid. If stochastic=True, can be a callable that generates
        fresh training batches when called.
    lr : float, optional
        Learning rate for Adam optimizer (default: 1e-3)
    print_loss : bool, optional
        Whether to print loss during training (default: True)
    loss_frequency : int, optional
        Frequency of loss printing in epochs (default: 10)
    gpu : bool, optional
        Whether to use GPU if available (default: True)
    stochastic : bool, optional
        If True, input_grid should be a callable that generates fresh training
        batches each epoch. If False, uses fixed input_grid. (default: False)
    batch_size : int, optional
        Batch size for stochastic training. Only used if stochastic=True.

    Returns
    -------
    list
        List of loss values over training epochs

    Examples
    --------
    Standard fixed grid training:
    >>> train_block_policy_nn(policy_net, loss_fn, 1000, fixed_grid)
    
    Stochastic training with fresh batches:
    >>> def batch_generator():
    ...     return generate_fresh_batch(batch_size=128)
    >>> train_block_policy_nn(policy_net, loss_fn, 1000, batch_generator, 
    ...                      stochastic=True, batch_size=128)
    """
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    import time

    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    policy_net = policy_net.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    losses = []

    # Training loop
    if print_loss:
        print(f"Training for {epochs} epochs on {device}...")
        
    start_time = time.time()
    
    # Use tqdm for progress bar
    epoch_iterator = tqdm(range(epochs), desc="Training") if print_loss else range(epochs)
    
    for epoch in epoch_iterator:
        optimizer.zero_grad()
        
        # Generate training data
        if stochastic:
            # Generate fresh batch each epoch
            if callable(input_grid):
                current_grid = input_grid()
            else:
                raise ValueError("When stochastic=True, input_grid must be callable")
        else:
            # Use fixed grid
            current_grid = input_grid
        
        # Ensure grid is on the correct device
        if hasattr(current_grid, 'to'):
            current_grid = current_grid.to(device)
        
        # Compute loss
        loss = loss_function(policy_net.decision_function, current_grid)
        
        # Handle loss tensor reduction
        if hasattr(loss, 'mean'):
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_item = loss.item() if hasattr(loss, 'item') else float(loss)
        losses.append(loss_item)
        
        # Print progress
        if print_loss and epoch % loss_frequency == 0:
            if hasattr(epoch_iterator, 'set_postfix'):
                epoch_iterator.set_postfix(loss=f"{loss_item:.6f}")
            else:
                print(f"Epoch {epoch}: Loss = {loss_item:.6f}")
    
    train_time = time.time() - start_time
    
    if print_loss:
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Final loss: {losses[-1]:.6f}")
    
    return losses
