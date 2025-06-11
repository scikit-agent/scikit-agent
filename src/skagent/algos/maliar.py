import numpy as np
import torch
from skagent.grid import Grid
from skagent.simulation.monte_carlo import draw_shocks

"""
Implementation of Maliar, Maliar, and Winant (Journal of Monetary Economics, 2021) Methods.

This module implements the complete MMW JME '21 framework for solving dynamic economic 
models using neural networks and all-in-one (AiO) loss functions. The MMW approach 
provides three fundamental loss function formulations for training neural network 
approximations of policy and value functions.

MMW JME '21 All-in-One Loss Functions
====================================

The MMW JME '21 paper "Deep learning for solving dynamic economic models" introduces 
three fundamental all-in-one (AiO) loss functions for solving dynamic economic models:

1. **Expected Discounted Lifetime Reward (EDLR)** - Primary Method
   - Mathematical foundation (MMW Definition 2.4):
     Ξ(θ) = E_ω[∑_{t=0}^{T} β^t r(m_t, s_t, φ(m_t, s_t; θ))]
   - Forward simulation approach maximizing expected lifetime utility
   - Uses big_t shock copies for proper integration over shock sequences
   - AiO integration: Single composite Monte Carlo draws for both shock integration
     and decision function approximation
   - Implemented in: get_expected_discounted_lifetime_reward_loss()

2. **Bellman Residual Minimization** - Alternative Method  
   - Mathematical foundation (MMW Section 2.4):
     Loss = |V(s_t) - [u(s_t,c_t) + β * E[V(s_{t+1}) | s_t,c_t]]|²
   - Joint training of policy and value functions via Bellman equation violations
   - Computationally efficient: uses only 2 shock copies (t and t+1)
   - Theoretical compliance: Direct implementation of Bellman equation residuals
   - Implemented in: get_bellman_residual_loss()

3. **Euler Residual Minimization** - Future Implementation
   - Would minimize first-order condition violations (MMW Section 2.3)
   - Uses 2 shock copies like Bellman approach
   - Not yet implemented in this module

Method Comparison
=================

| Method            | Shock Copies | Computational Cost | Theoretical Foundation     | Status          |
|-------------------|-------------|-------------------|---------------------------|----------------|
| EDLR (Primary)    | big_t       | Higher            | Direct utility maximization| ✅ Implemented |
| Bellman Residual  | 2           | Lower             | Bellman equation          | ✅ Implemented |
| Euler Residual    | 2           | Lower             | First-order conditions    | ❌ Not implemented |

Key Features
============

- **Theoretical Compliance**: Direct implementation of MMW JME '21 mathematical formulations
- **Computational Efficiency**: Optimized shock handling and grid generation
- **Neural Network Integration**: Seamless integration with PyTorch networks via skagent.ann
- **Block System Compatibility**: Works with existing skagent DBlock economic models
- **Device Handling**: Automatic CUDA/CPU tensor management

Core Functions
==============

Training Grid Generation:
- generate_bellman_training_grid(): Creates 2-shock grids for Bellman training

Loss Function Factories:
- get_expected_discounted_lifetime_reward_loss(): Primary EDLR method (MMW main approach)
- get_bellman_residual_loss(): Alternative Bellman method (MMW Section 2.4)

Helper Functions:
- create_transition_function(): State transition dynamics T(s_t, ε_t, c_t)
- create_decision_function(): Policy function wrappers
- create_reward_function(): Reward function wrappers  
- estimate_discounted_lifetime_reward(): Forward simulation engine

Usage Examples
==============

Primary EDLR Method (MMW Main Approach)
---------------------------------------

```python
from skagent.algos.maliar import get_expected_discounted_lifetime_reward_loss
from skagent.ann import BlockPolicyNet, train_block_policy_nn

# Create EDLR loss function (main MMW method)
edlr_loss = get_expected_discounted_lifetime_reward_loss(
    state_variables=['m', 'a'], 
    block=consumption_block,
    discount_factor=0.96,
    big_t=100,
    parameters=calibration
)

# Train policy network
policy_net = BlockPolicyNet(consumption_block)
trained_net = train_block_policy_nn(
    policy_net, training_grid, edlr_loss, epochs=1000
)
```

Alternative Bellman Method
-------------------------

```python
from skagent.algos.maliar import get_bellman_residual_loss
from skagent.ann import BlockPolicyNet, BlockValueNet, train_bellman_nets

# Create Bellman loss function (MMW alternative)
bellman_loss = get_bellman_residual_loss(
    state_variables=['m', 'a'],
    block=consumption_block, 
    discount_factor=0.96,
    parameters=calibration
)

# Train both policy and value networks
policy_net = BlockPolicyNet(consumption_block)
value_net = BlockValueNet(consumption_block, ['m', 'a'])
trained_policy, trained_value = train_bellman_nets(
    policy_net, value_net, training_grid, bellman_loss, epochs=1000
)
```

Testing and Validation
======================

Comprehensive tests are provided in tests/test_bellman_loss.py covering:
- Loss function computation and numerical stability
- Grid generation with 2-shock structure for Bellman training
- Neural network training convergence
- Comparison between EDLR and Bellman approaches
- Integration with skagent DBlock models

Run tests with: pytest tests/test_bellman_loss.py

References
==========

Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic 
economic models. Journal of Monetary Economics, 122, 76-101.
Original paper: https://web.stanford.edu/~maliars/Files/JME2021.pdf

Implementation Notes
===================

This implementation uses a simpler problem representation than the full skagent 
Block system to maintain compatibility with the original MMW formulations while
leveraging the skagent infrastructure for model definition and simulation.

The implementation provides both the **primary MMW EDLR method** and the 
**alternative Bellman method**, enabling researchers to use the complete MMW JME '21 
methodology for solving dynamic economic models with neural networks.
"""


def create_transition_function(block, state_syms):
    """
    Create a transition function from a block.
    
    Parameters
    -----------
    block : DBlock
        The economic model block
    state_syms : list of str
        A list of symbols for 'state variables at time t', aka arrival states.
        
    Returns
    --------
    callable
        Transition function that computes state variables at t+1
    """

    def transition_function(states_t, shocks_t, controls_t, parameters):
        vals = parameters | states_t | shocks_t | controls_t
        post = block.transition(vals, {}, fix=list(controls_t.keys()))

        return {sym: post[sym] for sym in state_syms}

    return transition_function


def create_decision_function(block, decision_rules):
    """
    Create a decision function from decision rules.
    
    Parameters
    -----------
    block : DBlock
        The economic model block
    decision_rules : dict
        Dictionary mapping control variable names to decision rules
        
    Returns
    --------
    callable
        Decision function that returns control variable values
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
    Create a reward function from a block.
    
    Parameters
    -----------
    block : DBlock
        The economic model block
    agent : str, optional
        Name of reference agent for rewards. If None, uses all rewards.
        
    Returns
    --------
    callable
        Reward function that computes rewards for given states and controls
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
    Estimate discounted lifetime reward by forward simulation.
    
    Parameters
    -----------
    block : DBlock
        The economic model block
    discount_factor : float
        Discount factor (currently only numerical values supported)
    dr : dict or callable
        Decision rules (dict of functions), or a decision function 
    states_0 : dict
        Initial states, mapping symbols to values
    big_t : int
        Number of time steps to simulate forward
    shocks_by_t : dict, optional
        Dictionary mapping shock symbols to big_t vectors of shock values
    parameters : dict, optional
        Calibration parameters
    agent : str, optional
        Name of reference agent for rewards
        
    Returns
    --------
    float or torch.Tensor
        Estimated discounted lifetime reward
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

    # this assumes only one reward is given.
    # can be generalized in the future.
    rsym = list(
        {sym for sym in block.reward if agent is None or block.reward[sym] == agent}
    )[0]

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

        # assumes torch
        if isinstance(reward_t[rsym], torch.Tensor) and torch.any(
            torch.isnan(reward_t[rsym])
        ):
            raise Exception(f"Calculated reward {[rsym]} is NaN: {reward_t}")
        if isinstance(reward_t[rsym], np.ndarray) and np.any(np.isnan(reward_t[rsym])):
            raise Exception(f"Calculated reward {[rsym]} is NaN: {reward_t}")

        total_discounted_reward += reward_t[rsym] * discount_factor**t

        # t + 1
        states_t = tf(states_t, shocks_t, controls_t, parameters)

    return total_discounted_reward


def generate_bellman_training_grid(state_config, block, n_samples=100):
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
        
    Returns
    --------
    Grid
        Training grid with combined state variables and shock realizations.
        Grid.labels contains all variable names including shock suffixes.
        Grid.values contains the full training data as a PyTorch tensor.
    """
    # Create base state grid
    states_grid = Grid(state_config)
    
    # Get number of state points
    n_states = len(states_grid.values)
    
    # Get device from states_grid
    device = states_grid.values.device if hasattr(states_grid.values, 'device') else torch.device('cpu')
    
    # Draw shocks for t and t+1
    shock_vars = block.get_shocks()
    new_shock_values = {}
    
    # Only add shocks if the block has any
    if shock_vars:
        for period in [0, 1]:  # t and t+1
            shocks = draw_shocks(shock_vars, np.zeros(n_states))  # conditions not used for most shocks
            for shock_name, shock_values in shocks.items():
                new_shock_values[f"{shock_name}_{period}"] = torch.tensor(shock_values, dtype=torch.float32, device=device)
    
    # Combine state grid with shock values
    states_dict = states_grid.to_dict()
    combined_dict = {**states_dict, **new_shock_values}
    
    # Create new grid with combined data
    combined_labels = list(combined_dict.keys())
    combined_values = torch.stack([combined_dict[label] for label in combined_labels]).T
    
    # Create a new Grid-like structure  
    result_grid = Grid(state_config)  # Start with state config
    result_grid.labels = combined_labels
    result_grid.values = combined_values
    
    return result_grid


def get_bellman_residual_loss(state_variables, block, discount_factor, parameters, disc_params=None):
    """
    Creates a Bellman equation all-in-one loss function for the Maliar method.
    
    This implements the alternative MMW JME '21 approach (Section 2.4) using 
    Bellman equation residuals instead of the primary EDLR method. This approach
    jointly trains policy and value functions by minimizing violations of the
    Bellman equation.
    
    Mathematical Foundation (MMW JME '21, Section 2.4):
    The Bellman equation for dynamic programming is:
    V(s_t) = u(s_t, c_t) + β * E[V(s_{t+1}) | s_t, c_t]
    
    This function creates a loss that minimizes Bellman residuals:
    Loss = |V(s_t) - [u(s_t,c_t) + β * E[V(s_{t+1}) | s_t,c_t]]|²
    
    Where:
    - V(s_t) is the value function approximation at state s_t
    - u(s_t, c_t) is the period utility function
    - c_t = φ(s_t, ε_t; θ) is the policy function output
    - β is the discount factor
    - E[·] is the expectation over next period shocks
    
    Key Advantages vs EDLR:
    - Computationally efficient: Uses only 2 shock copies (t and t+1) vs big_t
    - Theoretically grounded: Direct implementation of Bellman optimality
    - Joint optimization: Trains both policy and value functions simultaneously
    
    Parameters
    -----------
    state_variables : list of str
        Names of state variables that define the value function domain.
        These variables determine the input space for the value function V(s).
    block : DBlock
        The economic model block containing dynamics, shocks, and rewards.
        Must define transition function T(s_t, ε_t, c_t) and reward function u(·).
    discount_factor : float
        Discount factor β ∈ (0,1) from the Bellman equation
    parameters : dict
        Model calibration parameters used in dynamics and reward computations
    disc_params : dict, optional
        Discretization parameters for shock distributions when computing expectations.
        If None, uses Monte Carlo integration with provided shock realizations.
        
    Returns
    --------
    callable
        Bellman residual loss function with signature:
        (policy_function, value_function, input_grid) -> loss_tensor
        
        The returned function computes squared Bellman residuals for each point
        in the input grid, enabling batch training of neural networks.
    """
    from skagent.model import discretized_shock_dstn
    from HARK.distributions import expected
    
    # Set up shock handling - use 2 copies of shocks (t and t+1)
    shock_vars = block.get_shocks()
    shock_syms_t = [f"{sym}_0" for sym in shock_vars.keys()]
    shock_syms_t1 = [f"{sym}_1" for sym in shock_vars.keys()]
    
    # Create helper functions
    tf = create_transition_function(block, state_variables)
    rf = create_reward_function(block)
    
    # Get the reward symbol (assumes single reward for now)
    rsym = list(block.reward.keys())[0]
    
    def bellman_residual_loss(policy_function, value_function, input_grid):
        """
        Compute Bellman equation residuals for given policy and value functions.
        
        Parameters
        ----------
        policy_function : callable
            Current policy function c*(s,ε) 
        value_function : callable
            Current value function approximation V(s)
        input_grid : Grid
            Training data with states and shock realizations
            
        Returns
        -------
        torch.Tensor
            Bellman residuals for each point in the input grid
        """
        given_vals = input_grid.to_dict()
        
        # Extract current period states and shocks
        states_t = {var: given_vals[var] for var in state_variables}
        shocks_t = {sym.replace('_0', ''): given_vals[sym] for sym in shock_syms_t if sym in given_vals}
        
        # Get policy decision for current period
        controls_t = policy_function(states_t, shocks_t, parameters)
        
        # Compute current period reward u(s_t, c_t)
        reward_vals = rf(states_t, shocks_t, controls_t, parameters)
        current_reward = reward_vals[rsym]
        
        # Compute current value function V(s_t)
        current_value = value_function(states_t, parameters)
        
        # Compute next period states s_{t+1} = T(s_t, ε_t, c_t)
        states_t1 = tf(states_t, shocks_t, controls_t, parameters)
        
        # Compute continuation value β * E[V(s_{t+1}) | s_t, c_t]
        if disc_params is not None:
            # Use discretized expectation if discretization parameters provided
            ds = discretized_shock_dstn(shock_vars, disc_params)
            
            def continuation_integrand(shock_array):
                # Convert shock array to shock dictionary
                shock_dict = {var: shock_array[var] for var in ds.variables.keys()}
                
                # Compute next period states with these shocks
                next_states = tf(states_t, shock_dict, controls_t, parameters)
                
                # Evaluate value function at next period states
                return value_function(next_states, parameters)
            
            continuation_value = expected(func=continuation_integrand, dist=ds)
        else:
            # Use next period shocks from input grid for Monte Carlo integration
            shocks_t1 = {sym.replace('_1', ''): given_vals[sym] for sym in shock_syms_t1 if sym in given_vals}
            
            if shocks_t1:
                # Use provided next period shocks
                next_states = tf(states_t, shocks_t1, controls_t, parameters)
            else:
                # If no next period shocks provided, use current period shocks as approximation
                next_states = states_t1
                
            continuation_value = value_function(next_states, parameters)
        
        # Compute Bellman residual: |V(s_t) - [u(s_t,c_t) + β * E[V(s_{t+1})]]|²
        bellman_target = current_reward + discount_factor * continuation_value
        bellman_residual = (current_value - bellman_target) ** 2
        
        return bellman_residual
    
    return bellman_residual_loss


def get_expected_discounted_lifetime_reward_loss(
    state_variables, block, discount_factor, big_t, parameters
):
    """
    Creates an Expected Discounted Lifetime Reward (EDLR) loss function.
    
    This implements the primary MMW JME '21 all-in-one loss function approach
    that maximizes expected discounted lifetime utility through forward simulation.
    This is the main method described in MMW Definition 2.4.
    
    Mathematical Foundation (MMW JME '21):
    Ξ(θ) = E_ω[∑_{t=0}^{T} β^t r(m_t, s_t, φ(m_t, s_t; θ))]
    
    Where:
    - θ are neural network parameters
    - φ(·; θ) is the policy function approximation  
    - r(·) is the period reward function
    - β is the discount factor
    - The expectation is over initial conditions and all future shocks
    
    Uses big_t copies of shocks for multi-period simulation, enabling proper
    integration over the full shock sequence as required by the MMW approach.
    
    Parameters
    -----------
    state_variables : list of str
        Names of state variables that define the problem state space
    block : DBlock
        The economic model block containing dynamics, shocks, and rewards
    discount_factor : float
        Discount factor β from the MMW formulation
    big_t : int
        Number of periods T to simulate forward (finite horizon approximation)
    parameters : dict
        Model calibration parameters
        
    Returns
    --------
    callable
        EDLR loss function with signature: (policy_function, input_grid) -> loss_tensor
        Returns negative lifetime reward for minimization by optimizers
    """
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

    def expected_discounted_lifetime_reward_loss(df: callable, input_grid: Grid):
        """
        Compute Expected Discounted Lifetime Reward loss for given policy function.
        
        This implements the core MMW JME '21 computation by:
        1. Extracting state variables and big_t shock sequences from input grid
        2. Forward simulating the policy for big_t periods  
        3. Computing discounted sum of rewards over the trajectory
        4. Returning negative reward (for minimization by optimizers)
        
        Parameters
        -----------
        df : callable
            Policy function that maps (states, shocks, parameters) -> controls
        input_grid : Grid
            Training grid containing initial states and shock sequences
            
        Returns
        --------
        torch.Tensor
            Negative expected discounted lifetime rewards for each grid point
        """
        # Extract state variables and shock sequences from training grid
        given_vals = input_grid.to_dict()

        # Organize shocks by time period for forward simulation
        shock_vals = {sym: given_vals[sym] for sym in big_t_shock_syms}
        shocks_by_t = {
            sym: torch.stack([shock_vals[f"{sym}_{t}"] for t in range(big_t)])
            for sym in shock_vars
        }

        # Compute expected discounted lifetime reward via forward simulation
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
        # Return negative for minimization (optimizers minimize, but we want to maximize reward)
        return -edlr

    return expected_discounted_lifetime_reward_loss
