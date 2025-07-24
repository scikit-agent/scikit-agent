# Algorithms

scikit-agent includes several algorithms for solving economic models.

## Overview

The algorithms module provides implementations of various solution methods commonly used in computational economics:

- **Maliar, Maliar, and Winant (2021)**: Neural network-based solution methods
- **Value function iteration**: Traditional dynamic programming approaches
- **Simulation-based methods**: Monte Carlo techniques

## Maliar method

The Maliar method uses neural networks to approximate policy and value functions:

```python
import skagent.algos.maliar as maliar
from skagent.models.consumer import consumption_block

# Create transition function for the model
state_variables = ["assets", "income"]
transition_func = maliar.create_transition_function(
    consumption_block, 
    state_variables
)
```

## Value function iteration

Classical dynamic programming approach:

```python
from skagent.algos import vbi

# Set up value function iteration parameters
vfi_params = {
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "discount_factor": 0.95
}

# Run value function iteration
# (implementation depends on specific model structure)
```

## Monte Carlo simulation

For simulating model solutions:

```python
from skagent.simulation.monte_carlo import draw_shocks

# Draw random shocks for simulation
n_periods = 100
n_agents = 1000

# Define shock distributions
shock_config = {
    "income": ("MeanOneLogNormal", {"sigma": 0.1}),
    "survival": ("Bernoulli", {"p": 0.98})
}

# Generate shock realizations
shocks = draw_shocks(shock_config, n_periods, n_agents)
```

## Neural network approximation

Using PyTorch for function approximation:

```python
import skagent.ann as ann
import torch

# Define network architecture
network = ann.create_network(
    input_dim=2,    # state variables
    hidden_dims=[64, 64],
    output_dim=1    # policy/value function
)

# Train the network
# (training loop implementation depends on specific application)
```

## Algorithm selection

Choose algorithms based on your model characteristics:

- **Small state spaces**: Use value function iteration
- **Large/continuous state spaces**: Use neural network methods (Maliar)
- **Heterogeneous agent models**: Combine simulation with approximation methods
- **Stochastic models**: Use Monte Carlo simulation for shock realizations

## Performance considerations

- **GPU acceleration**: Use PyTorch backend for large-scale problems
- **Parallel processing**: Leverage multiple cores for simulation
- **Memory management**: Consider grid size and batch processing for large models