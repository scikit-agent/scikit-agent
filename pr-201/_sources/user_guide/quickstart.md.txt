# Quickstart Guide

Get up and running with scikit-agent in minutes. This guide covers the basic
concepts and shows you how to create, solve, and simulate your first economic
model.

## Installation

First, install scikit-agent:

```bash
pip install scikit-agent
```

For development installation:

```bash
git clone https://github.com/scikit-agent/scikit-agent.git
cd scikit-agent
pip install -e ".[dev,docs]"
```

## Core Concepts

scikit-agent is built around several key concepts:

### DBlock (Dynamic Block)

A `DBlock` represents a "block" of model behavior with:

- **Shocks**: Random variables that affect the model
- **Dynamics**: Equations describing how variables evolve
- **Controls**: Decision variables that agents optimize
- **Rewards**: Objective functions that agents maximize

### Grid

A `Grid` represents discrete state spaces for numerical solutions.

### Simulators

Monte Carlo simulation engines for analyzing model behavior.

## Your First Model: Consumption-Saving

Let's create a simple consumption-saving model step by step.

### Step 1: Import scikit-agent

```python
import numpy as np
import skagent as ska
from skagent.models.consumer import consumption_block, calibration
```

### Step 2: Examine the Pre-built Model

The consumption block defines a standard consumption-saving problem:

```python
print("Model dynamics:")
for var, eq in consumption_block.dynamics.items():
    print(f"  {var}: {eq}")

print("\nModel shocks:")
for shock, dist in consumption_block.shocks.items():
    print(f"  {shock}: {dist}")
```

### Step 3: Set Up Model Parameters

```python
# Use the default calibration or modify it
my_calibration = calibration.copy()
my_calibration.update(
    {
        "DiscFac": 0.95,  # Discount factor
        "CRRA": 2.5,  # Risk aversion
        "R": 1.04,  # Interest rate
    }
)

print("Calibration:")
for param, value in my_calibration.items():
    print(f"  {param}: {value}")
```

### Step 4: Construct Shocks

```python
# Build the shock distributions using calibration
consumption_block.construct_shocks(my_calibration)
```

### Step 5: Create a Simple Decision Rule

For this example, we'll use a simple consumption rule (consume 90% of
resources):

```python
def simple_consumption_rule(m):
    """Simple rule: consume 90% of market resources"""
    return 0.9 * m


# Wrap in the format expected by the simulator
decision_rules = {"c": simple_consumption_rule}
```

### Step 6: Run a Simulation

```python
# Set up initial conditions
initial_conditions = {
    "k": 1.0,  # Initial capital
    "p": 1.0,  # Initial permanent income
}

# Create simulator
simulator = ska.MonteCarloSimulator(
    calibration=my_calibration,
    block=consumption_block,
    dr=decision_rules,
    initial=initial_conditions,
    agent_count=1000,
    T_sim=50,
    seed=42,
)

# Run simulation
print("Running simulation...")
results = simulator.simulate()

print(f"Simulation completed. History keys: {list(results.keys())}")
```

### Step 7: Analyze Results

```python
import matplotlib.pyplot as plt

# Plot average consumption over time
if "c" in simulator.history:
    consumption_data = np.array(simulator.history["c"])
    mean_consumption = np.mean(consumption_data, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_consumption)
    plt.title("Average Consumption Over Time")
    plt.xlabel("Period")
    plt.ylabel("Consumption")
    plt.grid(True)
    plt.show()
```

## Working with Grids

For more sophisticated solution methods, you'll work with grids:

```python
# Create a grid for wealth
wealth_grid_config = {"m": {"min": 0.1, "max": 10.0, "count": 50}}

wealth_grid = ska.Grid.from_config(wealth_grid_config)
print(f"Grid shape: {wealth_grid.shape()}")
print(f"First few wealth points: {wealth_grid['m'][:5]}")
```

## Neural Network Solutions

scikit-agent supports neural network-based solution methods:

```python
# Create a neural network policy
policy_net = ska.BlockPolicyNet(consumption_block, width=64)

print(f"Neural network input size: {policy_net.hidden1.in_features}")
print(f"Neural network output size: {policy_net.output.out_features}")
```

## Next Steps

After completing this quickstart:

- **Learn more about blocks**: Read the {doc}`blocks` guide to understand
  different model types and how to create custom models
- **Explore algorithms**: Check out the {doc}`algorithms` guide for different
  solution methods including value function iteration and neural network
  approaches
- **Master simulation**: See the {doc}`simulation` guide for advanced simulation
  techniques and analysis methods
- **Browse examples**: Look at the {doc}`../auto_examples/index` for more
  complex examples and use cases

## Key Takeaways

- **DBlocks** are the fundamental building blocks for economic models
- **Grids** discretize continuous state spaces for numerical methods
- **Simulators** generate synthetic data from solved models
- **Decision rules** map states to optimal actions
- The package follows scikit-learn conventions for a familiar API

You're now ready to build more sophisticated economic models with scikit-agent!

---

_This page is under construction. More detailed examples will be added._
