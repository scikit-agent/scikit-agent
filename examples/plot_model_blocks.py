"""
Model Building with Blocks
==========================

This example demonstrates how to work with scikit-agent's block-based
modeling system for building economic models.
"""

# %%
# Import the model building components
from skagent.model import DBlock, RBlock, Control
from skagent.distributions import Bernoulli, MeanOneLogNormal
import numpy as np

# %%
# Examine the consumption block from the consumer model
from skagent.models.consumer import consumption_block, calibration

print("Consumption Block Structure:")
print(f"Name: {consumption_block.name}")
print(f"Block type: {type(consumption_block).__name__}")

# %%
# Look at the shocks defined in the consumption block
print("\nShocks defined in consumption block:")
for shock_name, (dist_class, params) in consumption_block.shocks.items():
    print(f"  {shock_name}:")
    print(f"    Distribution: {dist_class.__name__}")
    print(f"    Parameters: {params}")

# %%
# Look at the dynamics in the consumption block  
print("\nDynamics defined in consumption block:")
dynamics_keys = list(consumption_block.dynamics.keys())
print(f"Number of dynamics: {len(dynamics_keys)}")
print(f"First few dynamics: {dynamics_keys[:5]}")

# %%
# Examine the calibration parameters
print("\nCalibration parameters:")
for param, value in calibration.items():
    print(f"  {param}: {value}")

# %%
# Create a simple example block for demonstration
simple_block = DBlock(
    name="simple_example",
    shocks={
        "epsilon": (MeanOneLogNormal, {"sigma": 0.1}),
        "alive": (Bernoulli, {"p": 0.98})
    },
    dynamics={
        "income": "w * epsilon",
        "assets_next": "assets * R + income - consumption",
        "utility": "consumption ** (1 - CRRA) / (1 - CRRA)"
    }
)

print(f"\nSimple block name: {simple_block.name}")
print(f"Simple block shocks: {list(simple_block.shocks.keys())}")
print(f"Simple block dynamics: {list(simple_block.dynamics.keys())}")

# %%
# Create an even simpler demonstration of model components
print("\nModel building components:")
print("- DBlock: Dynamic blocks that define state transitions")
print("- RBlock: Reward/return blocks for optimization") 
print("- Control: Variables that agents choose optimally")
print("- Shocks: Random variables that affect the model")
print("- Dynamics: Equations that govern state evolution")

# %%
# Show how parameters work
sample_params = {
    "w": 1.0,      # wage rate
    "R": 1.03,     # interest rate  
    "CRRA": 2.0,   # risk aversion
}

print(f"\nSample parameters for economic model:")
for param, value in sample_params.items():
    print(f"  {param}: {value}")
    
print("\nThese parameters would be used in the dynamics equations")
print("to compute outcomes like income, asset evolution, and utility.")