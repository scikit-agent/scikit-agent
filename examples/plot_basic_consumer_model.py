"""
Basic Consumer Model Example
============================

This example demonstrates how to create and work with a basic consumer model
using scikit-agent's modeling framework.
"""

# %%
# First, let's import the necessary modules
import numpy as np
from skagent.models.consumer import consumption_block, calibration

# %%
# Examine the default calibration parameters
print("Default calibration parameters:")
for key, value in calibration.items():
    print(f"  {key}: {value}")

# %%
# The consumption block contains the core dynamics of consumer behavior
print(f"\nConsumption block name: {consumption_block.name}")
print(f"Shocks defined: {list(consumption_block.shocks.keys())}")
print(f"Dynamics defined: {list(consumption_block.dynamics.keys())}")

# %%
# Let's examine the shock distributions
print("\nShock distributions:")
for shock_name, (dist_class, params) in consumption_block.shocks.items():
    print(f"  {shock_name}: {dist_class.__name__} with params {params}")