"""
Grid and Simulation Basics
==========================

This example demonstrates how to use scikit-agent's grid and simulation
functionality for economic modeling.
"""

# %%
# Import necessary modules
import numpy as np
from skagent.grid import Grid, make_grid
import matplotlib.pyplot as plt

# %%
# Create a simple grid configuration for asset values
# This is commonly used in dynamic programming solutions
asset_config = {
    "assets": {
        "min": 0.0,
        "max": 50.0,
        "count": 100
    }
}

# Create the grid
asset_grid_values = make_grid(asset_config)
asset_grid = Grid(["assets"], asset_grid_values)

print(f"Grid has {asset_grid.n()} points")
print(f"Grid shape: {asset_grid.shape()}")
print(f"First 5 grid points: {asset_grid.values[:5].flatten()}")
print(f"Last 5 grid points: {asset_grid.values[-5:].flatten()}")

# %%
# Visualize the grid
plt.figure(figsize=(12, 4))
asset_values = asset_grid.values.flatten()
plt.plot(asset_values, np.zeros_like(asset_values), 'bo', markersize=2)
plt.xlabel('Asset Value')
plt.title('Linear Asset Grid')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Create a 2D grid with assets and income
two_d_config = {
    "assets": {"min": 0.0, "max": 20.0, "count": 20},
    "income": {"min": 0.5, "max": 2.0, "count": 15}
}

grid_2d_values = make_grid(two_d_config)
grid_2d = Grid(["assets", "income"], grid_2d_values)

print(f"2D Grid has {grid_2d.n()} points")
print(f"2D Grid shape: {grid_2d.shape()}")

# %%
# Visualize the 2D grid
plt.figure(figsize=(10, 8))
asset_vals = grid_2d.values[:, 0]
income_vals = grid_2d.values[:, 1]
plt.scatter(asset_vals, income_vals, alpha=0.6, s=20)
plt.xlabel('Assets')
plt.ylabel('Income')
plt.title('2D Grid: Assets vs Income')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Convert grid to dictionary format for easier access
grid_dict = grid_2d.to_dict()
print("Grid as dictionary:")
print(f"Assets column shape: {grid_dict['assets'].shape}")
print(f"Income column shape: {grid_dict['income'].shape}")
print(f"First 5 asset values: {grid_dict['assets'][:5]}")
print(f"First 5 income values: {grid_dict['income'][:5]}")