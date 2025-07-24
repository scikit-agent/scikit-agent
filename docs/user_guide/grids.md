# Grids

Grids are essential for discretizing continuous state spaces in dynamic programming and numerical methods.

## Grid creation

### Basic grid configuration

```python
from skagent.grid import Grid, make_grid

# Define grid configuration
config = {
    "assets": {"min": 0.0, "max": 50.0, "count": 100},
    "income": {"min": 0.5, "max": 2.0, "count": 50}
}

# Create grid values
grid_values = make_grid(config)
grid = Grid(["assets", "income"], grid_values)

print(f"Grid shape: {grid.shape()}")
print(f"Number of points: {grid.n()}")
```

### Single-dimension grids

For simple one-dimensional grids:

```python
asset_config = {
    "assets": {"min": 0.0, "max": 20.0, "count": 50}
}

asset_values = make_grid(asset_config)
asset_grid = Grid(["assets"], asset_values)
```

## Grid operations

### Converting to dictionary format

```python
grid_dict = grid.to_dict()
print(f"Asset values: {grid_dict['assets'][:5]}")
print(f"Income values: {grid_dict['income'][:5]}")
```

### Tensor operations

Grids support PyTorch tensors for GPU acceleration:

```python
# Convert to PyTorch tensors
grid.torch()
print(f"Grid values type: {type(grid.values)}")
```

### Grid from existing values

You can also create grids from pre-computed values:

```python
import numpy as np

# Create custom grid points
asset_points = np.linspace(0, 20, 30)
income_points = np.linspace(0.5, 2.0, 20)

# Create meshgrid
assets_mesh, income_mesh = np.meshgrid(asset_points, income_points, indexing='ij')
values = np.column_stack([assets_mesh.flatten(), income_mesh.flatten()])

# Create grid object
custom_grid = Grid(["assets", "income"], values)
```

## Common patterns

### Logarithmic spacing

For variables that span multiple orders of magnitude:

```python
import numpy as np

# Create logarithmically spaced asset grid
log_assets = np.logspace(np.log10(0.1), np.log10(100), 50)
config = {
    "assets": {"min": log_assets.min(), "max": log_assets.max(), "count": len(log_assets)}
}
```

### Non-uniform grids

For better resolution in important regions:

```python
# More points near zero for assets
fine_region = np.linspace(0, 5, 30)
coarse_region = np.linspace(5, 50, 20)[1:]  # Exclude duplicate at 5
combined_assets = np.concatenate([fine_region, coarse_region])
```