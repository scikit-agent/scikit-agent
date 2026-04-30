# Utils

This section contains the API documentation for utility functions and helpers.

## General Utilities

The utils module contains general-purpose utility functions.

```{eval-rst}
.. automodule:: skagent.utils
   :members:
```

## Example Usage

### Working with Grids

```python
import skagent as ska

# Create a grid configuration
config = {
    "wealth": {"min": 0.0, "max": 10.0, "count": 50},
    "income": {"min": 0.5, "max": 2.0, "count": 20},
}

# Create grid
grid = ska.Grid.from_config(config)

# Access grid points
wealth_points = grid["wealth"]
income_points = grid["income"]

# Convert to dictionary
grid_dict = grid.to_dict()
```

---

_This page is under construction. Content will be added as the API develops._
