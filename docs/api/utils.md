# Utils

This section contains the API documentation for utility functions and helpers.

## Parser Utilities

The parser module provides tools for converting mathematical expressions to
callable functions.

```{eval-rst}
.. automodule:: skagent.parser
   :members:
```

### Core Parser Functions

```{eval-rst}
.. autofunction:: skagent.parser.math_text_to_lambda
   :no-index:
```

### Parser Classes

```{eval-rst}
.. autoclass:: skagent.parser.ControlToken
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. autoclass:: skagent.parser.Expression
   :members:
   :undoc-members:
   :show-inheritance:
```

## General Utilities

The utils module contains general-purpose utility functions.

```{eval-rst}
.. automodule:: skagent.utils
   :members:
```

## Example Usage

### Parsing Mathematical Expressions

```python
from skagent.parser import math_text_to_lambda

# Convert string expression to callable function
utility_func = math_text_to_lambda("c**(1-gamma)/(1-gamma)")

# Use the function
gamma = 2.0
c = 1.5
utility = utility_func(c, gamma)
```

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
