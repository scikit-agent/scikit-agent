# Parsing

The parser module provides tools for converting mathematical expressions to
callable functions.

```{eval-rst}
.. automodule:: skagent.parser
   :members:
```

## Core Parser Functions

```{eval-rst}
.. autofunction:: skagent.parser.math_text_to_lambda
   :no-index:
```

## Parser Classes

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
