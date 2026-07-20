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
   :no-index:
```

## Rules

The rule module builds on the parser to extract dependencies and formulas from
model rules.

```{eval-rst}
.. automodule:: skagent.rule
   :members:
```

## Example Usage

### Parsing Mathematical Expressions

```python
import inspect

from skagent.parser import math_text_to_lambda

# Convert a string expression to a callable function
reward_func = math_text_to_lambda("c**(1-rho)/(1-rho)")

# The positional argument order follows the expression's free symbols.
# Check it before calling the function positionally:
inspect.signature(reward_func)  # (c, rho)

reward = reward_func(1.5, 2.0)
```

```{warning}
Variable names that collide with built-in SymPy objects are parsed as those
objects rather than as free variables. For example, `gamma` is parsed as the
gamma *function*, so `math_text_to_lambda("c**(1-gamma)/(1-gamma)")` raises a
`TypeError`. Prefer names like `rho` or `CRRA` for curvature parameters.
```
