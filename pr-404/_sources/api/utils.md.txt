# Utils

This section contains the API documentation for utility functions and helpers.

## General Utilities

The utils module contains general-purpose utility functions.

```{eval-rst}
.. automodule:: skagent.utils
   :members:
```

## Example Usage

### Calling a Function from a Dictionary of Values

`apply_fun_to_vals` calls a function using only the entries of a dictionary that
match the function's named arguments, ignoring the rest:

```python
from skagent.utils import apply_fun_to_vals


def transition(a, b):
    return a + b


# The extra key "c" is ignored
apply_fun_to_vals(transition, {"a": 1.0, "b": 2.0, "c": 99.0})  # 3.0
```

### Smooth Complementarity Conditions

`fischer_burmeister` replaces the complementarity conditions
$a \geq 0,\; h \geq 0,\; ah = 0$ with a single smooth equation, which is useful
for occasionally binding constraints in loss functions:

```python
import torch

from skagent.utils import fischer_burmeister

a = torch.tensor([0.0, 1.0, 3.0])
h = torch.tensor([2.0, 0.0, 4.0])
fischer_burmeister(a, h)  # tensor([0., 0., 2.])
```

```{note}
Grid construction tools (`Grid`, `make_grid`, `cartesian_product`) live in
`skagent.grid` and are documented on the {doc}`algorithms` page.
```
