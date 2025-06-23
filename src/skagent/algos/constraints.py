import torch
from inspect import signature

"""
Constraint handling utilities for economic models.

This module provides functions for handling inequality constraints and
complementarity conditions in economic models, particularly for neural
network-based solution methods like those in the Maliar, Maliar, and Winant
(2021) framework.

Key Functions
=============
- `fischer_burmeister()`: Smooth, differentiable constraint handling using FB function
- `get_constraint_violations()`: Extract constraint violations from DBlock models

Mathematical Foundation
=====================
The Fischer-Burmeister function provides a smooth, differentiable way to handle
complementarity conditions a ≥ 0, b ≥ 0, a·b = 0 that arise in economic models
with occasionally binding constraints.

Applications
============
- Borrowing constraints in consumption-savings models
- Non-negativity constraints on consumption and assets
- Kuhn-Tucker conditions in optimization problems
- Complementarity conditions in equilibrium models

Usage
=====
This module is used by the MMW loss functions in `skagent.algos.maliar` to handle
constraint violations in a differentiable manner suitable for neural network training.

Example:
--------
```python
from skagent.algos.constraints import fischer_burmeister, get_constraint_violations

# Extract constraint violations
violations = get_constraint_violations(block, states, controls, parameters)

# Apply Fischer-Burmeister penalty
for constraint_name, violation in violations.items():
    multiplier = torch.ones_like(violation) * 0.1
    fb_residual = fischer_burmeister(violation, multiplier)
    penalty += torch.mean(fb_residual**2)
```

References
==========
Fischer, A. (1992). A special Newton-type optimization method.
Optimization, 24(3-4), 269-284.

Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic
economic models. Journal of Monetary Economics, 122, 76-101.
"""


def fischer_burmeister(a, b):
    """
    Fischer-Burmeister function for handling complementarity constraints.

    The Fischer-Burmeister function provides a smooth, differentiable way to
    handle complementarity conditions a ≥ 0, b ≥ 0, a·b = 0. This is essential
    for economic models with occasionally binding constraints like borrowing limits.

    Mathematical Definition:
    FB(a,b) = a + b - sqrt(a² + b²)

    Properties:
    - FB(a,b) = 0 when complementarity conditions are satisfied
    - Smooth and differentiable everywhere (unlike min(a,b))
    - FB(a,b) < 0 when constraints are violated
    - FB(a,b) = 0 when constraints are exactly satisfied

    Applications in Economic Models:
    - Borrowing constraints: FB(c - c_max, λ) = 0
    - Non-negativity: FB(c, λ_c) = 0 where λ_c is multiplier on c ≥ 0
    - Asset constraints: FB(a - a_min, λ_a) = 0

    Parameters
    -----------
    a : torch.Tensor or float
        First argument (often constraint violation: constraint - bound)
    b : torch.Tensor or float
        Second argument (often Lagrange multiplier)

    Returns
    --------
    torch.Tensor or float
        Fischer-Burmeister function value. Zero when complementarity satisfied.

    Examples
    --------
    >>> # Borrowing constraint: consumption ≤ cash_on_hand
    >>> constraint_violation = cash_on_hand - consumption  # Should be ≥ 0
    >>> multiplier = lagrange_multiplier  # Should be ≥ 0
    >>> fb_residual = fischer_burmeister(constraint_violation, multiplier)
    >>> constraint_penalty = torch.mean(fb_residual**2)

    References
    ----------
    Fischer, A. (1992). A special Newton-type optimization method.
    Optimization, 24(3-4), 269-284.

    Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving
    dynamic economic models. Journal of Monetary Economics, 122, 76-101.
    """
    # Ensure inputs are tensors for proper gradient computation
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)

    # FB(a,b) = a + b - sqrt(a² + b²)
    # Note: This equals 0 when complementarity satisfied, negative when violated
    return a + b - torch.sqrt(a**2 + b**2)


def get_constraint_violations(block, states, controls, parameters):
    """
    Extract constraint violations for Fischer-Burmeister function application.

    Analyzes the economic model block to identify constraint violations that
    should be penalized using Fischer-Burmeister complementarity conditions.
    This enables proper handling of occasionally binding constraints.

    Common Economic Constraints:
    - Borrowing limit: consumption ≤ cash-on-hand
    - Non-negativity: consumption ≥ 0, assets ≥ 0
    - Resource constraints: income ≥ consumption + savings

    Parameters
    -----------
    block : DBlock
        Economic model block containing constraint specifications
    states : dict
        Current state variables {symbol: values}
    controls : dict
        Current control variables {symbol: values}
    parameters : dict
        Model calibration parameters

    Returns
    --------
    dict
        Dictionary of constraint violations {constraint_name: violation_amount}
        Positive violations indicate constraint is satisfied
        Negative violations indicate constraint is violated
    """
    violations = {}

    # Check upper bound constraints from DBlock Control specifications
    dynamics = block.get_dynamics()
    for control_name in block.get_controls():
        control = dynamics[control_name]

        if hasattr(control, "upper_bound") and control.upper_bound is not None:
            # Handle upper bound constraint
            upper_bound = control.upper_bound

            if isinstance(upper_bound, str):
                # Upper bound is a symbol name (e.g., 'c <= m')
                if upper_bound in states:
                    upper_bound_value = states[upper_bound]
                elif upper_bound in controls:
                    upper_bound_value = controls[upper_bound]
                elif upper_bound in parameters:
                    upper_bound_value = parameters[upper_bound]
                else:
                    continue  # Skip if symbol not found
            elif callable(upper_bound):
                # Upper bound is a function
                bound_args = signature(upper_bound).parameters
                bound_vals = {}
                all_vals = {**parameters, **states, **controls}
                for arg_name in bound_args:
                    if arg_name in all_vals:
                        bound_vals[arg_name] = all_vals[arg_name]

                if not bound_vals:
                    continue

                try:
                    upper_bound_value = upper_bound(**bound_vals)
                except Exception:
                    continue
            else:
                # Upper bound is a constant
                upper_bound_value = upper_bound

            control_value = controls[control_name]

            # Constraint violation: upper_bound - control_value
            # Positive when constraint satisfied, negative when violated
            violations[f"{control_name}_upper"] = upper_bound_value - control_value

        # Check lower bound constraints from DBlock Control specifications
        if hasattr(control, "lower_bound") and control.lower_bound is not None:
            # Handle lower bound constraint
            lower_bound = control.lower_bound

            if isinstance(lower_bound, str):
                # Lower bound is a symbol name
                if lower_bound in states:
                    lower_bound_value = states[lower_bound]
                elif lower_bound in controls:
                    lower_bound_value = controls[lower_bound]
                elif lower_bound in parameters:
                    lower_bound_value = parameters[lower_bound]
                else:
                    continue  # Skip if symbol not found
            elif callable(lower_bound):
                # Lower bound is a function
                bound_args = signature(lower_bound).parameters
                bound_vals = {}
                all_vals = {**parameters, **states, **controls}
                for arg_name in bound_args:
                    if arg_name in all_vals:
                        bound_vals[arg_name] = all_vals[arg_name]

                if not bound_vals:
                    continue

                try:
                    lower_bound_value = lower_bound(**bound_vals)
                except Exception:
                    continue
            else:
                # Lower bound is a constant
                lower_bound_value = lower_bound

            control_value = controls[control_name]

            # Constraint violation: control_value - lower_bound
            # Positive when constraint satisfied, negative when violated
            violations[f"{control_name}_lower"] = control_value - lower_bound_value
        else:
            # Default non-negativity constraint if no lower bound specified
            if control_name in controls:
                control_value = controls[control_name]
                violations[f"{control_name}_lower"] = control_value  # c ≥ 0

    # Non-negativity constraints for controls without explicit lower bounds
    for control_name, control_value in controls.items():
        if f"{control_name}_lower" not in violations:
            violations[f"{control_name}_nonnegativity"] = torch.clamp(
                -control_value, min=0
            )

    return violations
