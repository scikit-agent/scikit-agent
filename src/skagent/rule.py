"""
Universal rule processing module for scikit-agent models.

This module provides utilities for working with "rules" - the right-hand side
expressions in structural statements that define model dynamics, controls, and rewards.

A rule can be:
- A callable (function, lambda)
- A Control object
- A Distribution or tuple with distribution parameters
- A string expression
- A constant value

Key functions:
- format_rule: Get printable string version of a rule
- extract_dependencies: Get variables that a rule depends on
"""

import re
import inspect
from skagent.model import Control
from HARK.distributions import Distribution

_TOKEN_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def extract_dependencies(rule):
    """
    Extract variable dependencies from different rule types.

    Takes the "right hand side" of a model statement -- which can be a callable
    'structural equation', or the information needed to construct a distribution,
    and so on -- and returns the variables that statement depends on.

    Parameters
    ----------
    rule : various
        Can be Control, Distribution, callable, tuple, or string

    Returns
    -------
    list
        List of dependency variable names
    """
    deps = []

    if isinstance(rule, Control):
        # Control has explicit information set
        deps = list(rule.iset)
    elif isinstance(rule, Distribution):
        # Distribution might depend on calibration parameters
        # This would require inspecting the distribution parameters
        # For now, we'll need to handle this case-by-case
        pass
    elif isinstance(rule, tuple) and len(rule) == 2:
        # Shock definition with (Distribution, params)
        dist_class, params = rule
        if isinstance(params, dict):
            for param_expr in params.values():
                if isinstance(param_expr, str):
                    # Extract variables from string expressions
                    deps.extend(_TOKEN_RE.findall(param_expr))
    elif isinstance(rule, str):
        # String expression
        deps = _TOKEN_RE.findall(rule)
    elif callable(rule):
        # Callable function
        try:
            deps = list(inspect.signature(rule).parameters.keys())
        except Exception:
            # Fallback: parse source code
            try:
                src = inspect.getsource(rule)
                deps = _TOKEN_RE.findall(src)
            except Exception:
                pass

    return deps


