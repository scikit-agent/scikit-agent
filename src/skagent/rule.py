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

import inspect
from skagent.model import Control
from skagent.distributions import Distribution
from sympy.parsing.sympy_parser import parse_expr


def math_text_to_free_variable_names(txt):
    """
    Extract free variable names from mathematical text using SymPy.

    Parameters
    ----------
    txt : str
        Mathematical expression as string

    Returns
    -------
    list
        List of free variable names

    Examples
    --------
    >>> math_text_to_free_variable_names("10 * x + y **2 - z")
    ['x', 'y', 'z']
    """
    try:
        return [sym.name for sym in parse_expr(txt).free_symbols]
    except Exception:
        # If SymPy fails, return empty list
        return []


def extract_dependencies(rule):
    """
    Extract variable dependencies from a model rule.

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
        deps = list(rule.iset)
    elif isinstance(rule, Distribution):
        pass
    elif isinstance(rule, tuple) and len(rule) == 2:
        dist_class, params = rule
        if isinstance(params, dict):
            for param_expr in params.values():
                if isinstance(param_expr, str):
                    deps.extend(math_text_to_free_variable_names(param_expr))
    elif isinstance(rule, str):
        deps = math_text_to_free_variable_names(rule)
    elif callable(rule):
        # Check if it's a built-in or module function
        module = getattr(rule, "__module__", None)
        if module in ("builtins", "math", "numpy", "scipy") or module is None:
            # Built-in functions, math functions, etc. don't depend on model variables
            deps = []
        else:
            try:
                deps = list(inspect.signature(rule).parameters.keys())
            except Exception:
                # For other callables that fail signature inspection,
                # just return empty list since they don't depend on model variables
                deps = []

    return deps
