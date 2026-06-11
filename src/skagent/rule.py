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
from skagent.distributions import Distribution
from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify, lambdify


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
    from skagent.block import Control  # TODO: move to separate module

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


def extract_formula(rule):
    """
    Extract formula as string from a rule.

    Parameters
    ----------
    rule : various
        The rule to extract formula from

    Returns
    -------
    str
        The formula as string
    """
    from skagent.block import Control

    if isinstance(rule, Control):
        deps = ", ".join(sorted(rule.iset))
        return f"Control({deps})"

    elif isinstance(rule, str):
        return rule

    elif callable(rule):
        try:
            src = inspect.getsource(rule).strip()
            if "lambda" in src:
                # Extract lambda body
                return src.split(":", 1)[1].strip().rstrip(",)")
            else:
                # Try to get function name and params
                params = list(inspect.signature(rule).parameters.keys())
                return f"Function({', '.join(params)})"
        except (OSError, TypeError):
            print(rule)
            return "Function()"

    elif isinstance(rule, tuple) and len(rule) == 2:
        dist_class, params = rule
        if isinstance(params, dict):
            param_strs = [f"{k}={v}" for k, v in params.items()]
            return f"{dist_class.__name__}({', '.join(param_strs)})"
        return str(rule)

    else:
        return str(rule)


def format_rule(var, rule):
    """
    Get a printable (string) version of a rule.

    Parameters
    ----------
    var : str
        The variable name (LHS of structural statement)
    rule : callable, Control, str, or any
        The rule to format (RHS of structural statement)

    Returns
    -------
    str
        A human-readable string representation of the rule
    """
    formula = extract_formula(rule)
    return f"{var} = {formula}"


class Rule:
    def __init__(self, expression_string):
        """
        Initialize a Rule from a mathematical expression string.

        Args:
            expression_string: A string representing a mathematical expression
        """
        self._original_string = expression_string
        self._sympy_expr = sympify(expression_string)

        # Extract free symbols from the expression for lambdify
        self._symbols = sorted(self._sympy_expr.free_symbols, key=lambda s: s.name)

        self._lambda_func = lambdify(self._symbols, self._sympy_expr)

    @property
    def sympy(self):
        """Access the underlying SymPy expression."""
        return self._sympy_expr

    def update_func(self):
        """
        Returns the update rule as a function
        """
        return self._lambda_func

    def __str__(self):
        """Return the original string representation."""
        return self._original_string

    def __repr__(self):
        """Return a developer-friendly representation."""
        return f"Rule('{self._original_string}')"
