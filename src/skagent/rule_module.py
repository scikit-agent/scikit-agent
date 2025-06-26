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


def format_rule(var, rule):
    """
    Get a printable (string) version of a rule.
    
    This is the main entry point for rule formatting that handles
    all types of rules uniformly.
    
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
        
    Examples
    --------
    >>> format_rule("c", lambda m: m * 0.8)
    "c = m * 0.8"
    
    >>> format_rule("a", Control(["m"], agent="consumer"))
    "a = Control(m)"
    
    >>> format_rule("y", "theta * k")
    "y = theta * k"
    """
    if isinstance(rule, Control):
        return format_control_rule(var, rule)
    elif callable(rule):
        return format_callable_rule(var, rule)
    elif isinstance(rule, str):
        return format_string_rule(var, rule)
    else:
        return format_constant_rule(var, rule)


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


def format_callable_rule(var, rule):
    """
    Format a callable rule (function/lambda) into a string.
    
    Parameters
    ----------
    var : str
        The variable name
    rule : callable
        The callable rule
        
    Returns
    -------
    str
        Formatted rule string
    """
    try:
        src = inspect.getsource(rule).strip()
        if "lambda" in src:
            # Extract lambda body
            body = src.split(":", 1)[1].strip()
            # Remove trailing comma or parenthesis
            body = body.rstrip(",)")
            return f"{var} = {body}"
        else:
            return f"{var} = [Function]"
    except Exception:
        return f"{var} = [Unknown]"


def format_control_rule(var, control):
    """
    Format a Control rule into a string.
    
    Parameters
    ----------
    var : str
        The variable name
    control : Control
        The Control object
        
    Returns
    -------
    str
        Formatted Control rule string
    """
    deps = sorted(control.iset)
    bounds_info = []
    if control.lower_bound:
        bounds_info.append("lower_bound")
    if control.upper_bound:
        bounds_info.append("upper_bound")
    bounds_str = f", {', '.join(bounds_info)}" if bounds_info else ""
    return f"{var} = Control({', '.join(deps)}{bounds_str})"


def format_string_rule(var, expression):
    """
    Format a string expression rule.
    
    Parameters
    ----------
    var : str
        The variable name
    expression : str
        The mathematical expression string
        
    Returns
    -------
    str
        Formatted rule string
    """
    return f"{var} = {expression}"


def format_constant_rule(var, value):
    """
    Format a constant value rule.
    
    Parameters
    ----------
    var : str
        The variable name
    value : any
        The constant value
        
    Returns
    -------
    str
        Formatted rule string
    """
    return f"{var} = {value}"


def validate_rule(rule):
    """
    Validate that a rule is well-formed.
    
    Parameters
    ----------
    rule : any
        The rule to validate
        
    Returns
    -------
    bool
        True if rule is valid
        
    Raises
    ------
    ValueError
        If rule is invalid
    """
    if isinstance(rule, Control):
        if not rule.iset:
            raise ValueError("Control rule must have non-empty information set")
        return True
    elif callable(rule):
        try:
            inspect.signature(rule)
            return True
        except Exception as e:
            raise ValueError(f"Invalid callable rule: {e}")
    elif isinstance(rule, str):
        if not rule.strip():
            raise ValueError("String rule cannot be empty")
        return True
    else:
        # Constants are always valid
        return True


def get_rule_type(rule):
    """
    Determine the type of a rule.
    
    Parameters
    ----------
    rule : any
        The rule to classify
        
    Returns
    -------
    str
        Rule type: 'control', 'callable', 'string', 'distribution', 'tuple', or 'constant'
    """
    if isinstance(rule, Control):
        return 'control'
    elif isinstance(rule, Distribution):
        return 'distribution'
    elif isinstance(rule, tuple):
        return 'tuple'
    elif callable(rule):
        return 'callable'
    elif isinstance(rule, str):
        return 'string'
    else:
        return 'constant'