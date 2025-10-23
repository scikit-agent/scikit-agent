"""
FormulaGenerator: Extracts and formats human-readable formulas from
scikit-agent DBlock/RBlock models.
"""

import inspect
from skagent.model import Control


def _extract_formula(rule):
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
    formula = _extract_formula(rule)
    return f"{var} = {formula}"


class FormulaGenerator:
    """
    Analyzes a scikit-agent model to generate a dictionary of formulas.
    """

    def __init__(self, model, calibration):
        """
        Parameters
        ----------
        model : DBlock or RBlock
            The model to analyze
        calibration : dict
            Calibration parameters
        """
        self.model = model
        self.calibration = calibration
        self._blocks = list(self.model.iter_dblocks())

    def generate(self):
        """
        Generate and return a dictionary of all formulas in the model.
        """
        formulas = {}
        # Process dynamics from all blocks
        for blk in self._blocks:
            for var, rule in blk.get_dynamics().items():
                formulas[var] = format_rule(var, rule)

        # Process parameters
        for param, value in self.calibration.items():
            formulas[param] = format_rule(param, value)

        return formulas
