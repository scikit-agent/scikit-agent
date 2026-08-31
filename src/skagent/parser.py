from skagent.distributions import Bernoulli, Lognormal, MeanOneLogNormal
from skagent.rule import Rule
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import yaml


class Expression:
    def __init__(self, text):
        self.txt = text
        self.expr = parse_expr(text)
        self.npf = self.func()

        # first derivatives.
        self.grad = {
            sym.__str__(): self.expr.diff(sym) for sym in list(self.expr.free_symbols)
        }

    def func(self):
        return lambdify(list(self.expr.free_symbols), self.expr, "numpy")


def tuple_constructor_from_class(cls):
    def constructor(loader, node):
        value = loader.construct_mapping(node)
        return (cls, value)

    return constructor


CONTROL_FIELDS = ("iset", "lower_bound", "upper_bound", "agent")


def bound_from_text(bound):
    """
    Returns a control bound declared in a document as something
    :func:`skagent.block.normalize_bound` accepts.

    A document cannot hold a callable, so a bound given as an expression is
    compiled into one whose parameter names are the expression's free
    variables. Other declarations are passed through unchanged.
    """
    if isinstance(bound, str):
        return Rule(bound).update_func()
    return bound


def control_constructor(loader, node):
    """
    A PyYAML constructor building a :class:`skagent.block.Control`.
    """
    from skagent.block import Control  # TODO: move to separate module

    args = loader.construct_mapping(node)

    unknown = set(args) - set(CONTROL_FIELDS)
    if unknown:
        raise ValueError(
            f"Control has unknown field(s) {sorted(unknown)}; "
            f"expected some of {list(CONTROL_FIELDS)}."
        )
    if "iset" not in args:
        raise ValueError("Control must declare an information set as 'iset'.")

    iset = args["iset"]
    if isinstance(iset, str):
        iset = [iset]
    if not all(isinstance(sym, str) for sym in iset):
        raise ValueError(f"Control's iset must name variables; got {args['iset']!r}.")

    return Control(
        list(iset),
        lower_bound=bound_from_text(args.get("lower_bound")),
        upper_bound=bound_from_text(args.get("upper_bound")),
        agent=args.get("agent"),
    )


def math_text_to_lambda(text):
    """
    Returns a function represented by the given mathematical text.
    """
    expr = parse_expr(text)
    func = lambdify(list(expr.free_symbols), expr, "numpy")
    return func


def skagent_loader():
    """
    A PyYAML loader that supports tags for scikit-agent,
    such as random variables and model tags.
    """
    loader = yaml.SafeLoader
    yaml.SafeLoader.add_constructor(
        "!Bernoulli", tuple_constructor_from_class(Bernoulli)
    )
    yaml.SafeLoader.add_constructor(
        "!MeanOneLogNormal", tuple_constructor_from_class(MeanOneLogNormal)
    )
    yaml.SafeLoader.add_constructor(
        "!Lognormal", tuple_constructor_from_class(Lognormal)
    )
    yaml.SafeLoader.add_constructor("!Control", control_constructor)

    return loader
