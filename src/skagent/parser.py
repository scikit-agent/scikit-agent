from collections.abc import Mapping
import dataclasses

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


def _block_class(document):
    """The block class a document builds: recursive if it holds sub-blocks."""
    from skagent.block import DBlock, RBlock  # TODO: move to separate module

    return RBlock if "blocks" in document else DBlock


def validate_block(document, name=None):
    """Refuse a block document whose keys are not a block's keys.

    A block declares its parts at one level -- ``shocks``, ``dynamics``,
    ``reward`` -- and names its variables one level below. A part indented one
    step too deep is still valid YAML: it becomes a VARIABLE of the part above
    it, so the block silently loses that whole section and gains a symbol
    nobody declared. Nothing downstream can tell that apart from a block that
    never had the section, which is why it is caught here rather than left to
    fail later.

    A document holding ``blocks`` is read as a recursive block, and its
    sub-blocks are validated too.

    Parameters
    ----------
    document : Mapping
        One block, as a document holds it. A model document -- a calibration
        beside a list of blocks -- is not a block; validate its blocks.
    name : str, optional
        What to call the block when something is wrong with it. Defaults to
        the document's own ``name``.

    Raises
    ------
    ValueError
        If a key of the document is not one the block has a place for, or if
        one of the block's own keys is used as a variable inside another.
    """
    cls = _block_class(document)
    fields = frozenset(f.name for f in dataclasses.fields(cls))
    label = name or document.get("name") or "<unnamed>"

    unknown = sorted(set(document) - fields)
    if unknown:
        raise ValueError(
            f"block {label!r} declares {unknown}, which a {cls.__name__} has no "
            f"place for; a block's keys are {sorted(fields)}."
        )

    for section in ("shocks", "dynamics", "reward"):
        part = document.get(section)
        if not isinstance(part, Mapping):
            continue
        misplaced = sorted(set(part) & fields)
        if misplaced:
            raise ValueError(
                f"block {label!r} names {misplaced} as a variable of {section!r}, "
                f"and those are the block's own keys -- so they are indented one "
                f"level too deep. The block has no {misplaced[0]!r} of its own, "
                f"and {section!r} has gained a variable nothing declared."
            )

    for sub in document.get("blocks", []):
        validate_block(sub)


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
