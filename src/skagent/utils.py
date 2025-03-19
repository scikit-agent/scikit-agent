from inspect import signature


def apply_fun_to_vals(fun, vals):
    """
    Applies a function to the arguments defined in `vals`.
    This is equivalent to `fun(**vals)`, except
    that `vals` may contain keys that are not named arguments
    of `fun`.

    Parameters
    ----------
    fun: callable

    vals: dict
    """
    return fun(*[vals[var] for var in signature(fun).parameters])
