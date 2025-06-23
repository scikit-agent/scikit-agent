from inspect import signature
import numpy as np
import torch


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


def reconcile(vec_a, vec_b):
    """
    Returns a new vector with the values of vec_b but with
    the object type and shape of vec_a.
    """
    target_shape = vec_a.shape

    if isinstance(vec_a, np.ndarray):
        if isinstance(vec_b, torch.Tensor):
            vec_b_np = vec_b.cpu().numpy()
        else:
            vec_b_np = vec_b

        if vec_b_np.shape == target_shape:
            return vec_b_np
        else:
            return vec_b_np.reshape(target_shape)
    if isinstance(vec_a, torch.Tensor):
        if isinstance(vec_b, np.ndarray):
            vec_b_torch = torch.FloatTensor(vec_b).to(vec_a.device)
        else:
            vec_b_torch = vec_b.to(vec_a.device)

        if vec_b_torch.shape == target_shape:
            return vec_b_torch
        else:
            return vec_b_torch.reshape(target_shape)
