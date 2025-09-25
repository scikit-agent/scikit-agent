import inspect
import logging
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
    # TODO: Can this just be `fun(**vals)` ?
    return fun(*[vals[var] for var in inspect.signature(fun).parameters])


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


def create_vectorized_function_wrapper_with_mapping(lambda_func, param_to_column):
    """
    Create a vectorized wrapper that automatically maps lambda parameters
    to correct tensor columns based on a parameter-to-column mapping.

    Args:
        lambda_func: Original lambda function
        param_to_column: A mapping from parameter names to columns of a tensor
    """

    def wrapper(input_tensor):
        # Extract the relevant columns for each parameter
        param_tensors = {}
        for param_name, col_idx in param_to_column.items():
            param_tensors[param_name] = input_tensor[:, col_idx]

        # Apply function with correct parameter mapping
        try:
            # Try vectorized application first
            result = lambda_func(**param_tensors)
            if not torch.is_tensor(result):
                result = torch.tensor(
                    result, dtype=input_tensor.dtype, device=input_tensor.device
                )
            return result.unsqueeze(1) if result.dim() == 1 else result
        except Exception as e:
            # Fallback to row-by-row if vectorization fails
            logging.warning(f"Vectorization failed ({e}), falling back to row-by-row")
            results = []
            for i in range(input_tensor.shape[0]):
                row_params = {
                    param_name: input_tensor[i, col_idx].item()
                    for param_name, col_idx in param_to_column.items()
                }
                result = lambda_func(**row_params)
                results.append(result)
            return torch.tensor(
                results, dtype=input_tensor.dtype, device=input_tensor.device
            ).unsqueeze(1)

    return wrapper


def extract_parameters(network):
    """Extract all parameters from a PyTorch network into a flat tensor."""
    params = []
    for param in network.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params) if params else torch.tensor([])


def compute_parameter_difference(params1, params2):
    """Compute the L2 norm of the difference between two parameter vectors."""
    if len(params1) != len(params2):
        return float("inf")
    return torch.norm(params1 - params2).item()
