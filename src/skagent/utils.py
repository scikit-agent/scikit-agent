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


def compute_gradients_for_tensors(tensors_dict, wrt):
    """
    Compute gradients for a dictionary of tensors with respect to variables.

    This function computes gradients using PyTorch's autograd, handling both
    scalar and batched tensor cases. It is used by grad_reward_function and
    grad_transition_function in BellmanPeriod.

    Parameters
    ----------
    tensors_dict : dict
        Dictionary mapping symbol names to tensors to compute gradients for
    wrt : dict
        Dictionary of variables to compute gradients with respect to.
        Keys are variable names, values are tensors with requires_grad=True

    Returns
    -------
    dict
        Nested dictionary of gradients for each tensor symbol and variable:
        {tensor_sym: {var_name: gradient}}
    """
    from torch.autograd import grad

    gradients = {}
    for tensor_sym in tensors_dict:
        gradients[tensor_sym] = {}
        for var_name, var_tensor in wrt.items():
            # Skip if variable doesn't require gradients
            if not var_tensor.requires_grad:
                gradients[tensor_sym][var_name] = None
                continue

            # Compute gradient of this tensor with respect to this variable
            target_tensor = tensors_dict[tensor_sym]

            # For batched computations, we need to compute gradients for each element
            if target_tensor.dim() > 0 and target_tensor.numel() > 1:
                # Handle batched case: compute gradients for each element in the batch
                batch_gradients = []
                for i in range(target_tensor.shape[0]):
                    grad_result = grad(
                        target_tensor[i],
                        var_tensor,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grad_result[0] is not None:
                        g = grad_result[0]
                        if g.dim() > 0 and g.shape[0] > i:
                            batch_gradients.append(g[i])
                        else:
                            # Scalar or non-matching shape - use as-is
                            batch_gradients.append(g)
                    else:
                        batch_gradients.append(None)

                # Stack the gradients if they're not None
                if all(g is not None for g in batch_gradients):
                    gradients[tensor_sym][var_name] = torch.stack(batch_gradients)
                else:
                    gradients[tensor_sym][var_name] = None
            else:
                # Handle scalar case
                grad_result = grad(
                    target_tensor, var_tensor, retain_graph=True, allow_unused=True
                )
                gradients[tensor_sym][var_name] = (
                    grad_result[0] if grad_result[0] is not None else None
                )

    return gradients
