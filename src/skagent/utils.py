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


def compute_gradients_for_tensors(
    tensors_dict: dict[str, torch.Tensor],
    wrt: dict[str, torch.Tensor],
    create_graph: bool = False,
) -> dict[str, dict[str, torch.Tensor | None]]:
    """
    Compute gradients for a dictionary of tensors with respect to variables.

    This function computes gradients using PyTorch's autograd. It is used by
    the gradient methods on BellmanPeriod (``grad_reward_function``,
    ``grad_transition_function``, and ``grad_pre_state_function``).

    For batched inputs, this computes the diagonal of the Jacobian: for each
    batch element *i*, we compute ∂target[i]/∂var[i]. The implementation
    uses ``grad(target.sum(), var)``, which yields the correct diagonal
    whenever target[i] depends only on var[i] (agent-independence). This
    assumption holds in the Bellman context because each agent's
    state and reward are computed independently.

    For scalar inputs, ``.sum()`` is a no-op, so the same code path handles
    both cases without branching.

    Parameters
    ----------
    tensors_dict : dict[str, torch.Tensor]
        Dictionary mapping symbol names to tensors to compute gradients for.
    wrt : dict[str, torch.Tensor]
        Dictionary of variables to compute gradients with respect to.
        Keys are variable names, values are tensors with ``requires_grad=True``.
    create_graph : bool, optional
        If True, the graph of the derivative is constructed, allowing
        higher-order derivatives and end-to-end training. Default: False.

    Returns
    -------
    dict[str, dict[str, torch.Tensor | None]]
        Nested dictionary of gradients for each tensor symbol and variable:
        ``{tensor_sym: {var_name: gradient}}``. Gradient is ``None`` if the
        variable does not require gradients or the target does not depend on it.
    """
    from torch.autograd import grad

    gradients: dict[str, dict[str, torch.Tensor | None]] = {}
    for tensor_sym, target_tensor in tensors_dict.items():
        gradients[tensor_sym] = {}
        for var_name, var_tensor in wrt.items():
            if not var_tensor.requires_grad:
                gradients[tensor_sym][var_name] = None
                continue

            grad_result = grad(
                target_tensor.sum(),
                var_tensor,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
            gradients[tensor_sym][var_name] = grad_result[0]

    return gradients
