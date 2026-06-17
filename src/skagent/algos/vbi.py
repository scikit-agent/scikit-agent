"""
Use backwards induction to derive the arrival value function
from a continuation value function and stage dynamics.
"""

from skagent.block import DBlock
from inspect import signature
import itertools
import numpy as np
from scipy.optimize import minimize
from typing import Mapping, Sequence
import xarray as xr


def get_action_rule(action):
    """
    Produce a function from any inputs to a given value.
    This is useful for constructing decision rules with fixed actions.
    """

    def ar():
        return action

    return ar


def ar_from_data(da):
    """
    Build a decision rule from a fitted policy ``DataArray``.

    The returned rule follows the library's decision-rule calling convention:
    it takes the control's information-set values as *positional* arguments, in
    the order of ``da.dims`` (which :func:`solve` aligns to the control's
    ``iset``). This matches how ``block.transition`` invokes a rule,
    ``dr(*[vals[v] for v in iset])``, so a VBI-fitted rule is a drop-in for the
    rest of the stack.

    Interpolation runs in numpy/xarray space. Scalar arguments return a Python
    scalar; array-like arguments are interpolated *pointwise* (not as an outer
    product) and return a numpy array. For a torch-tensor interface, wrap this
    with :func:`tensor_decision_rule`.
    """
    dims = list(da.dims)

    def ar(*args):
        if len(args) != len(dims):
            raise TypeError(
                f"decision rule for dims {dims} expects {len(dims)} positional "
                f"argument(s), got {len(args)}"
            )
        if len(dims) == 0:
            # empty information set: a constant rule
            return da.values.tolist()

        batched = any(np.ndim(a) > 0 for a in args)
        if batched:
            # vectorized (pointwise) interpolation: share a single dimension
            # across all coordinate indexers so xarray does not take the outer
            # product of the inputs.
            coords = {
                dim: xr.DataArray(np.asarray(arg).ravel(), dims="_point")
                for dim, arg in zip(dims, args)
            }
            return da.interp(**coords).values

        coords = {dim: arg for dim, arg in zip(dims, args)}
        return da.interp(**coords).values.tolist()

    return ar


def tensor_decision_rule(np_rule, dtype=None, device=None):
    """
    Wrap a numpy-space decision rule (e.g. from :func:`ar_from_data`, including
    the rules returned by :func:`solve`) so it speaks torch tensors, for interop
    with the torch solving stack (``BellmanPeriod`` / ``loss`` / ``solver``).

    Inputs may be torch tensors or numpy/scalars; outputs are torch tensors.
    Because interpolation runs in numpy, the graph is *severed*: the returned
    controls are detached. This rule is therefore valid as a fixed /
    ground-truth / warm-start policy (e.g. an ``other_dr`` or in a value-
    residual loss), but not as a trainable policy in a loss that differentiates
    through the control (FOC-weighted Bellman, Euler).

    Defaults to float32 on the stack's device to match grid tensors
    (``grid.py`` builds them with ``torch.FloatTensor``).
    """
    import torch
    from skagent.grid import device as default_device

    dtype = torch.float32 if dtype is None else dtype
    device = default_device if device is None else device

    def tdr(*args):
        np_args = [a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args]
        out = np_rule(*np_args)
        return torch.as_tensor(np.asarray(out), dtype=dtype, device=device)

    return tdr


Grid = Mapping[str, Sequence]


def align_to_iset(da, iset):
    """
    Reduce a fitted policy ``DataArray`` to a control's information set.

    The state grid may carry dimensions outside the control's ``iset`` (e.g. a
    shock that only enters the transition, like ``psi`` in conftest ``case_3``).
    A decision rule is a function of the iset alone, so those extra dimensions
    are reduced by selecting the first slice. This assumes the optimal policy
    does not vary across non-iset dimensions, which is exactly what it means for
    a variable to be outside the information set. Remaining dimensions are
    ordered to match ``iset`` so positional calls line up.
    """
    non_iset = [d for d in da.dims if d not in iset]
    if non_iset:
        da = da.isel({d: 0 for d in non_iset})
    if iset:
        da = da.transpose(*iset)
    return da


def grid_to_data_array(
    grid: Grid = {},  ## TODO: Better data structure here.
):
    """
    Construct a zero-valued DataArray with the coordinates
    based on the Grid passed in.

    Parameters
    ----------
    grid: Grid
        A mapping from variable labels to a sequence of numerical values.

    Returns
    --------
    da xarray.DataArray
        An xarray.DataArray with coordinates given by both grids.
    """

    coords = {**grid}

    da = xr.DataArray(
        np.empty([len(v) for v in coords.values()]), dims=coords.keys(), coords=coords
    )

    return da


def solve(
    block: DBlock, continuation, state_grid: Grid, disc_params={}, calibration={}
):
    """
    Solve a DBlock using backwards induction on the value function.

    Parameters
    -----------
    block
    continuation

    state_grid: Grid
        This is a grid over all variables that the optimization will range over.
        This should be just the information set of the decision variables.

    disc_params
    calibration
    """

    # state-rule value function
    srv_function = block.get_state_rule_value_function_from_continuation(
        continuation, screen=True
    )

    # get_controls() returns a dict[sym, Control]; VBI works with the
    # ordered list of control symbols.
    controls = list(block.get_controls())

    # pseudo
    policy_data = grid_to_data_array(state_grid)
    value_data = grid_to_data_array(state_grid)

    # loop through every point in the state grid
    for state_point in itertools.product(*state_grid.values()):
        # build a dictionary from these states, as scope for the optimization
        state_vals = {k: v for k, v in zip(state_grid.keys(), state_point)}

        # The value of the action is computed given
        # the problem calibration and the states for the current point on the
        # state-grid.
        pre_states = calibration.copy()
        pre_states.update(state_vals)

        # prepare function to optimize
        def negated_value(a):
            dr = {c: get_action_rule(a[i]) for i, c in enumerate(controls)}

            # negative, for minimization later
            return -srv_function(pre_states, dr)

        if len(controls) == 0:
            # if no controls, no optimization is necessary
            pass
        elif len(controls) == 1:
            ## get lower bound.
            ## assumes only one control currently
            lower_bound = -1e12  # a very low number
            feq = block.dynamics[controls[0]].lower_bound
            if feq is not None:
                lower_bound = feq(
                    *[pre_states[var] for var in signature(feq).parameters]
                )

            ## get upper bound
            ## assumes only one control currently
            upper_bound = 1e12  # a very high number
            feq = block.dynamics[controls[0]].upper_bound

            if feq is not None:
                upper_bound = feq(
                    *[pre_states[var] for var in signature(feq).parameters]
                )

            bounds = ((lower_bound, upper_bound),)

            res = minimize(  # choice of
                negated_value,
                1,  # x0 is starting guess, here arbitrary.
                bounds=bounds,
            )

            dr_best = {c: get_action_rule(res.x[i]) for i, c in enumerate(controls)}

            if res.success:
                policy_data.sel(**state_vals).variable.data.put(
                    0, res.x[0]
                )  # will only work for scalar actions
                value_data.sel(**state_vals).variable.data.put(
                    0, srv_function(pre_states, dr_best)
                )
            else:
                print(f"Optimization failure at {state_vals}.")
                print(res)

                dr_best = {c: get_action_rule(res.x[i]) for i, c in enumerate(controls)}

                policy_data.sel(**state_vals).variable.data.put(0, res.x[0])  # ?
                value_data.sel(**state_vals).variable.data.put(
                    0, srv_function(pre_states, dr_best)
                )
        elif len(controls) > 1:
            raise Exception(
                f"Value backup iteration is not yet implemented for stages with {len(controls)} > 1 control variables."
            )

    # use the xarray interpolator to create a decision rule, reduced to each
    # control's information set so the rule is a function of the iset alone.
    dr_from_data = {
        c: ar_from_data(align_to_iset(policy_data, list(block.dynamics[c].iset)))
        for c in controls
    }

    dec_vf = block.get_decision_value_function(dr_from_data, continuation)
    arr_vf = block.get_arrival_value_function(disc_params, dr_from_data, continuation)

    return dr_from_data, dec_vf, arr_vf
